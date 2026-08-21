# -*- coding: utf-8 -*-
"""Protein Benchmark Suite -- frozen-probe evaluation of protein models.

Embeds sequences once with a frozen model, fits a small probe (linear, knn or
histgb) on those embeddings, and writes one CSV row per task x seed x probe x
split. Also runs non-neural baselines (k-mer, and via sibling scripts MMseqs2
and phmmer) through the identical splits and metrics, so they are comparable.

    python protein_benchmark_suite.py --list_tasks
    python protein_benchmark_suite.py -m facebook/esm2_t6_8M_UR50D \
        --tasks solubility -p linear --eval_split test

Task counts and preset membership are deliberately not written here; they go
stale. Ask the code: --list_tasks. Full usage is in README.md.
"""

import argparse
import functools
import gc
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for thread_env_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(thread_env_var, "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

BENCHMARK_SEED = 42


def seed_all(seed: int) -> None:
    """Seed every global RNG the benchmark can reach.

    BENCHMARK_SEED alone only covers what it is explicitly threaded into --
    sklearn's random_state and datasets.shuffle. Anything reaching for a global
    RNG instead (torch init, an unseeded permutation, dataloader shuffling,
    fine-tuning dropout) was previously free to vary between runs that both
    reported the same seed.

    Deliberately not calling torch.use_deterministic_algorithms(True): it makes
    several embedding kernels error or fall back to far slower paths, and
    embedding is inference-only here, so it buys nothing.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # covers CUDA too


# Bootstrap resamples for metric CIs. 0 disables; --bootstrap N sets it.
BOOTSTRAP_N = 0
# Above this many evaluation rows the bootstrap draw count is scaled down: the
# interval is already ~+/-0.001 and each draw recomputes the full metric block.
_BOOTSTRAP_FULL_ROWS = 50_000
_BOOTSTRAP_MIN_DRAWS = 100
# Training proteins for the pairwise contact probe; --contact_train_proteins.
CONTACT_TRAIN_PROTEINS = 400


def _boot_ci(metric_fn, y_true, y_pred, n_boot: int, seed: int) -> Dict[str, float]:
    """Percentile CIs by resampling test predictions and recomputing metrics.

    ``metric_fn(y_true, y_pred) -> dict`` is the caller's existing metric block,
    so the interval is computed for exactly the metrics it already reports and
    nothing has to be reimplemented per metric. No refitting: the probe is
    fixed, and this is the sampling distribution of the *test estimate*, which
    is the quantity a "is this gap real?" question is about.
    """
    if not n_boot:
        return {}
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_boot = bootstrap_draws_for(n_boot, len(y_true))
    draws: Dict[str, List[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            sample = metric_fn(y_true[idx], y_pred[idx])
        except ValueError:
            # A resample can drop a class entirely; skip rather than distort.
            continue
        for key, value in sample.items():
            draws.setdefault(key, []).append(value)

    # Refuse to report an interval built from a handful of surviving resamples.
    # On a task with rare classes most draws can fail, and a 2.5th percentile
    # over a dozen values looks exactly like one over a thousand in the CSV.
    usable = {k: v for k, v in draws.items() if len(v) >= max(20, n_boot // 10)}
    if len(usable) < len(draws):
        logger.warning(
            "  Bootstrap: dropped CIs for %s (too few valid resamples of %d; "
            "resampling kept dropping a class)",
            sorted(set(draws) - set(usable)),
            n_boot,
        )
    return {
        f"{key}_CI_{side}": float(np.percentile(values, pct))
        for key, values in usable.items()
        for side, pct in (("low", 2.5), ("high", 97.5))
    }


DEFAULT_EMBED_MAX_LENGTH = 1024
DEFAULT_BLAS_THREAD_LIMIT = 1
# KNN keeps n_jobs=1 (threading backend; higher values can hit OpenBLAS thread limits)
DEFAULT_KNN_N_JOBS = 1
# OvR linear probes use process-based parallelism (loky); each worker inherits
# OPENBLAS_NUM_THREADS=1 so no thread explosion regardless of cpu_count.
# Use a bounded value: n_jobs=-1 spawns cpu_count() workers which causes IPC
# overhead to dominate for the ~12k-sample tasks here.
DEFAULT_OVR_N_JOBS = int(os.environ.get("PROTEIN_BENCH_OVR_JOBS", "8"))
# Legacy alias kept for NearestNeighbors retrieval call (n_jobs=1 is safe there).
DEFAULT_SKLEARN_N_JOBS = DEFAULT_KNN_N_JOBS

# NOTE: datasets is imported locally in task evaluation functions to avoid
# corrupting ESMplusplus models (which have issues with pyarrow/datasets library)
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MultiLabelBinarizer,
    StandardScaler,
)
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from benchmark_comparison import compare_benchmarks, display_comparison
from kmer_baseline import kmer_features, parse_kmer_model_name
from benchmark_tasks import (
    DEFAULT_TASKS,
    FAST_MAX_SAMPLES,
    FAST_TOKEN_CLASS_MAX_SAMPLES,
    FAST_TASKS,
    VERY_FAST_TASKS,
    RETRIEVAL_TASKS,
    PROTEINGYM_TASKS,
    TASK_NAME_TO_KEY as _TASK_NAME_TO_KEY,
    TASKS,
    TaskConfig,
)
from benchmark_utils import (
    DEFAULT_RESULT_EVAL_MODE,
    DEFAULT_RESULT_EVAL_SPLIT,
    DEFAULT_RESULT_EVAL_STRATEGY,
    DEFAULT_RESULT_PROBE,
)
from model_utils import (
    _assert_no_wrapper_prefixes,
    apply_esmplusplus_compat_patch,
    detect_model_type,
    disable_esm2_token_dropout,
    fix_amplify_meta_tensors,
    fix_proteva_rope_buffer,
    from_pretrained_with_flash,
    get_torch_compile_settings,
    needs_esm2_token_dropout_workaround,
    patch_amplify_attention_fallback,
    patch_unknown_residue_tokens,
    _prepare_amplify_inputs,
)

# Reduce TensorFlow log noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORT", "1")

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("default", category=ConvergenceWarning)

apply_esmplusplus_compat_patch()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Reduce noisy HTTP logs from Hugging Face hubs/datasets
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
DEFAULT_BENCHMARK_EVAL_SPLIT = "validation"
SUPPORTED_EVAL_SPLITS = {"validation", "test"}
_VALIDATION_SPLIT_ALIASES = ("validation", "valid", "val", "dev", "eval")
PROBE_LABELS = {
    "auto": "Auto (fastest probe per task)",
    "linear": "Linear",
    "torch_linear": "Torch Linear (AdamW, early stopping)",
    "histgb": "HistGradientBoosting",
    "knn": "K-Nearest Neighbors",
}
# Multilabel tasks (GO/EC) only have linear heads: OvR LogisticRegression or
# one multi-output torch head. knn/histgb are silently the linear evaluator there.
MULTILABEL_PROBES = frozenset({DEFAULT_RESULT_PROBE, "torch_linear"})
# Tasks where sklearn's solver scales badly in the number of outputs or rows, so
# ``-p auto`` routes them to the torch head. Measured on ESM-C 300M, full data:
# remote_homology (1195 classes) 61s lbfgs vs 2.2s; ec_classification (572
# labels, one liblinear fit per label) 156s vs 15s; ss3 615s and
# conservation_flip 749s single-core at 38GB RSS (~2.8M residues). Everything
# else is faster in sklearn.
_AUTO_TORCH_TASKS = frozenset({"remote_homology", "cath_eat"})
_MODEL_SIGNATURE_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.bin",
    "*/*.json",
    "*/*.safetensors",
    "*/*.bin",
)


def safe_model_name(model_name: str) -> str:
    """Convert a model name/path into a filesystem-safe identifier."""
    return model_name.replace("/", "_").replace("\\", "_")


def probe_label(probe_type: str) -> str:
    """Return a human-readable label for a probe type."""
    return PROBE_LABELS[probe_type]


def _model_signature_paths(model_path: Path) -> list[Path]:
    """Collect a small, stable set of checkpoint files for cache namespacing."""
    if model_path.is_file():
        return [model_path]

    candidates: set[Path] = set()
    for pattern in _MODEL_SIGNATURE_PATTERNS:
        candidates.update(path for path in model_path.glob(pattern) if path.is_file())
    if not candidates:
        return [model_path]
    return sorted(candidates)[:32]


def _model_cache_namespace(model_name: str) -> str:
    """Build a cache namespace that changes when a local checkpoint changes."""
    model_path = Path(model_name)
    safe_name = safe_model_name(model_name)
    if not model_path.exists():
        return safe_name

    signature_parts = [str(model_path.resolve())]
    for path in _model_signature_paths(model_path):
        try:
            stat = path.stat()
        except OSError:
            continue
        relative_path = path.name
        if model_path.is_dir():
            relative_path = str(path.relative_to(model_path))
        signature_parts.append(f"{relative_path}:{stat.st_size}:{stat.st_mtime_ns}")

    digest = hashlib.sha256("|".join(signature_parts).encode("utf-8")).hexdigest()
    return f"{safe_name}_{digest[:12]}"


def _clear_model_cache_dirs(embed_cache_dir: str, model_name: str) -> int:
    """Remove all cache directories for a model name, regardless of version suffix."""
    cache_root = Path(embed_cache_dir)
    if not cache_root.exists():
        return 0

    removed = 0
    for cache_dir in cache_root.glob(f"{safe_model_name(model_name)}*"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            removed += 1
    return removed


def _result_eval_mode(cfg: TaskConfig) -> str:
    """Return the persisted evaluation mode for result rows."""
    return cfg.eval_mode or DEFAULT_RESULT_EVAL_MODE


def _auto_probe_type(cfg: TaskConfig) -> str:
    """Resolve ``-p auto`` to the probe that is fastest for this task's shape.

    Single source of the routing rule: the CLI, ``scripts/run_bench.py`` and any
    other caller all come through here, so a task cannot get a different probe
    depending on how the benchmark was invoked.
    """
    task_key = _TASK_NAME_TO_KEY.get(cfg.name, cfg.name)
    if (
        cfg.problem_type in {"multilabel", "token_classification"}
        or task_key in _AUTO_TORCH_TASKS
    ):
        return "torch_linear"
    return DEFAULT_RESULT_PROBE


def effective_probe_type(cfg: TaskConfig, requested_probe: str) -> str:
    """Return the probe label that reflects the evaluator actually used.

    ``auto`` resolves per task (see ``_auto_probe_type``) and is never persisted
    as-is -- the CSV records what actually ran. Retrieval and ProteinGym
    zero-shot evaluations have no probe; multilabel only supports the linear
    heads (``MULTILABEL_PROBES``). Anything else is persisted as the default
    linear probe identity for apples-to-apples comparisons.
    """
    if requested_probe == "auto":
        requested_probe = _auto_probe_type(cfg)
    if cfg.problem_type == "retrieval":
        return DEFAULT_RESULT_PROBE
    if cfg.problem_type == "multilabel":
        return requested_probe if requested_probe in MULTILABEL_PROBES else DEFAULT_RESULT_PROBE
    if cfg.eval_mode == "proteingym_zeroshot":
        return DEFAULT_RESULT_PROBE
    return requested_probe


def _make_probe_variant_label(
    probe_type: str,
    l2_normalize_embeddings: bool = False,
    knn_weights: str = "uniform",
) -> str:
    """Return probe label with KNN variant suffixes for result tracking."""
    if probe_type != "knn":
        return probe_type

    label = "knn"
    if l2_normalize_embeddings:
        label += "_l2"
    if knn_weights == "distance":
        label += "_dist"
    return label


def _progress_bars_enabled(local_rank: Optional[int] = None) -> bool:
    """Return whether tqdm-style progress bars should be shown.

    Resolution order:
    1) `PROTEIN_PROGRESS_BARS=on|off` forces behavior.
    2) `auto` (default) enables bars on rank 0.
    """
    raw = os.environ.get("PROTEIN_PROGRESS_BARS", "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False

    rank = local_rank
    if rank is None:
        try:
            rank = int(os.environ.get("LOCAL_RANK", "0"))
        except ValueError:
            rank = 0
    if rank > 0:
        return False

    return True


def _configure_tqdm_defaults(progress_enabled: bool) -> float:
    """Configure tqdm update cadence to reduce log spam.

    Returns the effective min-interval in seconds.
    """
    if not progress_enabled:
        return 0.0

    raw = os.environ.get("PROTEIN_PROGRESS_MIN_INTERVAL", "5.0").strip()
    try:
        interval = float(raw)
    except ValueError:
        interval = 5.0
    interval = max(0.5, interval)

    os.environ["TQDM_MININTERVAL"] = f"{interval}"
    os.environ.setdefault("TQDM_MINITERS", "1")
    os.environ.setdefault("TQDM_DYNAMIC_NCOLS", "1")

    if sys.stderr.isatty() and os.environ.get("TERM", "") != "dumb":
        os.environ.setdefault("TQDM_POSITION", "0")

    return interval


def parse_seed_list(seed_text: str) -> list[int]:
    """Parse a comma-separated list of integer benchmark seeds.

    Args:
        seed_text: String such as ``"42,67,73"``.

    Returns:
        Parsed benchmark seeds in input order.

    Raises:
        ValueError: If no valid seeds are provided.
    """

    seeds: list[int] = []
    for token in seed_text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        seeds.append(int(stripped))
    if not seeds:
        raise ValueError("At least one benchmark seed is required.")
    return seeds


# =============================================================================
# Model Loading (SentenceTransformer + HF Fallback)
# =============================================================================


def _fix_sbert_tokenizer(model) -> None:
    """Fix tokenizer for SentenceTransformer-wrapped models.

    Handles three cases:
    - ESMplusplus/FastPLM: replace ST tokenizer with the model's native tokenizer
    - ESM2: disable token_dropout bug (HuggingFace transformers >=5.x)
    - Other models: resize embeddings if tokenizer/model vocab sizes mismatch
    """
    if not (hasattr(model, "_modules") and len(model._modules) > 0):
        return

    first_module = list(model._modules.values())[0]
    if not hasattr(first_module, "auto_model"):
        return

    auto_model = first_module.auto_model

    if needs_esm2_token_dropout_workaround(auto_model):
        disable_esm2_token_dropout(auto_model)

    # Check if model has a native tokenizer (ESMplusplus)
    native_tokenizer = None
    if hasattr(auto_model, "tokenizer") and auto_model.tokenizer is not None:
        native_tokenizer = auto_model.tokenizer
    elif hasattr(auto_model, "model") and hasattr(auto_model.model, "tokenizer"):
        native_tokenizer = auto_model.model.tokenizer

    if native_tokenizer is not None:
        logger.info("-> Detected ESMplusplus model, using native tokenizer")
        first_module.tokenizer = native_tokenizer
    elif hasattr(first_module, "tokenizer"):
        # Check for vocab mismatch on non-ESMplusplus models
        tokenizer = first_module.tokenizer
        embedding_layer = auto_model.get_input_embeddings()
        if embedding_layer is not None:
            model_vocab_size = embedding_layer.num_embeddings
            tokenizer_vocab_size = len(tokenizer)

            if tokenizer_vocab_size != model_vocab_size:
                logger.warning(
                    f"VOCAB MISMATCH: Tokenizer={tokenizer_vocab_size}, Model={model_vocab_size}"
                )
                if tokenizer_vocab_size > model_vocab_size:
                    logger.info(
                        f"Resizing embeddings: {model_vocab_size} -> {tokenizer_vocab_size}"
                    )
                    auto_model.resize_token_embeddings(tokenizer_vocab_size)
                    logger.info("Embeddings resized successfully")


def resolve_proteva_runtime(torch_dtype, is_cpu):
    """Weight dtype + ``flash_attn_mode`` for the Proteva encoder, honoring the
    requested precision instead of force-casting bf16.

    The frozen probe defaults to fp32 (``torch_dtype`` None) so precision is a
    non-confound across models and near-degenerate-sequence tasks (GB1, DMS subs)
    keep their sub-bf16-noise-floor signal. ``fa2-varlen`` is a bf16/fp16-only
    kernel, so fp32 MUST use the dense SDPA path (``"off"``) — bit-identical to
    AMPLIFY in fp32 (verified). Explicit bf16 keeps the fast flash kernel. CPU is
    always fp32 + ``"off"`` (no flash_attn; bf16 matmul unsupported/slow there).
    """
    if is_cpu:
        return torch.float32, "off"
    if torch_dtype == torch.bfloat16:
        return torch.bfloat16, "fa2-varlen"
    return torch.float32, "off"


class _KmerEmbedder:
    """Marker for the k-mer baseline, carrying k through to embed_sequences."""

    def __init__(self, k: int):
        self.k = k


def load_model(
    model_name: str,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
):
    """Load a model, then make its tokenizer tolerant of unknown residues.

    A thin wrapper over ``_load_model_impl`` rather than a patch inside it: the
    impl has eight return paths and the guard has to cover every one. FastPLM's
    tokenizer raises ``KeyError`` on any out-of-vocabulary character, which takes
    down a whole task -- CATH lookup69k alone contains a NUL byte that does it.
    """
    obj, is_sbert, dev = _load_model_impl(
        model_name, device, torch_dtype, attn_implementation
    )
    if is_sbert:
        tokenizer = getattr(obj, "tokenizer", None)
    elif isinstance(obj, tuple):
        tokenizer = obj[0]
    else:
        tokenizer = None
    # The embedding path prefers model.tokenizer over the one returned here, so
    # patch both -- they are often different objects.
    inner = obj[1] if isinstance(obj, tuple) and len(obj) == 2 else obj
    for tok in (tokenizer, getattr(inner, "tokenizer", None)):
        if tok is not None:
            patch_unknown_residue_tokens(tok)
    return obj, is_sbert, dev


def _load_model_impl(
    model_name: str,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
):
    """
    Load a model with SentenceTransformer priority, then HF AutoModel fallback.

    Args:
        model_name: HuggingFace model name or local path
        device: Device to load model on
        torch_dtype: Optional dtype for model weights (e.g., torch.bfloat16)
        attn_implementation: Optional explicit attention backend override for
            HF model loads. Supported values: ``flash_attention_2``, ``sdpa``,
            ``eager``. ``None`` uses model_utils auto-selection.

    Returns:
        Tuple of (model_obj, is_sbert, device)
        - model_obj: SentenceTransformer, (tokenizer, model) tuple
        - is_sbert: Boolean indicating if it's a SentenceTransformer
        - device: The device being used
    """
    # "kmer" / "kmer4" is the no-learning baseline: fixed k-mer frequency
    # vectors, no weights to load and nothing to put on a device. It rides the
    # normal path from here on so it gets the same probes, splits and metrics.
    kmer_k = parse_kmer_model_name(model_name)
    if kmer_k is not None:
        logger.info("Using the k-mer baseline (k=%d, %d dims)", kmer_k, 20**kmer_k)
        return _KmerEmbedder(kmer_k), False, "cpu"

    if not torch.cuda.is_available():
        print("WARNING! No GPU/CUDA available!")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model: {model_name}")

    model_type = detect_model_type(model_name)

    # Prefer SentenceTransformer when a local ST checkpoint is detected,
    # except for ESMplusplus/AMPLIFY/FastPLM/DPLM2/E1 where we need custom embedding handling.
    model_path = Path(model_name)
    if model_path.exists() and model_type not in {
        "amplify",
        "esmplusplus",
        "fastplm_esm2",
        "dplm2",
        "profluent_e1",
        "proteva",
    }:
        if (model_path / "modules.json").exists() or (
            model_path / "config_sentence_transformers.json"
        ).exists():
            _assert_no_wrapper_prefixes(model_name)
            try:
                from sentence_transformers import SentenceTransformer

                model_kwargs = {}
                if torch_dtype is not None:
                    model_kwargs["dtype"] = torch_dtype

                model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                    device=device,
                    model_kwargs=model_kwargs,
                )
                _fix_sbert_tokenizer(model)

                logger.info("-> Loaded as SentenceTransformer (local)")
                return model, True, device
            except Exception as e:
                logger.info(
                    f"SentenceTransformer load failed ({type(e).__name__}: {e})"
                )
                logger.info("Falling back to HuggingFace AutoModel...")

    hf_load_kwargs: Dict[str, Any] = {}
    if torch_dtype is not None:
        hf_load_kwargs["dtype"] = torch_dtype
    if attn_implementation is not None:
        hf_load_kwargs["attn_implementation"] = attn_implementation

    if model_type == "amplify":
        logger.info("-> Detected AMPLIFY model, loading with AutoModel")
        model = from_pretrained_with_flash(AutoModel, model_name, **hf_load_kwargs)
        fix_amplify_meta_tensors(model)
        patch_amplify_attention_fallback(model)
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("-> Loaded as HF AutoModel (AMPLIFY)")
        return (tokenizer, model), False, device

    if model_type == "fastplm_esm2":
        logger.info("-> Detected FastPLM ESM2 model, loading with AutoModelForMaskedLM")
        model = from_pretrained_with_flash(
            AutoModelForMaskedLM, model_name, **hf_load_kwargs
        )
        model.to(device).eval()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        logger.info("-> Loaded as HF AutoModelForMaskedLM (FastPLM ESM2)")
        return (tokenizer, model), False, device

    if model_type == "dplm2":
        logger.info("-> Detected DPLM2 model, loading with AutoModel")
        model = from_pretrained_with_flash(AutoModel, model_name, **hf_load_kwargs)
        model.to(device).eval()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        logger.info("-> Loaded as HF AutoModel (DPLM2)")
        return (tokenizer, model), False, device

    if model_type == "profluent_e1":
        logger.info("-> Detected Profluent-E1 model, loading with AutoModelForMaskedLM")
        model = from_pretrained_with_flash(
            AutoModelForMaskedLM, model_name, **hf_load_kwargs
        )
        model.to(device).eval()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        logger.info("-> Loaded as HF AutoModelForMaskedLM (Profluent-E1)")
        return (tokenizer, model), False, device

    if model_type == "esmplusplus":
        logger.info("-> Detected ESMplusplus model, loading with AutoModelForMaskedLM")
        model = from_pretrained_with_flash(
            AutoModelForMaskedLM, model_name, **hf_load_kwargs
        )
        model.to(device).eval()
        tokenizer = model.tokenizer
        logger.info("-> Loaded as HF AutoModelForMaskedLM (ESMplusplus)")
        return (tokenizer, model), False, device

    if model_type == "proteva":
        # Proteva is this project's HF-native encoder. It is benchmarked the
        # SAME way it was trained: BF16 weights + the model's own
        # ``flash_attn_mode="fa2-varlen"`` attention (block-diagonal varlen over
        # packed, UNPADDED sequences). We deliberately do NOT pass an HF
        # ``attn_implementation`` here — the encoder reads ``flash_attn_mode``
        # from ``config.encoder_config`` and dispatches flash-attn internally;
        # HF's attn dispatch does not apply to this custom model. Embeddings are
        # produced by the varlen path in ``embed_sequences`` (pack -> forward ->
        # segment-mean-pool), never the padded/SDPA-dense path.
        import plm.hf  # noqa: F401  (registers ProtevaConfig/ProtevaForPretraining)

        logger.info("-> Detected Proteva model, loading with AutoModel (fp32->BF16, fa2-varlen override)")
        # Build with INFERENCE-SAFE kernels, overriding whatever the checkpoint
        # trained with (weights are identical; outputs are equivalent — both
        # validated):
        #   * flash_attn_mode="fa2-varlen": fa3-varlen NaNs on real multi-segment
        #     packed inputs in eval-mode bf16 (100% NaN, verified); fa2-varlen is
        #     block-diagonal + bit-exact-validated. (fa3 is fine for TRAINING.)
        #   * fused_rmsnorm=False: native RMSNorm — no FLA/FLA_TILELANG dependency
        #     in the bench env; numerically equivalent to the fused kernel.
        # Load in fp32 FIRST then cast to bf16: loading directly in bf16 makes
        # __init__ compute the RoPE sin/cos cache in bf16 -> NaN embeddings.
        from plm.hf.config import ProtevaConfig

        _cfg = ProtevaConfig.from_pretrained(model_name)
        _is_cpu = str(device).startswith("cpu")
        # Honor the requested precision (was force-bf16, which made AMPLIFY-fp32
        # vs Proteva-bf16 an unfair comparison on precision-sensitive tasks).
        # fp32 (default) -> dense SDPA ("off") since fa2-varlen is bf16-only;
        # explicit bf16 -> the fast flash kernel.
        _weight_dtype, _flash_mode = resolve_proteva_runtime(torch_dtype, _is_cpu)
        if isinstance(getattr(_cfg, "encoder_config", None), dict):
            _cfg.encoder_config["flash_attn_mode"] = _flash_mode
            _cfg.encoder_config["fused_rmsnorm"] = False
        # Load fp32 FIRST then cast (loading directly in bf16 makes __init__
        # compute the RoPE sin/cos cache in bf16 -> NaN); fix_proteva_rope_buffer
        # below recomputes it in fp32 regardless. fp32 weight_dtype -> no-op cast.
        # torch.compile saves every state-dict key '_orig_mod.'-prefixed. Loading
        # such a checkpoint matches NOTHING and HF returns a SILENTLY
        # randomly-initialized body -> every probe scores at chance (AUC 0.5000,
        # Spearman 0.0000) while looking like a real result. The final root save
        # (<out>/model.safetensors) is written unwrapped and is CLEAN; the periodic
        # checkpoint-N/ saves are written from the compiled module and are PREFIXED.
        # This branch calls AutoModel.from_pretrained directly, so it bypasses the
        # strip + key-overlap guard inside model_utils.from_pretrained_with_flash —
        # invoke both explicitly here. _validate_local_checkpoint_integrity is the
        # hard gate: it RAISES rather than let a mismatched load proceed.
        from model_utils import _validate_local_checkpoint_integrity
        from plm.hf.checkpoint_utils import strip_orig_mod_prefix

        # NOT wrapped in try/except: strip_orig_mod_prefix returns 0 WITHOUT writing
        # when the checkpoint is already clean, so it can only fail while fixing a
        # genuinely prefixed checkpoint — exactly the case where continuing would
        # benchmark random weights. Swallowing that is what caused the 2026-07-21
        # incident in the first place. Let it raise.
        if os.path.isdir(model_name):
            n_stripped = strip_orig_mod_prefix(model_name)
            if n_stripped:
                logger.info(
                    "-> Stripped torch.compile '_orig_mod.' prefix from %d keys in %s",
                    n_stripped, model_name,
                )
        model = AutoModel.from_pretrained(model_name, config=_cfg)
        _validate_local_checkpoint_integrity(model_name, model)
        model.to(device).to(_weight_dtype).eval()
        # ProteinEncoder registers rope_cs as a NON-persistent buffer, so HF's
        # from_pretrained leaves it as uninitialized meta/garbage memory (it is
        # absent from the checkpoint and never re-run through __init__'s
        # _precompute_rope). That silently DISABLES RoPE for the benched model and
        # was the root cause of the constant ~0.03-0.32 downstream gap vs native
        # AMPLIFY benched on identical weights. Recompute it explicitly (analogous
        # to fix_amplify_meta_tensors for AMPLIFY's freqs_cis).
        fix_proteva_rope_buffer(model)
        enc_mode = getattr(getattr(model, "encoder", None), "config", None)
        enc_mode = getattr(enc_mode, "flash_attn_mode", "?")
        logger.info(f"-> Proteva encoder flash_attn_mode={enc_mode}")
        # The HF checkpoint ships weights only; load the project tokenizer.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as exc:
            # Some Proteva checkpoints ship weights without a tokenizer. Point
            # $PROTEVA_TOKENIZER at one rather than hardcoding a path that only
            # exists on the machine this was written on.
            fallback = os.environ.get("PROTEVA_TOKENIZER")
            if not fallback:
                raise RuntimeError(
                    f"No tokenizer in {model_name!r}. Set $PROTEVA_TOKENIZER to a "
                    f"directory containing one."
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(fallback)
            logger.info("-> Using $PROTEVA_TOKENIZER: %s", fallback)
        logger.info("-> Loaded as HF AutoModel (Proteva)")
        return (tokenizer, model), False, device

    # 1. Try SentenceTransformer first (preferred for non-ESMplusplus pretrained models)
    try:
        from sentence_transformers import SentenceTransformer

        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype

        model = SentenceTransformer(
            model_name, trust_remote_code=True, device=device, model_kwargs=model_kwargs
        )
        _fix_sbert_tokenizer(model)

        logger.info("-> Loaded as SentenceTransformer")
        return model, True, device
    except Exception as e:
        logger.info(f"SentenceTransformer load failed ({type(e).__name__}: {e})")
        logger.info("Falling back to HuggingFace AutoModel...")

    # 2. Try HF AutoModel (for base models)
    try:
        model = from_pretrained_with_flash(AutoModel, model_name, **hf_load_kwargs)
        if needs_esm2_token_dropout_workaround(model):
            disable_esm2_token_dropout(model)
        model.to(device).eval()

        # Get tokenizer - prefer model's own tokenizer if available
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            tokenizer = model.tokenizer
            logger.info("-> Using tokenizer from model attribute")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info("-> Using AutoTokenizer")

        logger.info("-> Loaded as HF AutoModel")
        return (tokenizer, model), False, device

    except Exception as e:
        logger.error(f"AutoModel load failed: {e}")
        raise RuntimeError(f"Failed to load model: {model_name}") from e


# =============================================================================
# Data Loading & Processing
# =============================================================================


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching column from candidates."""
    for c in candidates:
        if c in columns:
            return c
    return None


def get_split_data(dataset, split_name: str, all_keys: List[str]):
    """Get data for a split, with fallback for fuzzy matching."""
    if split_name in dataset:
        return dataset[split_name]

    for key in all_keys:
        if split_name in str(key).lower():
            logger.info(f"  Using '{key}' for requested split '{split_name}'")
            return dataset[key]

    raise KeyError(f"Split '{split_name}' not found. Available: {all_keys}")


def _normalize_split_value(value: Any) -> str:
    """Normalize a split value for case-insensitive comparisons."""
    return str(value).strip().lower()


def _resolve_local_dataset_path(dataset_name: str) -> Optional[Path]:
    """Resolve a dataset specifier to a local dataset directory if it exists."""
    dataset_path = Path(dataset_name).expanduser()
    candidate_paths = [dataset_path]
    if not dataset_path.is_absolute():
        candidate_paths.append(Path(__file__).resolve().parent / dataset_path)

    for candidate_path in candidate_paths:
        if candidate_path.is_dir():
            return candidate_path.resolve()
    return None


# data/<name> -> the script that builds it. Kept explicit rather than guessed
# from the name, so a rename breaks the test instead of the error message.
_PREP_SCRIPTS = {
    "data/conservation_flip": "scripts/prep_conservation.py",
    "data/disprot": "scripts/prep_disprot.py",
    "data/flip2_amylase": "scripts/prep_flip2.py",
    "data/flip2_rhomax": "scripts/prep_flip2.py",
}


def require_local_dataset(dataset_name: str) -> Path:
    """Resolve a `data/...` dataset, or explain how to build it.

    Without this, `datasets` reports "doesn't exist on the Hub", which sends the
    reader looking for a deleted HuggingFace dataset rather than at a prep
    script in this repo.
    """
    path = _resolve_local_dataset_path(dataset_name)
    if path is not None:
        return path
    script = _PREP_SCRIPTS.get(dataset_name)
    hint = (
        f"Build it first:\n    python {script}"
        if script
        else f"Expected a dataset directory at {dataset_name!r}."
    )
    raise FileNotFoundError(
        f"Local dataset {dataset_name!r} is not present. It is built from source "
        f"rather than downloaded, so a fresh clone does not have it.\n{hint}\n"
        f"See docs/DATASETS.md."
    )


def _clean_chezod_scores(values: Any) -> List[float]:
    """Normalize a CheZOD score payload into finite per-residue floats."""

    if values is None:
        return []
    if not isinstance(values, list):
        values = list(values)

    cleaned: List[float] = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value) or numeric_value >= 900.0:
            continue
        cleaned.append(numeric_value)
    return cleaned


def _chezod_rows_from_payload(payload: Any) -> List[Dict[str, Any]]:
    """Convert one CheZOD raw JSON payload into benchmark rows."""

    if isinstance(payload, list):
        frame = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        frame = pd.DataFrame(payload)
    else:
        raise TypeError(f"Unsupported CheZOD payload type: {type(payload).__name__}")

    if "zscore" not in frame.columns and "z-score" in frame.columns:
        frame = frame.rename(columns={"z-score": "zscore"})
    if "brmid" not in frame.columns:
        raise KeyError("CheZOD payload missing 'brmid' column")
    if "sequence" not in frame.columns:
        raise KeyError("CheZOD payload missing 'sequence' column")
    if "zscore" not in frame.columns:
        raise KeyError("CheZOD payload missing 'zscore'/'z-score' column")

    rows: List[Dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        scores = _clean_chezod_scores(record.get("zscore"))
        if not scores:
            continue
        sequence = str(record["sequence"])
        rows.append(
            {
                "sequence": sequence,
                "entry_id": str(record["brmid"]),
                "disorder_mean": float(np.mean(scores)),
                "disorder_std": float(np.std(scores)),
                "seq_length": len(sequence),
                "num_residues_scored": len(scores),
            }
        )
    return rows


def _load_chezod_from_raw(local_dataset_path: Path):
    """Rebuild the local CheZOD DatasetDict from raw JSON exports when needed."""

    from datasets import Dataset, DatasetDict

    raw_dir = local_dataset_path / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"CheZOD raw directory missing: {raw_dir}")

    raw_payloads: List[Tuple[str, List[Dict[str, Any]]]] = []
    for raw_path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        rows = _chezod_rows_from_payload(payload)
        if rows:
            raw_payloads.append((raw_path.name, rows))

    if len(raw_payloads) < 2:
        raise FileNotFoundError(
            f"Expected at least two raw CheZOD JSON files under {raw_dir}, found {len(raw_payloads)} usable payloads."
        )

    split_rows: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
    for raw_name, rows in raw_payloads:
        lowered = raw_name.lower()
        if any(token in lowered for token in ("1159", "train")):
            split_rows["train"] = (raw_name, rows)
        elif any(token in lowered for token in ("117", "test")):
            split_rows["test"] = (raw_name, rows)

    if {"train", "test"} <= split_rows.keys():
        train_name, train_rows = split_rows["train"]
        test_name, test_rows = split_rows["test"]
    else:
        raw_payloads.sort(key=lambda item: len(item[1]), reverse=True)
        train_name, train_rows = raw_payloads[0]
        test_name, test_rows = raw_payloads[1]
        if len(train_rows) < len(test_rows):
            raise ValueError(
                "Unable to infer CheZOD train/test split names from raw filenames "
                "and row-count fallback would invert the expected split sizes."
            )
    logger.warning(
        "  Reconstructed CheZOD dataset from raw JSON because Arrow shards are missing "
        "(train=%s rows=%d, test=%s rows=%d).",
        train_name,
        len(train_rows),
        test_name,
        len(test_rows),
    )
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        }
    )


def _load_local_dataset(local_dataset_path: Path):
    """Load a local benchmark dataset, with CheZOD raw fallback when required."""

    from datasets import load_from_disk

    try:
        return load_from_disk(str(local_dataset_path))
    except (FileNotFoundError, OSError, ValueError):
        if local_dataset_path.name == "chezod":
            return _load_chezod_from_raw(local_dataset_path)
        raise


def _select_rows_by_column_values(data, column: str, allowed_values: set[str]):
    """Select dataset rows whose column value matches one of the allowed values."""
    indices = [
        index
        for index, value in enumerate(data[column])
        if _normalize_split_value(value) in allowed_values
    ]
    return data.select(indices)


def _find_named_split(all_keys: List[str], candidates: List[str]) -> Optional[str]:
    """Resolve an explicit split name from dataset keys, case-insensitively."""
    key_map = {_normalize_split_value(key): key for key in all_keys}
    for candidate in candidates:
        if not candidate:
            continue
        matched = key_map.get(_normalize_split_value(candidate))
        if matched is not None:
            return matched
    return None


def _is_supervised_problem(cfg: TaskConfig) -> bool:
    """Return True for tasks that train probes on labels."""
    return cfg.problem_type in {"binary", "multiclass", "multilabel", "regression"}


def _normalize_sequence_value(sequence: Any) -> Any:
    """Normalize whitespace-delimited amino-acid strings for token-free models."""
    if isinstance(sequence, str):
        return "".join(sequence.split())
    return sequence


def extract_sequences(
    data, input_map: Dict[str, str], remove_sequence_whitespace: bool = False
) -> List:
    """Extract sequences from dataset using input mapping with fallback heuristics."""
    available_cols = data.column_names

    # Resolve actual column names (with fallback heuristics)
    resolved_cols = {}
    for key, col in input_map.items():
        if col in available_cols:
            resolved_cols[key] = col
        else:
            # Try common alternatives
            alternatives = {
                "seq": [
                    "sequence",
                    "sequences",
                    "primary",
                    "aa_seq",
                    "seq",
                    "input",
                    "protein",
                ],
                "seq1": ["protein1_sequence", "SeqA", "seq_1", "protein1", "peptide"],
                "seq2": [
                    "protein2_sequence",
                    "SeqB",
                    "seq_2",
                    "protein2",
                    "HLA_sequence",
                ],
            }
            found = find_column(available_cols, alternatives.get(key, []))
            if found:
                resolved_cols[key] = found
                logger.info(f"  Column '{col}' not found, using '{found}' instead")
            else:
                raise KeyError(
                    f"Cannot find column for '{key}'. "
                    f"Tried: {col}, alternatives: {alternatives.get(key, [])}. "
                    f"Available: {available_cols}"
                )

    # Extract based on number of sequence inputs
    if len(resolved_cols) == 1:
        col = list(resolved_cols.values())[0]
        sequences = list(data[col])
        if remove_sequence_whitespace:
            return [_normalize_sequence_value(sequence) for sequence in sequences]
        return sequences
    else:
        # Multiple inputs (e.g., PPI) - return as tuples
        ordered_keys = sorted(resolved_cols.keys())
        columns_data = [data[resolved_cols[k]] for k in ordered_keys]
        sequences = list(zip(*columns_data))
        if remove_sequence_whitespace:
            return [
                tuple(_normalize_sequence_value(sequence) for sequence in pair)
                for pair in sequences
            ]
        return sequences


def extract_labels(data, label_col: str, problem_type: str) -> Tuple[List, str]:
    """Extract and process labels from dataset."""
    available_cols = data.column_names

    # Find label column (with fallbacks)
    actual_col = label_col
    if label_col not in available_cols:
        alternatives = [
            "label",
            "labels",
            "target",
            "targets",
            "go_terms",
            "solubility",
        ]
        actual_col = find_column(available_cols, alternatives)
        if actual_col is None:
            raise KeyError(
                f"Label column '{label_col}' not found. Available: {available_cols}"
            )
        logger.info(f"  Label column '{label_col}' not found, using '{actual_col}'")

    raw_labels = data[actual_col]

    def _parse_multilabel(lbl):
        if isinstance(lbl, list):
            return [str(x) for x in lbl]
        if isinstance(lbl, str):
            return [x for x in lbl.replace(",", " ").split() if x]
        return [str(lbl)]

    def _parse_regression(lbl):
        return float(lbl[0]) if isinstance(lbl, (list, tuple)) else float(lbl)

    def _parse_classification(lbl):
        val = lbl[0] if isinstance(lbl, (list, tuple)) else lbl
        try:
            return int(val)
        except (ValueError, TypeError):
            return str(val)

    _parsers = {
        "multilabel": _parse_multilabel,
        "regression": _parse_regression,
        "binary": _parse_classification,
        "multiclass": _parse_classification,
    }
    parse = _parsers[problem_type]

    return [parse(lbl) for lbl in raw_labels], actual_col


def _apply_label_map(labels: List, label_map: Dict[str, Any]) -> List:
    if not label_map:
        return labels
    logger.info(f"  Applying label map: {label_map}")
    return [label_map.get(str(lbl), lbl) for lbl in labels]


def _filter_multilabel_top_k(
    labels: List[List], top_k: int
) -> Tuple[List[List], MultiLabelBinarizer]:
    logger.info(f"  Filtering to top {top_k} labels...")
    all_labels = [lbl for sub in labels for lbl in sub]
    top_k_set = set(pd.Series(all_labels).value_counts().head(top_k).index)
    filtered = [[lbl for lbl in sub if lbl in top_k_set] for sub in labels]
    return filtered, MultiLabelBinarizer(classes=sorted(list(top_k_set)))


def _filter_multiclass_top_k(
    sequences: List,
    labels: List,
    top_k: int,
) -> Tuple[List, List]:
    """Keep only samples whose label is in the top-K most frequent classes."""
    logger.info("  Filtering multiclass task to top %d labels...", top_k)
    top_k_set = set(pd.Series(labels).value_counts().head(top_k).index)
    filtered_pairs = [
        (sequence, label)
        for sequence, label in zip(sequences, labels)
        if label in top_k_set
    ]
    if not filtered_pairs:
        return [], []
    filtered_sequences, filtered_labels = zip(*filtered_pairs)
    return list(filtered_sequences), list(filtered_labels)


def _subsample_paired_data(
    sequences: List, labels: List, max_samples: int
) -> Tuple[List, List]:
    """Subsample paired sequence/label arrays deterministically."""
    if len(sequences) <= max_samples:
        return sequences, labels
    random_gen = np.random.RandomState(BENCHMARK_SEED)
    indices = np.arange(len(sequences))
    random_gen.shuffle(indices)
    keep = indices[:max_samples]
    return [sequences[i] for i in keep], [labels[i] for i in keep]


# Per-residue label meaning "no ground truth here" -- excluded from fitting and
# from scoring, so it never becomes a predictable pseudo-class.
IGNORE_LABEL = -1

_SS3_ALPHABET = "HEC"
# The 8 DSSP states, and only those -- this is the published Q8 label set, so
# `ss8` (GleghornLab) and `ss8_cb513` (proteinea) assign the same id to the same
# state and their F1_Macro values are comparable.
_SS8_ALPHABET = "GHIBESTC"
# GleghornLab/SS8 additionally marks unassigned residues `D`: 6.4% of train and
# 11.5% of test residues, 91% of them in terminal runs and so trivially
# predictable from position alone. Scoring them as a 9th class would inflate
# Accuracy and turn F1_Macro into a 9-class average that no published Q8 number
# can be compared against, so they are marked IGNORE_LABEL and dropped at fit
# time. Standard Q8 evaluation masks them out the same way.
_DISORDER_ALPHABET = "01"


def _decode_residue_label(task_name: str, label_col: str, raw: Any) -> List[int]:
    """Decode per-residue labels.

    SS3 / SS8 / Disorder use string alphabets (``HEC`` / ``GHIBESTC`` / ``01``);
    other tasks expect already-tokenized integer lists or comma-separated
    strings. Robust to lists, strings, and falls back to ``int(c)``.

    Residues with no ground truth decode to ``IGNORE_LABEL`` and are dropped
    later by ``token_classification_probe.drop_ignored_residues``.
    """
    if isinstance(raw, list):
        return [int(x) for x in raw]
    s = str(raw)
    name_lower = task_name.lower()
    # Branch ORDER is load-bearing. Task names overlap -- "Disorder (NetSurfP-SS3
    # mask)" contains "ss3", and an 8-state task is also named "Secondary
    # Structure ..." -- and every alphabet branch DROPS symbols it does not
    # recognise rather than raising. A mis-routed task therefore returns a short
    # or empty label list that the residue probe silently truncates against,
    # producing a plausible number computed on the wrong residues. Most specific
    # match first: disorder, then 8-state, then 3-state.
    if "disorder" in name_lower or label_col.startswith("disorder"):
        # NetSurfP ships this column as a stringified float list,
        # "['0.0', '1.0', ...]", not as bare 0/1 characters.
        if "." in s or "[" in s:
            return [int(float(tok)) for tok in re.findall(r"-?\d+(?:\.\d+)?", s)]
        return [_DISORDER_ALPHABET.index(c) for c in s if c in _DISORDER_ALPHABET]
    if label_col == "dssp8" or "ss8" in name_lower or "dssp8" in name_lower:
        # Unassigned residues map to IGNORE_LABEL rather than being dropped:
        # dropping them would shorten the list and SHIFT every later label
        # against its residue embedding.
        return [
            _SS8_ALPHABET.index(c) if c in _SS8_ALPHABET else IGNORE_LABEL for c in s
        ]
    if "ss3" in name_lower or "secondary structure" in name_lower:
        # Same ignore-don't-drop rule as the SS8 branch, so the two alphabets
        # agree on what an unrecognised symbol means. No shipped dataset emits
        # one, but dropping would silently shift every later label.
        return [
            _SS3_ALPHABET.index(c) if c in _SS3_ALPHABET else IGNORE_LABEL for c in s
        ]
    if "," in s:
        return [int(tok) for tok in s.split(",") if tok.strip()]
    # Fallback: try character-by-character int parsing
    return [int(c) for c in s if c.isdigit()]


def _prepare_token_classification_data(
    cfg: TaskConfig,
    max_samples: Optional[int],
    eval_split: str,
) -> Tuple[List[str], List[List[int]], Optional[List[str]], Optional[List[List[int]]], Dict[str, Any]]:
    """Load a residue-level dataset and decode per-residue labels.

    Mirrors the split-selection logic of ``prepare_data`` but emits
    per-residue label lists (not sequence-level scalars).
    """
    train_data, eval_data, split_metadata = _load_residue_splits(
        cfg, max_samples, eval_split
    )
    seq_col = cfg.input_map.get("seq", "sequence")

    def _decode_split(split_data) -> Tuple[List[str], List[List[int]]]:
        seqs = [str(x) for x in split_data[seq_col]]
        if cfg.remove_sequence_whitespace:
            seqs = ["".join(s.split()) for s in seqs]
        labs = [
            _decode_residue_label(cfg.name, cfg.label_col, raw)
            for raw in split_data[cfg.label_col]
        ]
        return seqs, labs

    train_seqs, train_labels = _decode_split(train_data)
    if eval_data is not None and len(eval_data) > 0:
        test_seqs, test_labels = _decode_split(eval_data)
    else:
        test_seqs, test_labels = None, None

    logger.info(
        "  Residue task: %d train sequences, %s eval sequences (%s)",
        len(train_seqs),
        len(test_seqs) if test_seqs is not None else "CV-fallback",
        split_metadata["eval_strategy"],
    )
    return train_seqs, train_labels, test_seqs, test_labels, split_metadata


def load_dataset_splits(dataset: str, data_files: Optional[Dict[str, str]] = None, **kwargs):
    """``load_dataset`` that tolerates per-split files with different columns.

    Handing several CSVs to one ``load_dataset`` call makes the builder unify
    their schemas, and raw hub CSV repos rarely agree: in
    ``proteinea/secondary_structure_prediction``, `CASP13.csv` carries
    `xyz_coordinates`, `CASP14.csv` also carries an `Unnamed: 0` index column,
    and `training_hhblits.csv` carries `cb513_mask`. Unifying them raises
    "Please either edit the data files to have matching columns".

    Loading each split on its own sidesteps that entirely -- the tasks only ever
    read the sequence and label columns, which every file does share.
    """
    from datasets import DatasetDict, load_dataset

    if not data_files:
        return load_dataset(dataset, **kwargs)
    return DatasetDict(
        {
            split: load_dataset(
                dataset, data_files={split: files}, split=split, **kwargs
            )
            for split, files in data_files.items()
        }
    )


def _unwrap_encoder_tokenizer(model_obj, is_sbert: bool):
    """Return ``(tokenizer, encoder)`` for the per-token extraction paths.

    A SentenceTransformer wraps the encoder, so reach through to the underlying
    HF model and its native tokenizer -- ``.encode()`` only ever hands back a
    pooled vector, and both the residue probe and the contact probe need the
    per-token hidden states.
    """
    if is_sbert:
        first_module = list(model_obj._modules.values())[0]
        return first_module.tokenizer, first_module.auto_model
    tokenizer, encoder = model_obj
    return tokenizer, encoder


def _prepare_contact_data(
    cfg: TaskConfig,
    max_samples: Optional[int],
    eval_split: str,
    train_proteins: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """Load a contact-prediction dataset: sequences plus CB coordinates.

    Returns one record per protein, ``{"seq", "tertiary", "valid_mask"}``. The
    coordinates build the contact LABELS in ``contact_metrics``; the model is
    only ever shown ``seq``.

    ``train_proteins`` truncates the train split BEFORE the rows are pulled out
    of Arrow. The probe only ever uses that many, and materialising all 25k
    coordinate arrays as Python lists costs an order of magnitude more memory
    than the 167 MB the dataset occupies on disk.
    """
    train_data, eval_data, split_metadata = _load_residue_splits(
        cfg, max_samples, eval_split
    )
    seq_col = cfg.input_map.get("seq", "seq")

    def _records(split_data) -> List[Dict[str, Any]]:
        return [
            {
                "seq": str(row[seq_col]),
                "tertiary": row[cfg.label_col],
                "valid_mask": row.get("valid_mask"),
            }
            for row in split_data
        ]

    if train_proteins and len(train_data) > train_proteins:
        # Shuffle first, matching every other subsampling site. Taking rows in
        # file order would make --fast (which shuffles via max_samples) and
        # --no-fast train on different subsets and report non-comparable numbers.
        train_data = train_data.shuffle(seed=BENCHMARK_SEED).select(
            range(train_proteins)
        )
    train_records = _records(train_data)
    eval_records = (
        _records(eval_data) if eval_data is not None and len(eval_data) > 0 else None
    )
    logger.info(
        "  Contact task: %d train proteins, %s eval proteins (%s)",
        len(train_records),
        len(eval_records) if eval_records is not None else "CV-fallback",
        split_metadata["eval_strategy"],
    )
    return train_records, eval_records, split_metadata


def _load_residue_splits(
    cfg: TaskConfig,
    max_samples: Optional[int],
    eval_split: str,
):
    """Split selection for the per-protein task paths (residue and contact).

    Mirrors ``prepare_data``'s split logic but returns the raw HF splits, since
    neither per-residue labels nor coordinate arrays survive the sequence-level
    parser. Returns ``(train_data, eval_data_or_None, split_metadata)``.
    """
    from datasets import load_dataset

    normalized_eval_split = _normalize_split_value(eval_split)
    if normalized_eval_split not in SUPPORTED_EVAL_SPLITS:
        raise ValueError(
            f"eval_split must be one of {sorted(SUPPORTED_EVAL_SPLITS)}; "
            f"got '{eval_split}'"
        )
    split_metadata: Dict[str, Any] = {
        "requested_eval_split": normalized_eval_split,
        "resolved_eval_split": normalized_eval_split,
        "eval_strategy": (
            "validation_split"
            if normalized_eval_split == "validation"
            else "test_split"
        ),
        "cv_fallback": False,
    }
    logger.info("Loading residue-level dataset: %s", cfg.dataset)
    if str(cfg.dataset).startswith("data/"):
        ds = _load_local_dataset(require_local_dataset(cfg.dataset))
    elif (local_dataset_path := _resolve_local_dataset_path(cfg.dataset)) is not None:
        ds = _load_local_dataset(local_dataset_path)
    else:
        load_kwargs = {}
        if cfg.dataset_config:
            load_kwargs["name"] = cfg.dataset_config
        if cfg.data_dir:
            load_kwargs["data_dir"] = cfg.data_dir
        ds = load_dataset_splits(
            cfg.dataset, data_files=cfg.data_files, **load_kwargs
        )

    all_keys = [str(k) for k in ds.keys()]

    train_data = None
    eval_data = None

    if cfg.split_column:
        source = get_split_data(ds, cfg.train_split, all_keys)
        train_data = _select_rows_by_column_values(
            source, cfg.split_column, {_normalize_split_value(cfg.train_split)}
        )
        if normalized_eval_split == "validation":
            allowed = {
                _normalize_split_value(v)
                for v in (cfg.validation_column_values or _VALIDATION_SPLIT_ALIASES)
            }
            eval_data = _select_rows_by_column_values(source, cfg.split_column, allowed)
            split_metadata["eval_strategy"] = "validation_split_column"
        else:
            eval_data = _select_rows_by_column_values(
                source, cfg.split_column, {_normalize_split_value(cfg.test_split)}
            )
            split_metadata["resolved_eval_split"] = "test"
            split_metadata["eval_strategy"] = "test_split_column"
    else:
        train_data = get_split_data(ds, cfg.train_split, all_keys)
        validation_candidates = []
        if cfg.validation_split:
            validation_candidates.append(cfg.validation_split)
        validation_candidates.extend(_VALIDATION_SPLIT_ALIASES)
        if normalized_eval_split == "validation":
            val_key = _find_named_split(all_keys, validation_candidates)
            if val_key:
                eval_data = get_split_data(ds, val_key, all_keys)
                split_metadata["eval_strategy"] = "validation_split"
            else:
                split_metadata["eval_strategy"] = "validation_cv4_train"
                split_metadata["cv_fallback"] = True
                # A task can have a real held-out test set and no validation
                # split -- the CASP/CB513 secondary-structure sets are exactly
                # that. Falling back to CV on train then answers a different
                # question than the task exists to ask, and every sibling task
                # sharing the train file reports the same number. It is recorded
                # in the EvalStrategy column, but say so out loud too.
                if cfg.test_split in all_keys:
                    logger.warning(
                        "%s has no validation split, so --eval_split validation "
                        "is 4-fold CV over TRAIN -- not the held-out '%s' set. "
                        "Use --eval_split test to score the held-out set.",
                        cfg.name,
                        cfg.test_split,
                    )
        else:
            if cfg.test_split in all_keys:
                eval_data = get_split_data(ds, cfg.test_split, all_keys)
                split_metadata["resolved_eval_split"] = "test"
                split_metadata["eval_strategy"] = "test_split"

    if max_samples and len(train_data) > max_samples:
        train_data = train_data.shuffle(seed=BENCHMARK_SEED).select(range(max_samples))
    if max_samples and eval_data is not None and len(eval_data) > max_samples:
        eval_data = eval_data.shuffle(seed=BENCHMARK_SEED).select(range(max_samples))

    return train_data, eval_data, split_metadata


def prepare_data(
    cfg: TaskConfig,
    max_samples: Optional[int] = None,
    eval_split: str = DEFAULT_BENCHMARK_EVAL_SPLIT,
    top_k_labels_override: Optional[int] = None,
) -> Tuple[
    List,
    List,
    Optional[List],
    Optional[List],
    Optional[MultiLabelBinarizer | np.ndarray],
    Dict[str, Any],
]:
    """Load and prepare train/eval data for a task."""
    from datasets import load_dataset

    normalized_eval_split = _normalize_split_value(eval_split)
    if normalized_eval_split not in SUPPORTED_EVAL_SPLITS:
        raise ValueError(
            f"eval_split must be one of {sorted(SUPPORTED_EVAL_SPLITS)}; "
            f"got '{eval_split}'"
        )

    split_metadata: Dict[str, Any] = {
        "requested_eval_split": normalized_eval_split,
        "resolved_eval_split": normalized_eval_split,
        "eval_strategy": (
            "validation_split"
            if normalized_eval_split == "validation"
            else "test_split"
        ),
        "cv_fallback": False,
    }
    effective_top_k_labels = (
        top_k_labels_override if top_k_labels_override is not None else cfg.top_k_labels
    )
    defer_sample_cap = (
        cfg.problem_type == "multiclass" and effective_top_k_labels is not None
    )

    logger.info(f"Loading dataset: {cfg.dataset}")

    # Load dataset — support both HF Hub datasets and local disk datasets
    load_kwargs = {}
    if cfg.dataset_config:
        load_kwargs["name"] = cfg.dataset_config
    if cfg.data_dir:
        load_kwargs["data_dir"] = cfg.data_dir

    # A data/... task is built by a prep script, never downloaded. Resolve it up
    # front so a missing one reports the script to run, rather than falling into
    # the Hub path below and reporting that the dataset does not exist there.
    if str(cfg.dataset).startswith("data/"):
        require_local_dataset(cfg.dataset)
    local_dataset_path = _resolve_local_dataset_path(cfg.dataset)
    try:
        if local_dataset_path is not None:
            logger.info("  Loading local dataset from disk: %s", local_dataset_path)
            ds = _load_local_dataset(local_dataset_path)
        else:
            ds = load_dataset_splits(
                cfg.dataset, data_files=cfg.data_files, **load_kwargs
            )
    except Exception as e:
        logger.warning(f"Standard load failed, trying with trust_remote_code: {e}")
        try:
            ds = load_dataset_splits(
                cfg.dataset,
                data_files=cfg.data_files,
                trust_remote_code=True,
                **load_kwargs,
            )
        except Exception as e2:
            raise RuntimeError(f"Failed to load dataset {cfg.dataset}: {e2}")

    ds_keys = getattr(ds, "keys", None)
    if ds_keys is None:
        raise TypeError(
            f"Expected dataset with split keys, got {type(ds).__name__}: {cfg.dataset}"
        )
    all_keys = [str(k) for k in ds_keys()]
    logger.info(f"  Available splits: {all_keys}")

    # ProteinGym tasks: load full dataset and return groups array for per-assay evaluation
    if cfg.eval_mode.startswith("proteingym"):
        split_metadata["eval_strategy"] = "proteingym_unchanged"
        train_data = get_split_data(ds, cfg.train_split, all_keys)
        if max_samples and not defer_sample_cap:
            train_data = train_data.shuffle(seed=BENCHMARK_SEED).select(
                range(min(len(train_data), max_samples))
            )
        # Zero-shot: verify the WT column exists before proceeding
        if cfg.eval_mode == "proteingym_zeroshot":
            wt_col = cfg.input_map.get("wt")
            if wt_col and wt_col not in train_data.column_names:
                logger.warning(
                    f"  Zero-shot: WT column '{wt_col}' not found "
                    f"(available: {train_data.column_names}). Skipping task."
                )
                return [], [], None, None, None, split_metadata
        seqs = extract_sequences(train_data, cfg.input_map)
        labels, _ = extract_labels(train_data, cfg.label_col, cfg.problem_type)
        groups = np.array(train_data[cfg.group_by])
        labels = _apply_label_map(labels, cfg.label_map)
        logger.info(
            f"  Loaded {len(seqs)} samples across {len(np.unique(groups))} groups"
        )
        return seqs, labels, None, None, groups, split_metadata

    if cfg.problem_type == "retrieval":
        split_metadata["eval_strategy"] = "retrieval_unchanged"
        data = get_split_data(ds, cfg.train_split, all_keys)
        if max_samples and not defer_sample_cap:
            data = data.shuffle(seed=BENCHMARK_SEED).select(
                range(min(len(data), max_samples))
            )

        seqs = extract_sequences(
            data,
            cfg.input_map,
            remove_sequence_whitespace=cfg.remove_sequence_whitespace,
        )
        labels, _ = extract_labels(data, cfg.label_col, "multiclass")
        labels = _apply_label_map(labels, cfg.label_map)
        logger.info(f"  Loaded {len(seqs)} retrieval queries/gallery sequences")
        return seqs, labels, seqs, labels, None, split_metadata

    train_data = None
    eval_data = None
    use_cv_fallback = False

    if cfg.split_column:
        source_data = get_split_data(ds, cfg.train_split, all_keys)
        if cfg.split_column not in source_data.column_names:
            raise KeyError(
                f"Split column '{cfg.split_column}' not found. "
                f"Available: {source_data.column_names}"
            )

        train_values = {_normalize_split_value(cfg.train_split)}
        train_data = _select_rows_by_column_values(
            source_data,
            cfg.split_column,
            train_values,
        )
        if normalized_eval_split == "validation" and _is_supervised_problem(cfg):
            validation_values = {
                _normalize_split_value(value)
                for value in (cfg.validation_column_values or _VALIDATION_SPLIT_ALIASES)
            }
            eval_data = _select_rows_by_column_values(
                source_data,
                cfg.split_column,
                validation_values,
            )
            if len(eval_data) > 0:
                split_metadata["eval_strategy"] = "validation_split_column"
            else:
                use_cv_fallback = True
                split_metadata["eval_strategy"] = "validation_cv4_train"
                split_metadata["cv_fallback"] = True
                logger.info(
                    "  Validation mode: no validation rows found in split column '%s'; "
                    "falling back to 4-fold CV on train rows",
                    cfg.split_column,
                )
        else:
            eval_data = _select_rows_by_column_values(
                source_data,
                cfg.split_column,
                {_normalize_split_value(cfg.test_split)},
            )
            split_metadata["resolved_eval_split"] = "test"
            split_metadata["eval_strategy"] = "test_split_column"

        if max_samples and not defer_sample_cap:
            if len(train_data) > max_samples:
                train_data = train_data.shuffle(seed=BENCHMARK_SEED).select(
                    range(max_samples)
                )
            if eval_data is not None and len(eval_data) > max_samples:
                eval_data = eval_data.shuffle(seed=BENCHMARK_SEED).select(
                    range(max_samples)
                )

        logger.info(
            "  Column split '%s': train='%s' -> %d rows, eval_target='%s' -> %d rows",
            cfg.split_column,
            cfg.train_split,
            len(train_data),
            normalized_eval_split,
            0 if eval_data is None else len(eval_data),
        )

        if len(train_data) == 0:
            raise ValueError(
                f"Column split '{cfg.split_column}' produced empty train data"
            )
        if eval_data is not None and len(eval_data) == 0:
            raise ValueError(
                f"Column split '{cfg.split_column}' produced empty eval data"
            )

    elif normalized_eval_split == "validation" and _is_supervised_problem(cfg):
        validation_candidates: List[str] = []
        if cfg.validation_split:
            validation_candidates.append(cfg.validation_split)
        validation_candidates.extend(_VALIDATION_SPLIT_ALIASES)
        validation_split_key = _find_named_split(all_keys, validation_candidates)

        train_data = get_split_data(ds, cfg.train_split, all_keys)
        if validation_split_key is not None:
            eval_data = get_split_data(ds, validation_split_key, all_keys)
            split_metadata["eval_strategy"] = "validation_split"
        else:
            use_cv_fallback = True
            split_metadata["eval_strategy"] = "validation_cv4_train"
            split_metadata["cv_fallback"] = True
            logger.info(
                "  Validation mode: no validation split found; using 4-fold CV on train split"
            )

        if max_samples and not defer_sample_cap:
            if len(train_data) > max_samples:
                train_data = train_data.shuffle(seed=BENCHMARK_SEED).select(
                    range(max_samples)
                )
            if eval_data is not None and len(eval_data) > max_samples:
                eval_data = eval_data.shuffle(seed=BENCHMARK_SEED).select(
                    range(max_samples)
                )

    # Handle auto-split for datasets with only train split in explicit test mode
    elif cfg.auto_split or (
        cfg.test_split not in all_keys and "test" not in str(all_keys).lower()
    ):
        logger.info("  Auto-splitting train into train/test (80/20)...")
        train_data = get_split_data(ds, cfg.train_split, all_keys)

        # Group-aware split: split by group (e.g., protein/DMS_id) to avoid leakage
        if cfg.group_by:
            if cfg.group_by not in train_data.column_names:
                logger.warning(
                    f"  Group column '{cfg.group_by}' not found. "
                    f"Available: {train_data.column_names}. Falling back to random split."
                )
            else:
                logger.info(f"  Splitting by group column: {cfg.group_by}")
                groups = train_data[cfg.group_by]
                unique_groups = list(set(groups))
                random_gen = np.random.RandomState(BENCHMARK_SEED)
                random_gen.shuffle(unique_groups)
                split_idx = int(len(unique_groups) * 0.8)
                train_groups = set(unique_groups[:split_idx])
                eval_groups = set(unique_groups[split_idx:])

                train_indices = [i for i, g in enumerate(groups) if g in train_groups]
                eval_indices = [i for i, g in enumerate(groups) if g in eval_groups]

                eval_data = train_data.select(eval_indices)
                train_data = train_data.select(train_indices)
                logger.info(
                    f"  Group split: {len(train_groups)} train groups, "
                    f"{len(eval_groups)} eval groups -> {len(train_data)} train, {len(eval_data)} eval samples"
                )

                # Apply max_samples AFTER group split to get balanced subsampling
                if max_samples and not defer_sample_cap:
                    if len(train_data) > max_samples:
                        train_data = train_data.shuffle(seed=BENCHMARK_SEED).select(
                            range(max_samples)
                        )
                    if len(eval_data) > max_samples:
                        eval_data = eval_data.shuffle(seed=BENCHMARK_SEED).select(
                            range(max_samples)
                        )
                    logger.info(
                        f"  After sampling: {len(train_data)} train, {len(eval_data)} eval"
                    )

                if len(eval_data) == 0:
                    raise ValueError(
                        "Group-based split resulted in empty eval set. "
                        "Try reducing split ratio or checking group distribution."
                    )

        # Fallback: standard random split (no group-by or group column missing)
        if not cfg.group_by or cfg.group_by not in train_data.column_names:
            if max_samples and not defer_sample_cap:
                total_needed = min(max_samples * 2, len(train_data))
                train_data = train_data.shuffle(seed=BENCHMARK_SEED).select(
                    range(total_needed)
                )

            train_data = train_data.shuffle(seed=BENCHMARK_SEED)
            split_idx = int(len(train_data) * 0.8)
            eval_data = train_data.select(range(split_idx, len(train_data)))
            train_data = train_data.select(range(split_idx))
        split_metadata["resolved_eval_split"] = "test"
        split_metadata["eval_strategy"] = "test_random_split"

    else:
        train_data = get_split_data(ds, cfg.train_split, all_keys)
        eval_data = get_split_data(ds, cfg.test_split, all_keys)

        if max_samples and not defer_sample_cap:
            train_data = train_data.shuffle(seed=BENCHMARK_SEED).select(
                range(min(len(train_data), max_samples))
            )
            eval_data = eval_data.shuffle(seed=BENCHMARK_SEED).select(
                range(min(len(eval_data), max_samples))
            )
        split_metadata["resolved_eval_split"] = "test"
        split_metadata["eval_strategy"] = "test_split"

    if train_data is None:
        raise RuntimeError("Failed to prepare training split")

    if use_cv_fallback:
        logger.info("  Train samples: %d (4-fold CV fallback)", len(train_data))

        train_seqs = extract_sequences(
            train_data,
            cfg.input_map,
            remove_sequence_whitespace=cfg.remove_sequence_whitespace,
        )
        train_labels, _ = extract_labels(train_data, cfg.label_col, cfg.problem_type)

        train_labels = _apply_label_map(train_labels, cfg.label_map)

        mlb = None
        if cfg.problem_type == "multilabel" and effective_top_k_labels:
            train_labels, mlb = _filter_multilabel_top_k(
                train_labels,
                effective_top_k_labels,
            )
        elif cfg.problem_type == "multiclass" and effective_top_k_labels:
            train_seqs, train_labels = _filter_multiclass_top_k(
                train_seqs,
                train_labels,
                effective_top_k_labels,
            )
            if max_samples is not None:
                train_seqs, train_labels = _subsample_paired_data(
                    train_seqs,
                    train_labels,
                    max_samples,
                )

        return train_seqs, train_labels, None, None, mlb, split_metadata

    if eval_data is None:
        raise RuntimeError("Failed to prepare evaluation split")

    logger.info(
        "  Train samples: %d, Eval samples: %d (%s)",
        len(train_data),
        len(eval_data),
        split_metadata["eval_strategy"],
    )

    # Extract sequences
    train_seqs = extract_sequences(
        train_data,
        cfg.input_map,
        remove_sequence_whitespace=cfg.remove_sequence_whitespace,
    )
    test_seqs = extract_sequences(
        eval_data,
        cfg.input_map,
        remove_sequence_whitespace=cfg.remove_sequence_whitespace,
    )

    # Extract labels
    train_labels, _ = extract_labels(train_data, cfg.label_col, cfg.problem_type)
    test_labels, _ = extract_labels(eval_data, cfg.label_col, cfg.problem_type)

    # Apply label_map if provided (e.g., mapping '0' -> 'Benign' for clinical_indels)
    if cfg.label_map:
        train_labels = _apply_label_map(train_labels, cfg.label_map)
        test_labels = _apply_label_map(test_labels, cfg.label_map)

    # Handle multilabel top-K filtering
    mlb = None
    if cfg.problem_type == "multilabel" and effective_top_k_labels:
        train_labels, mlb = _filter_multilabel_top_k(
            train_labels,
            effective_top_k_labels,
        )
        # Filter test_labels with the same top_k (we don't need a new mlb)
        test_labels, _ = _filter_multilabel_top_k(test_labels, effective_top_k_labels)
    elif cfg.problem_type == "multiclass" and effective_top_k_labels:
        train_seqs, train_labels = _filter_multiclass_top_k(
            train_seqs,
            train_labels,
            effective_top_k_labels,
        )
        test_seqs, test_labels = _filter_multiclass_top_k(
            test_seqs,
            test_labels,
            effective_top_k_labels,
        )
        if max_samples is not None:
            train_seqs, train_labels = _subsample_paired_data(
                train_seqs,
                train_labels,
                max_samples,
            )
            test_seqs, test_labels = _subsample_paired_data(
                test_seqs,
                test_labels,
                max_samples,
            )

    return train_seqs, train_labels, test_seqs, test_labels, mlb, split_metadata


# =============================================================================
# Embedding
# =============================================================================


def _sanitize_nan(embs: np.ndarray) -> np.ndarray:
    """Replace NaN embeddings with zeros; log a warning if any found."""
    nan_count = np.isnan(embs).sum()
    if nan_count > 0:
        nan_pct = 100 * nan_count / embs.size
        logger.warning(
            f"Embeddings contain {nan_count} NaN values ({nan_pct:.1f}%) — replacing with zeros"
        )
        embs = np.nan_to_num(embs, nan=0.0)
    return embs


def _l2_normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    """L2-normalize row embeddings while keeping zero rows unchanged."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return embs / safe_norms


def _pack_token_id_rows(rows: List[np.ndarray]) -> Dict[str, Any]:
    """Pack per-sequence token-id arrays into the fa2-varlen layout.

    Mirrors the packing produced by ``plm.hf.collator.ProteinPackedCollator``
    (the path the model was trained with): variable-length rows are
    concatenated into a single ``(1, total_tokens)`` sequence with no padding,
    accompanied by ``cu_seqlens`` (prefix-sum segment boundaries) and
    per-segment RoPE ``position_ids``. The model's flash-attn varlen kernel
    uses ``cu_seqlens`` for block-diagonal attention so segments never attend
    across protein boundaries.

    Args:
        rows: list of 1-D int token-id arrays, one per sequence (already
            tokenized + cropped; no padding).

    Returns:
        Dict with ``input_ids`` ``(1, T)``, ``attention_mask`` ``(1, T)``,
        ``cu_seqlens_q``/``cu_seqlens_k`` ``(num_segments + 1,)`` int32,
        ``position_ids`` ``(1, T)``, and ``max_seqlen_q``/``max_seqlen_k`` ints.
    """
    if not rows:
        raise ValueError("_pack_token_id_rows received no rows")

    seg_lens = [int(np.asarray(r).reshape(-1).shape[0]) for r in rows]
    packed_ids = np.concatenate([np.asarray(r, dtype=np.int64).reshape(-1) for r in rows])
    cu = np.zeros(len(seg_lens) + 1, dtype=np.int32)
    cu[1:] = np.cumsum(seg_lens, dtype=np.int32)
    max_seg = int(max(seg_lens))
    pos = np.concatenate([np.arange(n, dtype=np.int32) for n in seg_lens])
    total = int(cu[-1])

    cu_t = torch.from_numpy(cu)
    return {
        "input_ids": torch.from_numpy(packed_ids).long().unsqueeze(0),
        "attention_mask": torch.ones((1, total), dtype=torch.long),
        "cu_seqlens_q": cu_t,
        "cu_seqlens_k": cu_t,
        "max_seqlen_q": max_seg,
        "max_seqlen_k": max_seg,
        "position_ids": torch.from_numpy(pos).unsqueeze(0),
    }


def _segment_mean_pool(hidden: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Mean-pool packed hidden states back to one vector per segment.

    Inverse of :func:`_pack_token_id_rows`: given ``(1, total_tokens, H)``
    hidden states and the ``cu_seqlens`` boundaries, average each segment's
    token vectors to ``(num_segments, H)``. Output is always float32 (the
    model forwards in bf16; the linear probe wants float).
    """
    h = hidden.squeeze(0).float()  # (total_tokens, H)
    bounds = cu_seqlens.detach().cpu().to(torch.int64).tolist()
    pooled = [h[bounds[i] : bounds[i + 1]].mean(dim=0) for i in range(len(bounds) - 1)]
    return torch.stack(pooled, dim=0)


def embed_sequences(
    model_obj,
    is_sbert: bool,
    sequences: List,
    device: str,
    batch_size: int = 128,
    max_length: int = DEFAULT_EMBED_MAX_LENGTH,
    amp_dtype: Optional[torch.dtype] = None,
    embed_save_path: Optional[str] = None,
    l2_normalize_embeddings: bool = False,
    probe_embed_mode: str = "trunk",
) -> np.ndarray:
    """Generate embeddings for sequences (single or pairs).

    Supports:
    - SentenceTransformer models (is_sbert=True)
    - HuggingFace models (is_sbert=False, model_obj = (tokenizer, model))
    """

    if not sequences:
        return np.array([])

    # Check if input is pairs
    is_pair = isinstance(sequences[0], (tuple, list)) and len(sequences[0]) == 2

    # Flatten pairs for batch processing, deduplicating to avoid redundant embeddings
    if is_pair:
        unique_set = set()
        for pair in sequences:
            unique_set.update(pair)
        flat_seqs = list(unique_set)
        logger.info(
            f"  PPI dedup: {2 * len(sequences)} total -> {len(flat_seqs)} unique sequences"
        )
    else:
        flat_seqs = list(dict.fromkeys(sequences))  # deduplicate, preserve order
        if len(flat_seqs) < len(sequences):
            logger.info(
                f"  Dedup: {len(sequences)} -> {len(flat_seqs)} unique sequences"
            )

    show_progress = _progress_bars_enabled()
    _configure_tqdm_defaults(show_progress)

    if isinstance(model_obj, _KmerEmbedder):
        embs = kmer_features(flat_seqs, k=model_obj.k)
    elif is_sbert:
        if getattr(model_obj, "max_seq_length", None) != max_length:
            model_obj.max_seq_length = max_length
        # SentenceTransformer handles batching internally
        embs = model_obj.encode(
            flat_seqs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
    else:
        # Manual HuggingFace embedding with mean pooling
        # Handle tuple format: (tokenizer, model)
        tokenizer, model = model_obj

        is_amplify_model = (
            getattr(getattr(model, "config", None), "model_type", "") == "AMPLIFY"
        )
        is_proteva_model = (
            getattr(getattr(model, "config", None), "model_type", "") == "proteva"
        )
        if probe_embed_mode != "trunk" and not is_proteva_model:
            logger.warning(
                "probe_embed_mode=%s is only supported for Proteva models; falling back to trunk",
                probe_embed_mode,
            )
            probe_embed_mode = "trunk"
        # Synthyra models (ESM++) expose embed_dataset() for efficient batched
        # inference with length-sorted batches — use it when available.
        has_embed_dataset = hasattr(model, "embed_dataset") and callable(
            model.embed_dataset
        )
        if has_embed_dataset:
            # Only the legacy ESM++ signature (sequences=, max_len=, pooling_types=,
            # save_path=) is understood below. Current FastPLM builds ship
            # embed_dataset(inputs, *, pooling=, max_length=, output=, ...); calling
            # that with the old keywords raises "missing 1 required positional
            # argument: 'inputs'". The suite catches that per task and records it as
            # an Error row, so the sweep still exits 0 with an empty results table --
            # silent, and only found by reading the CSV. Detect the signature and
            # fall through to the generic batched path, which handles any HF model.
            import inspect as _inspect

            try:
                _params = _inspect.signature(model.embed_dataset).parameters
                has_embed_dataset = "sequences" in _params
            except (TypeError, ValueError):
                has_embed_dataset = False
            if not has_embed_dataset:
                logger.info(
                    "embed_dataset() has a non-legacy signature; using the generic "
                    "batched embedding path"
                )

        if is_proteva_model:
            # Proteva fa2-varlen path: tokenize each sequence, pack a chunk of
            # rows into a single UNPADDED (1, total_tokens) sequence with
            # cu_seqlens + per-segment position_ids, forward through the model's
            # block-diagonal flash-attn (BF16, the trained attention path), then
            # segment-mean-pool back to one (H,) vector per sequence. No padding
            # (mask is all-ones over the packed tokens), no SDPA-dense fallback.
            model_device = next(model.parameters()).device
            # CPU / flash-off fallback: the non-flash SDPA/manual attention path
            # attends densely and IGNORES cu_seqlens, so packing >1 sequence would
            # leak across segment boundaries. Forwarding ONE sequence per packed
            # call (single segment) makes the dense full-attention exact. Detect
            # via the encoder's resolved flash_attn_mode (set to "off" on CPU in
            # load_model). GPU fa2-varlen is unchanged (packs `batch_size` rows).
            _enc_cfg = getattr(getattr(model, "encoder", None), "config", None)
            _flash_off = getattr(_enc_cfg, "flash_attn_mode", "fa2-varlen") == "off"
            _pack_bs = 1 if (_flash_off or model_device.type == "cpu") else batch_size
            embs_list = []
            for i in range(0, len(flat_seqs), _pack_bs):
                batch = flat_seqs[i : i + _pack_bs]
                # Tokenize per sequence (no padding) -> per-row id arrays.
                tok = tokenizer(
                    batch,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                )
                rows = [np.asarray(ids, dtype=np.int64) for ids in tok["input_ids"]]
                packed = _pack_token_id_rows(rows)
                forward_inputs = {
                    k: (v.to(model_device) if isinstance(v, torch.Tensor) else v)
                    for k, v in packed.items()
                }
                with torch.inference_mode():
                    outputs = model(**forward_inputs, return_dict=True)
                hidden = outputs.last_hidden_state  # (1, total_tokens, H)
                if hidden is None:
                    raise RuntimeError(
                        "Proteva forward returned no last_hidden_state"
                    )
                pooled = _segment_mean_pool(
                    hidden, packed["cu_seqlens_q"]
                )  # (len(batch), H)
                if probe_embed_mode == "trunk":
                    embs_list.append(pooled.detach().cpu().numpy())
                else:
                    from aux_embed import build_probe_embedding, extract_aux_features

                    aux = extract_aux_features(
                        outputs, packed["cu_seqlens_q"], _log=(i == 0)
                    )
                    if probe_embed_mode == "aux_only" and aux is not None:
                        embs_list.append(aux.detach().cpu().numpy())
                    else:
                        embs_list.append(build_probe_embedding(pooled.detach().cpu().numpy(), aux))
            embs = np.concatenate(embs_list, axis=0)

        elif has_embed_dataset:
            # embed_dataset handles batching/sorting internally; returns dict[seq->tensor]
            # Note: embed_dataset truncates sequences to max_len, so dict keys are
            # truncated. We truncate the lookup keys to match.
            embed_tokenizer = getattr(model, "tokenizer", None)
            # Use a model-specific save path if provided; otherwise disable caching
            # entirely so different models never share stale cached embeddings.
            _use_cache = embed_save_path is not None
            if _use_cache:
                assert embed_save_path is not None
                os.makedirs(os.path.dirname(embed_save_path), exist_ok=True)
            # IMPORTANT: embed_dataset() unconditionally loads from save_path
            # when the file exists (regardless of the `save` flag).  When
            # caching is disabled we must pass a guaranteed-nonexistent path
            # so stale embeddings from a prior model are never loaded.
            if _use_cache:
                # Append PID so concurrent subprocesses never share a cache file.
                # The FastESM embed_dataset performs a non-atomic torch.save after
                # a read-modify-write, which races and corrupts the .pth when
                # multiple GPU workers target the same model path.
                base, ext = os.path.splitext(embed_save_path)
                _save_path = f"{base}.pid{os.getpid()}{ext}"
                # Seed this process's file from the shared cache so a second split
                # or a restart does not re-embed everything. This only ever *reads*
                # the shared file, so it cannot participate in the write race.
                if os.path.exists(embed_save_path) and not os.path.exists(_save_path):
                    try:
                        shutil.copyfile(embed_save_path, _save_path)
                    except OSError as exc:
                        logger.warning(
                            "Could not seed embedding cache from %s (%s); "
                            "embedding from scratch.",
                            embed_save_path,
                            exc,
                        )
            else:
                # embed_dataset gets save=False below, so it neither reads nor
                # writes this path -- it exists only so a stale file from a
                # previous model can never be picked up. Do NOT mkdtemp here:
                # the directory would never be written to and never cleaned up,
                # and since caching became opt-in that leaked on every run.
                _save_path = os.path.join(
                    tempfile.gettempdir(), f"_no_cache_embeddings.{os.getpid()}.pth"
                )
            kwargs = dict(
                sequences=flat_seqs,
                batch_size=batch_size,
                max_len=max_length,
                full_embeddings=False,
                embed_dtype=torch.float32,
                pooling_types=["mean"],
                save=_use_cache,
                save_path=_save_path,
            )
            if embed_tokenizer is not None:
                kwargs["tokenizer"] = embed_tokenizer
            with torch.inference_mode():
                emb_dict = model.embed_dataset(**kwargs)
            if not emb_dict:
                raise RuntimeError("embed_dataset returned no embeddings")
            if _use_cache and os.path.exists(_save_path):
                # Publish this process's cache for the next one. os.replace is
                # atomic within a filesystem, so a reader never sees a partial
                # file -- unlike the non-atomic torch.save that caused the
                # original corruption. Concurrent workers still write only their
                # own pid file; the last to publish wins, which costs reuse but
                # never validity.
                try:
                    os.replace(_save_path, embed_save_path)
                except OSError as exc:
                    logger.warning(
                        "Could not publish embedding cache to %s (%s); "
                        "the next process will re-embed.",
                        embed_save_path,
                        exc,
                    )
            # Re-order dict results to match original input order.
            # embed_dataset keys are truncated sequences; truncate lookups to match.
            embs_list = []
            missing_keys = []
            for s in flat_seqs:
                key = s[:max_length]
                val = emb_dict.get(key)
                if val is None:
                    # Fallback: try untruncated key (short sequences)
                    val = emb_dict.get(s)
                if val is None:
                    missing_keys.append(key)
                    continue
                embs_list.append(val.numpy() if isinstance(val, torch.Tensor) else val)
            if missing_keys:
                raise RuntimeError(
                    "embed_dataset returned incomplete embeddings for "
                    f"{len(missing_keys)} sequence(s); example key: {missing_keys[0]!r}"
                )
            embs = np.stack(embs_list, axis=0)
            # Fall through to shared reassembly logic below (pair concat / dedup restore)

        else:
            embs = []

            for i in range(0, len(flat_seqs), batch_size):
                batch = flat_seqs[i : i + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                amp_ctx = (
                    torch.autocast("cuda", dtype=amp_dtype)
                    if amp_dtype is not None and str(device).startswith("cuda")
                    else nullcontext()
                )

                try:
                    # Save boolean mask for pooling before any conversion
                    pooling_mask = inputs["attention_mask"]
                    orig_len = inputs["input_ids"].shape[1]

                    if is_amplify_model:
                        param = next(model.parameters(), None)
                        input_ids, additive_mask, orig_len, _ = _prepare_amplify_inputs(
                            inputs["input_ids"],
                            pooling_mask,
                            device=device,
                            dtype=(
                                amp_dtype
                                if amp_dtype is not None
                                else (param.dtype if param is not None else None)
                            ),
                        )
                        # Cast additive mask to match autocast dtype for xformers
                        if amp_dtype is not None:
                            additive_mask = additive_mask.to(amp_dtype)
                        with amp_ctx:
                            with torch.inference_mode():
                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=additive_mask,
                                    output_hidden_states=True,
                                    return_dict=True,
                                )
                    else:
                        with amp_ctx:
                            with torch.inference_mode():
                                outputs = model(**inputs, return_dict=True)
                except Exception as e:
                    logger.error(f"Model inference failed: {e}")
                    raise

                # Extract hidden states — models return them in various formats
                if (
                    hasattr(outputs, "last_hidden_state")
                    and outputs.last_hidden_state is not None
                ):
                    hidden = outputs.last_hidden_state
                elif (
                    hasattr(outputs, "hidden_states")
                    and outputs.hidden_states is not None
                ):
                    hidden = outputs.hidden_states[-1]
                elif outputs is not None and isinstance(outputs, torch.Tensor):
                    hidden = outputs
                elif outputs is not None:
                    try:
                        hidden = outputs[0]
                    except Exception:
                        raise RuntimeError(
                            f"Could not extract embeddings from model output: {type(outputs)}"
                        )

                # Slice back to original (pre-padding) length
                hidden = hidden[:, :orig_len, :]

                # Apply AMPLIFY's final layer norm (not included in hidden_states)
                # after slicing and under autocast to avoid bf16/fp32 mismatches.
                if is_amplify_model and hasattr(model, "layer_norm_2"):
                    with amp_ctx:
                        with torch.inference_mode():
                            hidden = model.layer_norm_2(hidden)

                # Mean pooling with attention mask (always use boolean mask, not additive)
                mask = pooling_mask.unsqueeze(-1).expand(hidden.size()).float()
                sum_embeddings = torch.sum(hidden * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                batch_embs = (sum_embeddings / sum_mask).detach().float().cpu().numpy()

                embs.append(batch_embs)

            embs = np.concatenate(embs, axis=0)

    embs = _sanitize_nan(embs)

    # Reassemble output in original order via lookup dict
    emb_dict = {seq: embs[i] for i, seq in enumerate(flat_seqs)}
    if is_pair:
        output_embs = np.array(
            [np.concatenate([emb_dict[s1], emb_dict[s2]]) for s1, s2 in sequences]
        )
    elif len(flat_seqs) < len(sequences):
        output_embs = np.stack([emb_dict[s] for s in sequences])
    else:
        output_embs = embs

    if l2_normalize_embeddings:
        output_embs = _l2_normalize_embeddings(output_embs)
    return output_embs


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_binary(X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Evaluate binary classification task."""
    return evaluate_classification_probe(
        DEFAULT_RESULT_PROBE,
        "binary",
        X_train,
        y_train,
        X_test,
        y_test,
    )


def evaluate_multiclass(X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Evaluate multiclass classification task."""
    return evaluate_classification_probe(
        DEFAULT_RESULT_PROBE,
        "multiclass",
        X_train,
        y_train,
        X_test,
        y_test,
    )


def evaluate_multilabel(
    X_train,
    y_train,
    X_test,
    y_test,
    mlb: Optional[MultiLabelBinarizer] = None,
    probe_type: str = DEFAULT_RESULT_PROBE,
) -> Dict[str, Any]:
    """Evaluate multilabel classification task.

    ``linear`` = one liblinear LogisticRegression per label (OvR); ``torch_linear``
    = one multi-output sigmoid head (one fit for all labels -- the OvR loop is
    what makes GO/EC slow). Other probes fall back to ``linear``.
    """
    if mlb is None:
        mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)

    y_test_bin = mlb.transform(y_test)

    # Filter out samples with no labels after filtering
    train_mask = y_train_bin.sum(axis=1) > 0
    test_mask = y_test_bin.sum(axis=1) > 0

    X_train_f = X_train[train_mask]
    y_train_f = y_train_bin[train_mask]
    X_test_f = X_test[test_mask]
    y_test_f = y_test_bin[test_mask]

    if len(X_train_f) == 0 or len(X_test_f) == 0:
        return {"Error": "No valid samples after label filtering"}

    if probe_type == "torch_linear":
        clf = make_probe_model(probe_type, "multilabel")
    else:
        clf = OneVsRestClassifier(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(solver="liblinear", random_state=BENCHMARK_SEED),
            ),
            n_jobs=DEFAULT_OVR_N_JOBS,
        )
    fit_seconds = timed_fit(clf, X_train_f, y_train_f)

    preds = clf.predict(X_test_f)

    # No MCC or balanced accuracy here: neither is defined over a multilabel
    # indicator matrix. Bootstrap resampling is fine though -- it resamples
    # rows, and every metric below accepts 2D indicator input.
    def _metrics(yt, yp):
        return {
            "Accuracy": accuracy_score(yt, yp),
            "F1_Macro": f1_score(yt, yp, average="macro", zero_division=0),
            "F1_Micro": f1_score(yt, yp, average="micro", zero_division=0),
        }

    metrics = _metrics(y_test_f, preds)
    metrics.update(_boot_ci(_metrics, y_test_f, preds, BOOTSTRAP_N, BENCHMARK_SEED))
    metrics["ProbeFitSec"] = fit_seconds
    return metrics


def evaluate_regression(X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Evaluate regression task."""
    return evaluate_regression_probe(
        DEFAULT_RESULT_PROBE,
        X_train,
        y_train,
        X_test,
        y_test,
    )


def make_probe_model(
    probe_type: str,
    problem_type: str,
    knn_k: int = 3,
    knn_weights: str = "uniform",
) -> Any:
    """Construct a probe model for a supported task type.

    Args:
        probe_type: Type of probe ("linear", "torch_linear", "histgb", "knn").
        problem_type: Type of problem ("regression", "binary", "multiclass").
        knn_k: Number of neighbors for KNN probes (default: 3).
        knn_weights: Weight function for KNN ("uniform" or "distance", default: "uniform").
    """
    if probe_type == DEFAULT_RESULT_PROBE:
        if problem_type == "regression":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        if problem_type in {"binary", "multiclass"}:
            # lbfgs = native multinomial (no OvR), quasi-Newton; with the
            # StandardScaler it converges in ~100 iters even for 1195-class
            # remote_homology. saga (stochastic) was ~750s/multiclass task here.
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    solver="lbfgs", max_iter=100, random_state=BENCHMARK_SEED
                ),
            )

    if probe_type == "torch_linear":
        from torch_linear_head import TorchLinearHead

        if problem_type == "regression":
            # Standardise y too: MSE from a zero-init head on raw-scale targets
            # (Topt in C, ddG in kcal) would need many more steps.
            return TransformedTargetRegressor(
                regressor=make_pipeline(
                    StandardScaler(), TorchLinearHead(task="regression", seed=BENCHMARK_SEED)
                ),
                transformer=StandardScaler(),
            )
        if problem_type in {"binary", "multiclass"}:
            return make_pipeline(
                StandardScaler(), TorchLinearHead(task="classification", seed=BENCHMARK_SEED)
            )
        if problem_type == "multilabel":
            # Sparse per-label gradients (EC: 572 labels, 0.3 % positives) need a
            # higher lr and more patience than a few-output sequence task; set
            # here, where the task shape is known, rather than hidden in the head.
            return make_pipeline(
                StandardScaler(),
                TorchLinearHead(
                    task="multilabel", lr=1e-2, patience=5, seed=BENCHMARK_SEED
                ),
            )

    if problem_type == "regression":
        if probe_type == "histgb":
            return HistGradientBoostingRegressor(random_state=BENCHMARK_SEED)
        if probe_type == "knn":
            return KNeighborsRegressor(
                n_neighbors=knn_k,
                weights=knn_weights,
                metric="euclidean",
                algorithm="brute",
                n_jobs=DEFAULT_SKLEARN_N_JOBS,
            )

    if problem_type in {"binary", "multiclass"}:
        if probe_type == "histgb":
            return HistGradientBoostingClassifier(random_state=BENCHMARK_SEED)
        if probe_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=knn_k,
                weights=knn_weights,
                metric="euclidean",
                algorithm="brute",
                n_jobs=DEFAULT_SKLEARN_N_JOBS,
            )

    raise ValueError(
        f"Unsupported probe/problem combination: {probe_type}/{problem_type}"
    )


def _make_probe_model_for_training_size(
    probe_type: str,
    problem_type: str,
    train_size: int,
    knn_k: int = 3,
    knn_weights: str = "uniform",
) -> Any:
    """Construct probe model with small-split safeguards for KNN."""
    if probe_type != "knn":
        return make_probe_model(probe_type, problem_type, knn_k, knn_weights)

    n_neighbors = max(1, min(knn_k, train_size))
    if problem_type == "regression":
        return KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=knn_weights,
            metric="euclidean",
            algorithm="brute",
            n_jobs=DEFAULT_SKLEARN_N_JOBS,
        )
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=knn_weights,
        metric="euclidean",
        algorithm="brute",
        n_jobs=DEFAULT_SKLEARN_N_JOBS,
    )


def timed_fit(model, X, y) -> float:
    """Fit ``model`` and return the wall seconds it took.

    One definition so ``ProbeFitSec`` means the same thing on every path
    (sequence, residue, multilabel, CV).
    """
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start


def bootstrap_draws_for(n_boot: int, n_rows: int) -> int:
    """Scale the bootstrap draw count down on very large evaluation sets.

    Residue tasks score 150k-600k rows; each draw recomputes the whole metric
    block, which measures ~0.13 s at 150k rows and ~0.5 s at 600k -- 1000 draws
    would cost minutes against a 60-120 s probe fit. The CI half-width is already
    ~+/-0.001 there, so the extra draws buy nothing. Small evaluation sets, where
    the interval is actually wide, keep the full count.
    """
    if n_boot <= 0 or n_rows <= _BOOTSTRAP_FULL_ROWS:
        return n_boot
    scaled = int(n_boot * _BOOTSTRAP_FULL_ROWS / n_rows)
    return max(_BOOTSTRAP_MIN_DRAWS, min(n_boot, scaled))


def classification_metrics(problem_type: str, y_true, y_pred) -> dict[str, float]:
    """Point metrics for binary / multiclass predictions.

    Single definition shared by the sequence-level probe evaluators and the
    residue-level probe (``token_classification_probe``), and the block that
    ``_boot_ci`` resamples -- so every path reports the same columns.

    Derived from ONE confusion matrix rather than 4-5 independent sklearn passes
    over the labels: each of those builds its own matrix internally, which is
    ~6x slower on the 150k-600k-row residue arrays this is the inner loop for.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        zeros = {"Accuracy": 0.0, "F1_Macro": 0.0, "BalancedAccuracy": 0.0, "MCC": 0.0}
        zeros["F1" if problem_type == "binary" else "F1_Weighted"] = 0.0
        return zeros
    labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    C = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    total = C.sum()

    tp = np.diag(C)
    pred_pos, actual_pos = C.sum(axis=0), C.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
        recall = np.where(actual_pos > 0, tp / actual_pos, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
        present = actual_pos > 0
        balanced = float(recall[present].mean()) if present.any() else 0.0

    # MCC from the matrix (Gorodkin's multiclass form); matches matthews_corrcoef.
    c, sk, pk = tp.sum(), actual_pos, pred_pos
    cov_ytyp = c * total - float(sk @ pk)
    cov_ytyt = total**2 - float(sk @ sk)
    cov_ypyp = total**2 - float(pk @ pk)
    mcc_denom = np.sqrt(cov_ytyt * cov_ypyp)
    metrics = {
        "Accuracy": float(tp.sum() / total),
        "F1_Macro": float(f1.mean()),
        "BalancedAccuracy": balanced,
        "MCC": float(cov_ytyp / mcc_denom) if mcc_denom > 0 else 0.0,
    }
    if problem_type == "binary":
        # Positive class = the larger label, matching sklearn's default pos_label.
        pos = int(np.argmax(labels))
        metrics["F1"] = float(f1[pos])
    else:
        metrics["F1_Weighted"] = float((f1 * actual_pos).sum() / total)
    return metrics


def evaluate_classification_probe(
    probe_type: str,
    problem_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    knn_k: int = 3,
    knn_weights: str = "uniform",
) -> dict[str, float]:
    """Evaluate binary or multiclass classification with the selected probe."""
    if len(y_train) > 0 and isinstance(y_train[0], str):
        label_encoder = LabelEncoder()
        all_labels = sorted(set(y_train) | set(y_test))
        label_encoder.fit(all_labels)
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)
        if probe_type == DEFAULT_RESULT_PROBE and problem_type == "binary":
            logger.info(
                "  Binary label mapping: %s",
                dict(
                    zip(
                        label_encoder.classes_,
                        label_encoder.transform(label_encoder.classes_),
                    )
                ),
            )

    classifier = _make_probe_model_for_training_size(
        probe_type,
        problem_type,
        train_size=len(X_train),
        knn_k=knn_k,
        knn_weights=knn_weights,
    )
    fit_seconds = timed_fit(classifier, X_train, y_train)
    predictions = classifier.predict(X_test)

    metrics = classification_metrics(problem_type, y_test, predictions)
    metrics.update(
        _boot_ci(
            functools.partial(classification_metrics, problem_type),
            y_test,
            predictions,
            BOOTSTRAP_N,
            BENCHMARK_SEED,
        )
    )
    metrics["ProbeFitSec"] = fit_seconds

    if problem_type == "multiclass":
        if hasattr(classifier, "predict_proba"):
            try:
                metrics["AUC"] = roc_auc_score(
                    y_test,
                    classifier.predict_proba(X_test),
                    multi_class="ovr",
                )
            except ValueError as exc:
                logger.warning("  Could not compute AUC for multiclass: %s", exc)
        return metrics

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X_test)
        if probabilities.shape[1] == 2:
            positive_prob = probabilities[:, 1]
            try:
                metrics["AUC"] = roc_auc_score(y_test, positive_prob)
                metrics["AP"] = average_precision_score(y_test, positive_prob)
            except ValueError as exc:
                logger.warning(
                    "  Binary probabilities were unsuitable for AUC/AP: %s",
                    exc,
                )
    return metrics


def evaluate_regression_probe(
    probe_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    knn_k: int = 3,
    knn_weights: str = "uniform",
) -> dict[str, float]:
    """Evaluate regression with the selected probe."""
    regressor = _make_probe_model_for_training_size(
        probe_type,
        "regression",
        train_size=len(X_train),
        knn_k=knn_k,
        knn_weights=knn_weights,
    )
    fit_seconds = timed_fit(regressor, X_train, y_train)
    predictions = regressor.predict(X_test)
    y_test_arr = np.asarray(y_test)

    try:
        spearman_corr, _ = spearmanr(y_test_arr, predictions)
        if np.isnan(spearman_corr):
            logger.warning("  Spearman correlation is NaN (constant predictions?)")
            spearman_corr = 0.0
    except Exception as exc:
        logger.warning("  Could not compute Spearman correlation: %s", exc)
        spearman_corr = 0.0

    mse = float(np.mean((y_test_arr - predictions) ** 2))
    try:
        pearson_corr, _ = pearsonr(y_test_arr, predictions)
        pearson_corr = 0.0 if np.isnan(pearson_corr) else float(pearson_corr)
    except Exception:
        pearson_corr = 0.0
    metrics = {
        "Spearman": float(spearman_corr),
        "Pearson": pearson_corr,
        "MSE": mse,
        "MAE": float(mean_absolute_error(y_test_arr, predictions)),
        "R2": float(r2_score(y_test_arr, predictions)),
        "ProbeFitSec": fit_seconds,
    }

    def _resampled(yt, yp):
        # NaN correlations on a degenerate resample fall back to 0.0, matching
        # how the point estimate above treats a constant prediction.
        rho, _ = spearmanr(yt, yp)
        r, _ = pearsonr(yt, yp)
        return {
            "Spearman": 0.0 if np.isnan(rho) else float(rho),
            "Pearson": 0.0 if np.isnan(r) else float(r),
            "MSE": float(np.mean((yt - yp) ** 2)),
            "MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp)),
        }

    metrics.update(
        _boot_ci(_resampled, y_test_arr, predictions, BOOTSTRAP_N, BENCHMARK_SEED)
    )
    return metrics


def _aggregate_cv_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate numeric metrics across CV folds."""
    if not fold_metrics:
        return {"Error": "No valid CV folds"}

    df = pd.DataFrame(fold_metrics)
    # Select only numeric columns and drop missing
    df_num = df.select_dtypes(include=[np.number])
    aggregated = {k: float(v) for k, v in df_num.mean().items() if np.isfinite(v)}
    # Fit time is a cost, not a score: report the total spent across folds so the
    # column means the same thing as it does on a holdout row.
    if "ProbeFitSec" in df_num:
        aggregated["ProbeFitSec"] = float(df_num["ProbeFitSec"].sum())
    aggregated["CV_Folds"] = len(fold_metrics)
    return aggregated


def evaluate_classification_probe_cv(
    probe_type: str,
    problem_type: str,
    X: np.ndarray,
    y: np.ndarray,
    knn_k: int = 3,
    knn_weights: str = "uniform",
    n_splits: int = 4,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate classification via deterministic cross-validation."""
    seed = BENCHMARK_SEED if seed is None else seed
    if len(X) < n_splits:
        return {"Error": f"Need at least {n_splits} samples for CV"}

    y_array = np.asarray(y)
    if len(np.unique(y_array)) < 2:
        return {"Error": "Need at least two classes for CV"}

    label_counts = Counter(y_array.tolist())
    can_stratify = min(label_counts.values()) >= n_splits
    if can_stratify:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y_array)
    else:
        logger.warning(
            "  Falling back to KFold CV because at least one class has fewer than %d samples",
            n_splits,
        )
        split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)

    fold_metrics: List[Dict[str, Any]] = []
    for train_idx, test_idx in split_iter:
        fold_result = evaluate_classification_probe(
            probe_type,
            problem_type,
            X[train_idx],
            y_array[train_idx],
            X[test_idx],
            y_array[test_idx],
            knn_k=knn_k,
            knn_weights=knn_weights,
        )
        if "Error" not in fold_result:
            fold_metrics.append(fold_result)

    return _aggregate_cv_metrics(fold_metrics)


def evaluate_regression_probe_cv(
    probe_type: str,
    X: np.ndarray,
    y: np.ndarray,
    knn_k: int = 3,
    knn_weights: str = "uniform",
    n_splits: int = 4,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate regression via deterministic cross-validation."""
    seed = BENCHMARK_SEED if seed is None else seed
    if len(X) < n_splits:
        return {"Error": f"Need at least {n_splits} samples for CV"}

    y_array = np.asarray(y, dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = [
        evaluate_regression_probe(
            probe_type,
            X[train_idx],
            y_array[train_idx],
            X[test_idx],
            y_array[test_idx],
            knn_k=knn_k,
            knn_weights=knn_weights,
        )
        for train_idx, test_idx in splitter.split(X)
    ]
    return _aggregate_cv_metrics(fold_metrics)


def evaluate_multilabel_cv(
    X: np.ndarray,
    y: np.ndarray,
    mlb: Optional[MultiLabelBinarizer] = None,
    n_splits: int = 4,
    seed: Optional[int] = None,
    probe_type: str = DEFAULT_RESULT_PROBE,
) -> Dict[str, Any]:
    """Evaluate multilabel classification via deterministic cross-validation."""
    seed = BENCHMARK_SEED if seed is None else seed
    if len(X) < n_splits:
        return {"Error": f"Need at least {n_splits} samples for CV"}

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: List[Dict[str, Any]] = []
    for train_idx, test_idx in splitter.split(X):
        fold_result = evaluate_multilabel(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            y[test_idx],
            mlb,
            probe_type=probe_type,
        )
        if "Error" not in fold_result:
            fold_metrics.append(fold_result)

    return _aggregate_cv_metrics(fold_metrics)


def _evaluate_proteingym_supervised_probe(
    cfg: TaskConfig,
    train_seqs: List,
    train_labels: List,
    extra_data: np.ndarray,
    model_obj: Any,
    is_sbert: bool,
    device: str,
    probe_type: str,
    *,
    knn_k: int,
    knn_weights: str,
    batch_size: int,
    max_length: int,
    amp_dtype: Optional[torch.dtype],
    embed_save_path: Optional[str],
    l2_normalize_embeddings: bool,
    probe_embed_mode: str = "trunk",
) -> Dict[str, Any]:
    """Evaluate supervised ProteinGym tasks with the selected non-linear probe."""
    logger.info(
        "  Generating embeddings (%s ProteinGym supervised)...",
        probe_label(probe_type),
    )
    embeddings = embed_sequences(
        model_obj,
        is_sbert,
        train_seqs,
        device,
        batch_size=batch_size,
        max_length=max_length,
        amp_dtype=amp_dtype,
        embed_save_path=embed_save_path,
        l2_normalize_embeddings=l2_normalize_embeddings,
        probe_embed_mode=probe_embed_mode,
    )
    labels = np.asarray(train_labels)
    groups = np.asarray(extra_data)
    metric_values: list[float] = []
    random_state = np.random.RandomState(BENCHMARK_SEED)

    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() < 10:
            continue

        X_group = embeddings[mask]
        y_group = labels[mask]
        shuffled = random_state.permutation(len(X_group))
        X_group = X_group[shuffled]
        y_group = y_group[shuffled]

        split_idx = int(len(X_group) * 0.8)
        if split_idx < 2 or split_idx >= len(X_group):
            continue

        X_train = X_group[:split_idx]
        X_test = X_group[split_idx:]
        y_train = y_group[:split_idx]
        y_test = y_group[split_idx:]

        try:
            if cfg.problem_type == "regression":
                metrics = evaluate_regression_probe(
                    probe_type,
                    X_train,
                    np.asarray(y_train, dtype=float),
                    X_test,
                    np.asarray(y_test, dtype=float),
                    knn_k=knn_k,
                    knn_weights=knn_weights,
                )
                metric_value = float(metrics[cfg.main_metric])
            else:
                y_train_arr = np.asarray(y_train)
                y_test_arr = np.asarray(y_test)
                if len(np.unique(y_train_arr)) < 2 or len(np.unique(y_test_arr)) < 2:
                    continue
                metrics = evaluate_classification_probe(
                    probe_type,
                    cfg.problem_type,
                    X_train,
                    y_train_arr,
                    X_test,
                    y_test_arr,
                    knn_k=knn_k,
                    knn_weights=knn_weights,
                )
                if cfg.main_metric in metrics:
                    metric_value = float(metrics[cfg.main_metric])
                elif "Accuracy" in metrics:
                    metric_value = float(metrics["Accuracy"])
                else:
                    continue
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "Skipping assay %s for task %s due to probe error: %s",
                str(group),
                cfg.name,
                exc,
            )
            continue

        if np.isfinite(metric_value):
            metric_values.append(metric_value)

    if not metric_values:
        return {"Error": "No valid groups for evaluation"}
    return {
        cfg.main_metric: float(np.mean(metric_values)),
        "Assays_Evaluated": len(metric_values),
    }


def evaluate_retrieval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: Tuple[int, ...] = (1, 10, 30),
) -> Dict[str, float]:
    """Evaluate structural retrieval with family-level Recall@K.

    Args:
        embeddings: Array of query/gallery embeddings with shape (n_samples, dim).
        labels: Family labels aligned to `embeddings`.
        k_list: Recall cutoffs to evaluate.

    Returns:
        Dictionary mapping Recall@K metric names to scores in [0, 1].

    Raises:
        ValueError: If embeddings and labels have mismatched lengths.
    """
    if len(embeddings) != len(labels):
        raise ValueError("Embeddings and labels must have the same number of rows")

    if len(embeddings) < 2:
        return {f"Recall@{k}": 0.0 for k in k_list}

    max_k = max(k_list)
    neighbor_count = min(len(embeddings), max_k + 1)
    nn = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        n_jobs=DEFAULT_SKLEARN_N_JOBS,
    )
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    label_array = np.asarray(labels)
    results: Dict[str, float] = {}

    queries = np.arange(len(embeddings))[:, None]
    valid_mask = indices != queries
    neighbor_labels = label_array[indices]

    for k in k_list:
        matches = [
            (nl[vm][:k] == ql).any()
            for nl, vm, ql in zip(neighbor_labels, valid_mask, label_array)
        ]
        results[f"Recall@{k}"] = float(np.mean(matches))

    return results


def _run_zeroshot_tta(
    cfg,
    model_obj,
    is_sbert,
    device,
    mutants,
    wts,
    groups,
    labels,
    tta_cfg,
    batch_size,
    max_length,
    amp_dtype,
    l2_normalize_embeddings,
):
    """Per-assay WT test-time training for the ProteinGym zero-shot path.

    Returns a list of per-group metric values, or ``None`` when TTA cannot apply
    to this model (SentenceTransformer, or no reachable MLM head) so the caller
    falls back to the standard embedding-cosine path. The embedding cache is
    bypassed (``embed_save_path=None``): per-assay adapted weights make any cached
    embedding stale by construction.
    """
    if is_sbert:
        logger.info(
            "  --tta requested but model is a SentenceTransformer (no reachable "
            "MLM head); running standard zero-shot instead."
        )
        return None

    from wt_test_time_training import resolve_mlm_head, run_tta_zeroshot

    tokenizer, model = model_obj
    try:
        refs = resolve_mlm_head(model, tokenizer)
    except ValueError as exc:
        logger.warning("  --tta disabled for this model: %s", exc)
        return None

    logger.info(
        "  WT test-time training: iters=%d lr=%g layers=%d mask_rate=%g "
        "train_head=%s (embedding cache bypassed)",
        tta_cfg.iters, tta_cfg.lr, tta_cfg.n_layers, tta_cfg.mask_rate,
        tta_cfg.train_head,
    )

    def embed_fn(seqs):
        return embed_sequences(
            model_obj, is_sbert, seqs, device,
            batch_size=batch_size, max_length=max_length, amp_dtype=amp_dtype,
            embed_save_path=None,
            l2_normalize_embeddings=l2_normalize_embeddings,
        )

    return run_tta_zeroshot(
        model, refs, tokenizer, mutants, wts, groups, labels,
        cfg.problem_type, embed_fn, tta_cfg, device=device, max_length=max_length,
    )


def evaluate_task(
    cfg: TaskConfig,
    model_obj,
    is_sbert: bool,
    device: str,
    max_samples: Optional[int] = None,
    top_k_labels_override: Optional[int] = None,
    amp_dtype: Optional[torch.dtype] = None,
    embed_save_path: Optional[str] = None,
    batch_size: int = 128,
    max_length: int = DEFAULT_EMBED_MAX_LENGTH,
    probe_type: str = DEFAULT_RESULT_PROBE,
    knn_k: int = 3,
    knn_weights: str = "uniform",
    l2_normalize_embeddings: bool = False,
    eval_split: str = DEFAULT_BENCHMARK_EVAL_SPLIT,
    tta_cfg=None,
    probe_embed_mode: str = "trunk",
) -> Tuple[Dict[str, Any], str, str]:
    """Run full evaluation for a single task.

    ``tta_cfg`` (a ``wt_test_time_training.TTAConfig`` or None) enables wild-type
    test-time training on the ProteinGym zero-shot path; ignored elsewhere.
    """

    # Resolve the probe ONCE, here, so every downstream evaluator receives a real
    # probe rather than a request: 'auto' picks per task shape, and a probe the
    # task cannot route through (knn on multilabel) collapses to the linear
    # identity. Doing it at the entry point is what lets the per-branch guards go.
    probe_type = effective_probe_type(cfg, probe_type)

    logger.info(f"Evaluating: {cfg.name}")

    # Residue-level (per-token) tasks use a separate linear-probe path that
    # extracts per-residue hidden states + fits a LogisticRegression. The
    # sequence-level ``prepare_data`` parser cannot handle per-residue
    # labels, so dispatch BEFORE it runs. See token_classification_probe.py.
    if cfg.problem_type == "token_classification":
        from token_classification_probe import (
            EmbeddingCache,
            evaluate_token_classification,
        )

        train_seqs, train_labels, test_seqs, test_labels, split_metadata = (
            _prepare_token_classification_data(cfg, max_samples, eval_split)
        )
        resolved_eval_split = str(
            split_metadata.get("resolved_eval_split", DEFAULT_RESULT_EVAL_SPLIT)
        )
        eval_strategy = str(
            split_metadata.get("eval_strategy", DEFAULT_RESULT_EVAL_STRATEGY)
        )
        tokenizer, encoder = _unwrap_encoder_tokenizer(model_obj, is_sbert)
        cache_root = None
        if embed_save_path:
            cache_root = Path(embed_save_path).parent / "residue_cache"
        cache = EmbeddingCache(cache_root) if cache_root else None
        model_hash = embed_save_path or cfg.dataset
        metrics = evaluate_token_classification(
            cfg=cfg,
            encoder=encoder,
            tokenizer=tokenizer,
            train_sequences=train_seqs,
            train_labels=train_labels,
            test_sequences=test_seqs,
            test_labels=test_labels,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            cache=cache,
            model_hash=model_hash,
            task_key=cfg.name,
            probe_type=probe_type,
        )
        return metrics, resolved_eval_split, eval_strategy

    # Pairwise residue-residue tasks. Same reason for dispatching early: the
    # sequence-level parser cannot carry coordinate arrays. See contact_probe.py.
    if cfg.problem_type == "contact_prediction":
        from contact_probe import evaluate_contact_prediction

        train_records, test_records, split_metadata = _prepare_contact_data(
            cfg, max_samples, eval_split, train_proteins=CONTACT_TRAIN_PROTEINS
        )
        resolved_eval_split = str(
            split_metadata.get("resolved_eval_split", DEFAULT_RESULT_EVAL_SPLIT)
        )
        eval_strategy = str(
            split_metadata.get("eval_strategy", DEFAULT_RESULT_EVAL_STRATEGY)
        )
        tokenizer, encoder = _unwrap_encoder_tokenizer(model_obj, is_sbert)
        metrics = evaluate_contact_prediction(
            encoder=encoder,
            tokenizer=tokenizer,
            train_records=train_records,
            test_records=test_records,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            train_proteins=CONTACT_TRAIN_PROTEINS,
            seed=BENCHMARK_SEED,
        )
        return metrics, resolved_eval_split, eval_strategy

    train_seqs, train_labels, test_seqs, test_labels, extra_data, split_metadata = (
        prepare_data(
            cfg,
            max_samples,
            eval_split=eval_split,
            top_k_labels_override=top_k_labels_override,
        )
    )

    resolved_eval_split = str(
        split_metadata.get("resolved_eval_split", DEFAULT_RESULT_EVAL_SPLIT)
    )
    eval_strategy = str(
        split_metadata.get("eval_strategy", DEFAULT_RESULT_EVAL_STRATEGY)
    )
    use_cv_fallback = bool(split_metadata.get("cv_fallback", False))

    if cfg.eval_mode == "proteingym_supervised":
        if not train_seqs:
            return (
                {"Error": "Missing required column (see logs)"},
                resolved_eval_split,
                eval_strategy,
            )
        if extra_data is None:
            return (
                {"Error": "Missing group labels for evaluation"},
                resolved_eval_split,
                eval_strategy,
            )
        if not isinstance(extra_data, np.ndarray):
            return (
                {"Error": "Invalid group labels for ProteinGym supervised mode"},
                resolved_eval_split,
                eval_strategy,
            )
        return (
            _evaluate_proteingym_supervised_probe(
                cfg,
                train_seqs,
                train_labels,
                extra_data,
                model_obj,
                is_sbert,
                device,
                probe_type,
                knn_k=knn_k,
                knn_weights=knn_weights,
                batch_size=batch_size,
                max_length=max_length,
                amp_dtype=amp_dtype,
                embed_save_path=embed_save_path,
                l2_normalize_embeddings=l2_normalize_embeddings,
                probe_embed_mode=probe_embed_mode,
            ),
            resolved_eval_split,
            eval_strategy,
        )

    # --- ProteinGym per-assay evaluation ---
    if cfg.eval_mode.startswith("proteingym"):
        if not train_seqs:
            return (
                {"Error": "Missing required column (see logs)"},
                resolved_eval_split,
                eval_strategy,
            )
        if extra_data is None:
            return (
                {"Error": "Missing group labels for evaluation"},
                resolved_eval_split,
                eval_strategy,
            )
        groups = np.asarray(extra_data)
        labels = np.array(train_labels)
        group_metrics: List[float] = []

        if cfg.eval_mode == "proteingym_zeroshot":
            # train_seqs = [(mutant, wt), ...] — keys sorted: "mutant" < "wt"
            mutants = [s[0] for s in train_seqs]
            wts = [s[1] for s in train_seqs]

            # Optional wild-type test-time training: adapt the model to each
            # assay's WT (a few MLM rounds) before the same cosine readout.
            # Returns None (and we fall through to baseline) when TTA is not
            # applicable to this model — e.g. SentenceTransformer or no MLM head.
            if tta_cfg is not None:
                tta_group_metrics = _run_zeroshot_tta(
                    cfg, model_obj, is_sbert, device, mutants, wts, groups, labels,
                    tta_cfg, batch_size=batch_size, max_length=max_length,
                    amp_dtype=amp_dtype,
                    l2_normalize_embeddings=l2_normalize_embeddings,
                )
                if tta_group_metrics is not None:
                    group_metrics = tta_group_metrics
                    valid_metrics = [x for x in group_metrics if np.isfinite(x)]
                    if not valid_metrics:
                        return (
                            {"Error": "No valid groups for evaluation"},
                            resolved_eval_split,
                            eval_strategy,
                        )
                    return (
                        {
                            cfg.main_metric: float(np.mean(valid_metrics)),
                            "Assays_Evaluated": len(valid_metrics),
                        },
                        resolved_eval_split,
                        eval_strategy,
                    )

            logger.info("  Generating embeddings (zero-shot)...")
            mutant_embs = embed_sequences(
                model_obj,
                is_sbert,
                mutants,
                device,
                batch_size=batch_size,
                max_length=max_length,
                amp_dtype=amp_dtype,
                embed_save_path=embed_save_path,
                l2_normalize_embeddings=l2_normalize_embeddings,
            )
            wt_embs = embed_sequences(
                model_obj,
                is_sbert,
                wts,
                device,
                batch_size=batch_size,
                max_length=max_length,
                amp_dtype=amp_dtype,
                embed_save_path=embed_save_path,
                l2_normalize_embeddings=l2_normalize_embeddings,
            )
            sims = F.cosine_similarity(
                torch.as_tensor(mutant_embs), torch.as_tensor(wt_embs)
            ).numpy()
            for g in np.unique(groups):
                mask = groups == g
                if mask.sum() < 2:
                    continue
                y_g, s_g = labels[mask].astype(float), sims[mask]
                if cfg.problem_type == "regression":
                    corr, _ = spearmanr(y_g, s_g)
                    group_metrics.append(float(corr) if not np.isnan(corr) else 0.0)
                else:
                    # Clinical pathogenicity: pathogenic = deleterious = mutant
                    # embedding FURTHER from WT = LOWER cosine. Negate so
                    # pathogenic ranks high (same sign convention as the MLM
                    # masked-marginal path). Raw cosine gives an inverted AUC
                    # (~0.32 → 0.68 after the flip).
                    try:
                        group_metrics.append(roc_auc_score(y_g, -s_g))
                    except ValueError:
                        pass

        else:
            return (
                {"Error": f"Unknown proteingym eval_mode: {cfg.eval_mode}"},
                resolved_eval_split,
                eval_strategy,
            )

        valid_metrics = [x for x in group_metrics if np.isfinite(x)]
        if not valid_metrics:
            return (
                {"Error": "No valid groups for evaluation"},
                resolved_eval_split,
                eval_strategy,
            )
        return (
            {
                cfg.main_metric: float(np.mean(valid_metrics)),
                "Assays_Evaluated": len(valid_metrics),
            },
            resolved_eval_split,
            eval_strategy,
        )

    if cfg.problem_type == "retrieval":
        if probe_type != DEFAULT_RESULT_PROBE:
            logger.info(
                "  Retrieval uses the built-in evaluator; ignoring probe_type=%s",
                probe_type,
            )
        logger.info("  Generating retrieval embeddings...")
        retrieval_embs = embed_sequences(
            model_obj,
            is_sbert,
            train_seqs,
            device,
            batch_size=batch_size,
            max_length=max_length,
            amp_dtype=amp_dtype,
            embed_save_path=embed_save_path,
            l2_normalize_embeddings=l2_normalize_embeddings,
            probe_embed_mode=probe_embed_mode,
        )
        return (
            evaluate_retrieval(retrieval_embs, np.asarray(train_labels)),
            resolved_eval_split,
            eval_strategy,
        )

    mlb = extra_data if isinstance(extra_data, MultiLabelBinarizer) else None

    if use_cv_fallback:
        logger.info("  Generating embeddings (4-fold CV fallback)...")
        X_train = embed_sequences(
            model_obj,
            is_sbert,
            train_seqs,
            device,
            batch_size=batch_size,
            max_length=max_length,
            amp_dtype=amp_dtype,
            embed_save_path=embed_save_path,
            l2_normalize_embeddings=l2_normalize_embeddings,
            probe_embed_mode=probe_embed_mode,
        )
        y_train = np.array(
            train_labels,
            dtype=object if cfg.problem_type == "multilabel" else None,
        )

        if cfg.problem_type == "binary":
            metrics = evaluate_classification_probe_cv(
                probe_type,
                "binary",
                X_train,
                y_train,
                knn_k=knn_k,
                knn_weights=knn_weights,
            )
        elif cfg.problem_type == "multiclass":
            metrics = evaluate_classification_probe_cv(
                probe_type,
                "multiclass",
                X_train,
                y_train,
                knn_k=knn_k,
                knn_weights=knn_weights,
            )
        elif cfg.problem_type == "multilabel":
            metrics = evaluate_multilabel_cv(X_train, y_train, mlb, probe_type=probe_type)
        else:
            metrics = evaluate_regression_probe_cv(
                probe_type,
                X_train,
                y_train,
                knn_k=knn_k,
                knn_weights=knn_weights,
            )
        return metrics, resolved_eval_split, eval_strategy

    # --- Standard evaluation path ---
    if test_seqs is None or test_labels is None:
        return (
            {"Error": "Missing eval data for standard evaluation"},
            resolved_eval_split,
            eval_strategy,
        )

    logger.info("  Generating embeddings...")
    # Content-addressed disk cache so the TRAIN embeddings extracted for the
    # validation probe are reused by the test probe (a separate process) instead
    # of re-extracted — the sequence path never persisted them (only the residue
    # + embed_dataset paths did), so heavy tasks paid their train extraction
    # twice. Safe: keyed on the exact seqs + embed config; ANY mismatch or cache
    # error re-extracts (perf-only, never changes results). See seq_embed_cache.
    from seq_embed_cache import cached_embed_sequences

    _seq_cache_root = (
        str(Path(embed_save_path).parent / "seq_cache") if embed_save_path else None
    )
    _cfg_key = (
        f"{probe_embed_mode}|l2={int(bool(l2_normalize_embeddings))}"
        f"|ml={max_length}|dt={amp_dtype}"
    )

    def _embed(_seqs):
        return embed_sequences(
            model_obj,
            is_sbert,
            _seqs,
            device,
            batch_size=batch_size,
            max_length=max_length,
            amp_dtype=amp_dtype,
            embed_save_path=embed_save_path,
            l2_normalize_embeddings=l2_normalize_embeddings,
            probe_embed_mode=probe_embed_mode,
        )

    X_train = cached_embed_sequences(
        lambda: _embed(train_seqs), train_seqs,
        cache_root=_seq_cache_root, cfg_key=_cfg_key,
    )
    X_test = cached_embed_sequences(
        lambda: _embed(test_seqs), test_seqs,
        cache_root=_seq_cache_root, cfg_key=_cfg_key,
    )

    y_train = np.array(
        train_labels, dtype=object if cfg.problem_type == "multilabel" else None
    )
    y_test = np.array(
        test_labels, dtype=object if cfg.problem_type == "multilabel" else None
    )

    # Apply L2 normalization if requested
    if l2_normalize_embeddings:
        norms_train = np.linalg.norm(X_train, axis=1, keepdims=True).clip(min=1e-12)
        X_train = X_train / norms_train
        norms_test = np.linalg.norm(X_test, axis=1, keepdims=True).clip(min=1e-12)
        X_test = X_test / norms_test
        logger.info("  Applied L2 normalization to embeddings")

    logger.info("  Training %s probe...", probe_label(probe_type))
    if cfg.problem_type == "binary":
        results = evaluate_classification_probe(
            probe_type,
            "binary",
            X_train,
            y_train,
            X_test,
            y_test,
            knn_k=knn_k,
            knn_weights=knn_weights,
        )
    elif cfg.problem_type == "multiclass":
        results = evaluate_classification_probe(
            probe_type,
            "multiclass",
            X_train,
            y_train,
            X_test,
            y_test,
            knn_k=knn_k,
            knn_weights=knn_weights,
        )
    elif cfg.problem_type == "multilabel":
        results = evaluate_multilabel(X_train, y_train, X_test, y_test, mlb, probe_type=probe_type)
    else:  # regression
        results = evaluate_regression_probe(
            probe_type,
            X_train,
            y_train,
            X_test,
            y_test,
            knn_k=knn_k,
            knn_weights=knn_weights,
        )

    return results, resolved_eval_split, eval_strategy


# =============================================================================
# Result Tracking
# =============================================================================


class ResultTracker:
    """Track and display benchmark results.

    Uses a stable filename per model (`bench_{model}.csv`) so that successive
    runs with different tasks can be appended into the same file.  Each row
    carries a `Date` column (YYYY-MM-DD).  When merging with an existing CSV:
        - Duplicate (Task, Samples, Date, Probe, EvalMode, EvalSplit, EvalStrategy)
            rows are overwritten by the new run.
    - Rows from different days are preserved (history).
    """

    round_decimals = 5

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = []
        self.date = datetime.now().strftime("%Y-%m-%d")

    @classmethod
    def _round_numeric_values(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Round numeric result columns to the configured decimal precision."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        rounded_df = df.copy()
        rounded_df[numeric_cols] = rounded_df[numeric_cols].round(cls.round_decimals)
        return rounded_df

    def add(
        self,
        task_name: str,
        metrics: Dict[str, Any],
        samples: Optional[int],
        probe: str = DEFAULT_RESULT_PROBE,
        eval_mode: str = DEFAULT_RESULT_EVAL_MODE,
        eval_split: str = DEFAULT_RESULT_EVAL_SPLIT,
        eval_strategy: str = DEFAULT_RESULT_EVAL_STRATEGY,
        benchmark_seed: Optional[int] = None,
        embedding_norm: str = "none",
    ):
        row = {
            "Model": self.model_name,
            "Task": task_name,
            "Samples": samples if samples else "Full",
            "Date": self.date,
            "Probe": probe,
            "EvalMode": eval_mode,
            "EvalSplit": eval_split,
            "EvalStrategy": eval_strategy,
            "EmbeddingNorm": embedding_norm,
        }
        if benchmark_seed is not None:
            row["BenchmarkSeed"] = benchmark_seed
        row.update(metrics)
        self.results.append(row)

    def display(self):
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        priority_cols = [
            "Task",
            "Samples",
            "BenchmarkSeed",
            "Probe",
            "EvalMode",
            "EvalSplit",
            "EvalStrategy",
            "EmbeddingNorm",
        ]
        present_priority_cols = [col for col in priority_cols if col in df.columns]
        other_cols = [
            c for c in df.columns if c not in present_priority_cols + ["Model", "Date"]
        ]
        cols = present_priority_cols + sorted(other_cols)

        print("\n" + "=" * 80)
        print(f" BENCHMARK RESULTS - {self.model_name}")
        print("=" * 80)
        print(df[cols].to_string(index=False))

    def save(self, output_dir: str = "."):
        """Save results, merging with any existing file for this model."""
        if not self.results:
            return None

        new_df = pd.DataFrame(self.results)
        defaults = {
            "Probe": DEFAULT_RESULT_PROBE,
            "EvalMode": DEFAULT_RESULT_EVAL_MODE,
            "EvalSplit": DEFAULT_RESULT_EVAL_SPLIT,
            "EvalStrategy": DEFAULT_RESULT_EVAL_STRATEGY,
            "EmbeddingNorm": "none",
            "BenchmarkSeed": "",
            "Samples": "Full",
        }
        for col, val in defaults.items():
            if col not in new_df.columns:
                new_df[col] = val
            new_df[col] = new_df[col].fillna(val).astype(str)

        safe_model_name = self.model_name.replace("/", "_").replace("\\", "_")
        filename = f"bench_{safe_model_name}.csv"
        # Create the directory here rather than relying on each caller. Saving is
        # the LAST thing a run does, so a missing output dir discards the whole
        # run's compute at the final step -- an hour of contact_catjac scoring,
        # in the case that prompted this.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(output_dir) / filename

        # Merge with existing results if the file already exists
        if filepath.exists():
            try:
                old_df = pd.read_csv(filepath)
                for col, val in defaults.items():
                    if col not in old_df.columns:
                        old_df[col] = val
                    old_df[col] = old_df[col].fillna(val).astype(str)
                # Concatenate old + new, then drop same-day duplicates (keep new)
                merged = pd.concat([old_df, new_df], ignore_index=True)
                dedup_cols = [
                    "Task",
                    "Samples",
                    "Date",
                    "BenchmarkSeed",
                    "Probe",
                    "EvalMode",
                    "EvalSplit",
                    "EvalStrategy",
                    "EmbeddingNorm",
                ]
                merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
                new_df = merged
                logger.info(
                    f"Merged with existing results ({len(old_df)} old rows -> "
                    f"{len(new_df)} total rows)"
                )
            except Exception as e:
                # Existing file is corrupt — save to recovery file, don't lose data
                recovery = filepath.with_name(
                    f"bench_{safe_model_name}_recovery_{self.date}.csv"
                )
                logger.warning(
                    f"Could not read existing {filepath} ({e}). "
                    f"Saving new results to {recovery}"
                )
                filepath = recovery

        new_df = self._round_numeric_values(new_df)
        new_df.to_csv(filepath, index=False)
        logger.info(
            "Results saved to: %s (numeric metrics rounded to %d decimals)",
            filepath,
            self.round_decimals,
        )

        return filepath


# =============================================================================
# CLI & Main
# =============================================================================


def print_task_table() -> None:
    """Print every task with its type, metric, and which preset selects it."""
    # Derived from the actual preset membership, not assumed. A few tasks
    # (cafa5, go_mf) are in no preset at all and must say so -- claiming
    # "default" would send people to --no-fast, which does not include them.
    groups = {
        k: (
            "very-fast" if k in VERY_FAST_TASKS
            else "fast" if k in FAST_TASKS or k in RETRIEVAL_TASKS
            else "proteingym" if k in PROTEINGYM_TASKS
            else "default" if k in DEFAULT_TASKS
            else "none"
        )
        for k in TASKS
    }
    width = max(len(k) for k in TASKS)
    print(f"{'TASK':<{width}}  {'TYPE':<20} {'METRIC':<12} {'PRESET':<11} NAME")
    for key in sorted(TASKS):
        cfg = TASKS[key]
        print(
            f"{key:<{width}}  {cfg.problem_type:<20} {cfg.main_metric:<12} "
            f"{groups[key]:<11} {cfg.name}"
        )
    print(
        f"\n{len(TASKS)} tasks. Presets: --very-fast ({len(VERY_FAST_TASKS)} scout "
        f"tasks), --fast ({len(FAST_TASKS) + len(RETRIEVAL_TASKS)}), "
        f"default ({len(DEFAULT_TASKS)}, excludes ProteinGym), "
        f"--proteingym adds the {len(PROTEINGYM_TASKS)} ProteinGym tasks."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate protein language models on benchmark tasks"
    )

    # Comparison mode
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two benchmark result files/models",
    )
    parser.add_argument(
        "--compare_model1",
        type=str,
        default=None,
        help="First model/directory/CSV file for comparison",
    )
    parser.add_argument(
        "--compare_model2",
        type=str,
        default=None,
        help="Second model/directory/CSV file for comparison",
    )

    # Evaluation mode (default)
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="facebook/esm2_t30_150M_UR50D",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--probe_type",
        "-p",
        choices=tuple(PROBE_LABELS),
        default=DEFAULT_RESULT_PROBE,
        help="Probe model type. binary/multiclass/regression/token tasks use the selected probe; "
        "multilabel supports linear and torch_linear; retrieval and ProteinGym zero-shot keep "
        "their built-in evaluators.",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Precision for embedding computation. Default fp32 for reproducibility; bf16 for speed (advanced).",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
        help=(
            "Attention backend override for Hugging Face loads. "
            "Default 'auto' prefers flash_attention_2 when installed, otherwise sdpa."
        ),
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(TASKS.keys()),
        # metavar keeps argparse from printing all 43 task names twice in the
        # usage line, which buried every other flag. --list_tasks shows them.
        metavar="TASK",
        help="Tasks to run. Overrides the presets. With no --tasks, --fast is on "
        "by default; pass --no-fast for the full set. See --list_tasks.",
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="Print the available tasks with their type and metric, then exit.",
    )
    parser.add_argument(
        "--max_samples",
        "-n",
        type=int,
        default=None,
        help="Max samples per split for quick testing",
    )
    parser.add_argument(
        "--top_k_labels",
        type=int,
        default=None,
        help="Optional override for keeping only the top-K most frequent labels for label-filtered tasks.",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a fast subset of tasks (default: True)",
    )
    parser.add_argument(
        "--very-fast",
        dest="very_fast",
        action="store_true",
        default=False,
        help=(
            "Run only the curated very-fast / low-variance subset "
            f"({', '.join(VERY_FAST_TASKS)}) for high-ROI scout comparisons. "
            "Takes precedence over --fast; ignored if --tasks is given."
        ),
    )
    parser.add_argument(
        "--contact_train_proteins",
        type=int,
        default=CONTACT_TRAIN_PROTEINS,
        help=(
            "Training proteins for the contact_probe pairwise probe "
            f"(default: {CONTACT_TRAIN_PROTEINS}). Lower it for a smoke test; "
            "the full train split is 25k proteins and does not fit in memory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="results/benchmarks",
        help="Directory to save results CSV",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (auto/cuda/cpu)"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="Resamples for percentile CIs on probe metrics, adding <Metric>_CI_low / "
        "_CI_high columns (default: 0, off; 1000 is plenty). Covers the label-only "
        "metrics -- AUC and AP are computed from predict_proba afterwards and get no "
        "interval. Do not combine with the cross-validation path (--eval_split "
        "validation on a task with no validation split): fold aggregation averages "
        "every numeric column, which turns the CIs into a mean of intervals.",
    )
    parser.add_argument(
        "--cache_embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache embeddings to disk under --embed_cache_dir/<model_name>/embeddings.pth "
        "(default: off). Opt in for static models you will re-benchmark. Note the cache "
        "key for a hub model id is just the name, so it does NOT invalidate when the "
        "upstream weights change; local paths are keyed on file size and mtime.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_EMBED_MAX_LENGTH,
        help="Maximum tokenized sequence length used for all embedding paths (default: 1024)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model-aware torch.compile for the manual HF embedding path (default: False). Uses backend=inductor, dynamic=False by default, and only enables extra Dynamo workarounds when the loaded model uses HF ESM rotary caches.",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear embedding cache for the model before running",
    )
    parser.add_argument(
        "--embed_cache_dir",
        type=str,
        default="embed_cache",
        help="Root directory for embedding caches (default: embed_cache/). "
        "Each model gets its own subfolder: <dir>/<safe_model_name>/embeddings.pth.",
    )
    parser.add_argument(
        "--proteingym",
        action="store_true",
        default=False,
        help="Add all 8 ProteinGym tasks to the run. "
        "These are large/slow and excluded from --fast and --no-fast by default.",
    )
    parser.add_argument(
        "--eval_split",
        "-e",
        choices=sorted(SUPPORTED_EVAL_SPLITS),
        default=DEFAULT_BENCHMARK_EVAL_SPLIT,
        help=(
            "Evaluation target for supervised tasks. "
            "validation (default) uses explicit validation splits when present and "
            "falls back to deterministic 4-fold CV on train when absent; "
            "test preserves historical test-set behavior. "
            "Retrieval and ProteinGym tasks are unchanged."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BENCHMARK_SEED,
        help="Seed used for benchmark subsampling, CV splits, and probe randomness.",
    )
    parser.add_argument(
        "--seed_list",
        type=str,
        default=None,
        help=(
            "Optional comma-separated benchmark seeds. When provided, the suite "
            "runs once per seed within a single process while reusing model load "
            "and embedding cache state."
        ),
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=3,
        help="Number of neighbors for KNN probes (default: 3). Used only when probe_type=knn.",
    )
    parser.add_argument(
        "--knn_weights",
        choices=["uniform", "distance"],
        default="uniform",
        help="Weight function for KNN probes: uniform or distance-based (default: uniform). Used only when probe_type=knn.",
    )
    parser.add_argument(
        "--l2_normalize_embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "L2-normalize pooled embeddings before probe training/evaluation. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--probe-embed-mode",
        choices=["trunk", "trunk_and_aux", "aux_only"],
        default="trunk",
        dest="probe_embed_mode",
        help=(
            "trunk (default): standard mean-pooled encoder hidden state. "
            "trunk_and_aux: concatenate non-None aux-head outputs (3Di, cons, tax, pLDDT…). "
            "aux_only: aux heads only (diagnostic). "
            "Proteva models only; other models log a warning and fall back to trunk."
        ),
    )

    # --- WT test-time training (TTT/TTA), opt-in; ProteinGym zero-shot only ---
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help=(
            "Enable wild-type test-time training: run a few MLM rounds on each "
            "assay's WT before the zero-shot embedding-cosine readout (ProteinTTT, "
            "arXiv 2411.02109). ProteinGym zero-shot tasks only; AMPLIFY/Proteva "
            "(non-SentenceTransformer) models with a reachable MLM head. Forces the "
            "embedding cache off. No-op (with a logged reason) elsewhere."
        ),
    )
    parser.add_argument(
        "--tta-iters", dest="tta_iters", type=int, default=20,
        help="WT-TTT SGD steps per assay (default: 20; ProteinTTT uses 10-30).",
    )
    parser.add_argument(
        "--tta-lr", dest="tta_lr", type=float, default=4e-4,
        help="WT-TTT learning rate; SGD momentum=0, weight_decay=0 (default: 4e-4).",
    )
    parser.add_argument(
        "--tta-layers", dest="tta_layers", type=int, default=2,
        help="Top encoder blocks to unfreeze for WT-TTT (default: 2).",
    )
    parser.add_argument(
        "--tta-mask-rate", dest="tta_mask_rate", type=float, default=0.15,
        help="MLM mask rate for WT-TTT, replicating pretraining (default: 0.15).",
    )
    parser.add_argument(
        "--tta-train-head", dest="tta_train_head",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Also fine-tune the MLM head during WT-TTT (default: True). "
            "--no-tta-train-head freezes the head (adapt backbone only, per ProteinTTT)."
        ),
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    global BENCHMARK_SEED, BOOTSTRAP_N, CONTACT_TRAIN_PROTEINS

    args = parse_args()

    if args.list_tasks:
        print_task_table()
        return
    benchmark_seeds = (
        parse_seed_list(args.seed_list) if args.seed_list is not None else [args.seed]
    )
    BENCHMARK_SEED = benchmark_seeds[0]
    seed_all(BENCHMARK_SEED)
    BOOTSTRAP_N = args.bootstrap
    if BOOTSTRAP_N:
        logger.info("Bootstrap CIs enabled: %d resamples per metric", BOOTSTRAP_N)
    CONTACT_TRAIN_PROTEINS = args.contact_train_proteins

    # Several older copies of this suite still exist on disk, and they disagree
    # about the task registry (ec_classification in particular). Record which one
    # actually ran, so a results directory can never be traced to the wrong code.
    logger.info(
        "Suite: %s (%d tasks)", Path(__file__).resolve(), len(TASKS)
    )

    # Handle comparison mode
    if args.compare:
        if not args.compare_model1 or not args.compare_model2:
            raise ValueError("--compare requires --compare_model1 and --compare_model2")

        logger.info("Running in comparison mode")
        comparison_df = compare_benchmarks(
            args.compare_model1,
            args.compare_model2,
            output_dir=args.output_dir,
        )
        display_comparison(comparison_df)

        # Save the comparison
        safe_name1 = Path(args.compare_model1).name
        safe_name2 = Path(args.compare_model2).name
        output_path = (
            Path(args.output_dir) / f"comparison_{safe_name1}_vs_{safe_name2}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Comparison saved to: {output_path}")
        return

    # Handle evaluation mode (default)
    config = {
        "model": args.model_name,
        "probe_type": args.probe_type,
        "amp_dtype": args.amp_dtype,
        "attn_implementation": args.attn_implementation,
        "tasks": args.tasks,
        "max_samples": args.max_samples,
        "top_k_labels": args.top_k_labels,
        "output_dir": args.output_dir,
        "device": args.device,
        "fast": args.fast,
        "very_fast": args.very_fast,
        "cache_embeddings": args.cache_embeddings,
        "embed_cache_dir": args.embed_cache_dir,
        "proteingym": args.proteingym,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "compile": args.compile,
        "eval_split": args.eval_split,
        "knn_k": args.knn_k,
        "knn_weights": args.knn_weights,
        "l2_normalize_embeddings": args.l2_normalize_embeddings,
        "probe_embed_mode": args.probe_embed_mode,
    }

    # Device selection
    device = config["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {config['model']}")

    # Performance tweaks
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # SDPA best practices: allow all standard optimized backends explicitly
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    # Model weight dtype follows --amp_dtype.
    # Default fp32 is safer/reproducible; bf16 is opt-in for speed.
    # Keeping weight dtype aligned with compute mode avoids mixed-dtype attention issues.
    torch_dtype = None
    if (
        config.get("amp_dtype") == "bf16"
        and device == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        torch_dtype = torch.bfloat16
        logger.info("BF16 supported: loading model weights in bfloat16")
    else:
        logger.info("Loading model weights in float32 (default).")

    attn_override = (
        None
        if config.get("attn_implementation", "auto") == "auto"
        else config["attn_implementation"]
    )
    if attn_override is not None:
        logger.info("Forcing attention backend: %s", attn_override)
    else:
        logger.info("Attention backend: auto (FA2 if available, else SDPA)")

    # Load model with torch_dtype and optional attention backend override.
    model_obj, is_sbert, device = load_model(
        config["model"],
        device,
        torch_dtype,
        attn_implementation=attn_override,
    )

    if config.get("compile"):
        if is_sbert:
            logger.info(
                "Compile requested; skipping SentenceTransformer path in benchmark suite"
            )
        elif not hasattr(torch, "compile"):
            logger.warning("torch.compile is unavailable in this PyTorch build")
        else:
            tokenizer, hf_model = model_obj
            has_embed_dataset = hasattr(hf_model, "embed_dataset") and callable(
                hf_model.embed_dataset
            )
            if has_embed_dataset:
                logger.info(
                    "Compile requested; skipping custom embed_dataset inference path"
                )
            else:
                compile_kwargs, needs_unspec_int = get_torch_compile_settings(hf_model)
                if (
                    needs_unspec_int
                    and hasattr(torch, "_dynamo")
                    and hasattr(torch._dynamo, "config")
                ):
                    torch._dynamo.config.allow_unspec_int_on_nn_module = True
                    logger.info(
                        "Enabled torch._dynamo.config.allow_unspec_int_on_nn_module for HF ESM rotary caches"
                    )
                model_obj = (tokenizer, torch.compile(hf_model, **compile_kwargs))
                logger.info(
                    "Compiled HF benchmark model with backend=%s, dynamic=%s, mode=%s",
                    compile_kwargs.get("backend", "default"),
                    compile_kwargs.get("dynamic", False),
                    compile_kwargs.get("mode", "default"),
                )

    # Select tasks. --very-fast (curated low-variance subset) takes precedence
    # over --fast; explicit --tasks overrides both.
    if config.get("tasks"):
        task_keys = [t for t in config["tasks"] if t in TASKS]
        if len(task_keys) != len(config["tasks"]):
            missing = set(config["tasks"]) - set(task_keys)
            raise ValueError(f"Unknown tasks provided: {sorted(missing)}")
    elif config.get("very_fast"):
        task_keys = list(VERY_FAST_TASKS)
    elif config.get("fast"):
        task_keys = list(FAST_TASKS) + list(RETRIEVAL_TASKS)
    else:
        task_keys = list(DEFAULT_TASKS)

    # --proteingym adds all 8 ProteinGym tasks (deduplicating)
    if config.get("proteingym"):
        existing = set(task_keys)
        for t in PROTEINGYM_TASKS:
            if t not in existing:
                task_keys.append(t)

    logger.info(f"Tasks to evaluate ({len(task_keys)}): {task_keys}")

    # Sample cap: ONLY in --very-fast (scout) mode. Default/--fast and --no-fast
    # run full data (no truncation); --very-fast caps sequences + residue-level
    # token-classification tasks to stay within a few minutes for quick scouts.
    if config.get("very_fast"):
        max_samples = config.get("max_samples")
        if max_samples is None or max_samples > FAST_MAX_SAMPLES:
            max_samples = FAST_MAX_SAMPLES
        config["max_samples"] = max_samples
        logger.info(
            "Very-fast mode: tasks=%s, max_samples=%s (default/--fast run full data)",
            ",".join(task_keys),
            max_samples,
        )

    # AMP setup: fp32 embeddings by default for reproducibility and numerical stability
    # (model weights still use bf16 if available, but embeddings computed in full precision)
    amp_dtype = None
    if (
        config.get("amp_dtype") == "bf16"
        and device == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.bfloat16
        logger.info("Using bfloat16 autocast for embeddings (advanced mode).")
    else:
        logger.info(
            "Using float32 for embedding computations (default: maximum reproducibility)."
        )

    # Embedding cache path (model-specific, or None to disable caching)
    embed_save_path = None
    if args.clear_cache:
        removed_dirs = _clear_model_cache_dirs(
            config.get("embed_cache_dir", "embed_cache"),
            config["model"],
        )
        logger.info(
            "Cleared %d cache director%s for model %s",
            removed_dirs,
            "y" if removed_dirs == 1 else "ies",
            config["model"],
        )
    if config.get("cache_embeddings"):
        cache_namespace = _model_cache_namespace(config["model"])
        embed_save_path = os.path.join(
            config.get("embed_cache_dir", "embed_cache"),
            cache_namespace,
            "embeddings.pth",
        )
        logger.info(f"Embedding cache enabled: {embed_save_path}")
    else:
        logger.info("Embedding cache disabled (use --cache_embeddings to enable).")

    # Run evaluations
    tracker = ResultTracker(config["model"])

    tta_cfg = None
    if getattr(args, "tta", False):
        from wt_test_time_training import TTAConfig

        tta_cfg = TTAConfig(
            iters=args.tta_iters,
            lr=args.tta_lr,
            n_layers=args.tta_layers,
            mask_rate=args.tta_mask_rate,
            train_head=args.tta_train_head,
            seed=args.seed,
        )
        logger.info(
            "WT test-time training ENABLED (zero-shot tasks only): %s", tta_cfg
        )

    total_runs = len(task_keys) * len(benchmark_seeds)
    completed_runs = 0
    for seed_index, benchmark_seed in enumerate(benchmark_seeds, start=1):
        BENCHMARK_SEED = benchmark_seed
        seed_all(BENCHMARK_SEED)
        logger.info(
            "Benchmark seed %d/%d: %s",
            seed_index,
            len(benchmark_seeds),
            benchmark_seed,
        )
        for key in task_keys:
            completed_runs += 1
            cfg = TASKS[key]
            requested_probe = config["probe_type"]
            effective_probe = effective_probe_type(cfg, requested_probe)
            probe_variant_label = _make_probe_variant_label(
                effective_probe,
                l2_normalize_embeddings=config.get("l2_normalize_embeddings", False),
                knn_weights=config.get("knn_weights", "uniform"),
            )
            probe_display = probe_label(effective_probe)
            if effective_probe != requested_probe:
                probe_display = f"{probe_display} (requested {probe_label(requested_probe)} ignored)"
            try:
                print(f"\n{'=' * 60}")
                print(
                    f"[{completed_runs}/{total_runs}] [seed={benchmark_seed}] "
                    f"[{key}] {cfg.name} [{probe_display}]"
                )
                print(f"{'=' * 60}")

                _raw_max_samples = config.get("max_samples")
                # Token-classification tasks run residue-level logistic
                # regression; cap sequences tighter than FAST_MAX_SAMPLES to
                # stay within a few minutes on CPU (~400k residues at 2k seqs).
                # Contact prediction is capped by --contact_train_proteins
                # instead; it embeds every sampled protein and then expands each
                # into O(L^2) pairs, so a sequence-count cap is the wrong knob.
                _task_max_samples = (
                    FAST_TOKEN_CLASS_MAX_SAMPLES
                    if (
                        cfg.problem_type
                        in ("token_classification", "contact_prediction")
                        and _raw_max_samples is not None
                        and _raw_max_samples > FAST_TOKEN_CLASS_MAX_SAMPLES
                    )
                    else _raw_max_samples
                )
                metrics, resolved_eval_split, eval_strategy = evaluate_task(
                    cfg,
                    model_obj,
                    is_sbert,
                    device,
                    _task_max_samples,
                    config.get("top_k_labels"),
                    amp_dtype,
                    embed_save_path=embed_save_path,
                    batch_size=config.get("batch_size", 128),
                    max_length=config.get("max_length", DEFAULT_EMBED_MAX_LENGTH),
                    probe_type=requested_probe,
                    eval_split=config.get("eval_split", DEFAULT_BENCHMARK_EVAL_SPLIT),
                    knn_k=config.get("knn_k", 3),
                    knn_weights=config.get("knn_weights", "uniform"),
                    l2_normalize_embeddings=config.get(
                        "l2_normalize_embeddings", False
                    ),
                    tta_cfg=tta_cfg,
                    probe_embed_mode=config.get("probe_embed_mode", "trunk"),
                )

                main_val = metrics.get(cfg.main_metric, None)
                if main_val is not None:
                    rounded_main = str(
                        round(float(main_val), ResultTracker.round_decimals)
                    )
                    print(f"  >> {cfg.main_metric}: {rounded_main}")
                else:
                    print(f"  >> Results: {metrics}")

                tracker.add(
                    cfg.name,
                    metrics,
                    config.get("max_samples"),
                    probe=probe_variant_label,
                    eval_mode=_result_eval_mode(cfg),
                    eval_split=resolved_eval_split,
                    eval_strategy=eval_strategy,
                    benchmark_seed=benchmark_seed,
                    embedding_norm=(
                        "l2" if config.get("l2_normalize_embeddings", False) else "none"
                    ),
                )

            except Exception as e:
                logger.error(
                    f"Task '{key}' failed for benchmark seed {benchmark_seed}: {e}"
                )
                traceback.print_exc()
                tracker.add(
                    cfg.name,
                    {"Error": str(e)},
                    config.get("max_samples"),
                    probe=probe_variant_label,
                    eval_mode=_result_eval_mode(cfg),
                    eval_split=config.get("eval_split", DEFAULT_BENCHMARK_EVAL_SPLIT),
                    eval_strategy="task_exception",
                    benchmark_seed=benchmark_seed,
                    embedding_norm=(
                        "l2" if config.get("l2_normalize_embeddings", False) else "none"
                    ),
                )

            # GPU memory cleanup between task evaluations
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

    # Display and save results
    tracker.display()

    os.makedirs(config.get("output_dir", "."), exist_ok=True)
    tracker.save(config.get("output_dir", "."))


# =============================================================================
# Execution
# =============================================================================

if __name__ == "__main__":
    main()
