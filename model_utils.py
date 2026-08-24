"""
Shared utilities for non-standard protein language model support.

Handles model-type detection, compatibility patches, and wrapper modules
for models that don't follow vanilla HuggingFace conventions:
  - AMPLIFY (chandar-lab): additive attention mask, xformers stride alignment
  - ESMplusplus (Synthyra): custom tokenizer, SentenceTransformer assembly
  - FastPLM ESM2 (Synthyra): bug-fixed ESM2 re-implementation
  - DPLM2 (Synthyra): protein diffusion language model
  - Profluent-E1 (Synthyra): retrieval-augmented protein encoder

Used by protein_benchmark_suite.py and the fine-tuning scripts (benchmarking).
"""

import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - optional dependency during import
    safe_open = None

try:
    import transformers

    _TRANSFORMERS_MAJOR = int(transformers.__version__.split(".", 1)[0])
except Exception:
    _TRANSFORMERS_MAJOR = 5

logger = logging.getLogger(__name__)

ModelType = Literal[
    "amplify",
    "esmplusplus",
    "fastplm_esm2",
    "dplm2",
    "profluent_e1",
    "proteva",
    "standard",
]

# Flash attention availability (cached at import time)
try:
    HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None
except Exception:
    HAS_FLASH_ATTN = False

# =============================================================================
# Model type detection
# =============================================================================


def detect_model_type(model_name: str) -> ModelType:
    """Detect model family from name/path. Checks HF name and local config.json."""
    name_lower = model_name.lower()

    if "amplify" in name_lower:
        return "amplify"

    if "esmplusplus" in name_lower or "esm++" in name_lower or "esm-c" in name_lower:
        return "esmplusplus"

    # Synthyra FastPLM ESM2 re-implementation (e.g. Synthyra/ESM2-150M)
    # Matches: "Synthyra/ESM2-*", names containing "fastplm"
    if "synthyra/esm2" in name_lower or "fastplm" in name_lower:
        return "fastplm_esm2"

    # Synthyra DPLM2 (e.g. Synthyra/DPLM2-150M, Synthyra/DPLM2-650M)
    if "dplm2" in name_lower:
        return "dplm2"

    # Synthyra Profluent-E1 (e.g. Synthyra/Profluent-E1-150M)
    if ("profluent" in name_lower and "synthyra" in name_lower) or "e1-" in name_lower:
        return "profluent_e1"

    # Proteva (this project's HF-native encoder; loaded fa2-varlen + BF16)
    if "proteva" in name_lower:
        return "proteva"

    # Check local config.json for non-obvious model paths
    config_path = Path(model_name) / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("model_type") == "AMPLIFY":
                return "amplify"
            if cfg.get("model_type") == "proteva":
                return "proteva"
            archs = cfg.get("architectures", [])
            auto_map = cfg.get("auto_map", {})
            all_vals = [*archs, *auto_map.values()]
            if any("ESMplusplus" in str(v) for v in all_vals):
                return "esmplusplus"
            # FastPLM ESM2: custom architectures from Synthyra ESM2 repos
            if any("FastESM" in str(v) or "FastEsmModel" in str(v) for v in all_vals):
                return "fastplm_esm2"
            # DPLM2: check for DPLM2 architectures
            if any("DPLM2" in str(v) or "Dplm2" in str(v) for v in all_vals):
                return "dplm2"
            # Profluent-E1: check for E1 architectures
            if any("E1" in str(v) or "ProfluentE1" in str(v) for v in all_vals):
                return "profluent_e1"
            # Proteva: ProtevaForPretraining architecture
            if any("Proteva" in str(v) for v in all_vals):
                return "proteva"
        except Exception:
            pass

    return "standard"


# =============================================================================
# Transformers compatibility patch (ESMplusplus)
# =============================================================================


def apply_esmplusplus_compat_patch():
    """Patch transformers to handle ESMplusplus models lacking 'all_tied_weights_keys'.

    Also ensures 'entrypoint_setup' (a side-effect config module shipped with the
    ESMplusplus Hub repo) is importable from the HF modules cache.  transformers'
    check_imports() treats any top-level import as a PyPI dependency; this module
    doesn't exist on PyPI so we copy it into the modules cache on demand.

    Safe to call multiple times — applies only once via a sentinel attribute.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel

        if getattr(PreTrainedModel, "_esmplusplus_patch_applied", False):
            return
        if hasattr(PreTrainedModel, "mark_tied_weights_as_initialized"):
            _orig = PreTrainedModel.mark_tied_weights_as_initialized

            def _patched(self, *args, **kwargs):
                if not hasattr(self, "all_tied_weights_keys"):
                    self.all_tied_weights_keys = []
                return _orig(self, *args, **kwargs)

            PreTrainedModel.mark_tied_weights_as_initialized = _patched
        PreTrainedModel._esmplusplus_patch_applied = True
    except ImportError:
        pass

    # ── entrypoint_setup: make importable ─────────────────────────────────────
    # Older ESMplusplus Hub code did `import entrypoint_setup` at the top level.
    # We search ALL Synthyra snapshot dirs (not just ESMplusplus_small) and add
    # them to sys.path so the import resolves.  Best-effort; no-op if not found.
    try:
        import sys
        from pathlib import Path as _P

        hub_dir = _P.home() / ".cache" / "huggingface" / "hub"
        if hub_dir.exists():
            for model_dir in hub_dir.iterdir():
                if not model_dir.name.startswith("models--Synthyra"):
                    continue
                snap_base = model_dir / "snapshots"
                if not snap_base.exists():
                    continue
                for snap_dir in snap_base.iterdir():
                    if (snap_dir / "entrypoint_setup.py").exists():
                        snap_str = str(snap_dir)
                        if snap_str not in sys.path:
                            sys.path.insert(0, snap_str)
                            logger.debug(
                                "Added Synthyra snapshot to sys.path: %s", snap_str
                            )
    except Exception:
        pass  # best-effort; failure handled at load time


def apply_esm_rotary_autograd_patch() -> None:
    """Patch ESM RotaryEmbedding.forward to sanitize inference-mode caches.

    During some SentenceTransformer cached-loss code paths, ESM rotary caches
    (``_cos_cached``/``_sin_cached``) can be refreshed under inference mode,
    then reused in a grad-enabled pass, triggering:
    ``RuntimeError: Inference tensors cannot be saved for backward``.

    This patch is idempotent and safe: in grad-enabled forwards, if a cached
    rotary tensor is inference-mode, it is cloned back to a normal tensor before
    continuing.
    """
    try:
        from transformers.models.esm.modeling_esm import RotaryEmbedding
    except Exception:
        return

    if getattr(RotaryEmbedding, "_rotary_autograd_patch_applied", False):
        return

    original_forward = RotaryEmbedding.forward

    def _patched_forward(self, q, k):
        if torch.is_grad_enabled():
            for attr in ("_cos_cached", "_sin_cached"):
                tensor = getattr(self, attr, None)
                if not isinstance(tensor, torch.Tensor):
                    continue
                is_inference = bool(
                    hasattr(tensor, "is_inference") and tensor.is_inference()
                )
                if is_inference:
                    setattr(self, attr, tensor.clone())
        return original_forward(self, q, k)

    RotaryEmbedding.forward = _patched_forward
    RotaryEmbedding._rotary_autograd_patch_applied = True


# =============================================================================
# Shared model-loading helpers
# =============================================================================


def from_pretrained_with_flash(model_cls, model_name: str, **extra_kwargs):
    """Load a model with flash attention if available, falling back to default.

    Args:
        model_cls: Model class to instantiate (AutoModel, AutoModelForMaskedLM, etc.)
        model_name: Model name or path
        **extra_kwargs: Additional kwargs to pass to from_pretrained (for example
            dtype or explicit ``attn_implementation``).

    Attention backend selection order:
        1. ``attn_implementation`` passed in ``extra_kwargs``
        2. ``PROTEIN_BENCH_ATTN_IMPLEMENTATION`` environment variable
        3. Auto-select (``flash_attention_2`` if available, otherwise ``sdpa``)
    """
    # transformers >=5.3 prefers dtype; transformers 4.x expects torch_dtype.
    if _TRANSFORMERS_MAJOR >= 5:
        if "torch_dtype" in extra_kwargs and "dtype" not in extra_kwargs:
            extra_kwargs["dtype"] = extra_kwargs.pop("torch_dtype")
    else:
        if "dtype" in extra_kwargs and "torch_dtype" not in extra_kwargs:
            extra_kwargs["torch_dtype"] = extra_kwargs.pop("dtype")
    extra_kwargs = {
        k: v
        for k, v in extra_kwargs.items()
        if v is not None or k not in ("dtype", "torch_dtype")
    }

    kwargs = {"trust_remote_code": True, **extra_kwargs}

    requested_attn = kwargs.pop("attn_implementation", None)
    env_attn = os.environ.get("PROTEIN_BENCH_ATTN_IMPLEMENTATION", "").strip()
    allowed_attn = {"flash_attention_2", "sdpa", "eager", "auto", ""}

    if requested_attn is not None and requested_attn not in {
        "flash_attention_2",
        "sdpa",
        "eager",
    }:
        raise ValueError(
            "Unsupported attn_implementation=%r. Expected one of "
            "{'flash_attention_2','sdpa','eager'}." % (requested_attn,)
        )
    if env_attn not in allowed_attn:
        raise ValueError(
            "Unsupported PROTEIN_BENCH_ATTN_IMPLEMENTATION=%r. Expected one of "
            "{'auto','flash_attention_2','sdpa','eager'} or empty." % (env_attn,)
        )

    selected_attn: str
    if requested_attn is not None:
        selected_attn = requested_attn
    elif env_attn and env_attn != "auto":
        selected_attn = env_attn
    elif HAS_FLASH_ATTN:
        # Prefer FlashAttention-2 when available unless explicitly overridden.
        selected_attn = "flash_attention_2"
    else:
        # Stable fallback that avoids custom eager-mode mask code paths.
        selected_attn = "sdpa"

    kwargs["attn_implementation"] = selected_attn

    # Auto-fix torch.compile checkpoints. A model saved while torch.compile-wrapped
    # has every state-dict key prefixed with '_orig_mod.', so from_pretrained matches
    # NOTHING and SILENTLY returns a randomly-initialized model (garbage benchmarks).
    # Strip the prefix in-place for LOCAL checkpoint dirs; idempotent no-op when the
    # prefix is absent. Hub ids (chandar-lab/AMPLIFY_120M, Synthyra/ESMplusplus_small
    # / ESM-C-600M, EvolutionaryScale/*, ...) are not local dirs -> skipped, so this
    # is safe for every model type. Only rewrites single-file model.safetensors
    # (sharded checkpoints are left untouched).
    if os.path.isdir(model_name):
        try:
            from plm.hf.checkpoint_utils import strip_orig_mod_prefix

            n_stripped = strip_orig_mod_prefix(model_name)
            if n_stripped:
                logger.info(
                    "Stripped torch.compile '_orig_mod.' prefix from %d keys in %s",
                    n_stripped, model_name,
                )
        except Exception as exc:  # never block loading on the auto-fix
            logger.warning("torch.compile prefix auto-strip skipped for %s: %s", model_name, exc)

    def _load_model(current_kwargs):
        loaded_model = model_cls.from_pretrained(model_name, **current_kwargs)
        _validate_local_checkpoint_integrity(model_name, loaded_model)
        return loaded_model

    try:
        return _load_model(kwargs)
    except (TypeError, ValueError):
        if "attn_implementation" in kwargs:
            kwargs.pop("attn_implementation", None)
            return _load_model(kwargs)
        raise


_WRAPPER_PREFIXES = (
    "student.",
    "teacher.",
    # torch.compile prefix (subsumes the JEPA _orig_mod.student./.teacher. forms).
    # MUST stay listed: from_pretrained matches nothing against it and silently
    # returns a RANDOM model, which benchmarks as plausible garbage (AUC 0.500,
    # F1 0.000) instead of crashing. The pre-load auto-strip is best-effort — a
    # no-op on sharded checkpoints, and its failures are swallowed — so this gate
    # is the real guarantee. Incident: v6-rtd re-bench 2026-07-21, ~3.5 GPU-h.
    "_orig_mod.",
)


def _assert_no_wrapper_prefixes(model_path: str) -> None:
    """Raise if a local safetensors file contains JEPA wrapper-prefixed keys.

    Intended as a pre-load guard for code paths that bypass
    ``_validate_local_checkpoint_integrity`` (e.g. SentenceTransformer loads).

    Args:
        model_path: Path to the model directory to inspect.

    Raises:
        RuntimeError: If wrapper-prefixed keys are found.
    """
    if safe_open is None:
        return
    safetensor_path = Path(model_path) / "model.safetensors"
    if not safetensor_path.is_file():
        return
    try:
        with safe_open(str(safetensor_path), framework="pt") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        logger.warning("Pre-load prefix scan skipped for %s: %s", model_path, exc)
        return
    bad = [k for k in keys if any(k.startswith(p) for p in _WRAPPER_PREFIXES)]
    if bad:
        raise RuntimeError(
            "Checkpoint appears unpatched (wrapper-prefixed keys found): "
            f"{model_path}. Use export patching before benchmarking. "
            f"Example bad key: {bad[0]!r}"
        )


def _validate_local_checkpoint_integrity(model_name: str, loaded_model) -> None:
    """Validate local checkpoint key integrity and emit actionable warnings.

    This catches common export mistakes (for example leftover ``student.`` keys)
    before benchmark runs silently proceed with mismatched parameters.
    """

    model_path = Path(model_name)
    if not model_path.is_dir():
        return

    safetensor_path = model_path / "model.safetensors"
    if not safetensor_path.is_file() or safe_open is None:
        return

    try:
        with safe_open(str(safetensor_path), framework="pt") as handle:
            checkpoint_keys = list(handle.keys())
    except Exception as exc:
        logger.warning("Checkpoint key validation skipped for %s: %s", model_name, exc)
        return

    wrapper_keys = [
        k for k in checkpoint_keys if any(k.startswith(p) for p in _WRAPPER_PREFIXES)
    ]
    if wrapper_keys:
        raise RuntimeError(
            "Checkpoint appears unpatched (wrapper-prefixed keys found): "
            f"{model_name}. Use export patching before benchmarking. "
            f"Example bad key: {wrapper_keys[0]!r}"
        )

    try:
        model_keys = set(loaded_model.state_dict().keys())
    except Exception as exc:
        logger.warning(
            "Checkpoint/model key overlap validation skipped for %s: %s",
            model_name,
            exc,
        )
        return

    if not model_keys:
        return

    checkpoint_key_set = set(checkpoint_keys)
    overlap_count = len(model_keys & checkpoint_key_set)
    overlap_ratio = overlap_count / float(len(model_keys))

    if overlap_ratio < 0.5:
        raise RuntimeError(
            "Low checkpoint/model key overlap for %s (%.1f%%, %d/%d keys matched). "
            "This indicates architecture mismatch or an incorrect export. "
            "Use export patching before benchmarking."
            % (model_name, overlap_ratio * 100.0, overlap_count, len(model_keys))
        )


def disable_esm2_token_dropout(model) -> bool:
    """Disable ESM2 token_dropout to work around HuggingFace transformers bug.

    HuggingFace transformers >=5.x broke ESM2's token_dropout: the attention_mask
    is no longer passed to the embeddings layer, causing incorrect scaling of
    embeddings during both training and inference.  See:
    https://github.com/huggingface/transformers/issues/44162

    This sets config.token_dropout = False on the model (and any wrapped inner
    model) so the broken code path is never entered.  Safe to call on any model;
    returns True if a fix was applied.
    """
    fixed = False
    if not needs_esm2_token_dropout_workaround(model):
        return False

    for obj in _iter_model_wrappers(model):
        if obj is None:
            continue
        cfg = getattr(obj, "config", None)
        if (
            cfg is not None
            and getattr(cfg, "model_type", None) == "esm"
            and getattr(cfg, "token_dropout", False)
        ):
            cfg.token_dropout = False
            fixed = True
            logger.info(
                "   Disabled ESM2 token_dropout (HF transformers bug workaround)"
            )
    return fixed


def _iter_model_wrappers(model):
    """Yield model plus common wrapped inner models exactly once."""
    stack = [model]
    seen: set[int] = set()

    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield obj

        for attr in ("model", "auto_model"):
            inner = getattr(obj, attr, None)
            if inner is not None and inner is not obj:
                stack.append(inner)


from string import ascii_uppercase


class _FallbackVocab(dict):
    """Vocabulary dict that returns a fallback id for any unknown token.

    Defined at module level, with ``__reduce__``, so it survives pickling into
    spawned dataloader workers.

    ``__missing__`` fires only on ``[]`` lookup, so ``in``, ``len()`` and
    ``get_vocab()`` still report the true vocabulary -- nothing downstream sees
    a larger vocab than the embedding matrix has rows.
    """

    def __init__(self, mapping, fallback_id):
        super().__init__(mapping)
        self.fallback_id = fallback_id

    def __missing__(self, key):
        return self.fallback_id

    def __reduce__(self):
        return (_FallbackVocab, (dict(self), self.fallback_id))


def patch_unknown_residue_tokens(tokenizer):
    """Map any token missing from the vocabulary onto 'X' instead of raising.

    FastPLM's ``_convert_token_to_id``
    (``fastplms/models/esm2/modeling_fastesm.py:198``) raises ``KeyError`` for an
    out-of-vocabulary token rather than falling back to ``unk_token``. Stock
    ``EsmTokenizer`` does not -- so this only bites checkpoints saved with
    FastPLM's tokenizer identity.

    Two families of offender show up in practice:

    * **residue codes**: 'J' (IUPAC ambiguity code for Leu/Ile) is the one letter
      absent from the ESM2 vocabulary. A single 'J' killed a dataloader worker at
      step 134 of a 4,244-step run, which took down the rank and then the DDP job.
    * **non-residue characters** carried in benchmark sequence fields: '|' in
      Peptide-HLA and '#' in Thermostability (FLIP) each errored a whole task in
      the ESM-2 35M benchmark arm.

    Enumerating A-Z covers the first family only, so the fallback is installed for
    every unknown key instead.

    'X' is used rather than ``<unk>`` deliberately: ESM2 was pretrained with 'X'
    as the unknown-residue symbol and has essentially never seen ``<unk>`` in a
    sequence, so 'X' keeps the input inside the pretraining distribution.

    Idempotent.
    """
    table = getattr(tokenizer, "_token_to_id", None)
    if not isinstance(table, dict) or isinstance(table, _FallbackVocab):
        return
    fallback_id = table.get("X", tokenizer.unk_token_id)
    if fallback_id is None:
        return
    tokenizer._token_to_id = _FallbackVocab(table, fallback_id)
    logger.info(
        "Tokenizer will map out-of-vocabulary tokens (e.g. %s) to 'X' (id %d)",
        ", ".join(repr(c) for c in ascii_uppercase if c not in table) or "'|', '#'",
        fallback_id,
    )


def is_fastplm_runtime_model(model) -> bool:
    """Return True when a loaded model originates from Synthyra FastPLM code."""
    for obj in _iter_model_wrappers(model):
        cls = obj.__class__
        module_name = getattr(cls, "__module__", "").lower()
        class_name = getattr(cls, "__name__", "").lower()
        if "modeling_fastesm" in module_name or "fastesm" in class_name:
            return True
    return False


def needs_esm2_token_dropout_workaround(model) -> bool:
    """Return True only for the stock HF ESM path affected by the token_dropout bug."""
    if is_fastplm_runtime_model(model):
        return False

    return any(
        getattr(getattr(obj, "config", None), "model_type", None) == "esm"
        and bool(getattr(getattr(obj, "config", None), "token_dropout", False))
        for obj in _iter_model_wrappers(model)
    )


def uses_hf_esm_rotary_embeddings(model) -> bool:
    """Return True when the model contains HF ESM RotaryEmbedding modules.

    This is capability-based rather than family-name-based, so it also catches
    wrappers/reimplementations that directly reuse HF's RotaryEmbedding class.
    """
    for obj in _iter_model_wrappers(model):
        if not isinstance(obj, nn.Module):
            continue
        for module in obj.modules():
            cls = module.__class__
            if (
                getattr(cls, "__name__", "") == "RotaryEmbedding"
                and getattr(cls, "__module__", "")
                == "transformers.models.esm.modeling_esm"
                and hasattr(module, "_seq_len_cached")
            ):
                return True
    return False


def get_torch_compile_settings(model) -> tuple[dict[str, object], bool]:
    """Return model-aware torch.compile kwargs and whether unspec-int should be enabled."""
    raw_dynamic = os.environ.get("PROTEIN_COMPILE_DYNAMIC", "0").strip().lower()
    dynamic = raw_dynamic not in {"0", "false", "no", "off"}
    backend = os.environ.get("PROTEIN_COMPILE_BACKEND", "inductor").strip()
    mode = os.environ.get("PROTEIN_COMPILE_MODE", "default").strip()

    compile_kwargs: dict[str, object] = {"dynamic": dynamic}
    if backend:
        compile_kwargs["backend"] = backend
    if mode and mode != "default":
        compile_kwargs["mode"] = mode

    return compile_kwargs, uses_hf_esm_rotary_embeddings(model)


# =============================================================================
# AMPLIFY helpers
# =============================================================================


def fix_amplify_meta_tensors(model):
    """Recompute freqs_cis if stuck on meta device (happens with from_pretrained)."""
    if hasattr(model, "freqs_cis") and model.freqs_cis.is_meta:
        mod = importlib.import_module(model.__class__.__module__)
        model.freqs_cis = mod.precompute_freqs_cis(
            model.config.hidden_size // model.config.num_attention_heads,
            model.config.max_length,
        )


def fix_proteva_rope_buffer(model):
    """Recompute Proteva's ``encoder.rope_cs`` RoPE cache after ``from_pretrained``.

    ``ProteinEncoder`` registers ``rope_cs`` (the precomputed cos/sin RoPE cache)
    as a NON-persistent buffer (``persistent=False``), so it is intentionally
    excluded from the safetensors checkpoint. HF's ``from_pretrained`` materializes
    the model on a meta device and copies tensors from the state dict; because the
    buffer is absent from the checkpoint, it is left as the meta/empty allocation
    and is NEVER re-run through ``__init__``'s ``_precompute_rope``. The result is a
    garbage ``rope_cs`` (observed: all-zeros, or non-deterministic memory junk that
    differs run-to-run) — i.e. RoPE is silently DISABLED for the benched model.

    This is the exact analogue of the AMPLIFY ``freqs_cis`` meta-tensor problem that
    :func:`fix_amplify_meta_tensors` repairs. Without this fix, benched Proteva
    embeddings carry no positional information and the linear-probe scores sit a
    constant ~0.03–0.32 below the same AMPLIFY weights benched natively.

    Recomputes the buffer from the encoder config (matching ``ProteinEncoder.__init__``)
    and writes it back on the buffer's existing device/dtype. No-op for non-Proteva
    models or when the buffer is already valid.
    """
    enc = getattr(model, "encoder", None)
    if enc is None or not hasattr(enc, "rope_cs"):
        return
    buf = enc.rope_cs
    # Reference cos at position 1, pair 0 must be cos(1 * inv_freq[0]) = cos(1) ≈ 0.5403.
    # A meta/zero/garbage buffer fails this; a correct buffer always has cos[0]==1.0.
    pos0_ok = bool(torch.allclose(buf[0, :, 0].float(), torch.ones_like(buf[0, :, 0].float()), atol=1e-3))
    pos0_sin_ok = bool(torch.allclose(buf[0, :, 1].float(), torch.zeros_like(buf[0, :, 1].float()), atol=1e-3))
    if pos0_ok and pos0_sin_ok and not bool(torch.isnan(buf).any()):
        return  # already a valid RoPE cache
    from plm.model import _precompute_rope, _precompute_rope_multi_scale

    ecfg = enc.config
    if getattr(ecfg, "multi_scale_rope", False):
        fresh = _precompute_rope_multi_scale(
            ecfg.head_dim,
            ecfg.max_position,
            thetas=(1_000.0, 10_000.0, 100_000.0),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    else:
        fresh = _precompute_rope(
            ecfg.head_dim,
            ecfg.max_position,
            ecfg.rope_theta,
            device=torch.device("cpu"),
            dtype=torch.float32,
            rope_frac=ecfg.rope_frac,
        )
    enc.rope_cs = fresh.to(device=buf.device, dtype=buf.dtype)
    logger.info("   Recomputed Proteva encoder.rope_cs (was uninitialized after from_pretrained)")


_AMPLIFY_PATCHED_MODULES: set[str] = set()


def patch_amplify_attention_fallback(model) -> None:
    """Patch AMPLIFY remote-code xformers call to fall back to PyTorch SDPA.

    AMPLIFY imports ``memory_efficient_attention`` from xformers at module-import
    time and calls it unconditionally.  When xformers is present but its CUDA
    kernels are not compatible with the running PyTorch/CUDA build the forward
    pass raises ``NotImplementedError``.  This monkey-patches the module-level
    symbol with a wrapper that tries xformers first and silently falls back to
    ``F.scaled_dot_product_attention``.

    Set ``PROTJEPA_AMPLIFY_FORCE_SDPA=1`` to skip xformers entirely.

    Safe to call multiple times – subsequent calls on already-patched modules
    are no-ops.
    """
    module_name = model.__class__.__module__
    if module_name in _AMPLIFY_PATCHED_MODULES:
        return

    mod = importlib.import_module(module_name)
    original_fn = getattr(mod, "memory_efficient_attention", None)
    if original_fn is None:
        return  # not an xformers-backed AMPLIFY module

    if getattr(original_fn, "_amplify_sdpa_patched", False):
        _AMPLIFY_PATCHED_MODULES.add(module_name)
        return  # already patched by another code path

    force_sdpa = os.environ.get("PROTJEPA_AMPLIFY_FORCE_SDPA", "0") == "1"

    def _sdpa_fallback(query, key, value, attn_bias=None, p=0.0, **kw):
        # xformers layout [B, S, H, D] → torch SDPA layout [B, H, S, D]
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        if attn_bias is not None:
            while attn_bias.dim() < 4:
                attn_bias = attn_bias.unsqueeze(1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=p if torch.is_grad_enabled() else 0.0,
        )
        return out.permute(0, 2, 1, 3)

    def _patched(query, key, value, attn_bias=None, p=0.0, **kw):
        if force_sdpa:
            return _sdpa_fallback(query, key, value, attn_bias, p)
        try:
            return original_fn(
                query=query, key=key, value=value, attn_bias=attn_bias, p=p, **kw
            )
        except (NotImplementedError, RuntimeError):
            return _sdpa_fallback(query, key, value, attn_bias, p)

    _patched._amplify_sdpa_patched = True
    setattr(mod, "memory_efficient_attention", _patched)
    _AMPLIFY_PATCHED_MODULES.add(module_name)
    logger.info("   Applied AMPLIFY xformers→SDPA fallback patch")


def _pad_to_multiple(tensor: torch.Tensor, multiple: int, value=0):
    """Pad last dimension to a multiple of `multiple`. Returns (padded, pad_len)."""
    remainder = tensor.shape[-1] % multiple
    if remainder == 0:
        return tensor, 0
    pad_len = multiple - remainder
    return nn.functional.pad(tensor, (0, pad_len), value=value), pad_len


def _prepare_amplify_inputs(input_ids, attention_mask, device=None, dtype=None):
    """Pad inputs and build additive mask for AMPLIFY/xformers.

    Returns (input_ids, additive_mask, orig_len, pad_len).
    The caller should slice hidden states back to [:, :orig_len, :] after inference.
    """
    orig_len = input_ids.shape[1]

    # xformers cutlass requires stride(-2) % 4 == 0 → pad to multiple of 8
    input_ids, pad_len = _pad_to_multiple(input_ids, 8, value=0)
    if attention_mask is not None:
        attention_mask, _ = _pad_to_multiple(attention_mask, 8, value=0)
    else:
        attention_mask = torch.ones(
            input_ids.shape[0], orig_len, device=input_ids.device
        )
        attention_mask, _ = _pad_to_multiple(attention_mask, 8, value=0)

    # Additive mask: 0.0 for real tokens, -inf for padding
    additive_mask = torch.where(attention_mask.bool(), 0.0, float("-inf"))

    # Match compute dtype (bf16 under autocast, or model param dtype)
    if torch.is_autocast_enabled():
        additive_mask = additive_mask.to(torch.get_autocast_dtype("cuda"))
    elif dtype is not None:
        additive_mask = additive_mask.to(dtype)
    if device is not None:
        additive_mask = additive_mask.to(device)

    return input_ids, additive_mask, orig_len, pad_len
