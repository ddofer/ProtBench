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

try:
    # Probe the symbol, not the spec: flash_attn_4 ships a namespace `flash_attn/`
    # tree, so find_spec reports True while the varlen import still raises.
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func  # noqa: F401

    HAS_FLASH_ATTN = True
except Exception:
    HAS_FLASH_ATTN = False

# Model type detection


def detect_model_type(model_name: str) -> ModelType:
    """Detect model family, treating a local ``config.json`` as authoritative."""
    name_lower = model_name.lower()

    # Config beats directory name: ``amplifyc_PTM_*`` (Proteva warm-started from
    # AMPLIFY-C) must not be routed through the stock-AMPLIFY loader.
    config_path = Path(model_name) / "config.json"
    if config_path.exists():
        try:
            with config_path.open() as handle:
                cfg = json.load(handle)
            model_type = str(cfg.get("model_type", "")).lower()
            if model_type == "amplify":
                return "amplify"
            if model_type == "proteva":
                return "proteva"
            archs = cfg.get("architectures", [])
            auto_map = cfg.get("auto_map", {})
            all_vals = [*archs, *auto_map.values()]
            if any("ESMplusplus" in str(value) for value in all_vals):
                return "esmplusplus"
            if any(
                "FastESM" in str(value) or "FastEsmModel" in str(value)
                for value in all_vals
            ):
                return "fastplm_esm2"
            if any("DPLM2" in str(value) or "Dplm2" in str(value) for value in all_vals):
                return "dplm2"
            if any("E1" in str(value) or "ProfluentE1" in str(value) for value in all_vals):
                return "profluent_e1"
            if any("Proteva" in str(value) for value in all_vals):
                return "proteva"
            return "standard"
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    if "amplify" in name_lower:
        return "amplify"

    if "esmplusplus" in name_lower or "esm++" in name_lower or "esm-c" in name_lower:
        return "esmplusplus"

    if "synthyra/esm2" in name_lower or "fastplm" in name_lower:
        return "fastplm_esm2"

    if "dplm2" in name_lower:
        return "dplm2"

    if ("profluent" in name_lower and "synthyra" in name_lower) or "e1-" in name_lower:
        return "profluent_e1"

    if "proteva" in name_lower:
        return "proteva"

    return "standard"


# Transformers compatibility patch (ESMplusplus)


def apply_esmplusplus_compat_patch():
    """Patch transformers to handle ESMplusplus models lacking 'all_tied_weights_keys'.

    Also puts Synthyra Hub snapshot dirs on sys.path so the non-PyPI
    ``entrypoint_setup`` module they import resolves. Idempotent.
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

    SentenceTransformer cached-loss paths can refresh ``_cos_cached``/``_sin_cached``
    under inference mode and reuse them with grad enabled ("Inference tensors cannot
    be saved for backward"); grad-enabled forwards clone them back. Idempotent.
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


# Shared model-loading helpers


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
        selected_attn = "flash_attention_2"
    else:
        selected_attn = "sdpa"

    kwargs["attn_implementation"] = selected_attn

    # torch.compile checkpoints carry '_orig_mod.' keys that from_pretrained matches
    # NOTHING against and SILENTLY loads random weights; strip in place (local dirs only).
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
    # MUST stay listed: from_pretrained silently returns a RANDOM model on this prefix
    # (benchmarks as AUC 0.500, not a crash); the pre-load strip is best-effort only.
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
    """Raise on wrapper-prefixed keys or <50% key overlap in a local checkpoint."""

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
    """Disable ESM2 token_dropout to work around a transformers >=5 bug.

    The attention_mask no longer reaches the embeddings layer, so they are mis-scaled
    (https://github.com/huggingface/transformers/issues/44162). Returns True if applied.
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
    """Return True when the model contains HF ESM RotaryEmbedding modules (capability-based)."""
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


# AMPLIFY helpers


def fix_amplify_meta_tensors(model):
    """Recompute freqs_cis if stuck on meta device (happens with from_pretrained)."""
    if hasattr(model, "freqs_cis") and model.freqs_cis.is_meta:
        mod = importlib.import_module(model.__class__.__module__)
        model.freqs_cis = mod.precompute_freqs_cis(
            model.config.hidden_size // model.config.num_attention_heads,
            model.config.max_length,
        )


def fix_proteva_rope_buffer(model):
    """Recompute Proteva's non-persistent ``encoder.rope_cs`` after ``from_pretrained``.

    The buffer is absent from the checkpoint, so HF leaves the meta allocation as
    garbage and RoPE is silently DISABLED (probe scores ~0.03-0.32 below native
    AMPLIFY). Analogue of :func:`fix_amplify_meta_tensors`. No-op when valid.
    """
    enc = getattr(model, "encoder", None)
    if enc is None or not hasattr(enc, "rope_cs"):
        return
    buf = enc.rope_cs
    # A valid cache has cos==1 / sin==0 at position 0; meta/zero/garbage fails this.
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

    xformers kernels incompatible with the running torch/CUDA build raise
    ``NotImplementedError`` in forward. ``PROTJEPA_AMPLIFY_FORCE_SDPA=1`` skips
    xformers entirely. Idempotent.
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

    additive_mask = torch.where(attention_mask.bool(), 0.0, float("-inf"))

    if torch.is_autocast_enabled():
        additive_mask = additive_mask.to(torch.get_autocast_dtype("cuda"))
    elif dtype is not None:
        additive_mask = additive_mask.to(dtype)
    if device is not None:
        additive_mask = additive_mask.to(device)

    return input_ids, additive_mask, orig_len, pad_len


# Out-of-vocabulary residue fallback


class _FallbackVocab(dict):
    """Vocabulary dict that returns a fallback id for any unknown token.

    Module-level with ``__reduce__`` so it pickles into dataloader workers. ``__missing__``
    fires only on ``[]``, so ``in``/``len()``/``get_vocab()`` still report the true vocab.
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

    FastPLM's ``_convert_token_to_id`` raises ``KeyError`` on OOV tokens ('J', '|',
    '#', a NUL byte -- all seen in real tasks); probing ``AutoTokenizer`` will not
    show it. 'X' rather than ``<unk>`` keeps input in ESM2's pretraining distribution.
    Idempotent.
    """
    table = getattr(tokenizer, "_token_to_id", None)
    if not isinstance(table, dict) or isinstance(table, _FallbackVocab):
        return
    fallback_id = table.get("X", getattr(tokenizer, "unk_token_id", None))
    if fallback_id is None:
        return
    tokenizer._token_to_id = _FallbackVocab(table, fallback_id)
    logger.info(
        "Tokenizer will map out-of-vocabulary tokens to 'X' (id %d)", fallback_id
    )


# ESMplusplus SDPA fix


def force_sdpa_backend(model):
    """Ensure SDPA attention backend on ESMplusplus / Synthyra models.

    Safety net against Hub code changes; no-op when ``transformer.attn_backend`` is absent.
    """
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        return
    if hasattr(transformer, "attn_backend"):
        transformer.attn_backend = "sdpa"
    for block in getattr(transformer, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None:
            if hasattr(attn, "attn_backend"):
                attn.attn_backend = "sdpa"
            if hasattr(attn, "flex_attention"):
                attn.flex_attention = None
    logger.info("   Forced SDPA attention backend (flex_attention disabled)")


# Model wrappers


class _PLMWrapperBase(nn.Module):
    """Base wrapper for PLMs that need hidden_states extraction.

    Subclasses override _prepare_inputs() to handle model-specific quirks.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def _prepare_inputs(self, input_ids, attention_mask):
        """Override to transform inputs before model call. Returns (kwargs, orig_len)."""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }, input_ids.shape[1]

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        embedding = outputs.hidden_states[-1][:, :orig_len, :]
        return (embedding,)

    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class ESMplusplusWrapper(_PLMWrapperBase):
    """ESMplusplus thin wrapper: extracts last_hidden_state for SentenceTransformer."""

    def _prepare_inputs(self, input_ids, attention_mask):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }, input_ids.shape[1]

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embedding = outputs.last_hidden_state[:, :orig_len, :]
        else:
            embedding = outputs.hidden_states[-1][:, :orig_len, :]
        return (embedding,)


class FastPLMESM2Wrapper(ESMplusplusWrapper):
    """Wrapper for Synthyra's FastPLM ESM2; raises on missing hidden states."""

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embedding = outputs.last_hidden_state[:, :orig_len, :]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            embedding = outputs.hidden_states[-1][:, :orig_len, :]
        else:
            raise RuntimeError("FastPLM ESM2 model returned no hidden states")
        return (embedding,)


class DPLM2Wrapper(ESMplusplusWrapper):
    """Wrapper for Synthyra's DPLM2 (AutoModel; has ``model.tokenizer``)."""

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embedding = outputs.last_hidden_state[:, :orig_len, :]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            embedding = outputs.hidden_states[-1][:, :orig_len, :]
        else:
            raise RuntimeError("DPLM2 model returned no hidden states")
        return (embedding,)


class ProfluentE1Wrapper(ESMplusplusWrapper):
    """Wrapper for Synthyra's Profluent-E1 (AutoModelForMaskedLM)."""

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embedding = outputs.last_hidden_state[:, :orig_len, :]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            embedding = outputs.hidden_states[-1][:, :orig_len, :]
        else:
            raise RuntimeError("Profluent-E1 model returned no hidden states")
        return (embedding,)


class AMPLIFYWrapper(_PLMWrapperBase):
    """AMPLIFY: pad to multiple of 8 + additive attention mask for xformers.

    Also applies the final layer_norm_2 which AMPLIFY omits from
    hidden_states (it is only applied internally before the decoder head).
    """

    def __init__(self, model):
        super().__init__(model)
        fix_amplify_meta_tensors(model)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        embedding = outputs.hidden_states[-1][:, :orig_len, :]
        if hasattr(self.model, "layer_norm_2"):
            embedding = self.model.layer_norm_2(embedding)
        return (embedding,)

    def _prepare_inputs(self, input_ids, attention_mask):
        param = next(self.model.parameters(), None)
        input_ids, additive_mask, orig_len, _ = _prepare_amplify_inputs(
            input_ids,
            attention_mask,
            dtype=param.dtype if param is not None else None,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": additive_mask,
            "output_hidden_states": True,
        }, orig_len
