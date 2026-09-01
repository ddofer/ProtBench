from __future__ import annotations

from scripts.profile_models import parameter_counts, reset_peak_memory


class _Parameter:
    def __init__(self, size: int) -> None:
        self._size = size

    def numel(self) -> int:
        return self._size


class _Module:
    def named_parameters(self):
        return iter(
            [
                ("encoder.block.weight", _Parameter(10)),
                ("aux_ptm_any.weight", _Parameter(3)),
                ("tax_class_head.weight", _Parameter(2)),
                ("mlm_head.weight", _Parameter(7)),
            ]
        )


def test_parameter_counts_separates_encoder_and_aux_parameters() -> None:
    assert parameter_counts(_Module()) == {
        "total_parameters": 22,
        "encoder_parameters": 10,
        "aux_parameters": 5,
    }


def test_reset_peak_memory_only_touches_cuda() -> None:
    calls = []

    class _Cuda:
        def reset_peak_memory_stats(self, device):
            calls.append(("reset", device))

        def empty_cache(self):
            calls.append(("empty",))

    class _Torch:
        cuda = _Cuda()

    reset_peak_memory(_Torch(), "cpu")
    assert calls == []
    reset_peak_memory(_Torch(), "cuda:0")
    assert calls == [("empty",), ("reset", "cuda:0")]
