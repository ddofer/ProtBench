"""fix_amplify_meta_tensors must tolerate AMPLIFY variants that set freqs_cis to None."""

from types import SimpleNamespace

from model_utils import fix_amplify_meta_tensors


def test_none_rotary_buffer_is_left_alone():
    model = SimpleNamespace(freqs_cis=None)
    fix_amplify_meta_tensors(model)  # hf_amplify_c raised AttributeError here
    assert model.freqs_cis is None
