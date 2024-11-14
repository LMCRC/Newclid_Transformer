from pathlib import Path
from typing import TYPE_CHECKING
import pytest
from alphageo.inference import priority_beam_search as beam_search
from alphageo.model import Decoder
import sentencepiece as spm
from numpy import isclose
import torch

if TYPE_CHECKING:
    ...


def test_beamsearch_outputs():
    check_point_path = Path("./pt_ckpt")
    if not check_point_path.exists():
        pytest.skip(f"No checkpoint found at {check_point_path}")

    torch.set_grad_enabled(False)

    tokenizer = spm.SentencePieceProcessor(str(check_point_path / "vocab.model"))
    cfg = torch.load(check_point_path / "cfg.sav")  # type: ignore
    model = Decoder(cfg)
    params = torch.load(check_point_path / "params.sav")  # type: ignore
    model.load_state_dict(params)
    model.bfloat16()

    # "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    test_str = (
        "{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c {F1} x00"
    )
    tokens = tokenizer.encode_as_ids(test_str)
    inp = torch.LongTensor([tokens])
    outp = beam_search(model, inp, beam_width=4, num_return_sequences=2)

    assert (
        tokenizer.decode_ids(outp[0][0].tolist())
        == test_str + " e : C a c e 02 C b d e 03 ;"
    )  # type: ignore
    assert isclose(-1.6585191699136992, outp[0][1], rtol=0.05)

    assert (
        tokenizer.decode_ids(outp[1][0].tolist())
        == test_str + " e : D a b c e 02 D a c b e 03 ;"
    )  # type: ignore
    assert isclose(-1.802338749355057, outp[1][1], rtol=0.05)
