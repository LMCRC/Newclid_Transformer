from pathlib import Path
import pytest
from alphageo.inference import priority_beam_search as beam_search
from alphageo.model import Decoder
import sentencepiece as spm


def test_beamsearch_output_strings():
    pytest.importorskip("torch")

    import torch

    check_point_path = Path("./pt_ckpt")
    if not check_point_path.exists():
        pytest.skip(f"No checkpoint found at {check_point_path}")

    torch.set_grad_enabled(False)

    tokenizer = spm.SentencePieceProcessor(
        str(check_point_path.joinpath("vocab.model"))
    )
    cfg = torch.load(str(check_point_path.joinpath("cfg.sav")))
    model = Decoder(cfg)
    params = torch.load(str(check_point_path.joinpath("params.sav")))
    model.load_state_dict(params)
    model.bfloat16()

    test_str = (
        "{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c {F1} x00"
    )
    tokens = tokenizer.encode(test_str)
    inp = torch.LongTensor([tokens])
    outp = beam_search(model, inp, beam_width=4, num_return_sequences=2)

    assert tokenizer.decode(outp[0][0]) == test_str + " e : C a c e 02 C b d e 03 ;"
    assert tokenizer.decode(outp[1][0]) == test_str + " e : D a b c e 02 D a c b e 03 ;"


def test_beamsearch_output_scores():
    pytest.importorskip("torch")

    import torch
    from numpy import isclose

    check_point_path = Path("./pt_ckpt")
    if not check_point_path.exists():
        pytest.skip(f"No checkpoint found at {check_point_path}")

    torch.set_grad_enabled(False)

    tokenizer = spm.SentencePieceProcessor(
        str(check_point_path.joinpath("vocab.model"))
    )
    cfg = torch.load(str(check_point_path.joinpath("cfg.sav")))
    model = Decoder(cfg)
    params = torch.load(str(check_point_path.joinpath("params.sav")))
    model.load_state_dict(params)
    model.bfloat16()

    test_str = (
        "{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c {F1} x00"
    )
    tokens = tokenizer.encode(test_str)
    inp = torch.LongTensor([tokens])
    outp = beam_search(model, inp, beam_width=4, num_return_sequences=2)

    assert isclose(-1.6585191699136992, outp[0][1], rtol=0.05)
    assert isclose(-1.802338749355057, outp[1][1], rtol=0.05)
