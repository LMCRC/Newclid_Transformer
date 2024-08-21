from __future__ import annotations
from pathlib import Path
import sys
import pytest
from alphageo.__main__ import main


def test_solve_orthocenter_aux():
    check_point_path = Path("./pt_ckpt")
    if not check_point_path.exists():
        pytest.skip(f"No checkpoint found at {check_point_path}")
    sys.argv = [
        "alphageo",
        "--device",
        "cpu",
        "-o",
        None,
        "--problems-file",
        "problems_datasets/examples.txt",
        "--problem",
        "orthocenter_aux",
        "--rules",
        "rule_sets/triangles.txt",
    ]
    success = main()
    assert success


def test_alphageometry_should_solve_orthocenter():
    check_point_path = Path("./pt_ckpt")
    if not check_point_path.exists():
        pytest.skip(f"No checkpoint found at {check_point_path}")

    sys.argv = [
        "alphageo",
        "--device",
        "cpu",
        "-o",
        None,
        "--problems-file",
        "problems_datasets/examples.txt",
        "--problem",
        "orthocenter",
        "--rules",
        "rule_sets/triangles.txt",
    ]
    success = main()
    assert success


@pytest.mark.skip("Too slow")
def test_imo_2018_p1():
    check_point_path = Path("./pt_ckpt")
    if not check_point_path.exists():
        pytest.skip(f"No checkpoint found at {check_point_path}")

    sys.argv = [
        "alphageo",
        "--device",
        "cpu",
        "-o",
        None,
        "--problems-file",
        "problems_datasets/examples.txt",
        "--problem",
        "translated_imo_2018_p1",
        "--search-width",
        "5",
        "--search-depth",
        "2",
        "--batch-size",
        "5",
        "--lm-beam-width",
        "5",
    ]
    success = main()
    assert success


@pytest.mark.skip("Too slow")
def test_imo_2012_p5():
    check_point_path = Path("./pt_ckpt")
    if not check_point_path.exists():
        pytest.skip(f"No checkpoint found at {check_point_path}")

    sys.argv = [
        "alphageo",
        "--device",
        "cpu",
        "-o",
        None,
        "--problems-file",
        "problems_datasets/examples.txt",
        "--problem",
        "translated_imo_2018_p1",
        "--search-width",
        "5",
        "--search-depth",
        "2",
        "--batch-size",
        "5",
        "--lm-beam-width",
        "5",
    ]
    success = main()
    assert success
