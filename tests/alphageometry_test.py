from pathlib import Path
import sys
import pytest
from alphageo.__main__ import main


def test_solver_only_should_solve_orthocenter_aux():
    sys.argv = [
        "alphageo",
        "--device",
        "cpu",
        "-o",
        None,
        "--problems",
        "problems_datasets/examples.txt",
        "--problem",
        "orthocenter_aux",
        "--solver-only",
    ]
    success = main()
    assert success


def test_solver_only_should_not_solve_orthocenter():
    sys.argv = [
        "alphageo",
        "--device",
        "cpu",
        "-o",
        None,
        "--problems",
        "problems_datasets/examples.txt",
        "--problem",
        "orthocenter",
        "--solver-only",
    ]
    success = main()
    assert not success


def test_alphageometry_should_solve_orthocenter():
    pytest.importorskip("torch")
    check_point_path = Path(".\pt_ckpt")
    if not check_point_path.exists():
        pytest.skip(f"No checkpoint found at {check_point_path}")

    sys.argv = [
        "alphageo",
        "--device",
        "cpu",
        "-o",
        None,
        "--problems",
        "problems_datasets/examples.txt",
        "--problem",
        "orthocenter",
    ]
    success = main()
    assert success
