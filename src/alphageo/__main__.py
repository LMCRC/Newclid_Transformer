from __future__ import annotations
from pathlib import Path
import json
from alphageo.alphageometry import get_lm, get_tokenizer, run_alphageometry
from alphageo.cli import run_cli
import logging

from geosolver import GeometricSolverBuilder
from geosolver.problem import Problem
import torch
RESULTS_DIR = Path("./results")

def main() -> bool:
    args = run_cli()

    logging.basicConfig(level=args.log_level)

    # when using the language model,
    # point names will be renamed to alphabetical a, b, c, d, e, ...
    # instead of staying with their original names,
    # in order to match the synthetic training data generation.

    out_folder = args.out_folder
    if out_folder is None:
        out_folder = RESULTS_DIR/args.exp/args.problem
    else:
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

    single_file_stats = out_folder / "stats.json"

    solver_builder = GeometricSolverBuilder()
    problem = Problem.from_file(
        problems_path=args.problems_file,
        problem_name=args.problem,
    )
    if args.defs:
        solver_builder.load_defs_from_file(Path(args.defs))
    if args.rules:
        solver_builder.load_rules_from_file(Path(args.rules))

    stats = {"problem": args.problem}

    with torch.no_grad():
        model = get_lm(Path(args.ckpt), args.device)
        tokenizer = get_tokenizer(Path(args.vocab))
        solver = run_alphageometry(
            solver_builder,
            problem,
            model,
            tokenizer,
            args.device,
            args.lm_beam_width,
            args.batch_size,
            args.search_depth,
            args.search_width,
        )

    stats.update(solver.run_infos)
    if stats["success"]:
        if out_folder is not None:
            solver.write_solution(out_folder / "proof_steps.txt")
            solver.draw_figure(False, out_folder / "proof_figure.png")
        else:
            solver.write_solution(out_folder)
        stats.update(solver.run_infos)

    logging.info(f"[{args.problem}] Stats={stats}")

    if out_folder is not None:
        with open(single_file_stats,"w") as out:
            out.write(json.dumps(stats,indent=2))
    return stats["success"]


if __name__ == "__main__":
    main()
