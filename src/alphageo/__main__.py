from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
from alphageo.alphageometry import get_lm, get_tokenizer, run_alphageometry
from alphageo.cli import run_cli
import logging

from geosolver import AGENTS_REGISTRY, GeometricSolverBuilder
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

    solver_builder = GeometricSolverBuilder()
    if args.defs:
        solver_builder.load_defs_from_file(Path(args.defs))
    if args.rules:
        solver_builder.load_rules_from_file(Path(args.rules))
    if args.agent:
        solver_builder.with_deductive_agent(AGENTS_REGISTRY[args.agent])

    stats = {"problem": args.problem}
    if args.have_aux:
        problems : list[Problem] = []
        with open(out_folder / "aux.txt", "r") as aux:
            for line in aux.readlines():
                problems.append(Problem.from_text(line))
        assert len(problems) > 0
        for problem in problems:
            solver = deepcopy(solver_builder).load_problem(problem).build()
            if solver.run():
                break
    else:
        assert args.problems_file
        problem = Problem.from_file(
            problems_path=args.problems_file,
            problem_name=args.problem,
        )
        with torch.no_grad():
            model = get_lm(Path(args.ckpt), args.device)
            tokenizer = get_tokenizer(Path(args.vocab))
            solver, problems = run_alphageometry(
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

    assert solver # type: ignore
    stats.update(solver.run_infos)
    if stats["success"]:
        solver.write_solution(out_folder / "proof_steps.txt")
        solver.draw_figure(False, out_folder / "proof_figure.png")
        stats.update(solver.run_infos)

    logging.info(f"[{args.problem}] Stats={stats}")

    with open(out_folder / "stats.json", "w") as out:
        out.write(json.dumps(stats,indent=2))
    with open(out_folder / "aux.txt", "w") as out:
        for problem in problems:
            print(str(problem), file=out)
    return stats["success"]


if __name__ == "__main__":
    main()
