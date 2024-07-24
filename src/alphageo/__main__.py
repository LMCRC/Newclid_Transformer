from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
from typing import Optional
from alphageo.alphageometry import get_lm, get_tokenizer, run_alphageometry
from alphageo.cli import run_cli
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from geosolver import AGENTS_REGISTRY, GeometricSolver, GeometricSolverBuilder
from geosolver.problem import Problem
import torch

RESULTS_DIR = Path("./results")


def solve_problem(
    problem: Problem, solver_builder: GeometricSolverBuilder
) -> GeometricSolver:
    logging.info(f"Building {problem}")
    solver = deepcopy(solver_builder).load_problem(problem).build()
    logging.info(f"Built. Now try to solve {problem}")
    solver.run()
    return solver


def parallel_solve(
    problems: list[Problem],
    solver_builder: GeometricSolverBuilder,
    max_workers: Optional[int] = None,
) -> GeometricSolver:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_problem = {
            executor.submit(solve_problem, problem, solver_builder): problem
            for problem in problems
        }
        for future in as_completed(future_to_problem):
            problem = future_to_problem[future]
            try:
                solver = future.result()
                if solver.run_infos["success"]:
                    logging.info(f"Solved: {problem}")
                    return solver
                else:
                    logging.error(f"Problem {problem} was not solved")
            except Exception as exc:
                logging.error(
                    f"Problem {problem} generated an exception when being solved: {exc}"
                )
    return solver  # type: ignore


def main() -> bool:
    args = run_cli()

    logging.basicConfig(level=args.log_level)

    # when using the language model,
    # point names will be renamed to alphabetical a, b, c, d, e, ...
    # instead of staying with their original names,
    # in order to match the synthetic training data generation.

    out_folder = args.out_folder
    if out_folder is None:
        out_folder = RESULTS_DIR / args.exp / args.problem
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
        problems: list[Problem] = []
        with open(out_folder / "aux.txt", "r") as aux:
            for line in aux.readlines():
                line = line.strip()
                if not line:
                    continue
                problems.append(Problem.from_text(line))
        assert len(problems) > 0
        solver = parallel_solve(problems, solver_builder)
    else:
        assert args.problems_file
        problem = Problem.from_file(
            problems_path=args.problems_file,
            problem_name=args.problem,
        ).renamed()
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

    stats.update(solver.run_infos)
    if stats["success"]:
        solver.write_solution(out_folder / "proof_steps.txt")
        solver.draw_figure(False, out_folder / "proof_figure.png")
        stats.update(solver.run_infos)

    logging.info(f"[{args.problem}] Stats={stats}")

    with open(out_folder / "stats.json", "w") as out:
        out.write(json.dumps(stats, indent=2))

    if not args.have_aux:
        with open(out_folder / "aux.txt", "w") as out:
            for problem in problems:
                print(str(problem), file=out)
    return stats["success"]


if __name__ == "__main__":
    main()
