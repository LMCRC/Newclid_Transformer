from pathlib import Path
import json
import os
from alphageo.alphageometry import get_lm, get_tokenizer, run_alphageometry
from alphageo.cli import DEFAULT_OUTPUT, run_cli

try:
    import torch
except ImportError:
    torch = object()

from newclid import GeometricSolverBuilder

RESULTS_DIR = "./results"


def main() -> bool:
    args = run_cli()

    if args.logging:
        import logging

        logging.basicConfig(level=logging.INFO)

    # when using the language model,
    # point names will be renamed to alphabetical a, b, c, d, e, ...
    # instead of staying with their original names,
    # in order to match the synthetic training data generation.
    need_rename = not args.solver_only

    out_folder = args.out_folder
    if out_folder == DEFAULT_OUTPUT:
        out_folder = f"{RESULTS_DIR}/{args.exp}/{args.problem}"
    if out_folder is not None:
        if out_folder == "None":
            out_folder = None
        else:
            out_folder = Path(out_folder)
            out_folder.mkdir(parents=True, exist_ok=True)

        single_file_stats = out_folder / "stats.json"
        if os.path.exists(single_file_stats):
            tmp = json.load(open(single_file_stats))
            if "success" in tmp:
                if tmp["success"] and tmp["success"] is not None:
                    if args.logging:
                        logging.info(f"[{args.problem}] stats found!")
                    return True

    solver_builder = GeometricSolverBuilder(seed=None).load_problem_from_file(
        problems_path=args.problems_file,
        problem_name=args.problem,
        translate=need_rename,
    )
    if args.defs is not None:
        solver_builder.load_defs_from_file(args.defs)
    if args.rules is not None:
        solver_builder.load_rules_from_file(args.rules)
    solver = solver_builder.build()

    success = False
    stats = {"name": args.problem}

    try:
        if args.solver_only:
            success = solver.run()
        else:
            with torch.no_grad():
                model = get_lm(args.ckpt, args.device)
                tokenizer = get_tokenizer(args.vocab)
                success = run_alphageometry(
                    solver,
                    model,
                    tokenizer,
                    args.device,
                    args.lm_beam_width,
                    args.batch_size,
                    args.search_depth,
                    args.search_width,
                    out_folder,
                )
        #
    except Exception as inst:
        stats["exception"] = str(inst)

    if success:
        if out_folder is not None:
            solver.write_solution(out_folder / "proof_steps.txt")
            solver.draw_figure(out_folder / "proof_figure.png")
        else:
            solver.write_solution(out_folder)
        stats.update(solver.run_infos)
    #
    else:
        stats["success"] = success  # case when llm can't solve it

    if args.logging:
        logging.info(f"[{args.problem}] Stats={stats}")
        logging.info(f"[{args.problem}] Success={success}")

    if out_folder is not None:
        with open(single_file_stats, "w") as out:
            out.write(json.dumps(stats, indent=2))

    return success


if __name__ == "__main__":
    main()
