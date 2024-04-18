from pathlib import Path

from alphageo.alphageometry import get_lm, get_tokenizer, run_alphageometry
from alphageo.cli import DEFAULT_OUTPUT, run_cli

try:
    import torch
except ImportError:
    torch = object()


from geosolver import GeometricSolverBuilder
import sys


def main() -> bool:
    sys.setrecursionlimit(2000)

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
        out_folder = f"results/{args.problem}"
    if out_folder is not None:
        if out_folder == "None":
            out_folder = None
        else:
            out_folder = Path(out_folder)
            out_folder.mkdir(parents=True, exist_ok=True)

    solver_builder = GeometricSolverBuilder().load_problem_from_file(
        problems_path=args.problems_file,
        problem_name=args.problem,
        translate=need_rename,
    )
    if args.defs is not None:
        solver_builder.load_defs_from_file(args.defs)
    if args.rules is not None:
        solver_builder.load_rules_from_file(args.rules)
    solver = solver_builder.build()

    if args.solver_only:
        success = solver.run()
        if success:
            if out_folder is not None:
                solver.write_solution(out_folder / "proof_steps.txt")
                solver.draw_figure(out_folder / "proof_figure.png")
            else:
                solver.write_solution(out_folder)
        return success

    torch.requires_grad = False
    model = get_lm(args.ckpt, args.device)
    tokenizer = get_tokenizer(args.vocab)
    return run_alphageometry(
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


if __name__ == "__main__":
    main()
