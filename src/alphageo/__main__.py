from pathlib import Path

from alphageo.alphageometry import get_lm, get_tokenizer, run_alphageometry
from alphageo.cli import run_cli


import torch


from alphageo.geosolver_facade import GeometricSolver


def main():
    args = run_cli()

    # when using the language model,
    # point names will be renamed to alphabetical a, b, c, d, e, ...
    # instead of staying with their original names,
    # in order to match the synthetic training data generation.
    need_rename = not args.solver_only

    out_folder = args.out_folder
    if out_folder is None:
        out_folder = f"results/{args.problem}"
    out_path = Path(out_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    solver = GeometricSolver.build_from_files(
        args.defs, args.rules, args.problems, args.problem, translate=need_rename
    )

    if args.solver_only:
        return solver.run_solver()

    torch.requires_grad = False
    model = get_lm(args.ckpt, args.device)
    tokenizer = get_tokenizer(args.vocab)
    run_alphageometry(
        solver,
        model,
        tokenizer,
        args.device,
        args.lm_beam_width,
        args.batch_size,
        args.search_depth,
        args.search_width,
        out_path,
    )


if __name__ == "__main__":
    main()
