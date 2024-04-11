from pathlib import Path

from alphageo.alphageometry import get_lm, get_tokenizer, run_alphageometry, run_ddar
from alphageo.cli import run_cli


import geosolver.graph as gh
import geosolver.problem as pr
import torch


def main():
    args = run_cli()

    # defs of terms used in our domain-specific language.
    defs = pr.Definition.from_txt_file(args.defs, to_dict=True)
    # load inference rules used in DD.
    rules = pr.Theorem.from_txt_file(args.rules, to_dict=True)

    # when using the language model,
    # point names will be renamed to alphabetical a, b, c, d, e, ...
    # instead of staying with their original names,
    # in order to match the synthetic training data generation.
    need_rename = not args.solver_only

    # load problems from the problems_file,
    problems = pr.Problem.from_txt_file(
        args.problems, to_dict=True, translate=need_rename
    )

    if args.problem not in problems:
        raise ValueError(
            f"Problem name `{args.problem}` not found in `{args.problems}`"
        )

    problem = problems[args.problem]

    out_path = Path(args.out_folder)
    if args.solver_only:
        g, _ = gh.Graph.build_problem(problem, defs)
        run_ddar(g, problem, rules, out_path)
        return

    torch.requires_grad = False
    model = get_lm(args.ckpt, args.device)
    tokenizer = get_tokenizer(args.vocab)
    run_alphageometry(
        problem,
        defs,
        rules,
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
