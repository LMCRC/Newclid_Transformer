from argparse import ArgumentParser, Namespace

DEFAULT_OUTPUT = "#Default"


def run_cli() -> Namespace:
    parser = ArgumentParser("alphageo")
    parser.add_argument(
        "--problems",
        default="problems_datasets/examples.txt",
        type=str,
        help="Path to the text file contains the problem strings.",
    )
    parser.add_argument(
        "--problem",
        default="orthocenter",
        type=str,
        help="text file contains the problem strings. See imo_ag_30.txt for example.",
    )
    parser.add_argument(
        "--defs",
        default=None,
        help="Path to definitions of available constructions to state a problem."
        " Defaults to geosolver's default. See geosolver for more details.",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Path to the list of deduction (explicit) rules used by DD."
        " Defaults to geosolver's default. See geosolver for more details.",
    )
    parser.add_argument(
        "--solver-only",
        action="store_true",
        help="If set, only runs geometric solver without the LLM.",
    )
    parser.add_argument(
        "--ckpt",
        default="pt_ckpt",
        help="Path to the checkpoint of the LM model.",
    )
    parser.add_argument(
        "--vocab",
        default="pt_ckpt/vocab.model",
        help="Path to the LM vocab file.",
    )
    parser.add_argument(
        "--lm-beam-width",
        "-B",
        default=4,
        help="Beam width for the LM decoder.",
    )
    parser.add_argument(
        "--batch-size",
        "-K",
        default=2,
        help="Number of sequences LM decoder returns for each input",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="compute device for LM",
    )
    parser.add_argument(
        "--out-folder",
        "-o",
        default=DEFAULT_OUTPUT,
        help="Path to the solution output folder.",
    )
    parser.add_argument(
        "--search-width",
        "-W",
        default=1,
        help="Beam width of the proof search across LM sugestions.",
    )
    parser.add_argument(
        "--search-depth",
        default=1,
        help="Depth of the proof search across LM sugestions.",
    )
    args, _ = parser.parse_known_args()
    return args
