from __future__ import annotations
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import logging


def run_cli() -> Namespace:
    parser = ArgumentParser("alphageo", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--problems-file",
        default=None,
        help="Path to the text file contains the problem strings.",
    )
    parser.add_argument(
        "--problem",
        default="orthocenter",
        help="text file contains the problem strings. See imo_ag_30.txt for example.",
    )
    parser.add_argument(
        "--have-aux",
        default=False,
        action="store_true",
        help="In the outfolder we have a file with a list of problems based on adding auxilliary points to an original problem"
    )
    parser.add_argument(
        "--exp",
        default="exp",
        help="experiment name, including dataset, modules, version",
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
        "--agent",
        default=None,
        help="Agent for geosolver"
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
        type=int,
        default=4,
        help="Beam width for the LM decoder.",
    )
    parser.add_argument(
        "--batch-size",
        "-K",
        type=int,
        default=2,
        help="Number of sequences LM decoder returns for each input",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="compute device for LM",
    )
    parser.add_argument(
        "--out-folder",
        "-o",
        default=None,
        help="Path to the solution output folder.",
    )
    parser.add_argument(
        "--search-width",
        "-W",
        type=int,
        default=2,
        help="Beam width of the proof search across LM sugestions.",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=2,
        help="Depth of the proof search across LM sugestions.",
    )
    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=int,
        help="Logging level.",
    )
    args, _ = parser.parse_known_args()
    return args
