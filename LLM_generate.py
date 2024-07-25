import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
from alphageo.__main__ import main

need_aux = [
    "translated_imo_1983_p2",
    "translated_imo_2009_sl_g3",
    "translated_imo_2009_sl_g6",
    "translated_imo_2010_sl_g2",
    "translated_imo_2012_sl_g3",
    "translated_imo_2012_sl_g4",
    "translated_imo_2013_sl_g2",
    "translated_imo_2014_sl_g3",
    "translated_imo_2015_sl_g3",
    "translated_imo_2015_sl_g5",
    "translated_imo_2016_sl_g2",
    "translated_imo_2016_sl_g4",
    "translated_imo_2016_sl_g5",
    "translated_imo_2016_sl_g6",
    "translated_imo_2017_sl_g3",
    "translated_imo_2017_sl_g4",
    "translated_imo_2017_sl_",
    "translated_imo_2018_sl_g2",
    "translated_imo_2018_sl_g5",
    "translated_imo_2019_sl_g2",
    "translated_imo_2019_sl_g7",
    "translated_imo_2020_sl_g7a",
    "translated_imo_2020_sl_g7b",
    "translated_imo_2020_sl_g8",
    "translated_imo_2021_sl_g4",
    "translated_imo_2022_sl_g2",
    "translated_imo_2022_sl_g3",
    "translated_usamo_1997_p2",
    "translated_usamo_1999_p6",
    "translated_usamo_2001_p2",
    "translated_usamo_2005_p3",
    "translated_usamo_2008_p2",
    "translated_usamo_2012_p5",
    "translated_usamo_2013_p1",
]


def run_solver_on_problem(problems_path: Path, problem_name: str):
    sys.argv = [
        "alphageo",
        "--problems-file",
        str(problems_path),
        "--problem",
        problem_name,
        "--agent",
        "flemmard",
        "--search-width",
        "512",
        "--search-depth",
        "4",
        "--batch-size" "32",
        "--lm-beam-width",
        "32",
        "--exp",
        "with_aux",
    ]
    main()


def run_geosolver(problems_path: Path, max_workers: int = 32):
    if not os.path.exists(problems_path):
        print(f"File {problems_path} not found.")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_solver_on_problem, [problems_path] * len(need_aux), need_aux)


if __name__ == "__main__":
    problems_path = Path("problems_datasets/new_benchmark_50.txt")
    run_geosolver(Path(problems_path))
