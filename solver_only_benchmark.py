import os
from pathlib import Path
from sys import stderr
from concurrent.futures import ProcessPoolExecutor
from geosolver.api import GeometricSolverBuilder


def run_solver_on_problem(problems_path: Path, problem_name: str):
    try:
        solver = (
            GeometricSolverBuilder()
            .load_problem_from_file(problems_path, problem_name)
            .build()
        )
        print(f"Start running {problem_name}")
        if solver.run():
            print(f"Solved problem {problem_name}")
        solver.write_all_outputs(Path("results") / "only_geosolver" / problem_name)
    except Exception as e:
        print(
            f"Warning: solver crashed on problem '{problem_name}' : ({type(e)}) {e}",
            file=stderr,
        )


def kill_processes(executor: ProcessPoolExecutor):
    # Access the internal processes managed by the executor
    for process in executor._processes.values():
        print(f"Terminating process {process.pid}")
        process.terminate()  # Forcefully terminate the process
        process.join()  # Wait for the process to finish
        print(f"Process {process.pid} terminated")


def run_geosolver(problems_path: Path, max_workers: int = 32):
    if not os.path.exists(problems_path):
        print(f"File {problems_path} not found.")
        return
    problem_names: list[str] = []

    with open(problems_path, "r") as file:
        for count, line in enumerate(file):
            if count % 2 == 0:
                problem_names.append(line.strip())

    print(f"{problem_names=}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(
            run_solver_on_problem, [problems_path] * len(problem_names), problem_names
        )


if __name__ == "__main__":
    problems_path = Path("problems_datasets/new_benchmark_50.txt")
    run_geosolver(Path(problems_path))
