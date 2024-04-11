from __future__ import annotations
from copy import deepcopy
import logging
from pathlib import Path
import traceback

from geosolver.graph import Graph
from geosolver.numericals import draw
from geosolver.ddar import solve, get_proof_steps
from geosolver.problem import Problem, Theorem, Definition, Dependency, Clause
from geosolver.geometry import Point, Circle, Line, Segment
from geosolver.pretty import pretty_nl


class GeometricSolver:
    def __init__(
        self,
        proof_state: "Graph",
        problem: "Problem",
        defs: dict[str, "Definition"],
        rules: list["Theorem"],
    ) -> None:
        self.proof_state = proof_state
        self.problem = problem
        self.defs = defs
        self.rules = rules
        self.pstring = problem.txt()

    @classmethod
    def build_from_files(
        cls,
        defs_path: Path,
        rules_path: Path,
        problems_path: Path,
        problem_name: str,
        translate: bool,
    ):
        defs = Definition.from_txt_file(defs_path, to_dict=True)
        rules = Theorem.from_txt_file(rules_path, to_dict=True)
        problems = Problem.from_txt_file(
            problems_path, to_dict=True, translate=translate
        )
        if problem_name not in problems:
            raise ValueError(f"Problem name `{problem_name}` not found in `{problems}`")

        problem = problems[problem_name]
        proof_state, _ = Graph.build_problem(problem, defs)
        return cls(proof_state, problem, defs, rules)

    def load_state(self, proof_state: Graph):
        self.proof_state = deepcopy(proof_state)

    def load_pstring(self, pstring: str):
        self.pstring = pstring

    def get_problem_string(self) -> str:
        return self.problem.txt()

    def get_proof_state(self) -> str:
        return deepcopy(self.proof_state)

    def get_defs(self):
        return self.defs

    def get_setup_string(self) -> str:
        return self.problem.setup_str_from_problem(self.defs)

    def run_solver(self) -> bool:
        solve(self.proof_state, self.rules, self.problem, max_level=1000)
        goal = self.problem.goal
        goal_args_names = self.proof_state.names2nodes(goal.args)
        if not self.proof_state.check(goal.name, goal_args_names):
            logging.info("Solver failed to solve the problem.")
            return False
        logging.info("Solved.")
        return True

    def write_solution(self, out_file: Path):
        write_solution(self.proof_state, self.problem, out_file)

    def draw_figure(self, out_file: Path):
        draw(
            self.proof_state.type2nodes[Point],
            self.proof_state.type2nodes[Line],
            self.proof_state.type2nodes[Circle],
            self.proof_state.type2nodes[Segment],
            block=False,
            save_to=out_file,
        )

    def get_existing_points(self) -> list[str]:
        return [p.name for p in self.proof_state.all_points()]

    def validate_clause_txt(self, clause_txt: str):
        if clause_txt.startswith("ERROR"):
            return clause_txt
        clause = Clause.from_txt(clause_txt)
        try:
            self.proof_state.copy().add_clause(clause, 0, self.defs)
        except Exception:
            return "ERROR: " + traceback.format_exc()
        return clause_txt

    def add_auxiliary_construction(self, aux_string: str):
        # Update the constructive statement of the problem with the aux point:
        candidate_pstring = insert_aux_to_premise(self.pstring, aux_string)
        logging.info('Solving: "%s"', candidate_pstring)
        p_new = Problem.from_txt(candidate_pstring)
        p_new.url = self.problem.url
        # This is the new proof state graph representation:
        g_new, _ = Graph.build_problem(p_new, self.defs)

        self.problem = p_new
        self.proof_state = g_new


def insert_aux_to_premise(pstring: str, auxstring: str) -> str:
    """Insert auxiliary constructs from proof to premise.

    Args:
      pstring: str: describing the problem to solve.
      auxstring: str: describing the auxiliar construction.

    Returns:
      str: new pstring with auxstring inserted before the conclusion.
    """
    setup, goal = pstring.split(" ? ")
    return setup + "; " + auxstring + " ? " + goal


def write_solution(proof: "Graph", problem: "Problem", out_file: str) -> None:
    """Output the solution to out_file.

    Args:
      g: gh.Graph object, containing the proof state.
      p: Problem object, containing the theorem.
      out_file: file to write to, empty string to skip writing to file.
    """
    setup, aux, proof_steps, refs = get_proof_steps(
        proof, problem.goal, merge_trivials=False
    )

    solution = "\n=========================="
    solution += "\n * From theorem premises:\n"
    premises_nl = []
    for premises, [points] in setup:
        solution += " ".join([p.name.upper() for p in points]) + " "
        if not premises:
            continue
        premises_nl += [
            _natural_language_statement(p) + " [{:02}]".format(refs[p.hashed()])
            for p in premises
        ]
    solution += ": Points\n" + "\n".join(premises_nl)

    solution += "\n\n * Auxiliary Constructions:\n"
    aux_premises_nl = []
    for premises, [points] in aux:
        solution += " ".join([p.name.upper() for p in points]) + " "
        aux_premises_nl += [
            _natural_language_statement(p) + " [{:02}]".format(refs[p.hashed()])
            for p in premises
        ]
    solution += ": Points\n" + "\n".join(aux_premises_nl)

    # some special case where the deduction rule has a well known name.
    r2name = {
        "r32": "(SSS)",
        "r33": "(SAS)",
        "r34": "(Similar Triangles)",
        "r35": "(Similar Triangles)",
        "r36": "(ASA)",
        "r37": "(ASA)",
        "r38": "(Similar Triangles)",
        "r39": "(Similar Triangles)",
        "r40": "(Congruent Triangles)",
        "a00": "(Distance chase)",
        "a01": "(Ratio chase)",
        "a02": "(Angle chase)",
    }

    solution += "\n\n * Proof steps:\n"
    for i, step in enumerate(proof_steps):
        _, [con] = step
        nl = _proof_step_string(step, refs, last_step=i == len(proof_steps) - 1)
        rule_name = r2name.get(con.rule_name, "")
        nl = nl.replace("\u21d2", f"{rule_name}\u21d2 ")
        solution += "{:03}. ".format(i + 1) + nl + "\n"

    solution += "==========================\n"
    logging.info(solution)
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(solution)
        logging.info("Solution written to %s.", out_file)


def _proof_step_string(
    proof_step: Dependency, refs: dict[tuple[str, ...], int], last_step: bool
) -> str:
    """Translate proof to natural language.

    Args:
      proof_step: Dependency with .name and .args
      refs: dict(hash: int) to keep track of derived predicates
      last_step: boolean to keep track whether this is the last step.

    Returns:
      a string of (pseudo) natural language of the proof step for human reader.
    """
    premises, [conclusion] = proof_step

    premises_nl = " & ".join(
        [
            _natural_language_statement(p) + " [{:02}]".format(refs[p.hashed()])
            for p in premises
        ]
    )

    if not premises:
        premises_nl = "similarly"

    refs[conclusion.hashed()] = len(refs)

    conclusion_nl = _natural_language_statement(conclusion)
    if not last_step:
        conclusion_nl += " [{:02}]".format(refs[conclusion.hashed()])

    return f"{premises_nl} \u21d2 {conclusion_nl}"


def _natural_language_statement(logical_statement: Dependency) -> str:
    """Convert logical_statement to natural language.

    Args:
      logical_statement: Dependency with .name and .args

    Returns:
      a string of (pseudo) natural language of the predicate for human reader.
    """
    names = [a.name.upper() for a in logical_statement.args]
    names = [(n[0] + "_" + n[1:]) if len(n) > 1 else n for n in names]
    return pretty_nl(logical_statement.name, names)
