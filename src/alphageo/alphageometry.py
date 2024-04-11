# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run DD+AR or AlphaGeometry solver.

Please refer to README.md for detailed instructions.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import geosolver.ddar as ddar
import geosolver.graph as gh
import geosolver.problem as pr
import geosolver.pretty as pt


import sentencepiece as spm
import torch

from alphageo.translate import try_translate_constrained_to_construct
from alphageo.inference import simple_beam_search

if TYPE_CHECKING:
    from pytorch.model import Decoder


def run_alphageometry(
    problem: pr.Problem,
    defs: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    model: "Decoder",
    tokenizer: spm.SentencePieceProcessor,
    device: str,
    model_beam_width: int,
    model_num_return_sequences: int,
    search_depth: int,
    beam_size: int,
    out_file: str,
) -> bool:
    """Simplified code to run AlphaGeometry proof search.

    We removed all optimizations that are infrastructure-dependent, e.g.
    parallelized model inference on multi GPUs,
    parallelized DD+AR on multiple CPUs,
    parallel execution of LM and DD+AR,
    shared pool of CPU workers across different problems, etc.

    Many other speed optimizations and abstractions are also removed to
    better present the core structure of the proof search.

    Args:
      model: Interface with inference-related endpoints to JAX's model.
      p: pr.Problem object describing the problem to solve.
      search_depth: max proof search depth.
      beam_size: beam size of the proof search.
      out_file: path to output file if solution is found.

    Returns:
      boolean of whether this is solved.
    """
    # translate the problem to a string of grammar that the LM is trained on.
    string = problem.setup_str_from_problem(defs)
    # special tokens prompting the LM to generate auxiliary points.
    string += " {F1} x00"
    # the graph to represent the proof state.
    graph, _ = gh.Graph.build_problem(problem, defs)

    # First we run the symbolic engine DD+AR:
    if run_ddar(graph, problem, rules, out_file):
        return True

    # beam search for the proof
    # each node in the search tree is a 3-tuple:
    # (<graph representation of proof state>,
    #  <string for LM to decode from>,
    #  <original problem string>)
    beam_queue = BeamQueue(max_size=beam_size)
    # originally the beam search tree starts with a single node (a 3-tuple):
    beam_queue.add(
        node=(graph, string, problem.txt()),
        val=0.0,  # value of the root node is simply 0.
    )

    for depth in range(search_depth):
        logging.info("Depth %s. There are %i nodes to expand:", depth, len(beam_queue))
        for _, (_, string, _) in beam_queue:
            logging.info(string)

        new_queue = BeamQueue(max_size=beam_size)  # to replace beam_queue.

        for prev_score, (graph, string, pstring) in beam_queue:
            logging.info("Decoding from %s", string)
            tokens = tokenizer.encode(string)
            inp = torch.LongTensor([tokens]).to(device)
            outs = simple_beam_search(
                model,
                inp,
                beam_width=model_beam_width,
                num_return_sequences=model_num_return_sequences,
            )
            outputs = {
                "seqs_str": [
                    tokenizer.decode(o[0][len(tokens) :]).strip() for o in outs
                ],
                "scores": [o[1] for o in outs],
            }
            # outputs = model.beam_decode(string, eos_tokens=[';'])

            # translate lm output to the constructive language.
            # so that we can update the graph representing proof states:
            translations = [
                try_translate_constrained_to_construct(output, graph, defs)
                for output in outputs["seqs_str"]
            ]

            # couple the lm outputs with its translations
            candidates = zip(outputs["seqs_str"], translations, outputs["scores"])

            # bring the highest scoring candidate first
            # candidates = reversed(list(candidates))

            for lm_out, translation, score in candidates:
                logging.info('LM output (score=%f): "%s"', score, lm_out)
                logging.info('Translation: "%s"\n', translation)

                if translation.startswith("ERROR:"):
                    # the construction is invalid.
                    continue

                # Update the constructive statement of the problem with the aux point:
                candidate_pstring = insert_aux_to_premise(pstring, translation)

                logging.info('Solving: "%s"', candidate_pstring)
                p_new = pr.Problem.from_txt(candidate_pstring)

                # This is the new proof state graph representation:
                g_new, _ = gh.Graph.build_problem(p_new, defs)
                if run_ddar(g_new, p_new, rules, out_file):
                    logging.info("Solved.")
                    return True

                # Add the candidate to the beam queue.
                new_queue.add(
                    # The string for the new node is old_string + lm output +
                    # the special token asking for a new auxiliary point ' x00':
                    node=(g_new, string + " " + lm_out + " x00", candidate_pstring),
                    # the score of each node is sum of score of all nodes
                    # on the path to itself. For beam search, there is no need to
                    # normalize according to path length because all nodes in beam
                    # is of the same path length.
                    val=prev_score + score,
                )
                # Note that the queue only maintain at most beam_size nodes
                # so this new node might possibly be dropped depending on its value.

        # replace the old queue with new queue before the new proof search depth.
        beam_queue = new_queue

    return False


def get_lm(ckpt_init: str, device: str) -> "Decoder":
    decoder = torch.load(ckpt_init)
    decoder.to(device)
    return decoder


def get_tokenizer(vocab_path: str) -> spm.SentencePieceProcessor:
    tokenizer = spm.SentencePieceProcessor(vocab_path)
    return tokenizer


class BeamQueue:
    """Keep only the top k objects according to their values."""

    def __init__(self, max_size: int = 512):
        self.queue = []
        self.max_size = max_size

    def add(self, node: object, val: float) -> None:
        """Add a new node to this queue."""

        if len(self.queue) < self.max_size:
            self.queue.append((val, node))
            return

        # Find the minimum node:
        min_idx, (min_val, _) = min(enumerate(self.queue), key=lambda x: x[1])

        # replace it if the new node has higher value.
        if val > min_val:
            self.queue[min_idx] = (val, node)

    def __iter__(self):
        for val, node in self.queue:
            yield val, node

    def __len__(self) -> int:
        return len(self.queue)


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


def run_ddar(
    proof: gh.Graph, problem: pr.Problem, rules: pr.Theorem, out_file: str
) -> bool:
    """Run DD+AR.

    Args:
      g: gh.Graph object, containing the proof state.
      p: pr.Problem object, containing the problem statement.
      out_file: path to output file if solution is found.

    Returns:
      Boolean, whether DD+AR finishes successfully.
    """
    ddar.solve(proof, rules, problem, max_level=1000)

    goal_args = proof.names2nodes(problem.goal.args)
    if not proof.check(problem.goal.name, goal_args):
        logging.info("DD+AR failed to solve the problem.")
        return False

    write_solution(proof, problem, out_file)

    gh.nm.draw(
        proof.type2nodes[gh.Point],
        proof.type2nodes[gh.Line],
        proof.type2nodes[gh.Circle],
        proof.type2nodes[gh.Segment],
    )
    return True


def natural_language_statement(logical_statement: pr.Dependency) -> str:
    """Convert logical_statement to natural language.

    Args:
      logical_statement: pr.Dependency with .name and .args

    Returns:
      a string of (pseudo) natural language of the predicate for human reader.
    """
    names = [a.name.upper() for a in logical_statement.args]
    names = [(n[0] + "_" + n[1:]) if len(n) > 1 else n for n in names]
    return pt.pretty_nl(logical_statement.name, names)


def proof_step_string(
    proof_step: pr.Dependency, refs: dict[tuple[str, ...], int], last_step: bool
) -> str:
    """Translate proof to natural language.

    Args:
      proof_step: pr.Dependency with .name and .args
      refs: dict(hash: int) to keep track of derived predicates
      last_step: boolean to keep track whether this is the last step.

    Returns:
      a string of (pseudo) natural language of the proof step for human reader.
    """
    premises, [conclusion] = proof_step

    premises_nl = " & ".join(
        [
            natural_language_statement(p) + " [{:02}]".format(refs[p.hashed()])
            for p in premises
        ]
    )

    if not premises:
        premises_nl = "similarly"

    refs[conclusion.hashed()] = len(refs)

    conclusion_nl = natural_language_statement(conclusion)
    if not last_step:
        conclusion_nl += " [{:02}]".format(refs[conclusion.hashed()])

    return f"{premises_nl} \u21d2 {conclusion_nl}"


def write_solution(g: gh.Graph, p: pr.Problem, out_file: str) -> None:
    """Output the solution to out_file.

    Args:
      g: gh.Graph object, containing the proof state.
      p: pr.Problem object, containing the theorem.
      out_file: file to write to, empty string to skip writing to file.
    """
    setup, aux, proof_steps, refs = ddar.get_proof_steps(
        g, p.goal, merge_trivials=False
    )

    solution = "\n=========================="
    solution += "\n * From theorem premises:\n"
    premises_nl = []
    for premises, [points] in setup:
        solution += " ".join([p.name.upper() for p in points]) + " "
        if not premises:
            continue
        premises_nl += [
            natural_language_statement(p) + " [{:02}]".format(refs[p.hashed()])
            for p in premises
        ]
    solution += ": Points\n" + "\n".join(premises_nl)

    solution += "\n\n * Auxiliary Constructions:\n"
    aux_premises_nl = []
    for premises, [points] in aux:
        solution += " ".join([p.name.upper() for p in points]) + " "
        aux_premises_nl += [
            natural_language_statement(p) + " [{:02}]".format(refs[p.hashed()])
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
        nl = proof_step_string(step, refs, last_step=i == len(proof_steps) - 1)
        rule_name = r2name.get(con.rule_name, "")
        nl = nl.replace("\u21d2", f"{rule_name}\u21d2 ")
        solution += "{:03}. ".format(i + 1) + nl + "\n"

    solution += "==========================\n"
    logging.info(solution)
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(solution)
        logging.info("Solution written to %s.", out_file)
