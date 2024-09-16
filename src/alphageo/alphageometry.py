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
from copy import deepcopy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator


import sentencepiece as spm
import os

from alphageo.model import Decoder
from alphageo.translate import (
    setup_str_from_problem,
    try_translate_constrained_to_construct,
)
from alphageo.inference import priority_beam_search as beam_search
from geosolver.proof import ConstructionError

from torch import load, LongTensor  # type: ignore

if TYPE_CHECKING:
    from geosolver import GeometricSolverBuilder, GeometricSolver
    from geosolver.formulations.problem import ProblemJGEX


def run_alphageometry(
    builder: "GeometricSolverBuilder",
    problem: "ProblemJGEX",
    model: "Decoder",
    tokenizer: "spm.SentencePieceProcessor",
    device: str,
    model_beam_width: int,
    model_num_return_sequences: int,
    search_depth: int,
    beam_size: int,
) -> tuple[GeometricSolver, list[ProblemJGEX], list[float]]:
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
    problems = [problem]
    problem_scores = [0.0]

    # First we run the symbolic engine DD+AR:
    solver = deepcopy(builder).load_problem(problem).build()
    success = solver.run()
    if success:
        return (solver, problems)

    # translate the problem to a string of grammar that the LM is trained on.
    string = setup_str_from_problem(problem, builder.defs)
    # special tokens prompting the LM to generate auxiliary points.
    string += " {F1} x00"

    # beam search for the proof
    # each node in the search tree is a 3-tuple:
    # (<graph representation of proof state>,
    #  <string for LM to decode from>,
    #  <original problem string>)
    beam_queue = BeamQueue(max_size=beam_size)
    # originally the beam search tree starts with a single node (a 3-tuple):
    beam_queue.add(
        val=0.0,  # value of the root node is simply 0.
        data=(string, problem),
    )

    for depth in range(search_depth):
        print(f"Depth {depth}. There are {len(beam_queue)} nodes to expand:")
        # logging.info(f"Depth {depth}. There are {len(beam_queue)} nodes to expand:")
        for _, (string, problem) in beam_queue:
            print(f"problem : {str(problem)} -- {string}")
            # logging.info(f"problem : {str(problem)} -- {string}")

        new_queue = BeamQueue(max_size=beam_size)

        for prev_score, (string, problem) in beam_queue:
            logging.info("Decoding from %s", string)
            tokens = tokenizer.encode_as_ids(string)
            inp = LongTensor([tokens]).to(device)
            outs = beam_search(
                model,
                inp,
                beam_width=model_beam_width,
                num_return_sequences=model_num_return_sequences,
            )
            seqs_str = [
                tokenizer.decode_ids(o[0][len(tokens) :].tolist()).strip()  # type: ignore
                for o in outs
            ]
            scores = [o[1] for o in outs]

            for i, out_string in enumerate(seqs_str):
                # logging.info(f"LM output {i+1}: {out_string} (score: {scores[i]})")
                print(f"LM output {i+1}: {out_string} (score: {scores[i]})")
            # outputs = model.beam_decode(string, eos_tokens=[';'])

            # translate lm output to the constructive language.
            # so that we can update the graph representing proof states:

            # bring the highest scoring candidate first
            # candidates = reversed(list(candidates))

            for lm_out, score in zip(seqs_str, scores):
                print('Trying LM output (score=%f): "%s"', score, lm_out)
                # logging.info('Trying LM output (score=%f): "%s"', score, lm_out)

                aux_string = try_translate_constrained_to_construct(
                    lm_out, problem.points()
                )

                if aux_string.startswith("ERROR:"):
                    # the construction is invalid.
                    logging.warning('Could not translate lm output: "%s"\n', aux_string)
                    continue
                logging.info('Translation: "%s"\n', aux_string)

                new_problem = problem.with_more_construction(aux_string)
                try:
                    logging.info(
                        'Try to build and solve: (%d points) "%s"\n',
                        len(new_problem.points()),
                        new_problem,
                    )
                    solver = deepcopy(builder).load_problem(new_problem).build()
                    problems.append(new_problem)
                    problem_scores.append(score)

                except ConstructionError as e:
                    logging.info('ConstructionError: "%s"\n', str(e))
                    continue
                except Exception as e:
                    logging.info('Exception: "%s"\n', str(e))
                    continue
                success = solver.run()
                if success:
                    return (solver, problems, problem_scores)

                # Add the candidate to the beam queue.
                new_queue.add(
                    # The string for the new node is old_string + lm output +
                    # the special token asking for a new auxiliary point ' x00':
                    data=(string + " " + lm_out + " x00", new_problem),
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

    return (solver, problems)


def get_lm(ckpt_init: Path, device: str) -> "Decoder":
    cfg = load(ckpt_init / "cfg.sav")
    decoder = Decoder(cfg)
    params = load(os.path.join(ckpt_init, "params.sav"))
    decoder.load_state_dict(params)
    decoder.to(device)
    decoder.bfloat16()
    return decoder


def get_tokenizer(vocab_path: Path) -> spm.SentencePieceProcessor:
    tokenizer = spm.SentencePieceProcessor(str(vocab_path))
    return tokenizer


class BeamQueue:
    """Keep only the top k objects according to their values."""

    def __init__(self, max_size: int = 512):
        self.queue: list[tuple[float, Any]] = []
        self.max_size = max_size

    def add(self, val: float, data: Any) -> None:
        """Add a new node to this queue."""

        if len(self.queue) < self.max_size:
            self.queue.append((val, data))
            return

        # Find the minimum node:
        min_idx, (min_val, _) = min(enumerate(self.queue), key=lambda x: x[1][0])

        # replace it if the new node has higher value.
        if val > min_val:
            self.queue[min_idx] = (val, data)

    def __iter__(self) -> Generator[tuple[float, Any], None, None]:
        for val, node in self.queue:
            yield val, node

    def __len__(self) -> int:
        return len(self.queue)
