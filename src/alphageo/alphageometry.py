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
from pathlib import Path
from typing import TYPE_CHECKING


import sentencepiece as spm
import torch

from alphageo.geosolver_facade import GeometricSolver
from alphageo.translate import try_translate_constrained_to_construct
from alphageo.inference import simple_beam_search

if TYPE_CHECKING:
    from pytorch.model import Decoder


def run_alphageometry(
    solver: "GeometricSolver",
    model: "Decoder",
    tokenizer: spm.SentencePieceProcessor,
    device: str,
    model_beam_width: int,
    model_num_return_sequences: int,
    search_depth: int,
    beam_size: int,
    out_folder: Path,
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
    # First we run the symbolic engine DD+AR:

    success = solver.run_solver()
    if success:
        solver.write_solution(out_folder / "proof_steps.txt")
        solver.draw_figure(out_folder / "proof_figure.png")
        return True

    # translate the problem to a string of grammar that the LM is trained on.
    string = solver.get_setup_string()
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
        node=(solver.get_proof_state(), string, solver.get_problem_string()),
        val=0.0,  # value of the root node is simply 0.
    )

    for depth in range(search_depth):
        logging.info("Depth %s. There are %i nodes to expand:", depth, len(beam_queue))
        for _, (_, string, _) in beam_queue:
            logging.info(string)

        new_queue = BeamQueue(max_size=beam_size)  # to replace beam_queue.

        for prev_score, (proof_state, string, pstring) in beam_queue:
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

            # couple the lm outputs with its translations
            candidates = zip(outputs["seqs_str"], outputs["scores"])

            # bring the highest scoring candidate first
            # candidates = reversed(list(candidates))

            for lm_out, score in candidates:
                solver.load_state(proof_state)
                solver.load_pstring(pstring)

                logging.info('LM output (score=%f): "%s"', score, lm_out)

                aux_string = try_translate_constrained_to_construct(
                    lm_out, solver.get_existing_points(), solver.get_defs()
                )
                aux_string = solver.validate_clause_txt(aux_string)

                if aux_string.startswith("ERROR:"):
                    # the construction is invalid.
                    logging.warning('Could not translate lm output: "%s"\n', aux_string)
                    continue
                logging.info('Translation: "%s"\n', aux_string)

                solver.add_auxiliary_construction(aux_string)
                success = solver.run_solver()
                if success:
                    solver.write_solution(out_folder / "proof_steps.txt")
                    solver.draw_figure(out_folder / "proof_figure.png")
                    return True

                # Add the candidate to the beam queue.
                new_queue.add(
                    # The string for the new node is old_string + lm output +
                    # the special token asking for a new auxiliary point ' x00':
                    node=(
                        solver.get_proof_state(),
                        string + " " + lm_out + " x00",
                        solver.get_problem_string(),
                    ),
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
