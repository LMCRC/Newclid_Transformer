from __future__ import annotations
from typing import Any
from torch import Tensor, cat
from torch.nn.functional import pad

from alphageo.model import Decoder


def brevity_penalty(length : int, alpha : float = 0.6, numerator_bias : int = 5, denominator_bias : int = 6) -> float:
    """
    Calculate a brevity penalty for short sequences.
    In Language Modeling, short sequences are usually inherently more likley than longer ones because of the overall probability of a sequence is the product of probabilities for each word P(w1...wn) = P(wn) * P(wn-1) * ... * P(w1).
    A brevity penalty aims to mitigate this somewhat, by dividing each token by a factor proportional to its position in [1, inf).

    Args:
        length: Length of current sequence (i.e., position of newly generated token).
        alpha: Power factor (default: 0.6, taken from original AG).
        numerator_bias: Bias added to original length (default: 5, taken from original AG).
        denominator_bias: Divisor for `length+numerator_bias` (default: 6, taken from original AG).

    Returns:
        Scaling factor for current token probability at position `length`.
    """
    return pow((length + numerator_bias) / denominator_bias, alpha)


def priority_beam_search(
    model : Decoder, inp : Tensor, beam_width : int = 4, num_return_sequences : int = 2, eos_id : int = 263, max_new_tokens : int = 512, tokenizer : Any = None
) -> list[tuple[Tensor, float]]:
    """
    Beam search with a priority queue, designed to be close to AG's beam search at
    https://github.com/google-deepmind/alphageometry/blob/main/beam_search.py

    Some simplifications to make it straight-forward; output is consistent with original AG.

    Args:
        model: A (Decoder) language model.
        inp: Original input sequence, a LongTensor of shape (1, seq_length).
        beam_width: Search width for beam search. At each step, we take this many potential sequences, and predict the top beam_width tokens for the next timestep for each.
        num_return_sequences: The number of final decoded sequences to be returned (default: 2).
        eos_id: Token ID for End-of-Sequence; used to stop decoding for a given sequence (default: 263, from AG vocabulary).
        max_new_tokens: Limit of generation length. If no EOS token is generated within this number of steps, disregard the sequence.

    Returns:
        A list of `[(seq1, score1), ..., (seq_n, score_n)]` tuples of sequence token IDs (list of int) and final sequence log prob (float).
    """
    live_sequences : list[tuple[Tensor, float]] = [
        (inp, 0.0)
    ]  # essentially a priority queue for unfinished sequences
    finished_sequences : list[tuple[Tensor, float]] = []
    start_len = inp.shape[-1]
    batch_size = beam_width  # at each time step, we expand the top beam_width sequences

    while live_sequences:
        cur_batch = live_sequences[
            :batch_size
        ]  # these are the currently most-likely unfinished sequences
        live_sequences = live_sequences[
            batch_size:
        ]  # this is the rest... better luck the next time :)

        # break condition:
        if (
            # we have at least as many finished sequences as we want to return
            finished_sequences and len(finished_sequences) >= num_return_sequences
            # the current best unfinished sequence is *worse* than the worst finished one.
            and cur_batch[0][1] < finished_sequences[-1][1]
        ):
            break

        batch_lens = [
            item[0].shape[-1] for item in cur_batch
        ]  # length of each batch sequence; for padding and brevity penalty
        max_len = max(batch_lens)

        batch_inp = cat(  # input for model; item[0] is the sequence Tensor (item[1] is current score)
            [pad(item[0], (0, max_len - item[0].shape[-1])) for item in cur_batch]
        )

        batch_log_p : list[Tensor] = model(batch_inp).log_softmax(dim=-1)

        for b_idx, (log_p, length) in enumerate(zip(batch_log_p, batch_lens)):
            cur_inp, cur_score = cur_batch[b_idx]  # input Tensor, current score

            values, indices = log_p[length - 1].topk(
                beam_width
            )  # top beam_width next tokens for input's next position
            penalty = brevity_penalty(
                length + 1 - start_len
            )  # brevity penalty for input's next position

            for val, idx in zip(values, indices):
                new_inp = cat([cur_inp, idx.view(1, 1)], dim=-1)  # potential next input
                new_score = (cur_score) + val.item() / penalty  # potential next score

                good_score = (
                    len(finished_sequences)
                    < num_return_sequences  # we're still generating anyways
                    or new_score
                    > finished_sequences[-1][
                        1
                    ]  # new score is *better* than worst finished sequence
                )

                if idx == eos_id and good_score:
                    # EOS was generated
                    new_seq = new_inp.flatten()
                    finished_sequences.append((new_seq, new_score))
                    finished_sequences.sort(key=lambda x: x[1], reverse=True)
                    finished_sequences = finished_sequences[
                        :num_return_sequences
                    ]  # we just want this many return sequences
                elif good_score:
                    # we're not done yet but the updated score warrants further genration of this sequence.
                    if new_inp.shape[-1] - start_len < max_new_tokens:
                        # also, we're not too long yet
                        live_sequences.append((new_inp, new_score))
                        live_sequences.sort(
                            key=lambda x: x[1], reverse=True
                        )  # we use live_sequences as a priority queue, need to keep it sorted

    return finished_sequences


def simple_beam_search(
    model : Decoder, inp : Tensor, beam_width : int =4, num_return_sequences : int =2, eos_idx : int =263, max_tokens : int=128
) -> list[tuple[Tensor, float]]:
    """
    Implementation of a very straight-forward beam search. A bit simpler than priority_beam_search, but not guranteed to generate the same sequences (though it often may).
    At each time step, the input to the model will be of shape (beam_width, current_sequence_length).

    Args:
        model: A (Decoder) language model.
        inp: Original input sequence, a LongTensor of shape (1, seq_length).
        beam_width: Search width for beam search. At each step, we take this many potential sequences, and predict the top beam_width tokens for the next timestep for each.
        num_return_sequences: The number of final decoded sequences to be returned (default: 2).
        eos_id: Token ID for End-of-Sequence; used to stop decoding for a given sequence (default: 263, from AG vocabulary).
        max_tokens: Limit of generation length. If no EOS token is generated within this number of steps, disregard the sequence.

    Returns:
        A list of `[(seq1, score1), ..., (seq_n, score_n)]` tuples of sequence token IDs (list of int) and final sequence log prob (float).

    """
    ...