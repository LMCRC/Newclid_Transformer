from alphageo.optional_imports import raise_if_called, raise_if_instanciated


try:
    from torch import cat, LongTensor
    from torch.nn.functional import pad
except ImportError:
    cat = raise_if_called("torch")
    LongTensor = raise_if_instanciated("torch")
    pad = raise_if_called("pad")


def decode(model, inp):
    """
    Get the log probability over the next token, given a model and input sequence.

    Args:
        model: A (Decoder) Language Model.
        inp: Input sequence, a LongTensor of shape (batch_size, sequence_length).

    Returns:
        A FloatTensor of shape (batch_size, sequence_length): log prob per token.
    """
    return model(inp).log_softmax(dim=-1)


def brevity_penalty(length, alpha=0.6, numerator_bias=5, denominator_bias=6):
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
    model, inp, beam_width=4, num_return_sequences=2, eos_id=263, max_new_tokens=512
):
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
    live_sequences = [
        (inp, 0.0)
    ]  # essentially a priority queue for unfinished sequences
    finished_sequences = []
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
            finished_sequences
            and len(finished_sequences)
            >= num_return_sequences  # we have at least as many finished sequences as we want to return
            and cur_batch[0][1]
            < finished_sequences[-1][
                1
            ]  # the current best unfinished sequence is *worse* than the worst finished one.
        ):
            break

        batch_lens = [
            item[0].shape[-1] for item in cur_batch
        ]  # length of each batch sequence; for padding and brevity penalty
        max_len = max(batch_lens)

        batch_inp = cat(  # input for model; item[0] is the sequence Tensor (item[1] is current score)
            [pad(item[0], (0, max_len - item[0].shape[-1])) for item in cur_batch]
        )

        batch_log_p = model(batch_inp).log_softmax(dim=-1)

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
                    new_seq = new_inp.flatten().tolist()
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
    model, inp, beam_width=4, num_return_sequences=2, eos_idx=263, max_tokens=128
):
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

    inp = inp.tile(beam_width, 1)  # first iteration, all sequences start the same
    scores = [0 for _ in range(beam_width)]  # keeping track of scores

    done_seqs = []  # finished sequences
    done_scores = []  # and scores

    num_new_tokens = 0

    while num_new_tokens < max_tokens and (  # we're not too long
        max(scores)
        >= min(
            done_scores, default=-1_000_000
        )  # best live score is better than worst finished
        or len(done_seqs) < num_return_sequences  # we don't have enough sequences yet
    ):
        next_scores = model(inp)[:, -1].log_softmax(dim=-1)  # log probs for next tokens
        next_candidates = next_scores.sort(dim=-1, descending=True).indices[
            :, :beam_width
        ]
        beam_items = []
        for idx, (cur_seq, cands) in enumerate(
            zip(inp, next_candidates)
        ):  # beam_width top next tokens per current input sequence
            for (
                cand
            ) in cands:  # next potential token for one of the current input sequences
                new_seq = cat([cur_seq, cand.unsqueeze(0)])
                new_score = scores[idx] + next_scores[idx][cand]
                if cand == eos_idx:
                    # done generating this sequence!
                    if (seq_list := new_seq.tolist()) not in done_seqs:
                        # this is a new sequence
                        done_seqs.append(seq_list)
                        done_scores.append(new_score.item())
                    else:
                        # we have generated this same sequence via a different decoding pass; keep the more likely one
                        seq_idx = done_seqs.index(seq_list)
                        done_scores[seq_idx] = max(done_scores[seq_idx], new_score)
                else:
                    beam_items.append((new_seq, new_score))

        beam_items.sort(key=lambda x: x[1], reverse=True)  # sort by score

        # build the beam input for next iteration
        new_inp = [beam_items[0][0].tolist()]
        new_scores = [beam_items[0][1].item()]
        for item in beam_items[1:]:
            if (x := item[0].tolist()) not in new_inp:
                # different new inputs
                new_inp.append(x)
                new_scores.append(item[1].item())
            else:
                # the same input sequence is already in the beam... keep the more likely one
                new_idx = new_inp.index(x)
                new_scores[new_idx] = max(new_scores[new_idx], item[1].item())

        # prune new input to beam_width
        new_inp = new_inp[:beam_width]
        inp = LongTensor(new_inp).to(inp.device)
        scores = new_scores[: inp.shape[0]]
        num_new_tokens += 1

    return sorted(zip(done_seqs, done_scores), key=lambda x: x[1], reverse=True)[
        :num_return_sequences
    ]
