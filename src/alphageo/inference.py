from alphageo.optional_imports import raise_if_called, raise_if_instanciated


try:
    from torch import cat, LongTensor
except ImportError:
    cat = raise_if_called("torch")
    LongTensor = raise_if_instanciated("torch")


def brevity_penalty(length, alpha=0.6, numerator_bias=5, denominator_bias=6):
    return pow((length + numerator_bias) / denominator_bias, alpha)


def priority_beam_search(model, inp, beam_width=4, num_return_sequences=2, eos_id=263):
    live_sequences = [(inp, 0.0)]
    finished_sequences = []
    start_len = inp.shape[-1]

    while live_sequences:
        cur_inp, cur_score = live_sequences.pop()

        if (
            finished_sequences
            and len(finished_sequences) >= num_return_sequences
            and cur_score < finished_sequences[-1][1]
        ):
            break

        log_p = model(cur_inp)[:, -1].log_softmax(dim=-1)
        values, indices = log_p.topk(beam_width)

        penalty = brevity_penalty(cur_inp.shape[-1] + 1 - start_len)

        for val, idx in zip(values.T, indices.T):
            new_inp = cat([cur_inp, idx.unsqueeze(0)], dim=-1)
            new_score = (cur_score) + val.item() / penalty

            if idx == eos_id:
                finished_sequences.append((new_inp.flatten().tolist(), new_score))
                finished_sequences.sort(key=lambda x: x[1], reverse=True)
                finished_sequences = finished_sequences[:num_return_sequences]
            else:
                live_sequences.append((new_inp, new_score))
                live_sequences.sort(key=lambda x: x[1])

    return finished_sequences


def simple_beam_search(
    model, inp, beam_width=4, num_return_sequences=2, eos_idx=263, max_tokens=128
):
    inp = inp.tile(beam_width, 1)
    scores = [0 for _ in range(beam_width)]

    done_seqs = []
    done_scores = []

    num_new_tokens = 0

    while num_new_tokens < max_tokens and (
        max(scores) >= max(done_scores, default=-1_000_000)
        or len(done_seqs) < num_return_sequences
    ):
        next_scores = model(inp)[:, -1].log_softmax(dim=-1)
        next_candidates = next_scores.sort(dim=-1, descending=True).indices[
            :, :beam_width
        ]
        beam_items = []
        for idx, (cur_seq, cands) in enumerate(zip(inp, next_candidates)):
            for cand in cands:
                new_seq = cat([cur_seq, cand.unsqueeze(0)])
                new_score = scores[idx] + next_scores[idx][cand]
                if cand == eos_idx:
                    if (seq_list := new_seq.tolist()) not in done_seqs:
                        done_seqs.append(seq_list)
                        done_scores.append(new_score.item())
                    else:
                        seq_idx = done_seqs.index(seq_list)
                        done_scores[seq_idx] = max(done_scores[seq_idx], new_score)
                else:
                    beam_items.append((new_seq, new_score))

        beam_items.sort(key=lambda x: x[1], reverse=True)
        new_inp = [beam_items[0][0].tolist()]
        new_scores = [beam_items[0][1].item()]
        for item in beam_items[1:]:
            if (x := item[0].tolist()) not in new_inp:
                new_inp.append(x)
                new_scores.append(item[1].item())
            else:
                new_idx = new_inp.index(x)
                new_scores[new_idx] = max(new_scores[new_idx], item[1].item())

        new_inp = new_inp[:beam_width]
        inp = LongTensor(new_inp).to(inp.device)
        scores = new_scores[: inp.shape[0]]
        num_new_tokens += 1

    return sorted(zip(done_seqs, done_scores), key=lambda x: x[1], reverse=True)[
        :num_return_sequences
    ]
