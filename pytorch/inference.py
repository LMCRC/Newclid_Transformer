import torch
import math


def simple_beam_search(model, inp, beam_width=2, num_return_sequences=2, eos_idx=263, max_tokens=128):
    inp = inp.tile(beam_width, 1)
    scores = [0 for _ in range(beam_width)]

    done_seqs = []
    done_scores = []

    while inp.shape[-1] < max_tokens and (max(scores) >= max(done_scores, default=-1_000_000) or len(done_seqs) < num_return_sequences):
        next_scores = model(inp)[:,-1].log_softmax(dim=-1)
        next_candidates = next_scores.sort(dim=-1, descending=True).indices[:,:beam_width]
        beam_items = []
        for idx, (cur_seq, cands) in enumerate(zip(inp, next_candidates)):
            for cand in cands:
                new_seq = torch.cat([cur_seq, cand.unsqueeze(0)])
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
            if not (x := item[0].tolist()) in new_inp:
                new_inp.append(x)
                new_scores.append(item[1].item())
            else:
                new_idx = new_inp.index(x)
                new_scores[new_idx] = max(new_scores[new_idx], item[1].item())

        new_inp = new_inp[:beam_width]
        inp = torch.LongTensor(new_inp)
        scores = new_scores[:inp.shape[0]]

    return sorted(zip(done_seqs, done_scores), key=lambda x: x[1], reverse=True)[:num_return_sequences]
