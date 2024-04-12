from get_model import *
from pytorch.model import *
import os
torch.set_grad_enabled(False)


def convert():
    _DATA_PATH, _MELIAD_PATH, _OUT_FOLDER = setup()
    ag_model = get_model(_DATA_PATH, _MELIAD_PATH)

    cfg = {"vocab_size": 1024, "embedding_dim": 1024, "num_layers": 12, "num_heads": 8, "mlp_num_layers": 2, "mlp_hidden_dim": 4096, "t5_num_buckets": 32, "t5_max_distance": 128}
    pt_model = Decoder(cfg)

    ag_state = ag_model.tstate.optimizer.target["decoder"]


    print("converting embedding")
    _ = pt_model.embedding.weight.data.copy_(torch.Tensor(ag_state["embed"]["embedding"]))

    for trans_layer_n in range(len(pt_model.layers)):
        print(f"converting layer {trans_layer_n}")
        print(f"\trelative attention")
        _ = pt_model.layers[trans_layer_n].relative_positions.relative_attention_bias.weight.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["relative_positions"]["rel_embedding"]).t())

        print(f"\tqueries layer")
        _ = pt_model.layers[trans_layer_n].qkv.queries_layer.weight.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["tbase"]["_kvq"]["queries_layer"]["kernel"]).t())
        print(f"\tkeys layer")
        _ = pt_model.layers[trans_layer_n].qkv.keys_layer.weight.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["tbase"]["_kvq"]["keys_layer"]["kernel"]).t())
        print(f"\tvalues layer")
        _ = pt_model.layers[trans_layer_n].qkv.values_layer.weight.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["tbase"]["_kvq"]["values_layer"]["kernel"]).t())

        print(f"\tpre attention layernorm")
        _ = pt_model.layers[trans_layer_n].qkv.pre_attn_layernorm.weight.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["tbase"]["_kvq"]["pre_attn_layernorm"]["scale"]))

        print(f"\tattention scale factors")
        _ = pt_model.layers[trans_layer_n].attention_scale_factors.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["tbase"]["_kvq"]["attention_scale"]))

        print(f"\tpre ffn layernorm")
        _ = pt_model.layers[trans_layer_n].pre_ffn_layernorm.weight.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["tbase"]["pre_ffn_layernorm"]["scale"]).t())

        print(f"\tffn")
        for ffn_layer_n in range(len(pt_model.layers[trans_layer_n].ffn.layers)-1):
            print(f"\t\tlayer {ffn_layer_n}")
            _ = pt_model.layers[trans_layer_n].ffn.layers[ffn_layer_n][0].weight.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["tbase"]["ffn"][f"hidden{ffn_layer_n}"]["kernel"]).t())

        print(f"\t\tfinal layer")
        _ = pt_model.layers[trans_layer_n].ffn.layers[-1].weight.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["tbase"]["ffn"]["output_layer"]["kernel"]).t())


        print(f"\tpost attention MLP")
        _ = pt_model.layers[trans_layer_n].post_attn_mlp.weight.data.copy_(torch.Tensor(ag_state[f"transformer{trans_layer_n}"]["tbase"]["post_attn_mlp"]["output_layer"]["kernel"]).t())

    print("final layer norm")
    _ = pt_model.final_layernorm.weight.data.copy_(torch.Tensor(ag_state["final_layernorm"]["scale"]))

    print("\nsaving model...", end="")
    os.makedirs(_OUT_FOLDER.value, exist_ok=True)
    torch.save(cfg, os.path.join(_OUT_FOLDER.value, "cfg.sav"))
    torch.save(pt_model.state_dict(), os.path.join(_OUT_FOLDER.value, "params.sav"))
    print("done")


if __name__ == "__main__":
    import sys
    convert()
