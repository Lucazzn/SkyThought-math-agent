#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob
from collections import defaultdict


def main():
    """
    GRPO过程结束后, verl保存的checkpoint并不是huggingface上的那种形式,
    而是连同优化器状态也一并保存,
    为了还原成huggingface中的形式,可以使用如下代码:
    """
    step = "50"
    fsdp_checkpoint_path = f"/.../global_step_{step}/actor"
    huggingface_model_path = f"/.../global_step_{step}/actor/huggingface"
    output_path = f"/.../huggingface_checkpoint/checkpoint_global_step_{step}"
    state_dict = defaultdict(list)

    world_size = 8   # 8卡
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
