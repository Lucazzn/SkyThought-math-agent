set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

nproc_per_node=8
CONFIG_PATH="/.../grpo_trainer.yaml"

python3 -m verl.trainer.main_ppo \
    --config_path=$CONFIG_PATH
