def load_config(config_path):
    """加载 YAML 配置文件"""
    from omegaconf import OmegaConf
    with open(config_path, 'r', encoding='utf-8') as file:
        config = OmegaConf.load(file)
    return config


# @hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(args):
    config = load_config(args.config_path)
    run_ppo(config)
