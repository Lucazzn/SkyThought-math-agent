# @hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(args):
    config = load_config(args.config_path)
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
                                           mesh_dim_names=('dp', 'sp'))
    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load and process YAML configuration.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    main(args)



# @hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
# def main(config):
#     local_rank, rank, world_size = initialize_global_process_group()

#     device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
#     dp_size = world_size // config.ulysses_sequence_parallel_size
#     ulysses_device_mesh = init_device_mesh(device_type='cuda',
#                                            mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
#                                            mesh_dim_names=('dp', 'sp'))
#     trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
#     trainer.fit()


# if __name__ == '__main__':
#     main()
