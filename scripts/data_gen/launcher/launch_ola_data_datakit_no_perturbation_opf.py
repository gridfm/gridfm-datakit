import os
import yaml
import subprocess

# Training configurations
grid_settings = [
    ("case118_ieee", 64, 500, 64 ),
]

# Base training config
default_yaml_path = "/u/apu/gridfm-datakit/scripts/config/ola/ola_dk_case118_base_topology_300000.yaml"

# Directory for generated configs
output_dir = "/u/apu/gridfm-datakit/scripts/config/ola/"
os.makedirs(output_dir, exist_ok=True)

# Load base config
with open(default_yaml_path, "r") as f:
    base_config = yaml.safe_load(f)

for grid_name, num_proc, chunk_size, ram in grid_settings:

    config = yaml.safe_load(yaml.dump(base_config))

    # Modify config if needed
    config["network"]["name"] = grid_name
    config["settings"]["num_processes"] = num_proc
    config["settings"]["large_chunk_size"] = chunk_size

    output_yaml_path = os.path.join(
        output_dir,
        f"{grid_name}_train.yaml"
    )

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # LSF submission
    job_name = f"train_{grid_name}"

    cmd = (
        f'bsub -q normal '
        f'-n {num_proc} '
        f'-R "rusage[mem={ram}GB]" '
        f'-J {job_name} '
        f'-o ~/.lsbatch/%J.out '
        f'"cd /u/apu/gridfm-datakit; '
        f'source venv/bin/activate; '
        f'export JULIA_CPU_TARGET=\'generic\'; '
        f'gridfm_datakit generate {output_yaml_path};"'
    )

    subprocess.run(cmd, shell=True)
    print(f"Submitted training job: {job_name}")