import os
import yaml
import subprocess
import copy

# PF dataset:
# (dataset_name, num_processes, large_chunk_size, max_iter,
#  k, n_topology_variants, RAM_in_GB, pf_fast)

grid_settings = [
    # ("training_dataset", 40, 200, 120, 2, 20, 256, False),
]

# Add test datasets
for k_test in range(2, 3):
    grid_settings.append(
        (f"test_{k_test}_lines_only", 40, 200, 200, k_test, 1, 256, False)
    )

# Path to default YAML config
default_yaml_path = "/u/apu/gridfm-datakit/scripts/config/default_pf.yaml"

# Directory to save new YAML files
output_dir = "/u/apu/gridfm-datakit/scripts/config/texas_contingency_lines_only"
os.makedirs(output_dir, exist_ok=True)

# Load default YAML once
with open(default_yaml_path, "r") as f:
    base_config = yaml.safe_load(f)

for (
    name_dataset,
    num_proc,
    chunk_size,
    max_iter,
    k,
    n_topology_variants,
    ram,
    pf_fast,
) in grid_settings:

    # Deep copy config
    config = copy.deepcopy(base_config)

    # --- Update YAML fields ---
    config["settings"]["data_dir"] = (
        f"/dccstor/gridfm/powermodels_data/v4/texas_contingency/{name_dataset}"
    )

    config["settings"]["num_processes"] = num_proc
    config["settings"]["large_chunk_size"] = chunk_size
    config["settings"]["max_iter"] = max_iter
    config["settings"]["pf_fast"] = pf_fast

    config["topology_perturbation"]["k"] = k
    config["topology_perturbation"]["n_topology_variants"] = n_topology_variants
    config["topology_perturbation"]["elements"] = ["branch"]

    config["network"]["source"] = "file"
    config["network"]["name"] = (
        "Texas2k_case1_2016summerpeak"
    )

    # Output YAML path
    output_yaml_path = os.path.join(output_dir, f"{name_dataset}.yaml")

    # Save updated YAML
    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(config, f)

    # --- Prepare LSF command ---
    job_name = f"texas_{name_dataset}"

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

    # Submit job
    print(cmd)
    # subprocess.run(cmd, shell=True, check=True)

    print(f"Submitted job for {name_dataset} using {num_proc} processes")