import os
import yaml
import subprocess
from math import ceil

# PF dataset: (grid_name, num_processes, large_chunk_size, max_iter, k, RAM, conv_rate, pf_fast)
grid_settings = [
    ("case14_ieee",     40, 1000, 50, 1, 64, 0.998135,  True),
    ("case30_ieee",     40, 1000, 50, 1, 64, 0.9884975, True),
    ("case57_ieee",     40, 1000, 100, 1, 64, 0.984705,  True),
    ("case118_ieee",    40, 1000, 100, 1, 64, 0.9954975, True),
    ("case500_goc",     40, 500,  120, 1, 128, 0.98, True),
    ("case1354_pegase", 60, 500,  120, 1, 128, 0.90526,   True),
    ("case2000_goc",    60, 500,  120, 1, 128, 0.9853825, False),
    # ("case10000_goc",   100, 100, 200, 1, 256, 0.9993975, False),
]

# Path to default YAML config
default_yaml_path = "/u/apu/gridfm-datakit/scripts/config/default_pf.yaml"

# Directory to save new YAML files
output_dir = "/u/apu/gridfm-datakit/scripts/config/finetuning/pf"
os.makedirs(output_dir, exist_ok=True)

# Load default YAML once
with open(default_yaml_path, "r") as f:
    base_config = yaml.safe_load(f)

for grid_name, num_proc, chunk_size, max_iter, k, ram, conv_rate, pf_fast in grid_settings:

    # Deep copy default config
    config = yaml.safe_load(yaml.dump(base_config))
    
    config['settings']['data_dir'] = "/dccstor/gridfm/powermodels_data/v4/finetuning/pf"

    # --- Update YAML fields ---
    _ = config['network']['name']
    config["network"]["name"] = grid_name
    
    _ = config['settings']['large_chunk_size']
    config["settings"]["large_chunk_size"] = chunk_size
    
    _ = config['settings']['max_iter']
    config["settings"]["max_iter"] = max_iter
    
    _ = config['topology_perturbation']['k']
    config["topology_perturbation"]["k"] = k
    
    _ = config['settings']['num_processes']
    config["settings"]["num_processes"] = num_proc
    
    _ = config['settings']['pf_fast']
    config["settings"]["pf_fast"] = pf_fast
    
    _ = config['topology_perturbation']['n_topology_variants']
    config["topology_perturbation"]["n_topology_variants"] = ceil(20 / conv_rate) + 1
    
    # Output YAML path
    output_yaml_path = os.path.join(output_dir, f"{grid_name}.yaml")

    # Save updated YAML
    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(config, f)

    # --- Prepare LSF command ---
    job_name = f"{grid_name}_pf"

    cmd = (
        f'w -q normal -M {ram}G -n {num_proc} -J {job_name} '
        f'-o ~/.lsbatch/%J.out '
        f'"cd /u/apu/gridfm-datakit; '
        f'source venv/bin/activate; '
        f'export JULIA_CPU_TARGET=\'generic\'; '
        f'gridfm_datakit generate {output_yaml_path};"'
    )

    # Submit job
    subprocess.run(cmd, shell=True)
    print(f"Submitted job for {grid_name} using {num_proc} processes")
