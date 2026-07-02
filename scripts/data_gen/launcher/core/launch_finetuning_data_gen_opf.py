import os
import yaml
import subprocess
from math import ceil

# (grid_name, num_processes, large_chunk_size, max_iter, k, RAM, n_topology_variants
grid_settings = [
    # ("case14_ieee", 40, 1000, 120, 2, 64,20/0.8626291666666667),
    # ("case30_ieee", 40, 1000, 140, 5, 64, 20/0.7170276923076923),
    # ("case57_ieee", 40, 1000, 120, 5, 64, 20/0.529676923076923),
    # ("case118_ieee", 40, 1000, 120, 10, 64,20/0.8842925),
    # ("case500_goc", 60, 500, 260, 10, 128,20/0.9443272727272727),
    # ("case1354_pegase", 60, 500, 300, 10, 128,20/0.86567), 
    # ("case2000_goc", 100, 500, 150, 10, 128, 20/0.9636272727272728),
    ("case10000_goc", 200, 200, 320, 10, 256,20/0.9455954545454546),
]

# Path to default YAML config
default_yaml_path = "/u/apu/gridfm-datakit/scripts/config/default_opf.yaml"

# Directory to save new YAML files
output_dir = "/u/apu/gridfm-datakit/scripts/config/finetuning/opf"
os.makedirs(output_dir, exist_ok=True)

# Load default YAML once (will deep-copy for each grid)
with open(default_yaml_path, "r") as f:
    base_config = yaml.safe_load(f)

for grid_name, num_proc, chunk_size, max_iter, k, ram, n_topology_variants in grid_settings:

    # Work on a fresh copy of the default config
    config = yaml.safe_load(yaml.dump(base_config))
    
    # check it exists
    _ = config['settings']['data_dir']
    config['settings']['data_dir'] = "/dccstor/gridfm/powermodels_data/v4/finetuning/opf"

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
    config["settings"]['num_processes'] = num_proc
        
    _ = config['topology_perturbation']['n_topology_variants']
    config["topology_perturbation"]["n_topology_variants"] = ceil(n_topology_variants) +1
    

    # Output YAML path
    output_yaml_path = os.path.join(output_dir, f"{grid_name}.yaml")

    # Save updated YAML
    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(config, f)

    # --- Prepare LSF command ---
    job_name = f"{grid_name}_opf"

    cmd = (
        f'bsub -q normal -M {ram}G -n {num_proc} -J {job_name} '
        f'-o ~/.lsbatch/%J.out '
        f'"cd /u/apu/gridfm-datakit; '
        f'source venv/bin/activate; '
        f'export JULIA_CPU_TARGET=\'generic\'; '
        f'gridfm_datakit generate {output_yaml_path};"'
    )

    # Submit job
    subprocess.run(cmd, shell=True)
    print(f"Submitted job for {grid_name} using {num_proc} processes")
