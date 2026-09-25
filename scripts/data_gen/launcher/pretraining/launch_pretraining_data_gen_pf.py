import os
import yaml
import subprocess
from glob import glob
import copy

# Path to default YAML config
default_yaml_path = "/u/apu/gridfm-datakit/scripts/config/default_pf.yaml"

# Directory to save new YAML files
output_dir = "/u/apu/gridfm-datakit/scripts/config/pretraining/pf"
os.makedirs(output_dir, exist_ok=True)

# Load default YAML once (read-only template)
with open(default_yaml_path, "r") as f:
    config = yaml.safe_load(f)

# Get all grid names
grid_files = glob("/dccstor/gridfm/powermodels_data/v4/subnets_casefiles/*.m")
grid_names = [os.path.basename(f).replace(".m", "") for f in grid_files]

print(f"Found {len(grid_names)} grid files to process")

# Base config for this run (copy once)
_ = config["network"]["source"]
config["network"]["source"] = "file"
_ = config["network"]["network_dir"]
config["network"]["network_dir"] = "/dccstor/gridfm/powermodels_data/v4/subnets_casefiles/"
_ = config["settings"]["num_processes"]
config["settings"]["num_processes"] = 20
_ = config["settings"]["data_dir"]
config["settings"]["data_dir"] = "/dccstor/gridfm/powermodels_data/v4/pretraining/pf/"
_ = config["settings"]["large_chunk_size"]
config["settings"]["large_chunk_size"] = 2000
_ = config["topology_perturbation"]["n_topology_variants"]
config["topology_perturbation"]["n_topology_variants"] = 1

for grid_name in grid_names:
    # Per-job isolated config
    new_config = copy.deepcopy(config)
    new_config["network"]["name"] = grid_name
    
    # Output YAML path
    output_yaml_path = os.path.join(output_dir, f"{grid_name}.yaml")
    
    # Save updated YAML
    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(new_config, f)
    
    # Prepare LSF command with better quoting
    job_name = f"{grid_name}_spf"
    cmd = [
        'bsub', '-q', 'normal', '-M', '16G', '-n', '20',
        '-J', job_name,
        '-o', f'~/.lsbatch/%J.out',
        'bash', '-c',
        f"cd /u/apu/gridfm-datakit && "
        f"source venv/bin/activate && "
        f"export JULIA_CPU_TARGET='generic' && "
        f"gridfm_datakit generate {output_yaml_path}"
    ]
    
    # Submit job with better error handling
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Submitted job for {grid_name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit job for {grid_name}: {e.stderr}")
    except Exception as e:
        print(f"✗ Error processing {grid_name}: {str(e)}")

print(f"\nFinished processing all {len(grid_names)} grids")