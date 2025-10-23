from gridfm_datakit.generate import generate_power_flow_data_distributed
import yaml
from gridfm_datakit.utils.param_handler import NestedNamespace


if __name__ == "__main__":
    yaml_path = "scripts/config/default.yaml"

    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)
    args.load.scenarios = 10
    args.settings.large_chunk_size = 10
    args.settings.num_processes = 2
    args.settings.data_dir = "debug_data"
    args.settings.mode = "secure"
    file_paths = generate_power_flow_data_distributed(args)
