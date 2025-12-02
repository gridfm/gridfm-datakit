import yaml
import os
import shutil
from gridfm_datakit.utils.param_handler import NestedNamespace
from gridfm_datakit.generate import generate_power_flow_data_distributed
from concurrent.futures import ProcessPoolExecutor
from gridfm_datakit.validation import validate_generated_data


def run_generation(config_params):
    """Run generation with specific perturbation configuration."""
    try:
        # Load base config
        with open("tests/config/default_pf_mode_test.yaml") as f:
            config_dict = yaml.safe_load(f)

        args = NestedNamespace(**config_dict)

        # Apply perturbation settings
        args.topology_perturbation.type = config_params["topology_type"]
        args.generation_perturbation.type = config_params["generation_type"]
        args.admittance_perturbation.type = config_params["admittance_type"]
        args.settings.mode = config_params["mode"]

        # Set test parameters
        args.load.scenarios = 5
        args.settings.large_chunk_size = 5
        args.settings.num_processes = 2
        args.settings.data_dir = (
            f"./tests/test_data_perturbations/{config_params['test_name']}"
        )

        # Generate data
        file_paths = generate_power_flow_data_distributed(args)
        validate_generated_data(file_paths, args.settings.mode, 100.0, n_partitions=10)

        # Clean up
        if os.path.exists(args.settings.data_dir):
            shutil.rmtree(args.settings.data_dir)

        return config_params["test_name"], "OK"

    except Exception as e:
        return config_params["test_name"], f"FAIL: {e}"


def test_all_perturbation_combinations():
    """Test all possible combinations of perturbation types."""

    # Define all possible perturbation types
    load_generators = ["agg_load_profile", "powergraph"]
    topology_types = ["none", "n_minus_k", "random"]
    generation_types = ["none", "cost_permutation", "cost_perturbation"]
    admittance_types = ["none", "random_perturbation"]

    base_load_generator = "agg_load_profile"
    base_topology_type = "random"
    base_generation_type = "cost_permutation"
    base_admittance_type = "random_perturbation"

    # Create all combinations
    test_configs = []

    for mode in ["opf", "pf"]:
        for load_generator in load_generators:
            if load_generator != base_load_generator:
                test_configs.append(
                    {
                        "test_name": f"load_{load_generator}_topo_{base_topology_type}_gen_{base_generation_type}_adm_{base_admittance_type}_mode_{mode}",
                        "load_generator": load_generator,
                        "topology_type": base_topology_type,
                        "generation_type": base_generation_type,
                        "admittance_type": base_admittance_type,
                        "mode": mode,
                    },
                )

        for topo in topology_types:
            if topo != base_topology_type:
                if topo == "n_minus_k" and mode == "opf":
                    continue  # this one is too slow
                # only change topology_type
                test_configs.append(
                    {
                        "test_name": f"load_{base_load_generator}_topo_{topo}_gen_{base_generation_type}_adm_{base_admittance_type}_mode_{mode}",
                        "load_generator": base_load_generator,
                        "topology_type": topo,
                        "generation_type": base_generation_type,
                        "admittance_type": base_admittance_type,
                        "mode": mode,
                    },
                )

        for gen in generation_types:
            if gen != base_generation_type:
                # only change generation_type
                test_configs.append(
                    {
                        "test_name": f"load_{base_load_generator}_topo_{base_topology_type}_gen_{gen}_adm_{base_admittance_type}_mode_{mode}",
                        "load_generator": base_load_generator,
                        "topology_type": base_topology_type,
                        "generation_type": gen,
                        "admittance_type": base_admittance_type,
                        "mode": mode,
                    },
                )

        for adm in admittance_types:
            if adm != base_admittance_type:
                # only change admittance_type
                test_configs.append(
                    {
                        "test_name": f"load_{base_load_generator}_topo_{base_topology_type}_gen_{base_generation_type}_adm_{adm}_mode_{mode}",
                        "load_generator": base_load_generator,
                        "topology_type": base_topology_type,
                        "generation_type": base_generation_type,
                        "admittance_type": adm,
                        "mode": mode,
                    },
                )

        # add base configuration
        test_configs.append(
            {
                "test_name": f"load_{base_load_generator}_topo_{base_topology_type}_gen_{base_generation_type}_adm_{base_admittance_type}_mode_{mode}",
                "load_generator": base_load_generator,
                "topology_type": base_topology_type,
                "generation_type": base_generation_type,
                "admittance_type": base_admittance_type,
                "mode": mode,
            },
        )

    print(f"Testing {len(test_configs)} perturbation combinations...")

    # Run tests in parallel
    with ProcessPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(run_generation, test_configs))

    # Check results
    failed_tests = []
    for test_name, status in results:
        print(f"{test_name}: {status}")
        if status != "OK":
            failed_tests.append((test_name, status))

    # Clean up any remaining test data
    test_dir = "./tests/test_data_perturbations"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Assert all tests passed
    if failed_tests:
        print(f"\nFailed tests: {len(failed_tests)}")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        assert False, f"{len(failed_tests)} perturbation combinations failed"
