#!/usr/bin/env python3

import ipywidgets as widgets
from IPython.display import display
import yaml
from pathlib import Path
from importlib import resources
from ipyfilechooser import FileChooser
from typing import List, Dict, Any

# Run the generation function directly
from gridfm_datakit.generate import generate_power_flow_data_distributed


def get_available_load_profiles() -> List[str]:
    """Get list of available aggregated load profiles."""
    try:
        # Get all .csv files from the load_profiles directory
        load_profiles_path = resources.files("gridfm_datakit.load_profiles")
        profiles = []

        for file in load_profiles_path.iterdir():
            if file.name.endswith(".csv") and file.name != "__init__.py":
                # Remove .csv extension to get profile name
                profile_name = file.name[:-4]
                profiles.append(profile_name)

        return sorted(profiles)
    except Exception as e:
        print(f"Warning: Could not load profiles dynamically: {e}")
        # Fallback to known profiles
        return [
            "default",
            "ercot_load_act_hr_2024_total",
            "ercot_load_act_hr_2024_coast",
            "ercot_load_act_hr_2024_east",
            "ercot_load_act_hr_2024_far_west",
            "ercot_load_act_hr_2024_north",
            "ercot_load_act_hr_2024_north_central",
            "ercot_load_act_hr_2024_south_central",
            "ercot_load_act_hr_2024_southern",
            "ercot_load_act_hr_2024_west",
        ]


def create_config() -> Dict[str, Any]:
    """Create configuration dictionary from widget values."""
    if network_source.value == "file":
        if not network_file_chooser.selected:
            raise ValueError("Please select a network file")
        file_path = Path(network_file_chooser.selected)
        name = file_path.stem
        network_dir = str(file_path.parent)
    elif network_source.value == "pglib":
        name = pglib_grid_dropdown.value
        network_dir = "none"

    config = {
        "network": {
            "name": name,
            "source": str(network_source.value),
            "network_dir": network_dir,
        },
        "load": {
            "generator": str(load_generator.value),
            "agg_profile": str(agg_profile.value),
            "scenarios": num_scenarios.value,
            "sigma": sigma.value,
            "change_reactive_power": change_reactive_power.value,
            "global_range": global_range.value,
            "max_scaling_factor": max_scaling_factor.value,
            "step_size": step_size.value,
            "start_scaling_factor": start_scaling_factor.value,
        },
        "topology_perturbation": {
            "k": k.value,
            "n_topology_variants": n_topology_variants.value,
            "type": str(perturbation_type.value),
            "elements": [str(e) for e in elements.value],
        },
        "generation_perturbation": {
            "type": gen_perturbation_type.value,
            "sigma": gen_sigma.value,
        },
        "admittance_perturbation": {
            "type": admittance_perturbation_type.value,
            "sigma": admittance_sigma.value,
        },
        "settings": {
            "num_processes": num_processes.value,
            "data_dir": str(data_dir.value),
            "large_chunk_size": large_chunk_size.value,
            "overwrite": overwrite.value,
            "mode": str(mode.value),
            "include_dc_res": include_dc_res.value,
            "pf_fast": pf_fast.value,
            "dcpf_fast": dcpf_fast.value,
            "enable_solver_logs": enable_solver_logs.value,
            "max_iter": max_iter.value,
        },
    }
    return config


def interactive_interface() -> None:
    """Main function to create and display the interactive interface."""
    # Get available load profiles
    available_profiles = get_available_load_profiles()

    # Dedicated output box
    output_box = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #bbb",
            padding="10px",
            margin="10px 0",
            height="300px",
            overflow="auto",
        ),
    )

    # Network Configuration
    global network_source, pglib_grid_dropdown, network_file_chooser

    network_source = widgets.Dropdown(
        options=[
            ("pglib - Power Grid Library (Recommended)", "pglib"),
            ("file - Custom network files (MATPOWER .m)", "file"),
        ],
        value="pglib",
        description="Network Source:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )

    # --- NEW: pglib grid dropdown ---
    # Full list of PGLib OPF cases (names without the 'pglib_opf_' prefix)
    pglib_grids = [
        "case3_lmbd",
        "case5_pjm",
        "case14_ieee",
        "case24_ieee_rts",
        "case30_as",
        "case30_ieee",
        "case39_epri",
        "case57_ieee",
        "case60_c",
        "case73_ieee_rts",
        "case89_pegase",
        "case118_ieee",
        "case162_ieee_dtc",
        "case179_goc",
        "case197_snem",
        "case200_activ",
        "case300_ieee",
        "case500_goc",
        "case588_sdet",  # Added
        "case793_goc",
        "case1354_pegase",  # Added
        "case1803_snem",
        "case1888_rte",
        "case1951_rte",
        "case2000_goc",
        "case2312_goc",
        "case2383wp_k",
        "case240_pserc",  # Note: Moved to its correct position by bus count (240 buses)
        "case2736sp_k",
        "case2737sop_k",
        "case2742_goc",
        "case2746wop_k",
        "case2746wp_k",
        "case2848_rte",
        "case2853_sdet",
        "case2868_rte",
        "case2869_pegase",
        "case3012wp_k",
        "case3022_goc",
        "case3120sp_k",
        "case3375wp_k",
        "case3970_goc",
        "case4020_goc",
        "case4601_goc",
        "case4619_goc",
        "case4661_sdet",
        "case4837_goc",
        "case4917_goc",
        "case5658_epigrids",
        "case6468_rte",
        "case6470_rte",
        "case6495_rte",
        "case6515_rte",
        "case7336_epigrids",
        "case8387_pegase",  # Added
        "case9241_pegase",
        "case9591_goc",
        "case10000_goc",  # Note: Moved from bottom to its correct position (10,000 buses)
        "case10192_epigrids",
        "case10480_goc",
        "case13659_pegase",
        "case19402_goc",
        "case20758_epigrids",
        "case24464_goc",
        "case30000_goc",  # Note: Moved from bottom to its correct position (30,000 buses)
        "case78484_epigrids",
    ]
    pglib_grid_dropdown = widgets.Dropdown(
        options=pglib_grids,
        value="case24_ieee_rts",
        description="PGLib Grid:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
    )

    # --- File chooser for network files ---
    network_file_chooser = FileChooser(
        title="Select Network File:",
        show_hidden=False,
        directory_only=False,
        filter_pattern="*.m",
    )

    def update_network_visibility(*args):
        if network_source.value == "pglib":
            pglib_grid_dropdown.layout.display = "block"
            network_file_chooser.layout.display = "none"
        else:  # file
            pglib_grid_dropdown.layout.display = "none"
            network_file_chooser.layout.display = "block"

    network_source.observe(update_network_visibility, names="value")
    update_network_visibility()

    # Load Configuration - Basic

    global load_generator, agg_profile, num_scenarios, sigma, change_reactive_power
    load_generator = widgets.Dropdown(
        options=[
            (
                "agg_load_profile - Real load profiles with scaling/noise (Recommended)",
                "agg_load_profile",
            ),
            ("powergraph", "powergraph"),
        ],
        value="agg_load_profile",
        description="Load Generator:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="650px"),
    )

    agg_profile = widgets.Dropdown(
        options=available_profiles,
        value="default",
        description="Load Profile:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )

    num_scenarios = widgets.IntSlider(
        value=10000,
        min=10,
        max=10000,
        step=10,
        description="Load Scenarios:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format="d",
    )

    sigma = widgets.FloatSlider(
        value=0.2,
        min=0.0,
        max=0.3,
        step=0.01,
        description="Local Noise (σ):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format=".2f",
    )

    change_reactive_power = widgets.Checkbox(
        value=True,
        description="Vary Reactive Power (Q) across scenarios",
        style={"description_width": "250px"},
        layout=widgets.Layout(width="400px"),
    )

    # Load Configuration - Advanced Scaling

    global global_range, max_scaling_factor, step_size, start_scaling_factor
    global_range = widgets.FloatSlider(
        value=0.4,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Global Range:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format=".1f",
    )

    max_scaling_factor = widgets.FloatSlider(
        value=4.0,
        min=0.7,
        max=4.0,
        step=0.1,
        description="Max Scaling:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format=".1f",
    )

    step_size = widgets.FloatSlider(
        value=0.1,
        min=0.05,
        max=0.2,
        step=0.005,
        description="Step Size:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format=".3f",
    )

    start_scaling_factor = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=2.0,
        step=0.1,
        description="Start Scaling:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format=".1f",
    )

    # Topology Perturbation Configuration

    global k, n_topology_variants, perturbation_type, elements
    k = widgets.IntSlider(
        value=1,
        min=1,
        max=10,
        step=1,
        description="Max Components (k):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format="d",
    )

    n_topology_variants = widgets.IntSlider(
        value=20,
        min=1,
        max=50,
        step=1,
        description="Top. Variants per load scenario:",
        style={"description_width": "200px"},
        layout=widgets.Layout(width="550px"),
        readout_format="d",
    )

    perturbation_type = widgets.Dropdown(
        options=[
            ("random", "random"),
            ("n_minus_k", "n_minus_k"),
            ("none", "none"),
        ],
        value="random",
        description="Perturbation Type:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="250px"),
    )

    elements = widgets.SelectMultiple(
        options=[
            ("branch", "branch"),
            ("gen", "gen"),
        ],
        value=["branch", "gen"],
        description="Elements to Drop:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px", height="120px"),
    )

    # Generation Configuration

    global gen_perturbation_type, gen_sigma

    gen_perturbation_type = widgets.Dropdown(
        options=[
            ("cost_permutation", "cost_permutation"),
            ("cost_perturbation", "cost_perturbation"),
            ("none", "none"),
        ],
        value="cost_permutation",
        description="Perturbation Type:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="250px"),
    )

    gen_sigma = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=3.0,
        step=0.1,
        description="Perturbation Range (σ):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format=".2f",
    )

    # Admittance Perturbation Configuration

    global admittance_perturbation_type, admittance_sigma

    admittance_perturbation_type = widgets.Dropdown(
        options=[
            ("random_perturbation", "random_perturbation"),
            ("none", "none"),
        ],
        value="random_perturbation",
        description="Perturbation Type:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="250px"),
    )

    admittance_sigma = widgets.FloatSlider(
        value=0.2,
        min=0.0,
        max=1.0,
        step=0.05,
        description="Perturbation Range (σ):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format=".2f",
    )

    # Execution Settings

    global \
        num_processes, \
        data_dir, \
        large_chunk_size, \
        overwrite, \
        mode, \
        include_dc_res, \
        pf_fast, \
        dcpf_fast, \
        enable_solver_logs, \
        max_iter
    num_processes = widgets.IntSlider(
        value=16,
        min=1,
        max=32,
        step=1,
        description="Parallel Processes:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format="d",
    )

    data_dir = widgets.Text(
        value="./data_out",
        description="Output Directory:",
        placeholder="e.g., data/, /path/to/output",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="500px"),
    )

    large_chunk_size = widgets.IntSlider(
        value=1000,
        min=10,
        max=10000,
        step=10,
        description="Chunk Size:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format="d",
    )

    # Optional Settings

    overwrite = widgets.Checkbox(
        value=True,
        description="Overwrite existing files (vs. append)",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="500px"),
    )

    # Mode selection
    mode = widgets.Dropdown(
        options=[
            (
                "PF mode - Generate Power flow data where one or more operating limits – the inequality constraints defined in OPF, e.g., voltage magnitude or branch limits – may be violated.",
                "pf",
            ),
            (
                "OPF Mode: Generate Optimal Power Flow data, with cost-optimal dispatches that satisfy all operating limits (OPF-feasible)",
                "opf",
            ),
        ],
        value="pf",
        description="Processing Mode:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="650px"),
    )

    # DC Power Flow option
    include_dc_res = widgets.Checkbox(
        value=True,
        description="Include DC PF/ OPF results",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="500px"),
    )

    # PF fast option
    pf_fast = widgets.Checkbox(
        value=True,
        description="Use fast PF when solving PF (does not work with all networks)",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="500px"),
    )

    dcpf_fast = widgets.Checkbox(
        value=True,
        description="Use fast DCPF when solving DC PF (compute_dc_pf)",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="500px"),
    )

    # Enable solver logs option
    enable_solver_logs = widgets.Checkbox(
        value=True,
        description="Enable OPF/PF solver logs",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="500px"),
    )

    # Max iterations for Ipopt-based solvers
    max_iter = widgets.IntSlider(
        value=200,
        min=50,
        max=1000,
        step=50,
        description="Max Iterations (Ipopt):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format="d",
    )

    # Load Configuration - Advanced Box
    load_advanced_box = widgets.VBox(
        [
            widgets.HTML(
                "<p style='margin: 5px 0; color: #666;'>Parameters for searching the aggregated load profile scaling factors</p>",
            ),
            global_range,
            max_scaling_factor,
            step_size,
            start_scaling_factor,
        ],
        layout=widgets.Layout(
            border="2px solid #FFF3E0",
            padding="15px",
            margin="5px 0",
            border_radius="10px",
        ),
    )

    # Only show advanced load scaling if agg_load_profile is selected
    def update_advanced_load_scaling(*args):
        load_advanced_box.layout.display = (
            "block" if load_generator.value == "agg_load_profile" else "none"
        )

    load_generator.observe(update_advanced_load_scaling, names="value")
    update_advanced_load_scaling()

    # Create organized widget layout

    # Network Configuration Box
    network_box = widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='color: #2196F3; margin: 10px 0;'>📡 Network Configuration</h3>",
            ),
            widgets.HTML(
                "<p style='margin: 5px 0; color: #666;'>Select the power grid network to analyze</p>",
            ),
            network_source,
            pglib_grid_dropdown,
            network_file_chooser,
        ],
        layout=widgets.Layout(
            border="2px solid #E3F2FD",
            padding="15px",
            margin="5px 0",
            border_radius="10px",
        ),
    )

    # Load Configuration - Basic Box
    load_basic_box = widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='color: #4CAF50; margin: 10px 0;'>⚡ Load Configuration - Basic Parameters</h3>",
            ),
            widgets.HTML(
                "<p style='margin: 5px 0; color: #666;'>Configure how load scenarios are generated</p>",
            ),
            load_generator,
            agg_profile,
            num_scenarios,
            sigma,
            change_reactive_power,
        ],
        layout=widgets.Layout(
            border="2px solid #E8F5E8",
            padding="15px",
            margin="5px 0",
            border_radius="10px",
        ),
    )

    # Create containers for topology perturbation components
    topology_components_container = widgets.VBox([k, n_topology_variants, elements])

    # Function to update topology perturbation component visibility
    def update_topology_perturbation_visibility(*args):
        if perturbation_type.value == "none":
            topology_components_container.layout.display = "none"
        elif perturbation_type.value == "n_minus_k":
            # For n_minus_k, only show k parameter, hide variants and elements
            topology_components_container.children = [k]
            topology_components_container.layout.display = "block"
        else:
            # For random, show all components
            topology_components_container.children = [k, n_topology_variants, elements]
            topology_components_container.layout.display = "block"

    perturbation_type.observe(update_topology_perturbation_visibility, names="value")
    update_topology_perturbation_visibility()  # Set initial state

    # Topology Configuration Box
    topology_box = widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='color: #9C27B0; margin: 10px 0;'>🔌 Topology Perturbation Configuration</h3>",
            ),
            widgets.HTML(
                "<p style='margin: 5px 0; color: #666;'>Simulate equipment failures and contingencies</p>",
            ),
            perturbation_type,
            topology_components_container,
        ],
        layout=widgets.Layout(
            border="2px solid #F3E5F5",
            padding="15px",
            margin="5px 0",
            border_radius="10px",
        ),
    )

    # Create container for generation perturbation components
    generation_components_container = widgets.VBox([gen_sigma])

    # Function to update generation perturbation component visibility
    def update_generation_perturbation_visibility(*args):
        if (
            gen_perturbation_type.value == "none"
            or gen_perturbation_type.value == "cost_permutation"
        ):
            generation_components_container.layout.display = "none"
        else:
            generation_components_container.layout.display = "block"

    gen_perturbation_type.observe(
        update_generation_perturbation_visibility,
        names="value",
    )
    update_generation_perturbation_visibility()  # Set initial state

    # Generation Configuration Box
    generation_box = widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='color: #9C27B0; margin: 10px 0;'>🔋 Generation Perturbation Configuration</h3>",
            ),
            widgets.HTML(
                "<p style='margin: 5px 0; color: #666;'>Configure how cost of generation components are perturbed</p>",
            ),
            gen_perturbation_type,
            generation_components_container,
        ],
        layout=widgets.Layout(
            border="2px solid #F3E5F5",
            padding="15px",
            margin="5px 0",
            border_radius="10px",
        ),
    )

    # Create container for admittance perturbation components
    admittance_components_container = widgets.VBox([admittance_sigma])

    # Function to update admittance perturbation component visibility
    def update_admittance_perturbation_visibility(*args):
        if admittance_perturbation_type.value == "none":
            admittance_components_container.layout.display = "none"
        else:
            admittance_components_container.layout.display = "block"

    admittance_perturbation_type.observe(
        update_admittance_perturbation_visibility,
        names="value",
    )
    update_admittance_perturbation_visibility()  # Set initial state

    # Admittance Configuration Box
    admittance_box = widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='color: #9C27B0; margin: 10px 0;'>🎛️ Line Admittance Perturbation Configuration</h3>",
            ),
            widgets.HTML(
                "<p style='margin: 5px 0; color: #666;'>Configure how line admittances are perturbed</p>",
            ),
            admittance_perturbation_type,
            admittance_components_container,
        ],
        layout=widgets.Layout(
            border="2px solid #F3E5F5",
            padding="15px",
            margin="5px 0",
            border_radius="10px",
        ),
    )

    # Execution Settings Box
    execution_box = widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='color: #795548; margin: 10px 0;'>⚙️ Execution Settings</h3>",
            ),
            widgets.HTML(
                "<p style='margin: 5px 0; color: #666;'>Configure computational and output settings</p>",
            ),
            num_processes,
            data_dir,
            large_chunk_size,
            mode,
            overwrite,
            include_dc_res,
            pf_fast,
            dcpf_fast,
            enable_solver_logs,
            max_iter,
        ],
        layout=widgets.Layout(
            border="2px solid #EFEBE9",
            padding="15px",
            margin="5px 0",
            border_radius="10px",
        ),
    )

    # Add a text field for YAML config file name
    config_filename = widgets.Text(
        value="user_config.yaml",
        description="Config Filename:",
        placeholder="e.g., user_config.yaml",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="400px"),
    )

    # Button to create YAML config only
    def save_config_only(b):
        with output_box:
            output_box.clear_output()
            try:
                config = create_config()
                config_path = Path(config_filename.value)
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"YAML configuration saved to {config_path.resolve()}")
            except Exception as e:
                print("Error while saving configuration:")
                print(e)

    def save_and_run_config(b):
        """Save configuration and run the generation script."""
        # Grey/disable both buttons immediately
        run_button.description = "Running…"
        run_button.disabled = True
        save_config_button.disabled = True

        with output_box:
            output_box.clear_output()
            try:
                config = create_config()

                # Save config to file
                config_path = Path(config_filename.value)
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"\nConfiguration written to: {config_path.resolve()}\n")

                # Run the generator (stdout/stderr will appear here)
                print("Starting generation...")
                generate_power_flow_data_distributed(str(config_path))
                print("\nGeneration completed successfully.")
            except Exception as e:
                print("Error during generation:")
                print(e)
            finally:
                # Re-enable buttons and restore label regardless of success/failure
                run_button.description = "Generate and Run Configuration"
                run_button.disabled = False
                save_config_button.disabled = False

    save_config_button = widgets.Button(
        description="Create YAML Config",
        button_style="info",
        tooltip="Save the current configuration to a YAML file.",
    )
    save_config_button.on_click(save_config_only)

    # Create and configure the run button
    run_button = widgets.Button(
        description="Generate and Run Configuration",
        button_style="success",
        tooltip="Click to generate configuration and run the script",
    )
    run_button.on_click(save_and_run_config)

    # Display all widget groups
    display(network_box)
    display(load_basic_box)
    display(load_advanced_box)
    display(topology_box)
    display(generation_box)
    display(admittance_box)
    display(execution_box)
    display(config_filename)
    display(save_config_button, run_button)

    # Display the dedicated output area last so it is close to the buttons
    display(widgets.HTML("<b>Output</b>"))
    display(output_box)
