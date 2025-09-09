#!/usr/bin/env python3

import ipywidgets as widgets
from IPython.display import display
import yaml
from pathlib import Path
from importlib import resources
from ipyfilechooser import FileChooser

# Run the generation function directly
from gridfm_datakit.generate import generate_power_flow_data_distributed


def get_available_load_profiles():
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


def create_config():
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
    else:  # pandapower
        name = pandapower_grid_dropdown.value
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
            "no_stats": no_stats.value,
            "overwrite": overwrite.value,
            "mode": str(mode),
        },
    }
    return config


def interactive_interface():
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
    global \
        network_source, \
        pglib_grid_dropdown, \
        pandapower_grid_dropdown, \
        network_file_chooser

    network_source = widgets.Dropdown(
        options=[
            ("pglib - Power Grid Library (Recommended)", "pglib"),
            ("pandapower - Built-in pandapower networks (beta)", "pandapower"),
            ("file - Custom network files (MATPOWER .m)", "file"),
        ],
        value="pglib",
        description="Network Source:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )

    # --- NEW: pglib grid dropdown ---
    pglib_grids = [
        "case24_ieee_rts",
        "case30_ieee",
        "case30_as",
        "case39_epri",
        "case57_ieee",
        "case60_c",
        "case73_ieee_rts",
        "case89_pegase",
        "case118_ieee",
        "case197_snem",
        "case240_pserc",
        "case793_goc",
        "case2312_goc",
        "case2869_pegase",
        "case3970_goc",
        "case4020_goc",
        "case4601_goc",
        "case4837_goc",
        "case4917_goc",
        "case5658_epigrids",
        "case7336_epigrids",
        "case9591_goc",
        "case10192_epigrids",
        "case19402_goc",
    ]
    pglib_grid_dropdown = widgets.Dropdown(
        options=pglib_grids,
        value="case24_ieee_rts",
        description="PGLib Grid:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
    )

    # --- NEW: pandapower grid dropdown ---
    pandapower_grids = [
        "GBreducednetwork",
        "case118",
        "case11_iwamoto",
        "case1354pegase",
        "case14",
        "case145",
        "case1888rte",
        "case24_ieee_rts",
        "case2848rte",
        "case2869pegase",
        "case30",
        "case300",
        "case3120sp",
        "case33bw",
        "case39",
        "case4gs",
        "case5",
        "case57",
        "case5_demo_gridcal",
        "case6470rte",
        "case6495rte",
        "case6515rte",
        "case6ww",
        "case89pegase",
        "case9",
        "case9241pegase",
        "case_ieee30",
        "case_illinois200",
        "iceland",
    ]
    pandapower_grid_dropdown = widgets.Dropdown(
        options=pandapower_grids,
        value="case14",
        description="Pandapower Grid:",
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
            pandapower_grid_dropdown.layout.display = "none"
            network_file_chooser.layout.display = "none"
        elif network_source.value == "pandapower":
            pglib_grid_dropdown.layout.display = "none"
            pandapower_grid_dropdown.layout.display = "block"
            network_file_chooser.layout.display = "none"
        else:  # file
            pglib_grid_dropdown.layout.display = "none"
            pandapower_grid_dropdown.layout.display = "none"
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
        value=200,
        min=10,
        max=30000,
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
        min=1.0,
        max=10.0,
        step=0.1,
        description="Max Scaling:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format=".1f",
    )

    step_size = widgets.FloatSlider(
        value=0.025,
        min=0.01,
        max=0.1,
        step=0.005,
        description="Step Size:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format=".3f",
    )

    start_scaling_factor = widgets.FloatSlider(
        value=0.8,
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
        value=2,
        min=1,
        max=10,
        step=1,
        description="Max Components (k):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format="d",
    )

    n_topology_variants = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
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
            ("line", "line"),
            ("trafo", "trafo"),
            ("gen", "gen"),
            ("sgen", "sgen"),
        ],
        value=["line", "trafo", "gen", "sgen"],
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

    global num_processes, data_dir, large_chunk_size, no_stats, overwrite, mode
    num_processes = widgets.IntSlider(
        value=10,
        min=1,
        max=16,
        step=1,
        description="Parallel Processes:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format="d",
    )

    data_dir = widgets.Text(
        value="data/",
        description="Output Directory:",
        placeholder="e.g., data/, /path/to/output",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="500px"),
    )

    large_chunk_size = widgets.IntSlider(
        value=200,
        min=10,
        max=500,
        step=10,
        description="Chunk Size:",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="550px"),
        readout_format="d",
    )

    # Optional Settings

    no_stats = widgets.Checkbox(
        value=False,
        description="Disable statistical calculations (faster)",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="500px"),
    )

    overwrite = widgets.Checkbox(
        value=True,
        description="Overwrite existing files (vs. append)",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="500px"),
    )

    # Set mode to "pf"
    mode = "pf"

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
            pandapower_grid_dropdown,
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
            k,
            n_topology_variants,
            elements,
        ],
        layout=widgets.Layout(
            border="2px solid #F3E5F5",
            padding="15px",
            margin="5px 0",
            border_radius="10px",
        ),
    )

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
            gen_sigma,
        ],
        layout=widgets.Layout(
            border="2px solid #F3E5F5",
            padding="15px",
            margin="5px 0",
            border_radius="10px",
        ),
    )

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
            admittance_sigma,
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
            widgets.HBox(
                [no_stats, overwrite],
                layout=widgets.Layout(justify_content="flex-start"),
            ),
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
