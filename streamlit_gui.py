import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import base64
from datetime import datetime

from run_examples import run_scenarios

# Set page config with enhanced styling
st.set_page_config(
    page_title="FleetPy Web Interface",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚗"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        font-size: 2rem;
        margin: 0 0.25rem;
        white-space: nowrap;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    h1 {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


# Define helper functions
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # Some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def create_sample_files():
    """Create sample files for users to download"""
    # Sample vehicle data
    vehicles_df = pd.DataFrame({
        'vehicle_id': ['v1', 'v2', 'v3'],
        'type': ['sedan', 'truck', 'van'],
        'capacity': [4, 2, 6],
        'start_location': ['depot', 'depot', 'depot'],
        'end_location': ['depot', 'depot', 'depot']
    })

    # Sample request data
    requests_df = pd.DataFrame({
        'request_id': ['r1', 'r2', 'r3', 'r4', 'r5'],
        'pickup_location': ['A', 'B', 'C', 'D', 'E'],
        'dropoff_location': ['F', 'G', 'H', 'I', 'J'],
        'pickup_time_window_start': [0, 10, 20, 30, 40],
        'pickup_time_window_end': [30, 40, 50, 60, 70],
        'dropoff_time_window_start': [30, 40, 50, 60, 70],
        'dropoff_time_window_end': [60, 70, 80, 90, 100]
    })

    # Sample configuration
    config = {
        "scenario_name": "sample_scenario",
        "algorithm": "insertion_heuristic",
        "objective": "minimize_vehicles",
        "max_computation_time": 60,
        "random_seed": 42
    }

    return vehicles_df, requests_df, config


# Main app
def main():
    st.title("FleetPy Web Interface")
    st.markdown("""
    <div style='color: #666; font-size: 1.1em; text-align: center;'>
    Optimize your fleet operations with advanced routing and scheduling algorithms
    </div>
    """, unsafe_allow_html=True)

    # Tab-based navigation with improved styling
    tabs = st.tabs([
        "🏠 Home", 
        "📋 Scenario Management", 
        "📤 Upload Files", 
        "⚙️ Simulation Config", 
        "🚗 Fleet Management", 
        "📊 Visualize Results"
    ])

    with tabs[0]:
        st.header("Welcome to FleetPy Web Interface")
        st.markdown("""
        <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #333;'>What is FleetPy?</h3>
            <p style='color: #666;'>FleetPy is a powerful tool for vehicle routing and scheduling optimization. 
            It helps you efficiently manage your fleet operations by finding optimal routes and schedules 
            for your vehicles.</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Quick Start")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4>1. Create or Load Scenario</h4>
                <p>Start with a sample scenario or create your own in the Scenario Management tab</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4>2. Configure Settings</h4>
                <p>Adjust simulation and fleet settings to match your requirements</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4>3. Run and Visualize</h4>
                <p>Run the simulation and analyze the results</p>
            </div>
            """, unsafe_allow_html=True)

    with tabs[1]:  # Scenario Management Tab
        st.header("Scenario Management")
        
        # Initialize session state for scenarios if not exists
        if 'scenarios' not in st.session_state:
            st.session_state.scenarios = {}
        if 'templates' not in st.session_state:
            st.session_state.templates = {}
        
        # Tab navigation within Scenario Management
        scenario_tabs = st.tabs(["Create New", "Samples", "Templates", "Compare Scenarios"])
        
        with scenario_tabs[0]:  # Create New Scenario
            st.subheader("Create New Scenario")
            
            # Basic Information
            col1, col2 = st.columns(2)
            with col1:
                scenario_name = st.text_input("Scenario Name", key="new_scenario_name")
            with col2:
                scenario_type = st.selectbox("Scenario Type", 
                                           ["Single Simulation", "Batch Simulation"],
                                           key="new_scenario_type")
            
            # Description and Tags
            scenario_description = st.text_area("Description", 
                                              help="Describe the purpose and key features of this scenario",
                                              key="new_scenario_desc")
            scenario_tags = st.multiselect("Tags", 
                                         ["Urban", "Rural", "Ridepooling", "Delivery", "Mixed"],
                                         help="Add tags to categorize the scenario",
                                         key="new_scenario_tags")
            
            # Algorithm Settings
            st.subheader("Algorithm Settings")
            col1, col2 = st.columns(2)
            with col1:
                algorithm = st.selectbox("Algorithm",
                                      ["insertion_heuristic", "local_search", "column_generation"],
                                      help="Select the optimization algorithm to use",
                                      key="scenario_algorithm")
            with col2:
                objective = st.selectbox("Objective",
                                      ["minimize_vehicles", "minimize_distance", "minimize_duration"],
                                      help="Select the optimization objective",
                                      key="scenario_objective")
            
            # Performance Settings
            st.subheader("Performance Settings")
            col1, col2 = st.columns(2)
            with col1:
                max_time = st.slider("Max Computation Time (seconds)", 10, 600, 60,
                                   help="Maximum time allowed for the solver to run",
                                   key="scenario_max_time")
            with col2:
                random_seed = st.number_input("Random Seed", 0, 9999, 42,
                                            help="Random seed for reproducible results",
                                            key="scenario_random_seed")
            
            # Configuration Source
            st.subheader("Configuration Source")
            config_source = st.radio("Load configuration from:",
                                   ["Current Settings", "Template", "New Configuration"],
                                   key="new_scenario_source")
            
            if config_source == "Template":
                template_name = st.selectbox("Select Template",
                                           list(st.session_state.templates.keys()),
                                           key="new_scenario_template")
            elif config_source == "New Configuration":
                st.info("Configure settings in the Simulation Config and Fleet Management tabs first")
            
            # Save Scenario
            if st.button("Save Scenario", key="save_new_scenario"):
                if not scenario_name:
                    st.error("Please enter a scenario name")
                else:
                    # Collect configuration from current settings
                    scenario_config = {
                        "name": scenario_name,
                        "type": scenario_type,
                        "description": scenario_description,
                        "tags": scenario_tags,
                        "algorithm": algorithm,
                        "objective": objective,
                        "max_time": max_time,
                        "random_seed": random_seed,
                        "sim_config": st.session_state.get('sim_config', {}),
                        "fleet_config": st.session_state.get('fleet_config', {}),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.session_state.scenarios[scenario_name] = scenario_config
                    st.success(f"Scenario '{scenario_name}' saved successfully!")
            
            # # TODO fix Navigation buttons
            # st.markdown("---")
            # col1, col2, col3 = st.columns([1, 1, 1])
            # with col2:
            #     if st.button("Next: Upload Files", key="next_upload_files"):
            #         st.session_state.current_tab = "Upload Files"
            #         st.rerun()
        
        with scenario_tabs[1]:  # Samples
            st.subheader("Sample Scenarios")
            st.info("These are pre-configured scenarios to help you get started. Load a sample to see how different configurations work.")
            
            # Sample scenarios
            sample_scenarios = {
                "Basic Urban Ridepooling": {
                    "description": "A simple urban ridepooling scenario with a small fleet",
                    "type": "Single Simulation",
                    "tags": ["Urban", "Ridepooling"],
                    "sim_config": {
                        "start_time": 0,
                        "end_time": 3600,
                        "time_step": 1,
                        "random_seed": 42,
                        "route_output": True,
                        "replay_output": False,
                        "network_type": "NetworkBase"
                    },
                    "fleet_config": {
                        "composition": [
                            {"type": "sedan", "quantity": 5, "capacity": 4}
                        ],
                        "control_strategy": "insertion_heuristic"
                    }
                },
                "Large Fleet Delivery": {
                    "description": "A delivery scenario with a large mixed fleet",
                    "type": "Single Simulation",
                    "tags": ["Urban", "Delivery"],
                    "sim_config": {
                        "start_time": 0,
                        "end_time": 7200,
                        "time_step": 1,
                        "random_seed": 42,
                        "route_output": True,
                        "replay_output": True,
                        "network_type": "NetworkBase"
                    },
                    "fleet_config": {
                        "composition": [
                            {"type": "van", "quantity": 10, "capacity": 20},
                            {"type": "truck", "quantity": 5, "capacity": 30}
                        ],
                        "control_strategy": "local_search"
                    }
                },
                "Mixed Fleet Ridepooling": {
                    "description": "A ridepooling scenario with different vehicle types",
                    "type": "Single Simulation",
                    "tags": ["Urban", "Ridepooling", "Mixed"],
                    "sim_config": {
                        "start_time": 0,
                        "end_time": 10800,
                        "time_step": 1,
                        "random_seed": 42,
                        "route_output": True,
                        "replay_output": True,
                        "network_type": "NetworkDynamicNFDClusters",
                        "density_bin_size": 300,
                        "density_avg_duration": 900
                    },
                    "fleet_config": {
                        "composition": [
                            {"type": "sedan", "quantity": 8, "capacity": 4},
                            {"type": "van", "quantity": 4, "capacity": 8}
                        ],
                        "control_strategy": "column_generation"
                    }
                }
            }
            
            # Display sample scenarios
            for name, scenario in sample_scenarios.items():
                with st.expander(f"📋 {name}"):
                    st.write(f"**Description:** {scenario['description']}")
                    st.write(f"**Type:** {scenario['type']}")
                    st.write(f"**Tags:** {', '.join(scenario['tags'])}")
                    
                    # Show configuration preview
                    st.markdown("**Configuration Preview:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Simulation Configuration**")
                        st.json(scenario['sim_config'])
                    
                    with col2:
                        st.markdown("**Fleet Configuration**")
                        st.json(scenario['fleet_config'])
                    
                    if st.button("Load Sample", key=f"load_sample_{name}"):
                        st.session_state.sim_config = scenario['sim_config']
                        st.session_state.fleet_config = scenario['fleet_config']
                        st.success(f"Sample scenario '{name}' loaded! Configure additional settings in the Simulation Config and Fleet Management tabs.")
                        st.info("Note: You may need to upload your own data files in the Upload Files tab.")
            
            # # Navigation buttons
            # st.markdown("---")
            # col1, col2, col3 = st.columns([1, 1, 1])
            # with col2:
            #     if st.button("Next: Upload Files", key="next_upload_files_samples"):
            #         st.session_state.current_tab = "Upload Files"
            #         st.rerun()
        
        with scenario_tabs[2]:  # Templates
            st.subheader("Scenario Templates")
            
            # Create Template
            st.markdown("### Create New Template")
            template_name = st.text_input("Template Name", key="new_template_name")
            template_description = st.text_area("Template Description", key="new_template_desc")
            
            # Show what will be saved in the template
            st.markdown("### Template Contents")
            st.info("This template will save the following configurations:")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Simulation Configuration**")
                if st.session_state.get('sim_config'):
                    sim_config = st.session_state.sim_config
                    st.json({
                        "Time Settings": {
                            "Start Time": sim_config.get('start_time', 'Not set'),
                            "End Time": sim_config.get('end_time', 'Not set'),
                            "Time Step": sim_config.get('time_step', 'Not set')
                        },
                        "Randomization": {
                            "Random Seed": sim_config.get('random_seed', 'Not set')
                        },
                        "Output Settings": {
                            "Route Output": sim_config.get('route_output', 'Not set'),
                            "Replay Output": sim_config.get('replay_output', 'Not set')
                        },
                        "Network Type": sim_config.get('network_type', 'Not set')
                    })
                else:
                    st.warning("No simulation configuration found. Configure settings in the Simulation Config tab first.")
            
            with col2:
                st.markdown("**Fleet Configuration**")
                if st.session_state.get('fleet_config'):
                    fleet_config = st.session_state.fleet_config
                    st.json({
                        "Fleet Composition": fleet_config.get('composition', []),
                        "Control Strategy": fleet_config.get('control_strategy', 'Not set')
                    })
                else:
                    st.warning("No fleet configuration found. Configure settings in the Fleet Management tab first.")
            
            if st.button("Save as Template", key="save_template"):
                if not template_name:
                    st.error("Please enter a template name")
                elif not st.session_state.get('sim_config') or not st.session_state.get('fleet_config'):
                    st.error("Please configure both simulation and fleet settings before saving as template")
                else:
                    template_config = {
                        "name": template_name,
                        "description": template_description,
                        "sim_config": st.session_state.get('sim_config', {}),
                        "fleet_config": st.session_state.get('fleet_config', {}),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.session_state.templates[template_name] = template_config
                    st.success(f"Template '{template_name}' saved successfully!")
            
            # Display Templates
            st.markdown("### Saved Templates")
            if st.session_state.templates:
                for template_name, template in st.session_state.templates.items():
                    with st.expander(f"📋 {template_name}"):
                        st.write(f"**Description:** {template['description']}")
                        st.write(f"**Created:** {template['created_at']}")
                        
                        # Show template contents
                        st.markdown("**Template Contents:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Simulation Configuration**")
                            st.json({
                                "Time Settings": {
                                    "Start Time": template['sim_config'].get('start_time', 'Not set'),
                                    "End Time": template['sim_config'].get('end_time', 'Not set'),
                                    "Time Step": template['sim_config'].get('time_step', 'Not set')
                                },
                                "Randomization": {
                                    "Random Seed": template['sim_config'].get('random_seed', 'Not set')
                                },
                                "Output Settings": {
                                    "Route Output": template['sim_config'].get('route_output', 'Not set'),
                                    "Replay Output": template['sim_config'].get('replay_output', 'Not set')
                                },
                                "Network Type": template['sim_config'].get('network_type', 'Not set')
                            })
                        
                        with col2:
                            st.markdown("**Fleet Configuration**")
                            st.json({
                                "Fleet Composition": template['fleet_config'].get('composition', []),
                                "Control Strategy": template['fleet_config'].get('control_strategy', 'Not set')
                            })
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Load Template", key=f"load_{template_name}"):
                                st.session_state.sim_config = template['sim_config']
                                st.session_state.fleet_config = template['fleet_config']
                                st.success(f"Template '{template_name}' loaded!")
                        with col2:
                            if st.button("Delete Template", key=f"delete_{template_name}"):
                                del st.session_state.templates[template_name]
                                st.success(f"Template '{template_name}' deleted!")
            else:
                st.info("No templates saved yet")
            
            # # Navigation buttons
            # st.markdown("---")
            # col1, col2, col3 = st.columns([1, 1, 1])
            # with col2:
            #     if st.button("Next: Upload Files", key="next_upload_files_templates"):
            #         st.session_state.current_tab = "Upload Files"
            #         st.rerun()
        
        with scenario_tabs[3]:  # Compare Scenarios
            st.subheader("Compare Scenarios")
            
            if len(st.session_state.scenarios) < 2:
                st.info("You need at least 2 scenarios to compare")
            else:
                # Select scenarios to compare
                selected_scenarios = st.multiselect("Select Scenarios to Compare",
                                                  list(st.session_state.scenarios.keys()),
                                                  default=list(st.session_state.scenarios.keys())[:2])
                
                if len(selected_scenarios) >= 2:
                    # Basic Comparison
                    st.markdown("### Basic Comparison")
                    comparison_data = []
                    for scenario_name in selected_scenarios:
                        scenario = st.session_state.scenarios[scenario_name]
                        comparison_data.append({
                            "Name": scenario_name,
                            "Type": scenario["type"],
                            "Created": scenario["created_at"],
                            "Last Modified": scenario["last_modified"],
                            "Tags": ", ".join(scenario["tags"]),
                            "Fleet Size": sum(item['quantity'] for item in scenario["fleet_config"].get("composition", [])),
                            "Simulation Duration": f"{scenario['sim_config'].get('end_time', 0) - scenario['sim_config'].get('start_time', 0)} seconds"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)
                    
                    # Configuration Comparison
                    st.markdown("### Configuration Comparison")
                    config_tabs = st.tabs(["Simulation Config", "Fleet Config"])
                    
                    with config_tabs[0]:
                        sim_configs = {}
                        for scenario_name in selected_scenarios:
                            sim_configs[scenario_name] = st.session_state.scenarios[scenario_name]["sim_config"]
                        
                        # Create a DataFrame for simulation configurations
                        sim_comparison = pd.DataFrame(sim_configs).T
                        st.dataframe(sim_comparison)
                    
                    with config_tabs[1]:
                        fleet_configs = {}
                        for scenario_name in selected_scenarios:
                            fleet_configs[scenario_name] = st.session_state.scenarios[scenario_name]["fleet_config"]
                        
                        # Create a DataFrame for fleet configurations
                        fleet_comparison = pd.DataFrame(fleet_configs).T
                        st.dataframe(fleet_comparison)
                    
                    # Export Comparison
                    if st.button("Export Comparison", key="export_comparison"):
                        # Create a dictionary with all comparison data
                        export_data = {
                            "basic_comparison": comparison_df.to_dict(),
                            "sim_config_comparison": sim_comparison.to_dict(),
                            "fleet_config_comparison": fleet_comparison.to_dict()
                        }
                        
                        # Convert to JSON
                        json_data = json.dumps(export_data, indent=2)
                        
                        # Create download link
                        st.markdown(download_link(json_data, "scenario_comparison.json", "📥 Download Comparison"),
                                  unsafe_allow_html=True)
            
            # # Navigation buttons
            # st.markdown("---")
            # col1, col2, col3 = st.columns([1, 1, 1])
            # with col2:
            #     if st.button("Next: Upload Files", key="next_upload_files_compare"):
            #         st.session_state.current_tab = "Upload Files"
            #         st.rerun()

    with tabs[2]:
        st.header("Upload Files")

        # Upload vehicles file
        st.subheader("Upload Vehicles File")
        vehicles_file = st.file_uploader("Choose a CSV file", type="csv", key="vehicles")
        if vehicles_file is not None:
            vehicles_df = pd.read_csv(vehicles_file)
            st.session_state['vehicles_df'] = vehicles_df
            st.success("Vehicles file uploaded successfully!")
            st.write(vehicles_df)

        # Upload requests file
        st.subheader("Upload Requests File")
        requests_file = st.file_uploader("Choose a CSV file", type="csv", key="requests")
        if requests_file is not None:
            requests_df = pd.read_csv(requests_file)
            st.session_state['requests_df'] = requests_df
            st.success("Requests file uploaded successfully!")
            st.write(requests_df)

        # Upload configuration file
        st.subheader("Upload Configuration File")
        config_file = st.file_uploader("Choose a JSON file", type="json")
        if config_file is not None:
            config = json.load(config_file)
            st.session_state['config'] = config
            st.success("Configuration file uploaded successfully!")
            st.json(config)

        # Run solver
        if st.button("Run FleetPy Solver"):
            if 'vehicles_df' in st.session_state and 'requests_df' in st.session_state and 'config' in st.session_state:
                st.info("Running solver...")

                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Here you would call FleetPy functions to run the solver
                # For demonstration, we'll simulate progress:

                import time

                for i in range(101):
                    # Update progress bar
                    progress_bar.progress(i)

                    # Update status text
                    if i < 30:
                        status_text.text(f"Setting up the problem... ({i}%)")
                    elif i < 60:
                        status_text.text(f"Optimizing routes... ({i}%)")
                    elif i < 90:
                        status_text.text(f"Finalizing solution... ({i}%)")
                    else:
                        status_text.text(f"Preparing results... ({i}%)")

                    # Simulate computation time
                    time.sleep(0.05)

                # For real implementation, you'd run your FleetPy solver here
                # and update the progress periodically based on solver status
                run_fleetpy()

                # Remove progress elements when done
                progress_bar.empty()
                status_text.empty()

                # TODO show actual results
                # For now, we'll just simulate a result
                st.session_state['solution'] = {
                    "routes": [
                        {"vehicle_id": "v1", "route": ["depot", "A", "F", "depot"]},
                        {"vehicle_id": "v2", "route": ["depot", "B", "G", "C", "H", "depot"]},
                        {"vehicle_id": "v3", "route": ["depot", "D", "I", "E", "J", "depot"]}
                    ],
                    "objective_value": 150.75,
                    "computation_time": 2.3
                }

                st.success("Solver completed!")
                st.balloons()

                # Save the solution timestamp
                st.session_state['solution_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Redirect to visualization page
                st.info("Redirecting to visualization page...")
                st.rerun()
            else:
                st.error("Please upload all required files first!")

        # # Navigation buttons
        # st.markdown("---")
        # col1, col2, col3 = st.columns([1, 1, 1])
        # with col1:
        #     if st.button("← Back: Scenario Management", key="back_scenario_management"):
        #         st.session_state.current_tab = "Scenario Management"
        #         st.rerun()
        # with col3:
        #     if st.button("Next: Simulation Config →", key="next_simulation_config"):
        #         st.session_state.current_tab = "Simulation Config"
        #         st.rerun()

    with tabs[3]:  # Simulation Configuration Tab
        st.header("Simulation Configuration")
        
        # Simulation Time Settings
        st.subheader("Time Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            start_time = st.number_input("Start Time (seconds)", 0, 86400, 0, 
                                       help="Simulation start time in seconds",
                                       key="sim_start_time")
        with col2:
            end_time = st.number_input("End Time (seconds)", 0, 86400, 86400, 
                                     help="Simulation end time in seconds",
                                     key="sim_end_time")
        with col3:
            time_step = st.number_input("Time Step (seconds)", 1, 3600, 1, 
                                      help="Simulation time step in seconds",
                                      key="sim_time_step")

        # Random Seed and Output Settings
        st.subheader("Randomization and Output")
        col1, col2 = st.columns(2)
        with col1:
            random_seed = st.number_input("Random Seed", 0, 9999, 42,
                                        help="Random seed for reproducible results",
                                        key="sim_random_seed")
        with col2:
            route_output = st.checkbox("Output Complete Routes", True,
                                     help="Output complete vehicle routes to files",
                                     key="sim_route_output")
            replay_output = st.checkbox("Output Node Passing Times", False,
                                      help="Output times when vehicles pass nodes",
                                      key="sim_replay_output")

        # Real-time Visualization Settings
        st.subheader("Real-time Visualization")
        realtime_plot = st.selectbox("Real-time Plot Mode", 
                                   ["None", "Live Plot", "Save Plots"],
                                   help="Choose real-time visualization mode",
                                   key="sim_realtime_plot")
        
        if realtime_plot != "None":
            col1, col2 = st.columns(2)
            with col1:
                plot_extent = st.text_input("Plot Extent (min_lon, max_lon, min_lat, max_lat)",
                                          help="Bounding box for visualization",
                                          key="sim_plot_extent")
            with col2:
                vehicle_status = st.multiselect("Vehicle Status to Display",
                                              ["driving", "charging", "idle", "boarding", "alighting"],
                                              default=["driving", "charging"],
                                              help="Select vehicle statuses to display in real-time",
                                              key="sim_vehicle_status")

        # Network Configuration
        st.subheader("Network Configuration")
        network_type = st.selectbox("Network Type",
                                  ["NetworkBase", "NetworkDynamicNFDClusters"],
                                  help="Select the network representation type",
                                  key="sim_network_type")
        
        if network_type == "NetworkDynamicNFDClusters":
            col1, col2 = st.columns(2)
            with col1:
                density_bin_size = st.number_input("Density Bin Size (seconds)", 1, 3600, 300,
                                                 help="Time bin size for network density calculation",
                                                 key="sim_density_bin_size")
            with col2:
                density_avg_duration = st.number_input("Density Average Duration (seconds)", 1, 3600, 900,
                                                     help="Duration for averaging network density",
                                                     key="sim_density_avg_duration")

        # Save Configuration
        if st.button("Save Simulation Configuration", key="sim_save_config"):
            sim_config = {
                "start_time": start_time,
                "end_time": end_time,
                "time_step": time_step,
                "random_seed": random_seed,
                "route_output": route_output,
                "replay_output": replay_output,
                "realtime_plot": realtime_plot,
                "plot_extent": plot_extent if realtime_plot != "None" else None,
                "vehicle_status": vehicle_status if realtime_plot != "None" else None,
                "network_type": network_type,
                "density_bin_size": density_bin_size if network_type == "NetworkDynamicNFDClusters" else None,
                "density_avg_duration": density_avg_duration if network_type == "NetworkDynamicNFDClusters" else None
            }
            st.session_state['sim_config'] = sim_config
            st.success("Simulation configuration saved!")

        # # Navigation buttons
        # st.markdown("---")
        # col1, col2, col3 = st.columns([1, 1, 1])
        # with col1:
        #     if st.button("← Back: Upload Files", key="back_upload_files"):
        #         st.session_state.current_tab = "Upload Files"
        #         st.rerun()
        # with col3:
        #     if st.button("Next: Fleet Management →", key="next_fleet_management"):
        #         st.session_state.current_tab = "Fleet Management"
        #         st.rerun()

    with tabs[4]:  # Fleet Management Tab
        st.header("Fleet Management")
        
        # Fleet Composition
        st.subheader("Fleet Composition")
        if 'fleet_composition' not in st.session_state:
            st.session_state.fleet_composition = []
        
        col1, col2, col3 = st.columns(3)
        with col1:
            vehicle_type = st.selectbox("Vehicle Type", ["sedan", "truck", "van", "bus"],
                                      key="fleet_vehicle_type")
        with col2:
            quantity = st.number_input("Quantity", 1, 100, 1,
                                     key="fleet_quantity")
        with col3:
            capacity = st.number_input("Capacity", 1, 50, 4,
                                     key="fleet_capacity")
        
        if st.button("Add Vehicle Type", key="fleet_add_vehicle"):
            st.session_state.fleet_composition.append({
                "type": vehicle_type,
                "quantity": quantity,
                "capacity": capacity
            })
        
        # Display Current Fleet
        st.subheader("Current Fleet")
        if st.session_state.fleet_composition:
            fleet_df = pd.DataFrame(st.session_state.fleet_composition)
            st.dataframe(fleet_df)
            
            # Fleet Statistics
            total_vehicles = sum(item['quantity'] for item in st.session_state.fleet_composition)
            avg_capacity = sum(item['capacity'] * item['quantity'] for item in st.session_state.fleet_composition) / total_vehicles
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Vehicles", total_vehicles)
            with col2:
                st.metric("Average Capacity", f"{avg_capacity:.1f}")
        else:
            st.info("No vehicles added to the fleet yet.")

        # Fleet Control Strategy
        st.subheader("Fleet Control Strategy")
        control_strategy = st.selectbox("Control Strategy",
                                      ["insertion_heuristic", "local_search", "column_generation"],
                                      help="Select the fleet control strategy",
                                      key="fleet_control_strategy")
        
        if control_strategy == "insertion_heuristic":
            st.info("Insertion Heuristic: Simple and fast, good for small to medium fleets")
        elif control_strategy == "local_search":
            st.info("Local Search: More sophisticated, better for larger fleets")
        else:
            st.info("Column Generation: Advanced optimization, best for complex scenarios")

        # Vehicle Status Monitoring
        st.subheader("Vehicle Status Monitoring")
        if 'vehicle_status' in st.session_state:
            status_df = pd.DataFrame(st.session_state.vehicle_status)
            st.dataframe(status_df)
            
            # Status Distribution Chart
            fig, ax = plt.subplots()
            status_counts = status_df['status'].value_counts()
            ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
            st.pyplot(fig)
        else:
            st.info("No vehicle status data available. Run a simulation to see status information.")

        # Save Fleet Configuration
        if st.button("Save Fleet Configuration", key="fleet_save_config"):
            fleet_config = {
                "composition": st.session_state.fleet_composition,
                "control_strategy": control_strategy
            }
            st.session_state['fleet_config'] = fleet_config
            st.success("Fleet configuration saved!")

        # # Navigation buttons
        # st.markdown("---")
        # col1, col2, col3 = st.columns([1, 1, 1])
        # with col1:
        #     if st.button("← Back: Simulation Config", key="back_simulation_config"):
        #         st.session_state.current_tab = "Simulation Config"
        #         st.rerun()
        # with col3:
        #     if st.button("Next: Visualize Results →", key="next_visualize_results"):
        #         st.session_state.current_tab = "Visualize Results"
        #         st.rerun()

    with tabs[5]:  # Visualize Results Tab
        st.header("Visualization")

        if 'solution' in st.session_state:
            # Solution overview in a card
            st.markdown("""
            <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
                <h3 style='color: #333; margin-bottom: 1rem;'>Solution Overview</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Solved at", st.session_state.get('solution_timestamp', 'Unknown'))
            with col2:
                st.metric("Objective Value", f"{st.session_state['solution']['objective_value']:.2f}")
            with col3:
                st.metric("Computation Time", f"{st.session_state['solution']['computation_time']:.2f} seconds")

            # Routes in a card
            st.markdown("""
            <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0;'>
                <h3 style='color: #333; margin-bottom: 1rem;'>Routes</h3>
            </div>
            """, unsafe_allow_html=True)

            for i, route in enumerate(st.session_state['solution']['routes']):
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                    <h4 style='color: #333; margin: 0;'>Vehicle {route['vehicle_id']}</h4>
                    <p style='color: #666; margin: 0.5rem 0;'>Route: {' → '.join(route['route'])}</p>
                </div>
                """, unsafe_allow_html=True)

            # Visualization tabs with improved styling
            viz_tabs = st.tabs(["🗺️ Route Map", "📅 Gantt Chart", "📊 Performance Metrics"])

            with viz_tabs[0]:
                st.markdown("""
                <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color: #333; margin-bottom: 1rem;'>Route Map</h3>
                </div>
                """, unsafe_allow_html=True)

                # Create a simple plot with improved styling
                fig, ax = plt.subplots(figsize=(12, 8))
                plt.style.use('default')  # Use default style as base
                
                # Set custom style parameters
                plt.rcParams.update({
                    'font.size': 12,
                    'axes.labelsize': 12,
                    'axes.titlesize': 16,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'axes.grid': True,
                    'grid.alpha': 0.3,
                    'axes.facecolor': 'white',
                    'figure.facecolor': 'white'
                })

                # Dummy locations with improved styling
                locations = {
                    'depot': (0, 0),
                    'A': (1, 2), 'F': (3, 2),
                    'B': (2, 3), 'G': (4, 3),
                    'C': (1, 4), 'H': (3, 4),
                    'D': (2, 1), 'I': (4, 1),
                    'E': (1, 0), 'J': (3, 0)
                }

                # Plot locations with improved styling
                for loc, (x, y) in locations.items():
                    color = '#4CAF50' if loc == 'depot' else '#2196F3'
                    ax.scatter(x, y, s=150, c=color, edgecolors='white', linewidth=2)
                    ax.text(x, y, loc, fontsize=14, ha='center', va='center', color='white')

                # Plot routes with improved styling
                colors = ['#FF5722', '#9C27B0', '#FFC107']
                for i, route in enumerate(st.session_state['solution']['routes']):
                    route_points = route['route']
                    for j in range(len(route_points) - 1):
                        start = locations[route_points[j]]
                        end = locations[route_points[j + 1]]
                        ax.plot([start[0], end[0]], [start[1], end[1]], 
                               color=colors[i], linewidth=3, alpha=0.8)

                ax.set_title("Route Map", pad=20)
                ax.set_xlabel("X coordinate")
                ax.set_ylabel("Y coordinate")
                plt.tight_layout()

                st.pyplot(fig)

            with viz_tabs[1]:
                st.markdown("""
                <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color: #333; margin-bottom: 1rem;'>Gantt Chart</h3>
                </div>
                """, unsafe_allow_html=True)

                # Create a Gantt chart with improved styling
                fig, ax = plt.subplots(figsize=(12, 6))
                plt.style.use('default')  # Use default style as base
                
                # Set custom style parameters
                plt.rcParams.update({
                    'font.size': 12,
                    'axes.labelsize': 12,
                    'axes.titlesize': 16,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'axes.grid': True,
                    'grid.alpha': 0.3,
                    'axes.facecolor': 'white',
                    'figure.facecolor': 'white'
                })

                # Dummy time data
                times = {
                    'v1': [(0, 10), (10, 20), (20, 30)],
                    'v2': [(0, 15), (15, 25), (25, 40), (40, 50)],
                    'v3': [(5, 15), (15, 25), (25, 35), (35, 45)]
                }

                # Plot Gantt chart with improved styling
                y_positions = {'v1': 3, 'v2': 2, 'v3': 1}
                colors = {'v1': '#FF5722', 'v2': '#9C27B0', 'v3': '#FFC107'}

                for vehicle, intervals in times.items():
                    for i, (start, end) in enumerate(intervals):
                        ax.barh(y_positions[vehicle], end - start, left=start, height=0.5,
                               color=colors[vehicle], alpha=0.8, edgecolor='white', linewidth=1)
                        ax.text(start + (end - start) / 2, y_positions[vehicle],
                               f"{i}", ha='center', va='center', color='white', fontweight='bold')

                ax.set_yticks(list(y_positions.values()))
                ax.set_yticklabels(list(y_positions.keys()))
                ax.set_title("Vehicle Schedules", pad=20)
                ax.set_xlabel("Time")
                ax.set_ylabel("Vehicle")
                plt.tight_layout()

                st.pyplot(fig)

            with viz_tabs[2]:
                st.markdown("""
                <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color: #333; margin-bottom: 1rem;'>Performance Metrics</h3>
                </div>
                """, unsafe_allow_html=True)

                # Create metrics with improved styling
                metrics = {
                    "Total Distance": 150.75,
                    "Total Duration": 180.5,
                    "Vehicle Utilization": 0.85,
                    "Average Waiting Time": 12.3,
                    "Requests Served": 5,
                    "Vehicles Used": 3
                }

                # Display metrics in a grid
                col1, col2, col3 = st.columns(3)
                with col1:
                    for metric, value in list(metrics.items())[:2]:
                        st.metric(metric, f"{value:.2f}")
                with col2:
                    for metric, value in list(metrics.items())[2:4]:
                        st.metric(metric, f"{value:.2f}")
                with col3:
                    for metric, value in list(metrics.items())[4:]:
                        st.metric(metric, value)

                # Add a chart with improved styling
                st.markdown("""
                <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 2rem;'>
                    <h4 style='color: #333; margin-bottom: 1rem;'>Vehicle Utilization</h4>
                </div>
                """, unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(10, 6))
                plt.style.use('default')  # Use default style as base
                
                # Set custom style parameters
                plt.rcParams.update({
                    'font.size': 12,
                    'axes.labelsize': 12,
                    'axes.titlesize': 16,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'axes.grid': True,
                    'grid.alpha': 0.3,
                    'axes.facecolor': 'white',
                    'figure.facecolor': 'white'
                })

                vehicle_utils = {
                    'v1': 0.75,
                    'v2': 0.90,
                    'v3': 0.85
                }

                colors = ['#FF5722', '#9C27B0', '#FFC107']
                bars = ax.bar(vehicle_utils.keys(), vehicle_utils.values(), color=colors)
                ax.set_ylim(0, 1)
                ax.set_title("Vehicle Utilization", pad=20)
                ax.set_xlabel("Vehicle")
                ax.set_ylabel("Utilization Rate")

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f"{height:.2f}", ha='center', va='bottom')

                plt.tight_layout()
                st.pyplot(fig)

                # Export options with improved styling
                st.markdown("""
                <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 2rem;'>
                    <h3 style='color: #333; margin-bottom: 1rem;'>Export Results</h3>
                </div>
                """, unsafe_allow_html=True)

                # Create a results dataframe
                results_df = pd.DataFrame([
                    {"vehicle": route["vehicle_id"], "route": " → ".join(route["route"])}
                    for route in st.session_state['solution']['routes']
                ])

                st.dataframe(results_df, use_container_width=True)

                # Download buttons with improved styling
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        download_link(results_df, "fleetpy_results.csv", "📥 Download Results CSV"),
                        unsafe_allow_html=True
                    )
                with col2:
                    results_json = json.dumps(st.session_state['solution'], indent=2)
                    st.markdown(
                        download_link(results_json, "fleetpy_results.json", "📥 Download Results JSON"),
                        unsafe_allow_html=True
                    )
        else:
            st.info("No solution available. Please run the solver first!")

        # # Navigation buttons
        # st.markdown("---")
        # col1, col2, col3 = st.columns([1, 1, 1])
        # with col1:
        #     if st.button("← Back: Fleet Management", key="back_fleet_management"):
        #         st.session_state.current_tab = "Fleet Management"
        #         st.rerun()
        # with col2:
        #     if st.button("Start Over", key="start_over"):
        #         st.session_state.current_tab = "Home"
        #         st.rerun()


def run_fleetpy():
    # TODO parameterize
    scs_path = os.path.join(os.path.dirname(__file__), "studies", "example_study", "scenarios")
    cc = os.path.join(scs_path, "constant_config_ir.csv")
    sc = os.path.join(scs_path, "example_ir_only.csv")
    log_level = "info"
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)


if __name__ == "__main__":
    main()
