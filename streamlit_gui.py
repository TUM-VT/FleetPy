import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import zipfile
import json
import base64
from datetime import datetime


from run_examples import run_scenarios

# Set page config
st.set_page_config(page_title="FleetPy Web Interface", layout="wide")


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

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page",
                                ["Home", "Create Scenario", "Upload Files", "Download Samples", "Visualize Results"])

    if page == "Home":
        st.header("Welcome to FleetPy Web Interface")
        st.write("""
        This application allows you to interact with the FleetPy library for vehicle routing and scheduling.
        Use the sidebar to navigate between different functionalities.
        """)

        st.subheader("Quick Start")
        st.markdown("""
        1. Download sample files to understand the required format
        2. Create a new scenario or upload your own files
        3. Run the solver and visualize results
        """)

    elif page == "Create Scenario":
        st.header("Create New Scenario")

        # Basic scenario setup
        scenario_name = st.text_input("Scenario Name", "my_scenario")

        st.subheader("Algorithm Settings")
        algorithm = st.selectbox("Algorithm",
                                 ["insertion_heuristic", "local_search", "column_generation"])

        objective = st.selectbox("Objective",
                                 ["minimize_vehicles", "minimize_distance", "minimize_duration"])

        max_time = st.slider("Max Computation Time (seconds)", 10, 600, 60)

        random_seed = st.number_input("Random Seed", 0, 9999, 42)

        st.subheader("Advanced Settings")
        show_advanced = st.checkbox("Show Advanced Settings")

        if show_advanced:
            st.write("Advanced settings go here...")

        # Create config dictionary
        config = {
            "scenario_name": scenario_name,
            "algorithm": algorithm,
            "objective": objective,
            "max_computation_time": max_time,
            "random_seed": random_seed
        }

        if st.button("Save Configuration"):
            # Store configuration in session state
            st.session_state['config'] = config
            st.success(f"Configuration for scenario '{scenario_name}' saved!")

            # Display the configuration
            st.json(config)

            # Create download link for the config
            config_json = json.dumps(config, indent=2)
            st.markdown(download_link(config_json, f"{scenario_name}_config.json", "Download Configuration"),
                        unsafe_allow_html=True)

    elif page == "Upload Files":
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
        config_file = st.file_uploader("Choose a JSON file", type="json", key="config")
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
                # st.rerun()
            else:
                st.error("Please upload all required files first!")

    elif page == "Download Samples":
        st.header("Download Sample Files")

        # Create sample files
        vehicles_df, requests_df, config = create_sample_files()

        # Display sample files
        st.subheader("Sample Vehicles File")
        st.write(vehicles_df)

        st.subheader("Sample Requests File")
        st.write(requests_df)

        st.subheader("Sample Configuration")
        st.json(config)

        # Create download links
        st.subheader("Download Links")

        # Convert dataframes to CSV
        vehicles_csv = vehicles_df.to_csv(index=False)
        requests_csv = requests_df.to_csv(index=False)
        config_json = json.dumps(config, indent=2)

        # Create download links
        st.markdown(download_link(vehicles_csv, "sample_vehicles.csv", "Download Sample Vehicles CSV"),
                    unsafe_allow_html=True)
        st.markdown(download_link(requests_csv, "sample_requests.csv", "Download Sample Requests CSV"),
                    unsafe_allow_html=True)
        st.markdown(download_link(config_json, "sample_config.json", "Download Sample Configuration JSON"),
                    unsafe_allow_html=True)

        # Create a zip file with all samples
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            zip_file.writestr('sample_vehicles.csv', vehicles_csv)
            zip_file.writestr('sample_requests.csv', requests_csv)
            zip_file.writestr('sample_config.json', config_json)

        zip_buffer.seek(0)
        b64 = base64.b64encode(zip_buffer.read()).decode()
        zip_link = f'<a href="data:application/zip;base64,{b64}" download="fleetpy_samples.zip">Download All Sample Files</a>'
        st.markdown(zip_link, unsafe_allow_html=True)

    elif page == "Visualize Results":
        st.header("Visualization")

        if 'solution' in st.session_state:
            st.subheader("Solution Overview")
            st.write(f"Solved at: {st.session_state.get('solution_timestamp', 'Unknown')}")
            st.write(f"Objective Value: {st.session_state['solution']['objective_value']}")
            st.write(f"Computation Time: {st.session_state['solution']['computation_time']} seconds")

            st.subheader("Routes")
            for i, route in enumerate(st.session_state['solution']['routes']):
                st.write(f"Vehicle {route['vehicle_id']}: {' → '.join(route['route'])}")

            # Visualization tabs
            viz_tabs = st.tabs(["Route Map", "Gantt Chart", "Performance Metrics"])

            with viz_tabs[0]:
                st.subheader("Route Map")
                # Here you would normally use fleetpy's mapping functionality
                # For now, we'll just create a simple plot

                # Create a simple plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # Dummy locations
                locations = {
                    'depot': (0, 0),
                    'A': (1, 2), 'F': (3, 2),
                    'B': (2, 3), 'G': (4, 3),
                    'C': (1, 4), 'H': (3, 4),
                    'D': (2, 1), 'I': (4, 1),
                    'E': (1, 0), 'J': (3, 0)
                }

                # Plot locations
                for loc, (x, y) in locations.items():
                    ax.scatter(x, y, s=100, c='blue' if loc == 'depot' else 'green')
                    ax.text(x, y, loc, fontsize=12)

                # Plot routes
                colors = ['red', 'orange', 'purple']
                for i, route in enumerate(st.session_state['solution']['routes']):
                    route_points = route['route']
                    for j in range(len(route_points) - 1):
                        start = locations[route_points[j]]
                        end = locations[route_points[j + 1]]
                        ax.plot([start[0], end[0]], [start[1], end[1]], color=colors[i], linewidth=2)

                ax.set_title("Route Map")
                ax.set_xlabel("X coordinate")
                ax.set_ylabel("Y coordinate")
                ax.grid(True)

                st.pyplot(fig)

            with viz_tabs[1]:
                st.subheader("Gantt Chart")
                # Create a simple Gantt chart
                fig, ax = plt.subplots(figsize=(10, 6))

                # Dummy time data
                times = {
                    'v1': [(0, 10), (10, 20), (20, 30)],
                    'v2': [(0, 15), (15, 25), (25, 40), (40, 50)],
                    'v3': [(5, 15), (15, 25), (25, 35), (35, 45)]
                }

                # Plot Gantt chart
                y_positions = {'v1': 3, 'v2': 2, 'v3': 1}
                colors = {'v1': 'red', 'v2': 'orange', 'v3': 'purple'}

                for vehicle, intervals in times.items():
                    for i, (start, end) in enumerate(intervals):
                        ax.barh(y_positions[vehicle], end - start, left=start, height=0.5,
                                color=colors[vehicle], alpha=0.7)
                        ax.text(start + (end - start) / 2, y_positions[vehicle],
                                f"{i}", ha='center', va='center')

                ax.set_yticks(list(y_positions.values()))
                ax.set_yticklabels(list(y_positions.keys()))
                ax.set_title("Vehicle Schedules")
                ax.set_xlabel("Time")
                ax.set_ylabel("Vehicle")
                ax.grid(True, axis='x')

                st.pyplot(fig)

            with viz_tabs[2]:
                st.subheader("Performance Metrics")

                # Create some dummy metrics
                metrics = {
                    "Total Distance": 150.75,
                    "Total Duration": 180.5,
                    "Vehicle Utilization": 0.85,
                    "Average Waiting Time": 12.3,
                    "Requests Served": 5,
                    "Vehicles Used": 3
                }

                # Display metrics
                col1, col2 = st.columns(2)

                with col1:
                    for metric, value in list(metrics.items())[:3]:
                        st.metric(metric, value)

                with col2:
                    for metric, value in list(metrics.items())[3:]:
                        st.metric(metric, value)

                # Add a chart
                fig, ax = plt.subplots(figsize=(10, 6))

                # Vehicle utilization chart
                vehicle_utils = {
                    'v1': 0.75,
                    'v2': 0.90,
                    'v3': 0.85
                }

                ax.bar(vehicle_utils.keys(), vehicle_utils.values(), color=['red', 'orange', 'purple'])
                ax.set_ylim(0, 1)
                ax.set_title("Vehicle Utilization")
                ax.set_xlabel("Vehicle")
                ax.set_ylabel("Utilization Rate")

                for i, v in enumerate(vehicle_utils.values()):
                    ax.text(i, v + 0.05, f"{v:.2f}", ha='center')

                st.pyplot(fig)

                # Export options
                st.subheader("Export Results")

                # Create a results dataframe
                results_df = pd.DataFrame([
                    {"vehicle": route["vehicle_id"], "route": " → ".join(route["route"])}
                    for route in st.session_state['solution']['routes']
                ])

                st.write(results_df)

                # Download results
                st.markdown(
                    download_link(results_df, "fleetpy_results.csv", "Download Results CSV"),
                    unsafe_allow_html=True
                )

                # JSON results
                results_json = json.dumps(st.session_state['solution'], indent=2)
                st.markdown(
                    download_link(results_json, "fleetpy_results.json", "Download Results JSON"),
                    unsafe_allow_html=True
                )
        else:
                st.info("No solution available. Please run the solver first!")
                st.button("Go to Upload Files", on_click=lambda: st.session_state.update({"page": "Upload Files"}))


def run_fleetpy():
    scs_path = os.path.join(os.path.dirname(__file__), "studies", "example_study", "scenarios")
    cc = os.path.join(scs_path, "constant_config_ir.csv")
    sc = os.path.join(scs_path, "example_ir_only.csv")
    log_level = "info"
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)


if __name__ == "__main__":
    main()
