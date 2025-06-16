import os
import sys
import streamlit as st
import multiprocessing as mp
import traceback
from pathlib import Path

# Add FleetPy path to system path
fleetpy_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(fleetpy_path)

from run_examples import run_scenarios
from src.scenario_gui.scenario_creator import ScenarioCreator, MODULE_PARAM_TO_DICT_LOAD, parameter_docs
from src.scenario_gui.parameter_utils import get_abnormal_param_options, categorize_parameters
from src.scenario_gui.ui_utils import render_parameter_input, apply_sidebar_styles
from src.FleetSimulationBase import INPUT_PARAMETERS_FleetSimulationBase


def create_scenario_page():
    st.title("Create Scenario")
    st.markdown("Use this page to create a new simulation scenario.")

    # Initialize the scenario creator if not already in session state
    if 'scenario_creator' not in st.session_state:
        st.session_state.scenario_creator = ScenarioCreator()
        st.session_state.current_step = "modules"
        st.session_state.network_selected = ""
        st.session_state.demand_selected = ""

    sc = st.session_state.scenario_creator

    # Initialize the active tab in session state if it doesn't exist
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "modules"

    # Create radio for tab selection
    selected_tab = st.radio("", ["1. Select Modules", "2. Configure Parameters"], 
                          index=0 if st.session_state.active_tab == "modules" else 1,
                          horizontal=True,
                          label_visibility="collapsed")

    # Create tabs and show content based on selection
    if selected_tab == "1. Select Modules":
        st.header("Module Selection")
        st.write("Select required and optional modules for your scenario.")

        # Mandatory Modules
        st.subheader("Mandatory Modules")
        for module in sc._current_mandatory_modules:
            options = ["Choose..."] + list(MODULE_PARAM_TO_DICT_LOAD[module]().keys()) if MODULE_PARAM_TO_DICT_LOAD.get(module) else ["Choose..."]
            selected = st.selectbox(
                f"{module}",
                options=options,
                key=f"mod_mandatory_{module}",
                help=parameter_docs[module]
            )
            if selected != "Choose...":
                sc.select_module(module, selected)

        # Optional Modules
        st.subheader("Optional Modules")
        for module in sc._current_optional_modules:
            options = ["None"] + list(MODULE_PARAM_TO_DICT_LOAD[module]().keys()) if MODULE_PARAM_TO_DICT_LOAD.get(module) else ["None"]
            selected = st.selectbox(
                f"{module}",
                options=options,
                key=f"mod_optional_{module}",
                help=parameter_docs[module]
            )
            if selected != "None":
                sc.select_module(module, selected)

        # Add Next button at the bottom of module selection
        st.markdown("---")  # Add a visual separator
        if st.button("Next", key="module_next_button"):
            # Check if all mandatory modules are selected
            mandatory_modules_selected = all(
                sc._currently_selected_modules.get(module) is not None
                for module in sc._current_mandatory_modules
            )
            
            if mandatory_modules_selected:
                # Use session state to switch to parameters tab
                st.session_state.active_tab = "parameters"
                # Force a rerun to switch tabs
                st.rerun()
            else:
                st.error("Please select all mandatory modules before proceeding.")

    elif selected_tab == "2. Configure Parameters":
        st.header("Parameter Selection")
        
        # Keep track of seen parameters
        seen_params = set()

        # Special handling for Basic Settings which includes network and demand selection
        with st.expander("Basic Settings", expanded=True):
            # Handle special case parameters (network, demand, rq_file)
            basic_params = ["network_name", "demand_name", "rq_file", "scenario_name", "study_name"]
            col1, col2 = st.columns(2)
            
            # First handle network and demand selection as they're needed for rq_file
            for param in ["network_name", "demand_name"]:
                if param not in sc._current_mandatory_params and param not in sc._current_optional_params:
                    continue
                    
                seen_params.add(param)
                col = col1 if param == "network_name" else col2
                
                with col:
                    abnormal_options = get_abnormal_param_options(param)
                    param_obj = sc.parameter_dict.get(param)
                    
                    if param_obj:
                        help_text = param_obj.doc_string if hasattr(param_obj, 'doc_string') else ""
                        
                        if abnormal_options:
                            selected = st.selectbox(
                                f"{param}",
                                options=abnormal_options,
                                key=f"param_mandatory_{param}",
                                help=help_text
                            )
                            
                            if param == "network_name" and selected:
                                st.session_state.network_selected = selected
                            elif param == "demand_name" and selected:
                                st.session_state.demand_selected = selected
                            
                            if selected:
                                sc.select_param(param, selected)
            
            # Now handle rq_file after network and demand are set
            if "rq_file" in sc._current_mandatory_params or "rq_file" in sc._current_optional_params:
                seen_params.add("rq_file")
                param_obj = sc.parameter_dict.get("rq_file")
                if param_obj:
                    help_text = param_obj.doc_string if hasattr(param_obj, 'doc_string') else ""
                    
                    if st.session_state.network_selected and st.session_state.demand_selected:
                        rq_path = os.path.join(fleetpy_path, "data", "demand", 
                                            st.session_state.demand_selected, "matched",
                                            st.session_state.network_selected)
                        if os.path.exists(rq_path):
                            rq_options = [""] + os.listdir(rq_path)
                            selected = st.selectbox(
                                "rq_file",
                                options=rq_options,
                                key="param_mandatory_rq_file",
                                help=help_text
                            )
                            if selected:
                                sc.select_param("rq_file", selected)
                        else:
                            st.warning("No request files found for the selected network and demand.")
                    else:
                        st.info("Please select both network and demand to view available request files.")
            
            # Handle remaining basic parameters
            for param in ["scenario_name", "study_name"]:
                if param not in sc._current_mandatory_params and param not in sc._current_optional_params:
                    continue
                    
                seen_params.add(param)
                col = col1 if param == "scenario_name" else col2
                
                with col:
                    param_obj = sc.parameter_dict.get(param)
                    if param_obj:
                        help_text = param_obj.doc_string if hasattr(param_obj, 'doc_string') else ""
                        value = st.text_input(
                            f"{param}",
                            key=f"param_mandatory_{param}",
                            help=help_text
                        )
                        if value:
                            sc.select_param(param, value)

        # Categorize remaining mandatory parameters
        remaining_mandatory = [p for p in sc._current_mandatory_params if p not in seen_params]
        mandatory_categories = categorize_parameters(remaining_mandatory, sc.parameter_dict)

        # Render mandatory parameters by category
        for category, params in mandatory_categories.items():
            with st.expander(category, expanded=True):
                col1, col2 = st.columns(2)
                for i, param in enumerate(params):
                    if param in seen_params:
                        continue
                    seen_params.add(param)
                    
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        param_obj = sc.parameter_dict.get(param)
                        if param_obj:
                            value = render_parameter_input(param, param_obj)
                            if value:
                                sc.select_param(param, value)

                            # Special handling for nr_mod_operators
                            if param == "nr_mod_operators" and value and value != "1":
                                try:
                                    if int(value) > 1:
                                        st.info("You have selected more than one operator. Please make sure to select different parameters for each operator. "
                                            "Otherwise, the parameters will be the same for all operators. You can do this by separating the different "
                                            "parameters by a comma ',' for all parameters starting with op_")
                                except ValueError:
                                    pass

        # Categorize optional parameters
        remaining_optional = [p for p in sc._current_optional_params if p not in seen_params]
        optional_categories = categorize_parameters(remaining_optional, sc.parameter_dict)

        # Render optional parameters by category
        st.subheader("Optional Parameters")
        for category, params in optional_categories.items():
            with st.expander(category, expanded=False):
                col1, col2 = st.columns(2)
                for i, param in enumerate(params):
                    if param in seen_params:
                        continue
                    seen_params.add(param)
                    
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        param_obj = sc.parameter_dict.get(param)
                        if param_obj:
                            value = render_parameter_input(param, param_obj, "optional_")
                            if value:
                                sc.select_param(param, value)

        # Save Button
        if st.button("Save Scenario"):
            try:
                scenario_path = sc.create_filled_scenario_df()
                st.success(f"Scenario saved successfully to {scenario_path}")
            except Exception as e:
                st.error(f"Error saving scenario: {str(e)}")


def run_simulation_page():
    st.title("FleetPy Simulation Manager")
    st.write("Upload or select configuration files to run FleetPy simulations")

    # Main area organization
    st.header("Configuration Files")

    # Create two columns for the settings
    col1, col2 = st.columns(2)

    with col1:
        # Log level selection
        log_level = st.selectbox(
            "Log Level",
            ["info", "debug", "warning", "error"],
            index=0
        )

    with col2:
        # CPU configuration
        n_cpu_per_sim = st.number_input(
            "CPUs per Simulation",
            min_value=1,
            max_value=mp.cpu_count(),
            value=1
        )

        n_parallel_sim = st.number_input(
            "Parallel Simulations",
            min_value=1,
            max_value=mp.cpu_count(),
            value=1
        )
    
    st.divider()

    # File upload option
    upload_method = st.radio(
        "Choose how to provide configuration files",
        ["Upload Files", "Select from Existing Files"]
    )

    const_cfg_file = None
    scenario_cfg_file = None

    if upload_method == "Upload Files":
        # Get list of existing studies for reference
        studies_path = os.path.join(fleetpy_path, "studies")
        existing_studies = [d for d in os.listdir(studies_path) 
                          if os.path.isdir(os.path.join(studies_path, d))]
        existing_studies.sort()
        
        # Study name input with existing studies as suggestions
        study_name = st.text_input(
            "Study Name",
            placeholder="Enter a name for your study",
            help="This will be used to organize your configuration files"
        )
        
        # Show existing studies as reference
        with st.expander("View Existing Studies"):
            st.write("Existing studies for reference:")
            for study in existing_studies:
                st.write(f"- {study}")
        
        const_cfg = st.file_uploader("Upload Constant Configuration File (YAML/CSV)", type=['yaml', 'csv'])
        scenario_cfg = st.file_uploader("Upload Scenario Configuration File (YAML/CSV)", type=['yaml', 'csv'])
        
        if study_name and const_cfg and scenario_cfg:
            # Create study directory structure
            study_path = os.path.join(studies_path, study_name)
            scenarios_path = os.path.join(study_path, "scenarios")
            
            # Create directories if they don't exist
            os.makedirs(scenarios_path, exist_ok=True)
            
            # Generate file paths preserving original extensions
            const_ext = os.path.splitext(const_cfg.name)[1] if const_cfg.name else ".yaml"
            scenario_ext = os.path.splitext(scenario_cfg.name)[1] if scenario_cfg.name else ".csv"
            
            const_cfg_file = os.path.join(scenarios_path, f"const_cfg{const_ext}")
            scenario_cfg_file = os.path.join(scenarios_path, f"scenario_cfg{scenario_ext}")
            
            # Save uploaded files
            with open(const_cfg_file, "wb") as f:
                f.write(const_cfg.getvalue())
            with open(scenario_cfg_file, "wb") as f:
                f.write(scenario_cfg.getvalue())
                
            st.success(f"Configuration files saved in study: {study_name}")
        elif const_cfg and scenario_cfg:
            st.error("Please enter a study name before uploading files")

    else:
        # Get list of available studies
        studies_path = os.path.join(fleetpy_path, "studies")
        studies = []
        
        # Get list of valid studies (ones with scenarios directory)
        for study in os.listdir(studies_path):
            study_path = os.path.join(studies_path, study)
            if os.path.isdir(study_path):
                scenarios_path = os.path.join(study_path, "scenarios")
                if os.path.exists(scenarios_path):
                    studies.append(study)
        
        # Sort studies alphabetically
        studies.sort()
        
        # Study selection dropdown
        selected_study = st.selectbox(
            "Select Study",
            studies,
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Get config files for selected study
        config_files = []
        if selected_study:
            scenarios_path = os.path.join(studies_path, selected_study, "scenarios")
            for file in os.listdir(scenarios_path):
                if file.endswith(('.yaml', '.csv')):
                    config_files.append(os.path.join(scenarios_path, file))

        # Sort and filter files for constant config
        def sort_key_for_const(filepath):
            filename = os.path.basename(filepath).lower()
            if "const" in filename:
                return (0, filename)
            return (1, filename)

        # Sort and filter files for scenario config
        def sort_key_for_scenario(filepath):
            filename = os.path.basename(filepath).lower()
            if "scenario" in filename:
                return (0, filename)
            if "example" in filename:
                return (1, filename)
            return (2, filename)

        # Sort files separately for each config type
        const_sorted_files = sorted(config_files, key=sort_key_for_const)
        scenario_sorted_files = sorted(config_files, key=sort_key_for_scenario)

        const_cfg_file = st.selectbox(
            "Select Constant Configuration File",
            const_sorted_files,
            format_func=lambda x: os.path.basename(x)
        )
        
        scenario_cfg_file = st.selectbox(
            "Select Scenario Configuration File",
            scenario_sorted_files,
            format_func=lambda x: os.path.basename(x)
        )

    # Add space before run section
    st.write("")
    st.write("")
    
    # Run simulation section
    st.header("Run Simulation")
    
    # Show run button and configuration summary
    if const_cfg_file and scenario_cfg_file:
        # Show configuration summary and preview
        st.write("Configuration Summary:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("📄 **Constant Config:**")
            st.write(f"`{os.path.basename(const_cfg_file)}`")
            with st.expander("Preview Constant Config", expanded=False):
                try:
                    with open(const_cfg_file, 'r') as f:
                        content = f.read()
                        if const_cfg_file.endswith('.yaml'):
                            st.code(content, language='yaml')
                        else:
                            st.code(content)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        with col2:
            st.write("📄 **Scenario Config:**")
            st.write(f"`{os.path.basename(scenario_cfg_file)}`")
            with st.expander("Preview Scenario Config", expanded=False):
                try:
                    with open(scenario_cfg_file, 'r') as f:
                        content = f.read()
                        if scenario_cfg_file.endswith('.yaml'):
                            st.code(content, language='yaml')
                        else:
                            st.code(content)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Show run button centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("▶️ Run Simulation", use_container_width=True):
                try:
                    with st.spinner("Running simulation..."):
                        # Run the simulation
                        run_scenarios(
                            const_cfg_file,
                            scenario_cfg_file,
                            log_level=log_level,
                            n_cpu_per_sim=n_cpu_per_sim,
                            n_parallel_sim=n_parallel_sim
                        )
                    
                    # Show success message with study name
                    study_name = os.path.basename(os.path.dirname(os.path.dirname(const_cfg_file)))
                    results_path = os.path.join(os.path.dirname(os.path.dirname(const_cfg_file)), "results")
                    st.success(f"✅ Simulation completed successfully!")
                    st.info(f"Results saved in: `{results_path}`")
                
                except Exception as e:
                    st.error("❌ An error occurred during simulation:")
                    st.error(str(e))
                    st.code(traceback.format_exc())
    else:
        st.info("Please select both configuration files to run the simulation")


def main():
    # Apply sidebar styles
    apply_sidebar_styles()
    
    st.sidebar.title("FleetPy")
    st.sidebar.markdown('<div style="margin-bottom: 0.5rem;"></div>', unsafe_allow_html=True)
    
    # Navigation buttons
    clicked = None
    if st.sidebar.button(
        "📝 Create Scenario", 
        key="nav_create", 
        help="Create a new simulation scenario",
        use_container_width=True,
        type="secondary"
    ):
        clicked = "create"
    
    st.sidebar.markdown('<div style="margin-bottom: 0.5rem;"></div>', unsafe_allow_html=True)
    
    if st.sidebar.button(
        "⚡️ Run Simulation", 
        key="nav_run", 
        help="Configure and run FleetPy simulations",
        use_container_width=True,
        type="secondary"
    ):
        clicked = "run"

    st.sidebar.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Initialize the page selection in session state if not already present
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "create"

    # Update current page based on button clicks
    if clicked:
        st.session_state.current_page = clicked

    # Show the appropriate page based on selection
    if st.session_state.current_page == "run":
        run_simulation_page()
    else:
        create_scenario_page()


if __name__ == "__main__":
    mp.freeze_support()
    main()
