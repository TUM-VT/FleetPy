# Exemplary FleetPy applications:

Steps for reproducing the Manhattan case study example:

1. Clone the [FleetPy repository](https://github.com/TUM-VT/FleetPy/)   
   ```bash
   git clone https://github.com/TUM-VT/FleetPy.git
   cd FleetPy
   ```
2. Set up a conda environment

    Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) installed.
   ```bash
   conda env create -f environment.yml
   conda activate fleetpy
   ```
   More details on installation instructions is described in the [README file](https://github.com/TUM-VT/FleetPy/blob/main/README.md#-installation) 
3. Download the Manhattan benchmark dataset [here](https://zenodo.org/records/15187906/files/FleetPy_Manhattan.zip?download=1)
   The demand, network, and zones folder should be unzipped to the `FleetPy/data/` directory. The folder structure should look like this:
   ```
    FleetPy/data/
    ├── demand/
    │   ├── Manhattan_2018
    │   ├── ...
    ├── network/
    │   ├── Manhattan_2019_corrected
    │   ├── ...
    ├── zones/
    │   ├── Manhattan_corrected_6min_max
    │   ├── Manhattan_corrected_8min_min
    │   ├── Manhattan_corrected_12min_max
    │   ├── Manhattan_Tax_Zones
    │   ├── ...
4. Place the unzipped folder in Fleetpy under the `studies` directory:
   ```
   FleetPy/studies/fleetpy_example_applications/
   ```
   The folder should contain the following files:
   - `README.md`
   - `run_manhattan_case_study_example.py`
   - `scenarios/`
     - `const_cfg_ex1_batch_assignment_manhattan.yaml`
     - `const_cfg_ex2_n_operator_manhattan.csv`
     - `scenario_cfg_ex1_batch_assignment_manhattan.csv`
     - `scenario_cfg_ex2_n_operator_manhattan.csv`
4. To run the scenarios, use the following command in the terminal:
   ```bash
   # Navigate to the example directory
   cd studies/fleetpy_example_applications/
   # Run the script
   python run_manhattan_case_study_example.py
   ```
   Optionally, you can run the scenarios in parallel. Change the `n_parallel_sim` argument in the `run_manhattan_case_study_example.py` file to the desired number of parallel simulations. For example, to run 4 simulations in parallel, set:
   ```python
   run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=4)
   ```
5. The results will be saved in the `results/` directory within the `fleetpy_example_applications` folder.  
