# 🚖 FleetPy – Open-Source Fleet Simulation Framework  

[![GitHub stars](https://img.shields.io/github/stars/TUM-VT/FleetPy?style=social)](https://github.com/TUM-VT/FleetPy)  
FleetPy is an open-source **fleet simulation framework** for modeling and controlling vehicle fleets in **ride-sharing, autonomous mobility, and on-demand transport** applications.  

> Whether you're a **researcher**, **transportation engineer**, or **mobility innovator**, FleetPy helps you analyze and optimize **fleet operations, routing strategies, and demand-responsive services**.

📖 **[Read the Wiki](https://github.com/TUM-VT/FleetPy/wiki)** | 🛠 **[Installation Guide](#-installation)** | 🚀 **[Quickstart](#-quickstart)**  

---

## 🎯 Key Features  

✅ **Agent-Based Simulation** – Models individual vehicles, passengers, and operators.  
✅ **Flexible User-Operator Interaction** – Supports multiple request-acceptance models.  
✅ **Multi-Fleet Management** – Simulates **ride-pooling, dispatching, and EV charging**.  
✅ **Customizable Routing Algorithms** – Choose between memory-efficient and fast methods.  
✅ **Modular & Extensible Design** – Easily integrate **new demand models, routing strategies, or data sources**.  
✅ **Optimized for Large-Scale Scenarios** – Handles thousands of vehicles and requests efficiently.  

---

## 🚀 Quickstart  

Get up and running in just a few commands!  

```bash
# Clone the repository
git clone https://github.com/TUM-VT/FleetPy.git
cd FleetPy

# Set up the virtual environment
conda env create -f environment.yml
conda activate fleetpy

# Run an example simulation
python3 run_examples.py
```

✔ The results will be saved in `FleetPy/studies/example_study/results/`  
✔ To visualize the results, run:  

```bash
python3 replay_pyplot.py scenario_result_directory
```

---

## 🛠 Installation  

### 1️⃣ Install with Conda (Recommended)

Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) installed.

```bash
# Fetch the latest updates
git clone https://github.com/TUM-VT/FleetPy.git
cd FleetPy

# Set up a Conda environment
conda env create -f environment.yml
conda activate fleetpy
```

### 2️⃣ Install C++ Router (Recommended)

For improved routing efficiency, compile the C++ router:

```bash
cd FleetPy/src/routing/cpp_router
python setup.py build_ext --inplace
```

**Ensure a C++ compiler and Cython are installed.**

### 3️⃣ Install Optimizer (Optional)

For advanced optimization tasks, install the necessary optimizers:

#### 🏆 **Gurobi**

To install Gurobi:

```bash
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```

You can acquire a **free academic license** from [Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/). 🎓

#### ⚡ **OR-Tools**

To install Google's OR-Tools:

```bash
pip install ortools
```

This is useful for combinatorial optimization and routing problems.

---

## 📂 Data Preparation and Study Setup

For now, you can inspect the data structures and files in the examples provided in GitHub:

* 📁 `FleetPy/data`
* 📁 `FleetPy/studies`

More detailed descriptions of the **data structure, preprocessing steps, and result data** will be provided in upcoming versions.
Additionally, a **GUI to set up scenarios** (with a choice of submodules and data) is planned for the future. 🎨

In general, you can **save your data and study definitions** in the mentioned directories. These are included in `.gitignore`.

<!-- ... (prepare study by config.csv and scenarios.csv) -->
<!-- ... (necessary modules for preprocessing can be installed by "pip3 install -r requirements_with_pp.txt") -->

---

## 📊 Running a Simulation  

To test an example scenario:  

```bash
python3 run_examples.py
```

✔ The output will be stored in:  
📂 `FleetPy/studies/example_study/results/`  

### 📊 Output Files

| File Name            | Description |
|----------------------|-------------|
| `1_user_stats.csv`   | User request statistics |
| `2_0_op-stats.csv`   | Fleet vehicle task logs |
| `standard_eval.csv`  | Aggregated evaluation results |

To analyze trends:

```bash
python3 FleetPy/src/evaluation/temporal.py scenario_result_directory
```

---

## 🎥 Watch a Replay  

To visualize a **previously simulated scenario**, run:  

```bash
python3 replay_pyplot.py scenario_result_directory <sim_seconds_per_real_second>
```

🛑 *(Optional: Specify start time as an additional argument.)*

---

## 🤝 Contributing  

We welcome contributions from the community! 🚀  
📌 **How to contribute:**  

1. Fork the repo & create a feature branch (`git checkout -b new-feature`).  
2. Commit your changes (`git commit -m "Added a cool feature"`).  
3. Push to your branch (`git push origin new-feature`).  
4. Open a Pull Request!  

<!-- ... TODO: contributing.md 
🔍 Check out [CONTRIBUTING.md](link) for guidelines.  
-->

---

## 📢 Get Involved  

💬 **Join the discussion:** [GitHub Discussions](https://github.com/TUM-VT/FleetPy/discussions)  
💻 **Contribute:** Open an issue or PR!  
⭐ **Support FleetPy:** Give us a **star ⭐** on GitHub!  
🚀 **Stay updated:** Follow us on [Homepage](https://www.mos.ed.tum.de/en/vt/home/) & [LinkedIn](https://www.linkedin.com/school/tum-chair-of-traffic-engineering-and-control/)  

---

## 📌 Tested on  

✔ **Windows 10 Pro x64**  
✔ **Chrome 79.0.3945**  
✔ **Python 3.7**  

---

## 🧪 Research Projects Using FleetPy

FleetPy has been applied in various academic and applied research projects across topics like shared mobility, autonomous vehicle operations, and electrified fleets:

### Ongoing Projects

* **[CONDUCTOR](https://conductor-project.eu/?show=consortium)** – Development of an integrated ride-parcel-pooling service using automated vehicles (CCAM), evaluated via FleetPy-Aimsun coupling *(multi-modal integration, AVs, cooperative routing, traffic control, simulation coupling)*
* **[metaCCAZE](https://www.metaccaze-project.eu/)** – Scientific support and simulation-based evaluation of Munich’s Living Lab innovations including multimodal logistics hubs, dynamic curbside management, and connected/autonomous last-mile vehicles. FleetPy contributes to monitoring and optimizing operations *(multimodal hubs, curbside management, AV logistics, LL evaluation, policy support)*
* **[MINGA](https://www.mos.ed.tum.de/en/vt/research/projects/current-projects/minga/)** – Evaluation of AV stop concepts and passenger interactions at TUM’s test field. FleetPy is used alongside MATSim and microscopic traffic simulations to model AV and non-AV traffic, evaluate climate and system-wide impacts, and support accessibility-focused user studies *(AV integration, stop concepts, MATSim coupling, KPI evaluation, accessibility research)*
* **[STADT:up](https://www.mos.ed.tum.de/en/vt/research/projects/current-projects/stadtup/)** – Simulation of autonomous shuttle bus in different inter-mobility scenarios with dynamic stops *(Operation, traffic evaluation)*
* **[SUM (Seamless Shared Urban Mobility)](https://sum-project.eu/)** – Simulation-based evaluation of operational strategies for automated on-demand fleets in Munich using FleetPy, including integration with public transport, pricing strategies, and transferability to future mobility hubs *(on-demand AV fleets, public transport integration, pricing, living lab scalability)*

### Completed Projects

* **[MOIA](https://www.mos.ed.tum.de/en/vt/forschung/projekte/abgeschlossene-projekte/moia-accompanying-research-completed-2021/)** – Implementation of algorithms to represent MOIA’s fleet control *(operation, ride-pooling)*

* **[MCube: STEAM](https://mcube-cluster.de/en/projekt/steam/)** – Simulation of semi-flexible bus lines in Munich using FleetPy, integrated with the TUM-OSM decision support system *(agent-based simulation, semi-flexible transit, DSS coupling, urban mobility, evaluation pipeline)*

> 🧠 Do you use FleetPy in your research? [Let us know!](mailto:florian.dandl@tum.de)

---

## Citation

If you find this framework useful for your work or use it in your project, please consider citing:
Engelhardt, R., Dandl, F., Syed, A., Zhang, Y., Fehn, F., Wolf, F., Bogenberger, K. (2022). FleetPy: A Modular Open-Source Simulation Tool for Mobility On-Demand Services. Arxiv pre-print

```
@misc{engelhardt2022fleetpymodularopensourcesimulation,
      title={FleetPy: A Modular Open-Source Simulation Tool for Mobility On-Demand Services}, 
      author={Roman Engelhardt and Florian Dandl and Arslan-Ali Syed and Yunfei Zhang and Fabian Fehn and Fynn Wolf and Klaus Bogenberger},
      year={2022},
      eprint={2207.14246},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2207.14246}, 
}
```
