# Creation of scenario_config files

### Attributes to modify and their corresponding levels ----

# Seed: 1,2,3
seed = c(1,2,3)

# Demand level (and fleet sizes)
demand_level = c(10, 20, 30, 40, 50, 100, 150)
# Fleet size -> calculated endogenously for each demand level (after expand.grid())

# Re-assignment activated (Y/N)
reassign = c("Y", "N")

# Max. relative error of the black box (excluding error 0)
max_rel_error = c(0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)

# PUDO duration distribution parameters:
pudo_duration_distr_parameters = c("var_mulog=0,var_sdlog=0",
                                    
                                   "var_mulog=0,var_sdlog=-0.5",
                                   "var_mulog=0,var_sdlog=-0.3",
                                   "var_mulog=0,var_sdlog=0.3",
                                   "var_mulog=0,var_sdlog=0.5",
                                   
                                   "var_mulog=-0.5,var_sdlog=0",
                                   "var_mulog=-0.3,var_sdlog=0",
                                   "var_mulog=0.3,var_sdlog=0",
                                   "var_mulog=0.5,var_sdlog=0")



### Pre-step: What is the mean duration of the base and +- delta scenarios? ----
# The +- should be roughly mean*delta
# TODO

### SCENARIO FAMILY 0: Fully informed deterministic scenarios -----
### (PUDO duration in the simulation is deterministic; operator has perfect knowledge about it)

# PUDO duration distribution parameters:
# Given by pre-step
pudo_duration_FAM_0 = c(30) # TODO

family_0_scenarios = expand.grid(seed = seed,
                                 demand_level = demand_level,
                                 reassign = reassign, 
                                 pudo_duration = pudo_duration_FAM_0)
# Post process scenario to match to scenario_config format

### SCENARIO FAMILY 1: Fully informed stochastic scenarios (operator has perfect knowledge; i.e., rel. error = 0) -----

pudo_duration_FAM_1 = pudo_duration_distr_parameters

max_rel_error_FAM_1 = 0

family_1_scenarios = expand.grid(seed = seed,
                                 demand_level = demand_level,
                                 reassign = reassign, 
                                 pudo_duration = pudo_duration_FAM_1,
                                 max_rel_error = max_rel_error_FAM_1)

### SCENARIO FAMILY 3: Semi-informed stochastic scenarios (different rel. error values) -----
family_3_scenarios = expand.grid(seed = seed,
                                 demand_level = demand_level,
                                 reassign = reassign, 
                                 pudo_duration = pudo_duration_FAM_1,
                                 max_rel_error = max_rel_error)
