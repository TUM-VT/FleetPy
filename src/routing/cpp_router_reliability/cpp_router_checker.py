import os
import sys
import random
import numpy as np
import pandas as pd

Fleetpy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(Fleetpy_dir)

from src.routing.NetworkBasicCpp import NetworkBasicCpp
from src.routing.NetworkBasic import NetworkBasic

"""
run this script to check if the C++ router returns the same results as the python router
    -> see if compiling worked out as intended
"""

example_nw_dir = os.path.join(Fleetpy_dir, "data", "networks", "example_network")

cpp_nw = NetworkBasicCpp(example_nw_dir)
python_nw = NetworkBasic(example_nw_dir)

res_list = []
for i in range(100):
    start_node = random.choice(range(len(cpp_nw.nodes)))
    end_node = random.choice(range(len(cpp_nw.nodes)))
    _, tt_cpp, dis_cpp = cpp_nw.return_travel_costs_1to1( (start_node, None, None), (end_node, None, None) )
    _, tt_python, dis_python = python_nw.return_travel_costs_1to1( (start_node, None, None), (end_node, None, None) )
    same = np.round(tt_cpp, 2) == np.round(tt_python,2) and np.round(dis_cpp,2) == np.round(dis_python,2)
    res_list.append({
        "start": start_node,
        "end": end_node,
        "same": same,
        "tt_cpp": tt_cpp,
        "tt_python": tt_python,
        "dis_cpp": dis_cpp,
        "dis_python": dis_python
    })
print(pd.DataFrame(res_list))
print("")
print("Number of different results:", len([x for x in res_list if not x["same"]]), "/", len(res_list))