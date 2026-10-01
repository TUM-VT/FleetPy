import os
import sys
import numpy as np

from src.routing.NetworkBasicWithStoreCpp import NetworkBasicWithStoreCpp as Network

import gurobipy as grp

def solve_max_coverage(nw: Network, max_cost_value, list_nodes_to_consider=None, list_fix_nodes=None, cost_value="travel_time"):
    """
    Solves the max coverage problem: find the minimum number of nodes such that all nodes in list_nodes_to_consider are covered (reachable within max_cost_value) by at least one of the selected nodes. 
    If list_nodes_to_consider is None, considers all nodes in the network. If list_fix_nodes is provided, these nodes are included in the solution and not considered for selection.
    :param nw: Network object
    :param max_cost_value: maximum cost value for coverage (e.g., travel time)
    :param list_nodes_to_consider: list of nodes (int) to consider for coverage
    :param list_fix_nodes: list of nodes (int) that are fixed and must be included in the solution
    :param cost_value: the cost metric to use for coverage (e.g., "travel_time", "distance")
    :return: list of selected nodes (int)
    """
    if list_nodes_to_consider is None:
        list_nodes_to_consider = [node.node_index for node in nw.get_node_list()]
    print(f"solve max coverage problem for {len(list_nodes_to_consider)} nodes with max_cost_value={max_cost_value} and cost_value={cost_value}")
    if cost_value == "travel_time":
        cscf = None
    elif cost_value == "distance":
        def cscf(travel_time, travel_distance, current_dijkstra_node):
            return travel_distance
    else:
        raise ValueError(f"Unsupported cost_value: {cost_value}")
    
    print(" -> compute reachable nodes for each node")
    node_to_reachable_nodes = {}
    list_nodes = [(k, None, None) for k in list_nodes_to_consider] # boarding nodes
    if list_fix_nodes is not None:
        for n in list_fix_nodes:
            res = nw.return_travel_costs_Xto1(list_nodes, (n, None, None), max_cost_value=max_cost_value, customized_section_cost_function=cscf)
            for o_pos, _, tt, _ in res:
                if o_pos in list_nodes:
                    list_nodes.remove(o_pos)
    print(f" -> {len(list_nodes)} nodes to process after removing nodes covered by fixed nodes")
    for k, pos in enumerate(list_nodes):
        i = pos[0]
        res = nw.return_travel_costs_Xto1(list_nodes, pos, max_cost_value=max_cost_value, customized_section_cost_function=cscf)
        node_to_reachable_nodes[i] = {}
        for o_pos, _, tt, _ in res:
            node_to_reachable_nodes[i][o_pos[0]] = tt
        if k % 500 == 0:
            print(f" -> processed {k}/{len(list_nodes)} nodes")
            
    print(" -> solve problem")
    m = grp.Model()
    var = {str(i) : m.addVar(name = str(i), obj = 1, vtype = grp.GRB.BINARY) for i in node_to_reachable_nodes.keys()}
    min_N = 1 
    for i, reachable_nodes in node_to_reachable_nodes.items():
        m.addConstr(sum(var[str(k)] for k in reachable_nodes.keys()) >= min(min_N, len(reachable_nodes.keys())), name = f"n_{i}" )

    m.setParam('TimeLimit', 5*60)
    m.optimize()
    vals = m.getAttr('X', var)
    #print(vals)
    assigned_nodes = []
    for x in vals:
        v = vals[x]
        #print(x, v)
        if int(np.round(v)) == 0:
            continue
        assigned_nodes.append(int(float(x)))
    print(" -> found solution with {} nodes".format(len(assigned_nodes)))
    if list_fix_nodes is not None:
        assigned_nodes = list(set(assigned_nodes + list_fix_nodes))
    print(f" -> selected {len(assigned_nodes)} nodes overall")
    return assigned_nodes