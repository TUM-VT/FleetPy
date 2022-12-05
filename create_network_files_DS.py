import os
import shutil
import pandas as pd

# create csv files and folder for speed variation of networks

speed_var_max = 20 #in percent reduction
for n in range(1, speed_var_max + 1):

    #create network files

    target_path_name = os.path.join("data", "networks", f'example_network_v-{n}', "base")
    #os.makedirs(target_path_name)
    source_path_name1 = os.path.join("data", "networks", "example_network", "base", "crs.info")
    shutil.copy(source_path_name1, target_path_name)

    source_path_name2 = os.path.join("data", "networks", "example_network", "base", "edges.csv")
    shutil.copy(source_path_name2, target_path_name)

    source_path_name3 = os.path.join("data", "networks", "example_network", "base", "edges_all_infos.geojson")
    shutil.copy(source_path_name3, target_path_name)

    source_path_name4 = os.path.join("data", "networks", "example_network", "base", "nodes.csv")
    shutil.copy(source_path_name4, target_path_name)

    source_path_name5 = os.path.join("data", "networks", "example_network", "base", "nodes_all_infos.geojson")
    shutil.copy(source_path_name5, target_path_name)

    change_path = os.path.join("data", "networks", f'example_network_v-{n}', "base", "edges.csv")

    edges_df = pd.read_csv(change_path)
    edges_df = edges_df.mul({'from_node': 1, 'to_node': 1, 'distance': 1, 'travel_time': (1 + n *5 * 0.01), 'source_edge_id': 1})
    edges_df = edges_df.astype({"from_node": int, "to_node": int})
    edges_df.to_csv(change_path, index=False)


    #create demand files

    target_path_name = os.path.join("data", "demand", "example_demand", "matched", f'example_network_v-{n}')
    #os.makedirs(target_path_name)
    source_path_name1 = os.path.join("data", "demand", "example_demand", "matched", "example_network", "example_100.csv")
    shutil.copy(source_path_name1, target_path_name)
