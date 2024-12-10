import pandas as pd
import pathlib


py_path = pathlib.Path(__file__)
NETWORK_NAME = "sumo_in"
MODE = "random_nodes"
NUMBER_OF_NODES = 150
SEED = 0

if __name__ == "__main__":  
    if MODE == "from_table":
        file_path = py_path.parent / "in_6_locations.csv"
        net_path = py_path.parent.parent.parent / "networks" / NETWORK_NAME /"base"/ "edges.csv"
        df = pd.read_csv(file_path)
        df_net = pd.read_csv(net_path)

        merged_df = pd.merge(df,df_net,left_on="SUMO_EDGE",right_on="source_edge_id")

        init_veh_distribution_df  = merged_df[["from_node"]]
        init_veh_distribution_df.rename(columns={"from_node":"node_index"})
        init_veh_distribution_df["probability"] = round(float(1/len(init_veh_distribution_df["from_node"])),3)
        init_veh_distribution_df.to_csv(py_path.parent/ "init_veh_dist.csv")

    elif MODE == "random_nodes":
        nodes_csv_path = py_path.parent.parent.parent / "networks" / NETWORK_NAME /"base"/ "nodes.csv"
        nodes_df = pd.read_csv(nodes_csv_path)  
        random_nodes = nodes_df['node_index'].sample(n=NUMBER_OF_NODES, random_state=SEED).tolist()
        init_veh_distribution_df = pd.DataFrame(random_nodes, columns=["node_index"])
        init_veh_distribution_df["probability"] = round(float(1/len(init_veh_distribution_df["node_index"])),3)
        
        directory_path = py_path.parent
        subdirectories = [f.name for f in directory_path.iterdir() if f.is_dir()]
        for sub_dir in subdirectories:
            if sub_dir.startswith(NETWORK_NAME):
                init_veh_distribution_df.to_csv(py_path.parent / sub_dir / f"{NETWORK_NAME}_init_veh_dist_n_{NUMBER_OF_NODES}_s_{SEED}.csv")