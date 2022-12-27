import os
import pandas as pd
import numpy as np

edge_path = os.path.join("data", "networks", "example_network_ID3_ref_vehtype", "base", "edges.csv")
edges_df = pd.read_csv(edge_path)

print(edges_df)