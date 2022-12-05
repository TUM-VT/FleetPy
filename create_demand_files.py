import os
import shutil
import pandas as pd
import numpy as np

demand_file_num = 100


for n in range(1, demand_file_num + 1):

    requests = np.random.randint(0, 7200, 100)
    requests_sort = np.sort(requests)
    start_points = np.random.randint(2966, 2993, 100)
    end_points = np.random.randint(2966, 2993, 100)
    ids = list(range(1, 101))

    demand = {'rq_time': requests_sort, 'start': start_points, 'end': end_points, 'request_id': ids}

    demand_df = pd.DataFrame(data=demand)

    demand_path = os.path.join("data", "demand", "example_demand", "matched", "example_network", f'example_100-{n}.csv')

    demand_df.to_csv(demand_path, index=False)