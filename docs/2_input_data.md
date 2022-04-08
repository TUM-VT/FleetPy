# Input Data

Input data shall be saved in the `data/` directory. The `data/` directory will not be part of the git-distribution. Generally, input data should be saved on the local disk.

Input data includes both mandatory and optional data. Mandatory data are the required data to initiate the simulation, including:

- Network
- Demand
- Vehicle
- Fleet-Control

Optional data contains all the information that can increase the quality of simulation:

- Zone Systems
- Infrastructure
- Public Transportation

## Network Data

### Basic Data Structure

Network data is stored in `data/networks`. For each network given a `network_name`, the following mandatory directory and file structure needs to be met:

```diff
    networks/  
    networks/{network_name}/  
    networks/{network_name}/base/  
    networks/{network_name}/base/nodes.csv  
    networks/{network_name}/base/edges.csv  
    networks/{network_name}/base/nodes_all_infos.geojson  
    networks/{network_name}/base/edges_all_infos.geojson
```

<!-- Yunfei: should we specify the data format here? -->
### Data Specification

A routable network consists of *nodes* and *edges*. Vehicles travel along *edges* containing the travel information. *Edges*, are defined as the connections between *nodes*, which represent the positions in the network, where different *edges* can be chosen as the next part of the route. Hence, *nodes* usually represent junctions/intersections of a street network.

```diff
- IMPORTANT: the network definition assumes that node indices are numbered from 0..|N-1|!
```

In the following contents, the columns of different network data files are described.

#### nodes.csv

##### Necessary Attributes

Column Name | Data Type | Description
-- | -- | --
node_index | int | ID of node
is_stop_only | bool | False: normal node; True: node can only be used as first or last part of a route leg
pos_x | float | x-position in projected coordinate system > unit: meters
pos_y | float | y-position in projected coordinate system > unit: meters

##### Optional Attributes

Column Name | Data Type | Description
-- | -- | --
node_order | int | only required for contraction hierarchy

#### edges.csv

##### Necessary Attributes

Column Name | Data Type | Description
-- | -- | --
from_node | int | ID of origin node of a street edge
to_node | int | ID of destination node of a street edge
distance | float | length of street edge in meters
travel_time | float | travel duration on street edge in seconds

##### Optional Attributes

Column Name | Data Type | Description
-- | -- | --
shortcut_def | str | only for contraction hierarchy; use “;” as separator between IDs
source_edge_id | str | depends on the source where edge are retrieved, could be, e.g., OSM-Edge ID, Aimsun Section ID...; can be "-" separated elements as well

<!-- Yunfei: do we need to cite the reference (osm, aimsun) here? -->

#### crs.info

`epsg:code`

- This file only contains one line: epsg-code 'code', which is valid for the pos_x, pos_y in the nodes.csv.

### Preprocessing Methods (Need to be supplemented)

There are various routing-modules that are based on different preprocessing scripts; the preprocessed data will also be saved in the respective network directory.

#### NN_FASTEST_TT.NPY / NN_FASTEST_DISTANCE.NPY

Fully preprocessed (according to the fastest route) node-to-node travel time or distance tables are saved as 2D-Numpy arrays.
The first index (row index) represents the origin node, the second index represents the destination node.
The data entries (travel time/distance) are of type float.
These files are saved under scenario_dir/tables/x.npy, where scenario_dir=ff for free-flow conditions.

#### Network Dynamics Files

These files can be used

* to define loading of corresponding travel time files at given simulation time (column "travel_time_folder")

or(!)

* to scale all network travel times with certain factors according to the simulation time. This input is used by the 'NetworkTTMatrix' module. (column "travel_time_factor")

Column Name | Data Type | Description
-- | -- | --
simulation_time | int | simulation time in seconds
travel_time_folder | str | corresponding folder name of travel time directory to be used from this simulation time on
travel_time_factor | float | general travel time factor that is used for complete network

```diff
- IMPORTANT: only one of the columns travel_time_folder/travel_time_factor is allowed to be given!
```

#### Partially preprocessed data

Partially preprocessed (according to the fastest route) node-to-node travel time or distance tables are saved as 2D-Numpy arrays.
The first index (row index) represents the origin node, the second index represents the destination node.
The data entries (travel time/distance) are of type float.
The travel time matrix is called tt_matrix.npy, the distance matrix is called dis_matrix.npy
These files are saved in the corresponding travel time folders; free-flow condition is stored in the base-folder.
Note that only the travel times/distances between the first x nodes are stored. x is defined by the shape of the matrix.
These matrices are used by NetworkPartiallyPreprocessed.py and NetworkPartiallyPreprocessedCpp.py.

<!-- (Yunfei): base directory 

(Roman): preprocessing methods  -->

## Zone Systems

<!-- (Yunfei) [optional input data]  -->

## Demand Data

<!-- (Yunfei) 

(disaggregated) trip data 

aggregated forecast data [optional]  -->

## Vehicle Data

<!-- (Yunfei)  -->

## Infrastructure Data

<!-- (Yunfei) [optional input data] 

* boarding points 

* depots 

* public charging infrastructure  -->

## Public Transportation Data

<!-- (Yunfei) [optional input data]  -->

## Fleet-Control Data

<!-- (Yunfei) [optional input data]  -->

## Data Collection

<!-- (Roman & Flo): access to prepared data sets  -->
