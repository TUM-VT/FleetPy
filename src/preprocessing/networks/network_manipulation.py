import os
from urllib.parse import non_hierarchical
fleet_sim_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
os.sys.path.append(fleet_sim_path)
from src.routing.NetworkBasic import Edge
import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely import ops
import numpy as np
import rtree

def convertCoordListToString(coord_list):
    l = []
    for coord in coord_list:
        l.append("{};{}".format(coord[0], coord[1]))
    return "|".join(l)

def convertStringToCoordList(string):
    coord_list = []
    for e in string.split("|"):
        x,y = e.split(";")
        coord_list.append( (float(x), float(y)) )
    return coord_list

def createDir(dirname):
    if os.path.isdir(dirname):
        print("{} allready here! Overwriting basic network files!".format(dirname))
    else:
        os.makedirs(dirname)

def getAllTravelTimeFolders(network_path):
    tt_folders = {}
    for f in os.listdir(network_path):
        if os.path.isfile(os.path.join(network_path, f, "edges_td_att.csv")):
            tt_folders[f] = os.path.join(network_path, f)
    return tt_folders

class NetworkNode():
    def __init__(self, node_index, pos_x, pos_y, is_stop_only = False, ch_value = -1, source_node_id = None):
        self.index = node_index
        self.coordinates = (pos_x, pos_y)
        self.is_stop_only = is_stop_only
        self.ch_value = ch_value

        self.outgoing_edges = {}
        self.incoming_edges = {}

        self.outgoing_ch_edges = {}
        self.incoming_ch_edges = {}

        self.is_deleted = False
        
        self.is_centroid = False
        
        self.source_node_id = source_node_id

        self.zoning_info = {} # zone_name -> zone_id
        self.boarding_point_info = {} # infra_name -> 1 if boarding point in infra
        self.depot_info = {}    # infra_name -> info dict if depot in infra

    def addIncomingEdge(self, edge):
        self.incoming_edges[edge.id] = edge

    def addOutgoingEdge(self, edge):
        self.outgoing_edges[edge.id] = edge

    def __str__(self):
        s = "{},{},{}".format(self.index, self.coordinates[0], self.coordinates[1])
        if self.is_stop_only:
            s += ",{}".format(1)
        else:
            s += ","
        s += ",{}".format(self.ch_value)
        return s

    def getAttributeDictGEOJSON(self):
        att_dict = {"node_index" : self.index, "is_stop_only" : self.is_stop_only, "source_node_id" : self.source_node_id, "geometry" : Point(self.coordinates)}
        return att_dict

    def getAttributeDictBaseFile(self):
        att_dict = {"node_index" : self.index, "is_stop_only" : self.is_stop_only, "source_node_id" : self.source_node_id, "pos_x" : self.coordinates[0], "pos_y" : self.coordinates[1] }
        return att_dict

    def register_zone(self, zoning_name, zone_id):
        self.zoning_info[zoning_name] = zone_id

    def register_boarding_point_info(self, infra_name):
        self.boarding_point_info[infra_name] = 1

    def register_depot_info(self, infra_name, info_dict):
        self.depot_info[infra_name] = info_dict.copy()

class NetworkSuperNode():
    def __init__(self, id, node_id_list, polygon = None, source_edge_id = None):
        self.node_collection = node_id_list[:]
        self.id = id
        self.polygon = polygon
        self.source_edge_id = source_edge_id
        self.node_collection_nodes = None

    def __str__(self):
        return "{},{},{},{}".format(self.id, self.source_edge_id, ";".join([str(x) for x in self.node_collection]), convertCoordListToString(self.polygon))

    def getAttributeDictGEOJSON(self):
        att_dict = {"index" : self.id, "node_collection" : ";".join([str(x) for x in self.node_collection]), "source_edge_id" : self.source_edge_id, "geometry" : Polygon(self.polygon) }
        #print(type(np.array(self.node_collection, dtype='int64')))
        return att_dict

class NetworkEdge():
    def __init__(self, start_node_index, end_node_index, travel_time, travel_distance, edge_index = None, source_edge_id = None, shortcut_def = None, customized_cost_val = None, polyline = None, roadtype = None):
        self.start_node_index = start_node_index
        self.start_node = None
        self.end_node_index = end_node_index
        self.end_node = None

        self.travel_time = travel_time
        self.travel_distance = travel_distance

        self.time_dependent_tt = {}        # time -> tt

        self.customized_cost_val = customized_cost_val

        self.id = edge_index
        if self.id is None:
            self.id = "{};{}".format(self.start_node_index, self.end_node_index)
        
        self.source_edge_id = source_edge_id
        self.shortcut_def = shortcut_def

        self.polyline = polyline
        #print(self.polyline)
        self.roadtype = roadtype

    def __str__(self):
        return f"edge: {self.start_node_index} -> {self.end_node_index} : source id: {self.source_edge_id}"

    def getStartNodeIndex(self):
        return self.start_node_index

    def getEndNodeIndex(self):
        return self.end_node_index

    def updateId(self):
        self.id = "{};{}".format(self.start_node_index, self.end_node_index)

    def __str__(self):
        s = "{},{},{},{},{},{},{},{}".format(self.start_node_index, self.end_node_index, self.travel_time, self.travel_distance, self.id, self.source_edge_id,self.roadtype,convertCoordListToString(self.polyline))
        return s

    def getAttributeDictGEOJSON(self):
        att_dict = {"from_node" : self.start_node_index, "to_node" : self.end_node_index, "distance" : self.travel_distance, "travel_time" : self.travel_time, "source_edge_id" : self.source_edge_id, "road_type" : self.roadtype, "geometry" : LineString(self.polyline)}
        return att_dict

    def getAttributeDictBaseFile(self):
        att_dict = {"from_node" : self.start_node_index, "to_node" : self.end_node_index, "distance" : self.travel_distance, "travel_time" : self.travel_time, "source_edge_id" : self.source_edge_id}
        return att_dict

class FullNetwork():
    def __init__(self, network_path, from_crs_to_crs=None, nodes_gdf=None, edges_gdf=None):
        """ this network is used to load all available information from an network folder 
        and perform manipulations like node adding or removing while maintaining all information
        this can be used espacially for matching and if node_indices have to change due to deletions or reordering
            while allready matched infra of demand files have to be transformed as-well
        :param network_path: path to the corresponding network (geojson-files have to be given!)
        :param from_crs_to_crs: if given: tuple (current crs (string epsg:...), crs to convert to (string epsg:...)) -> performs coordinate transform
        :param nodes_gdf: GeoDataFrame of nodes, if provided with edges_gdf the network is not loaded from file but directly created from the GeoDataFrames
        :param edges_gdf: GeoDataFrame of edges, if provided with nodes_gdf the network is not loaded from file but directly created from the GeoDataFrames
         """
        self.nodes = []
        self.original_node_index_to_node = {}   # node index from loaded network -> node-obj (to transfrom indices of matched data)
        self.edge_id_to_edge = {}
        self.source_edge_id_to_edge = {}
        self.super_nodes = {}

        self.network_tt_times = {}

        if nodes_gdf is None and edges_gdf is None:
            self.network_path = network_path
            self.network_name = os.path.basename(network_path)
            self._loadFullNetworkFromGEOJSONFiles(network_path, from_crs_to_crs=from_crs_to_crs)
        else:
            self._loadFullNetworkFromGEOJSONInputFiles(nodes_gdf, edges_gdf, from_crs_to_crs=from_crs_to_crs)

        self.registered_zonings = {}    # zoning_name -> zoning_path
        self.registered_infra = {}  # infra_name -> infra_path

        self.r_tree = None  #rtree for fast geometrical nearest node valuation

    def _loadFullNetworkFromGEOJSONFiles(self, network_folder, from_crs_to_crs = None):
        """ loads full network informations from ..._all_infos.geojson files
        into the buffer
        """
        node_gdf = gpd.read_file(os.path.join(network_folder, "base", "nodes_all_infos.geojson"))
        if from_crs_to_crs is not None:
            from_crs = from_crs_to_crs[0]
            node_gdf = gpd.read_file(os.path.join(network_folder, "base", "nodes_all_infos.geojson"), crs=from_crs)
            to_crs = from_crs_to_crs[1]
            print(node_gdf.crs)
            #node_gdf.crs = {'init': from_crs}
            node_gdf.to_crs(to_crs, inplace = True)
        else:
            node_gdf = gpd.read_file(os.path.join(network_folder, "base", "nodes_all_infos.geojson"))
        if not "node_index" in node_gdf.columns:
            node_gdf["node_index"] = node_gdf.index
        for key, entry in node_gdf.iterrows():
            index = entry['node_index']
            is_stop_only = entry['is_stop_only']
            point = entry['geometry']
            x = point.x
            y = point.y
            source_node_id = None
            if "source_node_id" in entry.keys():
                source_node_id = entry["source_node_id"]
            node = NetworkNode(int(index), float(x), float(y), is_stop_only=is_stop_only, source_node_id=source_node_id)
            self.addNode(node)
            self.original_node_index_to_node[node.index] = node

        edge_gdf = gpd.read_file(os.path.join(network_folder, "base", "edges_all_infos.geojson")) 
        if from_crs_to_crs is not None:
            from_crs = from_crs_to_crs[0]
            to_crs = from_crs_to_crs[1]
            edge_gdf.crs = from_crs
            edge_gdf.to_crs(to_crs, inplace = True)
        for key, entry in edge_gdf.iterrows():
            geo = entry["geometry"]
            if geo is None or len(geo.coords) == 0:
                start_node_id = int(entry["from_node"])
                end_node_id = int(entry["to_node"])
                polyline = [self.nodes[start_node_id].coordinates, self.nodes[end_node_id].coordinates]
            else:
                polyline = list(entry["geometry"].coords)
            edge = NetworkEdge(int(entry["from_node"]), int(entry["to_node"]), float(entry["travel_time"]), float(entry["distance"]), source_edge_id=entry.get("source_edge_id", None), roadtype=entry.get("road_type", None), polyline=polyline)
            self.addEdge(edge)

        supernode_df = None
        try:
            supernode_df = gpd.read_file(os.path.join(network_folder, "base", "supernodes_all_infos.geojson")) 
        except:
            print("... no supernode file found ...")
        if supernode_df is not None:
            if from_crs_to_crs is not None:
                from_crs = from_crs_to_crs[0]
                to_crs = from_crs_to_crs[1]
                supernode_df.crs = from_crs
                supernode_df.to_crs(to_crs, inplace = True)
            for key, entry in supernode_df.iterrows():
                try:
                    node_collection = [int(n) for n in entry["node_collection"].split(";")]
                except:
                    print("WARNING: couldnt create node collection from {}".format(entry["node_collection"]))
                    node_collection = []
                polygon = entry["geometry"].exterior.coords
                supernode = NetworkSuperNode(entry['index'], node_collection, polygon=polygon, source_edge_id=entry["source_edge_id"])
                self.addSuperNode(supernode)

        self.addNodeAttributesToClasses()

        tt_folders = getAllTravelTimeFolders(network_folder)
        for f, tt_folder in tt_folders.items():
            self.load_tt_file(f, tt_folder)
            self.network_tt_times[f] = 1
            
    def _loadFullNetworkFromGEOJSONInputFiles(self, nodes_gdf, edges_gdf, from_crs_to_crs=None):
        if from_crs_to_crs is not None:
            from_crs = from_crs_to_crs[0]
            nodes_gdf.crs = from_crs
            to_crs = from_crs_to_crs[1]
            print(nodes_gdf.crs)
            #node_gdf.crs = {'init': from_crs}
            nodes_gdf.to_crs(to_crs, inplace = True)
        if not "node_index" in nodes_gdf.columns:
            nodes_gdf["node_index"] = nodes_gdf.index
        for key, entry in nodes_gdf.iterrows():
            index = entry['node_index']
            is_stop_only = entry['is_stop_only']
            point = entry['geometry']
            x = point.x
            y = point.y
            node = NetworkNode(int(index), float(x), float(y), is_stop_only=is_stop_only, source_node_id=entry.get("source_node_id", None))
            self.addNode(node)
            self.original_node_index_to_node[node.index] = node

        if from_crs_to_crs is not None:
            from_crs = from_crs_to_crs[0]
            to_crs = from_crs_to_crs[1]
            edges_gdf.crs = from_crs
            edges_gdf.to_crs(to_crs, inplace = True)
        for key, entry in edges_gdf.iterrows():
            polyline = list(entry["geometry"].coords)
            if len(polyline) == 0:
                start_node_id = int(entry["from_node"])
                end_node_id = int(entry["to_node"])
                polyline = [self.nodes[start_node_id].coordinates, self.nodes[end_node_id].coordinates]
            edge = NetworkEdge(int(entry["from_node"]), int(entry["to_node"]), float(entry["travel_time"]), float(entry["distance"]), source_edge_id=entry.get("source_edge_id", None), roadtype=entry.get("road_type", None), polyline=polyline)
            self.addEdge(edge)    
            
        self.addNodeAttributesToClasses()    

    def load_tt_file(self, f, tt_folder):
        """ loads all travel time files into edge attributes
        :param f: travel time folder name
        :param tt_folder: folder with the file edges_td_att.csv
        """
        tt_file = os.path.join(tt_folder, "edges_td_att.csv")
        tmp_df = pd.read_csv(tt_file, index_col=[0,1])
        for edge_index_tuple, new_tt in tmp_df["edge_tt"].items():
            o_node = self.nodes[edge_index_tuple[0]]
            d_node = self.nodes[edge_index_tuple[1]]
            try:
                self.edge_id_to_edge["{};{}".format(edge_index_tuple[0],edge_index_tuple[1])].time_dependent_tt[f] = new_tt
            except KeyError:
                print("{};{}".format(edge_index_tuple[0],edge_index_tuple[1]) + " not found but mentioned in tt files")

    def register_matched_zoning(self, zoning_path, zoning_name):
        """ if a zone system has allready matched to the currently given network (no index changes may have happend yet!)
        this function loads the matched zones to the nodes into the network and stores the information in the nodes themselfes
        :param zoning_path: path to the folder src/data/zoning
        :param zoning_name: name of the given zone system
        """
        f = os.path.join(zoning_path, zoning_name, self.network_name, "node_zone_info.csv")
        try:
            node_zone_df = pd.read_csv(f)
        except:
            print("file not found! zoning not matched yet? {}".format(f))
            exit()
        for k, entry in node_zone_df.iterrows():
            node_index = int(entry["node_index"])
            try:
                zone_id = int(entry["zone_id"])
            except:
                continue
            self.nodes[node_index].register_zone(zoning_name, zone_id)
        self.registered_zonings[zoning_name] = os.path.join(zoning_path, zoning_name)

    def load_zoning_and_match_nodes(self, zoning_path, zoning_name):
        """ this function loads the unmatched polygon_definition.geojson and matches the nodes within the areas.
        it directly creates the matched files src/data/zoning/{zoning_name}/{network_name}/node_zone_info.csv and registers
        the zones in the network
        :param zoning_path: path to the folder src/data/zoning
        :param zoning_name: name of the given zone system
        """
        zoning = gpd.read_file(os.path.join(zoning_path, zoning_name, "polygon_definition.geojson"))
        node_zone_list = []
        for node in self.nodes:
            p = Point(node.coordinates)
            for k, row in zoning.iterrows():
                if p.within(row["geometry"]):
                    node_zone_list.append({"node_index" : node.index, "zone_id" : k})
        
        print("{}/{} nodes within zonesystem {}".format(len(node_zone_list), len(self.nodes), zoning_name))
        node_zone_df = pd.DataFrame(node_zone_list)
        output_dir = os.path.join(zoning_path, zoning_name, self.network_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        node_zone_df.to_csv(os.path.join(output_dir, "node_zone_info.csv"), index=False)
        self.register_matched_zoning(zoning_path, zoning_name)
        
    def load_centroids_and_match_nodes(self, zoning_path, zoning_name):
        """ this function loads a geojson file "centroid_definition.geojson" with corresponding Point geometries instead of zones.
        each node is assigned to the nearest centroid and thereby the zones are defined
        it directly creates the matched files src/data/zoning/{zoning_name}/{network_name}/node_zone_info.csv and registers
        the zones in the network
        :param zoning_path: path to the folder src/data/zoning
        :param zoning_name: name of the zone system (tbd)
        """
        centroids = gpd.read_file(os.path.join(zoning_path, zoning_name, "centroid_definition.geojson"))
        centroid_rtree = rtree.index.Index()
        for _, row in centroids.iterrows():
            zone_id = row["zone_id"]
            p = row["geometry"]
            coords = p.coords[0]
            centroid_rtree.insert(zone_id, (coords[0], coords[1], coords[0], coords[1]) )
        node_zone_list = []
        for node in self.nodes:
            nearest_cen =  list(centroid_rtree.nearest((node.coordinates[0], node.coordinates[1], node.coordinates[0], node.coordinates[1]), 1))[0]
            node_zone_list.append({"node_index" : node.index, "zone_id" : nearest_cen})
        
        print("{}/{} nodes within zonesystem {}".format(len(node_zone_list), len(self.nodes), zoning_name))
        node_zone_df = pd.DataFrame(node_zone_list)
        output_dir = os.path.join(zoning_path, zoning_name, self.network_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        node_zone_df.to_csv(os.path.join(output_dir, "node_zone_info.csv"), index=False)
        self.register_matched_zoning(zoning_path, zoning_name)

    def register_boarding_points(self, infra_path, infra_name): 
        """ if boarding points have allready been matched to the currently given network (no index changes may have happend yet!)
        this function loads the matched boarding points into the network and stores the information in the nodes themselfes
        :param infra_path: path to the folder src/data/infra
        :param infra_name: name of the given infra structure
        """
        self.registered_infra[infra_name] = os.path.join(infra_path, infra_name)
        file_path = os.path.join(infra_path, infra_name, self.network_name, "boarding_points.csv")
        if not os.path.isfile(file_path):
            print("file {} not found!".format(file_path))
            exit()
        bp_df = pd.read_csv(file_path)
        for bp in bp_df["node_index"].values:
            self.nodes[bp].register_boarding_point_info(infra_name)
            
    def set_boarding_points(self, infra_path, infra_name, list_node_indices):
        """ if boarding points have allready been matched to the currently given network (no index changes may have happend yet!)
        this function loads the matched boarding points into the network and stores the information in the nodes themselfes
        :param infra_path: path to the folder src/data/infra
        :param infra_name: name of the given infra structure
        :param list_node_indices: list of node indices that are a boarding point
        """
        self.registered_infra[infra_name] = os.path.join(infra_path, infra_name)
        for bp in list_node_indices:
            self.nodes[bp].register_boarding_point_info(infra_name)

    def load_boarding_points_and_match_closest_node(self, infra_path, infra_name, only_must_stop = False):
        """ this function loads the unmatched boarding_points.geojson and matches them to the closest network nodes.
        it directly creates the matched files src/data/infra/{infra_name}/{network_name}/boarding_points.csv and registers
        the boarding points in the network
        :param infra_path: path to the folder src/data/infra
        :param infra_name: name of the given infra structure
        """
        boarding_points = gpd.read_file(os.path.join(infra_path, infra_name, "boarding_points.geojson"))
        is_boarding_point_dict = {}
        longest_match_distance = 0
        if self.r_tree is None:
            self._initialize_rtree(only_must_stop=only_must_stop)
        gpd_index_to_matched_node = {}
        for key, entries in boarding_points.iterrows():
            print(" ... {}/{}".format(key, boarding_points.shape[0]))
            ap = entries["geometry"]
            o_point = list(ap.coords)[0]
            nearest_node =  list(self.r_tree.nearest((o_point[0], o_point[1], o_point[0], o_point[1]), 1))[0]
            nearest_distance = float("inf")
            p = Point(self.nodes[nearest_node].coordinates)
            nearest_distance = p.distance(ap)
            if nearest_distance > longest_match_distance:
                longest_match_distance = nearest_distance
            is_boarding_point_dict[nearest_node] = 1
            gpd_index_to_matched_node[key] = nearest_node
        print("{}/{} created boarding-points".format(len(is_boarding_point_dict.keys()), boarding_points.shape[0]))
        output_dir = os.path.join(infra_path, infra_name, self.network_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        bp_dict_list = [{"node_index" : n} for n in is_boarding_point_dict.keys()]
        bp_df = pd.DataFrame(bp_dict_list)
        bp_df.to_csv(os.path.join(output_dir, "boarding_points.csv"), index = False)
        self.register_boarding_points(infra_path, infra_name)
        return gpd_index_to_matched_node

    def register_depots(self, infra_path, infra_name, file_name):
        """ if depots have allready been matched to the currently given network (no index changes may have happend yet!)
        this function loads the matched depots into the network and stores the information in the nodes themselves
        :param infra_path: path to the folder src/data/infra
        :param infra_name: name of the given infra structure
        :param file_name: depots file name
        """
        self.registered_infra[infra_name] = os.path.join(infra_path, infra_name)
        file_path = os.path.join(infra_path, infra_name, self.network_name, file_name)
        if not os.path.isfile(file_path):
            print("file {} not found!".format(file_path))
            raise FileNotFoundError(file_path)
        depot_df = pd.read_csv(file_path)
        for key, entries in depot_df.iterrows():
            entry_dict = entries.to_dict()
            node_index = int(entry_dict["node_index"])
            del entry_dict["node_index"]
            self.nodes[node_index].register_depot_info(infra_name, entry_dict)

    def load_depots_and_match_closest_node(self, infra_path, infra_name, only_must_stop = False):
        """ this function loads the unmatched depots.geojson and matches them to the closest network nodes.
        it directly creates the matched files src/data/infra/{infra_name}/{network_name}/depots.csv and registers
        the depots points in the network
        :param infra_path: path to the folder src/data/infra
        :param infra_name: name of the given infra structure
        """
        depots = gpd.read_file(os.path.join(infra_path, infra_name, "depots.geojson"))
        depot_df_list = []
        longest_match_distance = 0
        if self.r_tree is None:
            self._initialize_rtree(only_must_stop=only_must_stop)
        for key, entries in depots.iterrows():
            ap = entries["geometry"]
            o_point = list(ap.coords)[0]
            nearest_node =  list(self.r_tree.nearest((o_point[0], o_point[1], o_point[0], o_point[1]), 1))[0]
            nearest_distance = float("inf")
            p = Point(self.nodes[nearest_node].coordinates)
            nearest_distance = p.distance(ap)
            if nearest_distance > longest_match_distance:
                longest_match_distance = nearest_distance
            depot_df_list.append( {"node_index" : nearest_node, "charging_station_id" : entries["charging_station_id"], "max_nr_parking" : entries["max_nr_parking"], "charging_units" : entries["charging_units"]} )
        print("{}/{} created depts".format(len(depot_df_list), depots.shape[0]))
        output_dir = os.path.join(infra_path, infra_name, self.network_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        depot_df = pd.DataFrame(depot_df_list)
        depot_df.to_csv(os.path.join(output_dir, "depots.csv"), index = False)
        self.register_depots(infra_path, infra_name)

    def addNode(self, node):
        """ adds a node to the network at last position
        node.index has to be set correctly allready!
        """
        self.nodes.append(node)
        if node.index != len(self.nodes) - 1:
            print("wrong node order!")
            exit()

    def deleteNode(self, node_index):
        """ marks a specific node as deleted
        node indeces are still maintained
        node indeces are updated with the function updateClassIndices
        multiple nodes can be deleted
        """
        self.nodes[node_index].is_deleted = True
        to_del_edge_ids = []
        for id, edge in self.nodes[node_index].outgoing_edges.items():
            to_del_edge_ids.append(id)
            del edge.end_node.incoming_edges[id]
        for id, edge in self.nodes[node_index].incoming_edges.items():
            to_del_edge_ids.append(id)
            del edge.start_node.outgoing_edges[id]
        for i in to_del_edge_ids:
            del self.edge_id_to_edge[i]

    def addNodeAttributesToClasses(self):
        """ this function sets correct start and end node objects in corresponding edge objects
        """
        for key, edge in self.edge_id_to_edge.items():
            edge.start_node = self.nodes[edge.start_node_index]
            edge.end_node = self.nodes[edge.end_node_index]
        for key, supernode in self.super_nodes.items():
            supernode.node_collection_nodes = [self.nodes[i] for i in supernode.node_collection]

    def updateClassIndices(self):
        """ this function updates all network indices in all classes in case
        some nodes have been deleted before
        it now removes deleted nodes entirely from the network
        """
        self.r_tree = None
        to_del = []
        c = 0
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_deleted:
                to_del.append(i)
                continue
            node.index = c
            new_nodes.append(node)
            c += 1
        self.nodes = new_nodes
        # for i in reversed(to_del):
        #     self.nodes = self.nodes[:i] + self.nodes[i+1:]
        #for i, node in enumerate(self.nodes):
            #print(i, node.index)
        new_edge_id_to_edge = {}
        for key, edge in self.edge_id_to_edge.items():
            edge.start_node_index = edge.start_node.index
            edge.end_node_index = edge.end_node.index
            edge.updateId()
            new_edge_id_to_edge[edge.id] = edge
        self.edge_id_to_edge = new_edge_id_to_edge
        for node in self.nodes:
            new_incoming = {}
            for key, val in node.incoming_edges.items():
                new_incoming[val.id] = val
            new_outgoing = {}
            for key, val in node.outgoing_edges.items():
                new_outgoing[val.id] = val
            node.incoming_edges = new_incoming
            node.outgoing_edges = new_outgoing
        for key, supernode in self.super_nodes.items():
            supernode.node_collection = [node.index for node in supernode.node_collection_nodes if not node.is_deleted]
            supernode.node_collection_nodes = [self.nodes[i] for i in supernode.node_collection]

    def addSuperNode(self, super_node):
        self.super_nodes[super_node.id] = super_node

    def getSuperNodeId(self, source_id):
        for super_node_id, super_node in self.super_nodes.items():
            if int(super_node.source_edge_id) == source_id:
                return super_node_id
        raise KeyError(source_id)

    def getSuperNodeOutgoingStartNodes(self, super_node_id):
        """ returns a list of node_indices with the startnode which are ther start node
        of an aimsun section outgoing of this super_node (aimsun node)
        Is usefull to create an extra stop node for this intersection/supernode/aimsunnode
        """
        super_node = self.super_nodes[super_node_id]
        outgoing_start_nodes = []
        for node_id in super_node.node_collection:
            node = self.nodes[node_id]
            for edge in node.outgoing_edges.values():
                if edge.roadtype != "Turn":
                    outgoing_start_nodes.append(node_id)
                    break
        return outgoing_start_nodes

    def getSuperNodeIncomingEndNodes(self, super_node_id):
        """ returns a list of node_indices with the startnode which are ther start node
        of an aimsun section outgoing of this super_node (aimsun node)
        Is usefull to create an extra stop node for this intersection/supernode/aimsunnode
        """
        super_node = self.super_nodes[super_node_id]
        incoming_end_nodes = []
        for node_id in super_node.node_collection:
            node = self.nodes[node_id]
            for edge in node.incoming_edges.values():
                if edge.roadtype != "Turn":
                    incoming_end_nodes.append(node_id)
                    break
        return incoming_end_nodes

    def getCurrentNodeNumber(self):
        return len(self.nodes)

    def getNodeCoordinates(self, node_id):
        return self.nodes[node_id].coordinates

    def addEdge(self, edge):
        start_index = edge.start_node_index
        end_index = edge.end_node_index
        #print("edge {} -> {} nodes {}".format(start_index, end_index, len(self.nodes)))
        self.nodes[start_index].addOutgoingEdge(edge)
        self.nodes[end_index].addIncomingEdge(edge)
        if self.edge_id_to_edge.get(edge.id):
            print("WARNING DOUBLE EDGE {} <-> {}".format(self.edge_id_to_edge[edge.id], edge))
        self.edge_id_to_edge[edge.id] = edge
        if edge.source_edge_id is not None:
            self.source_edge_id_to_edge[edge.source_edge_id] = edge
            
    def deleteEdge(self, edge):
        start = edge.start_node
        end = edge.end_node
        try:
            del start.outgoing_edges[edge.id]
        except KeyError:
            pass
        try:
            del end.incoming_edges[edge.id]
        except KeyError:
            pass     
        try:
            del self.edge_id_to_edge[edge.id]
        except KeyError:
            pass      
        try:
            del self.source_edge_id_to_edge[edge.source_edge_id]
        except KeyError:
            pass           
        

    def plotNetwork(self, outputpath = None, highlight_nodes = [], return_ax = False, ax = None):
        """creates a plot of the geometric informations of the network
        if outputpath == None: uses plt.show()
        else: stores png at ouputpath
        """

        from matplotlib import pyplot as plt
        from matplotlib import cm

        colors=[cm.Set1(i) for i in range(8)]  #cm.jet(np.linspace(0,1,5))
        c_index = 0

        node_gpd_list = []
        for node in self.nodes:
            node_gpd_list.append(node.getAttributeDictGEOJSON())
        node_gpd = gpd.GeoDataFrame(node_gpd_list)
        ax = node_gpd.plot(ax = ax, color = "gray", alpha = 0.2, markersize = 1, figsize = (7,7))
        
        edge_gpd_list = []
        for edge in self.edge_id_to_edge.values():
            edge_gpd_list.append(edge.getAttributeDictGEOJSON())
        edge_gpd = gpd.GeoDataFrame(edge_gpd_list)
        edge_gpd.plot(ax = ax, color = "gray", alpha = 0.2)

        supernode_gpd_list = []
        for supernode in self.super_nodes.values():
            supernode_gpd_list.append(supernode.getAttributeDictGEOJSON())
        if len(supernode_gpd_list) > 0:
            supernode_gpd = gpd.GeoDataFrame(supernode_gpd_list)
            supernode_gpd.plot(ax = ax, color = "gray", alpha = 0.2)

        for infra_name in self.registered_infra.keys():
            boarding_nodes_gpd_list = []
            for node in self.nodes:
                if node.boarding_point_info.get(infra_name):
                    boarding_nodes_gpd_list.append(node.getAttributeDictGEOJSON())
            if len(boarding_nodes_gpd_list) > 0:
                boarding_nodes_gpd = gpd.GeoDataFrame(boarding_nodes_gpd_list)
                boarding_nodes_gpd.plot(ax = ax, color = colors[c_index], alpha = 0.7, marker = ".", label = "AP {}".format(infra_name), markersize=10)
                c_index += 1
            depot_nodes_gpd_list = []
            for node in self.nodes:
                if node.depot_info.get(infra_name):
                    depot_nodes_gpd_list.append(node.getAttributeDictGEOJSON())
            if len(depot_nodes_gpd_list) > 0:
                depot_nodes_gpd = gpd.GeoDataFrame(depot_nodes_gpd_list)
                depot_nodes_gpd.plot(ax = ax, color = colors[c_index], alpha = 1.0, marker = "o", markersize = 60, label = "DEPOT {}".format(infra_name))
                c_index += 1
        if len(highlight_nodes) > 0:
            h_node_gpd_list = []
            for n in highlight_nodes:
                node = self.nodes[n]
                h_node_gpd_list.append(node.getAttributeDictGEOJSON())
            h_node_gpd = gpd.GeoDataFrame(h_node_gpd_list)
            h_node_gpd.plot(ax=ax, color = colors[c_index], alpha = 0.7, marker = ".", label = "highlight")
        
        plt.legend()
        plt.tight_layout()
        if not return_ax:
            if outputpath is None:
                plt.show()
            else:
                plt.savefig(outputpath)
                plt.close()
        else:
            return ax

    def storeNewFullNetwork(self, networks_dir, name):
        """ creates a new network folder with all information but updated structure
        also stores all network travel times, depot, boarding points and depots in the corresponding matched folders
        networks_dir : directory where the network should be stored
        name: name of the new network
        """
        try:
            os.mkdir(os.path.join(networks_dir, name))
        except:
            print("could not create {}".format(os.path.join(networks_dir, name)))
        try:
            os.mkdir(os.path.join(networks_dir, name, "base"))
        except:
            print("could not create {}".format(os.path.join(networks_dir, name, "base")))
        self.updateClassIndices()
        self._convertBaseInformationToCSV(os.path.join(networks_dir, name, "base"))
        self._convertFullInformationToGeoJSON(os.path.join(networks_dir, name, "base"))
        for foldername in self.network_tt_times.keys():
            self._storeTTData(os.path.join(networks_dir, name), foldername)
        for zoning_name, zoning_path in self.registered_zonings.items():
            self._store_zoning_information(zoning_name, zoning_path, name)
        for infra_name, infra_path in self.registered_infra.items():
            self._store_depot_information(infra_name, infra_path, name)
            self._store_boarding_point_information(infra_name, infra_path, name)

    def _convertFullInformationToGeoJSON(self, path):
        """ create the all_infos.geojson files """
        node_gpd_list = []
        for node in self.nodes:
            node_gpd_list.append(node.getAttributeDictGEOJSON())
        node_gpd = gpd.GeoDataFrame(node_gpd_list)
        node_gpd.to_file(os.path.join(path, "nodes_all_infos.geojson"), driver="GeoJSON")

        edge_gpd_list = []
        for edge in self.edge_id_to_edge.values():
            edge_gpd_list.append(edge.getAttributeDictGEOJSON())
        edge_gpd = gpd.GeoDataFrame(edge_gpd_list)
        edge_gpd.to_file(os.path.join(path, "edges_all_infos.geojson"), driver="GeoJSON")

        supernode_gpd_list = []
        for supernode in self.super_nodes.values():
            supernode_gpd_list.append(supernode.getAttributeDictGEOJSON())
        if len(supernode_gpd_list) > 0:
            supernode_gpd = gpd.GeoDataFrame(supernode_gpd_list)
            supernode_gpd.to_file(os.path.join(path, "supernodes_all_infos.geojson"), driver="GeoJSON")

    def _convertBaseInformationToCSV(self, path):
        """ creates the base csv files of the network """
        node_pd_list = []
        for node in self.nodes:
            node_pd_list.append(node.getAttributeDictBaseFile())
        node_pd = pd.DataFrame(node_pd_list)
        node_pd.to_csv(os.path.join(path, "nodes.csv"), index=False)

        edge_pd_list = []
        for edge in self.edge_id_to_edge.values():
            edge_pd_list.append( edge.getAttributeDictBaseFile() )
        edge_pd = pd.DataFrame(edge_pd_list)
        edge_pd.to_csv(os.path.join(path, "edges.csv"), index=False)

    def _storeTTData(self, networks_dir, folder_name):
        """ creates the travel time files"""
        tt_list = []
        for key, edge in self.edge_id_to_edge.items():
            if edge.time_dependent_tt.get(folder_name):
                tt = edge.time_dependent_tt[folder_name]
                o = edge.start_node.index
                d = edge.end_node.index
                tt_list.append({'from_node': o, 'to_node': d, 'edge_tt' : tt})
        new_df = pd.DataFrame(tt_list)
        try:
            os.mkdir(os.path.join(networks_dir, str(folder_name)))
        except:
            pass
        new_df.to_csv(os.path.join(networks_dir, str(folder_name), "edges_td_att.csv"), index=False)

    def _store_zoning_information(self, zoning_name, zoning_path, network_name):
        zoning_df_list = []
        for node in self.nodes:
            zone_id = node.zoning_info.get(zoning_name)
            if zone_id is not None:
                zoning_df_list.append({"node_index" : node.index, "zone_id" : zone_id, "is_centroid" : 1 if node.is_centroid else 0})
        if len(zoning_df_list) > 0:
            zoning_df = pd.DataFrame(zoning_df_list)
            p = os.path.join(zoning_path, network_name)
            if not os.path.isdir(p):
                os.mkdir(p)
            zoning_df.to_csv(os.path.join(p, "node_zone_info.csv"), index=False)

    def _store_depot_information(self, infra_name, infra_path, network_name):
        depot_df_list = []
        for node in self.nodes:
            depot_info_dict = node.depot_info.get(infra_name)
            if depot_info_dict is not None:
                out_dict = depot_info_dict.copy()
                out_dict["node_index"] = node.index
                depot_df_list.append(out_dict)
        if len(depot_df_list) > 0:
            depot_df = pd.DataFrame(depot_df_list)
            p = os.path.join(infra_path, network_name)
            if not os.path.isdir(p):
                os.mkdir(p)
            depot_df.to_csv(os.path.join(p, "depots.csv"), index=False)

    def _store_boarding_point_information(self, infra_name, infra_path, network_name):
        boarding_point_list = []
        for node in self.nodes:
            is_boarding_point = node.boarding_point_info.get(infra_name)
            if is_boarding_point is not None:
                boarding_point_list.append({"node_index" : node.index})
        if len(boarding_point_list) > 0:
            p = os.path.join(infra_path, network_name)
            if not os.path.isdir(p):
                os.mkdir(p)
            bp_df = pd.DataFrame(boarding_point_list)
            bp_df.to_csv(os.path.join(p, "boarding_points.csv"), index=False)

    def _initialize_rtree(self, only_must_stop = False):
        """ add all node coordinates to an rtree (needs to be reset if node_indices change) """
        self.r_tree = rtree.index.Index()
        for node in self.nodes:
            if not node.is_deleted:
                if only_must_stop and not node.is_stop_only:
                    continue
                index = node.index
                coords = node.coordinates
                self.r_tree.insert(index, (coords[0], coords[1], coords[0], coords[1]) )

    def get_nearest_node(self, coords, only_must_stop = True):
        if self.r_tree is None:
            self._initialize_rtree(only_must_stop=only_must_stop)
        nearest_node =  list(self.r_tree.nearest((coords[0], coords[1], coords[0], coords[1]), 1))
        nearest_node = nearest_node[0]
        return nearest_node

    def get_nearest_node_distance(self, coords, only_must_stop = True):
        nearest_node = self.get_nearest_node(coords, only_must_stop=only_must_stop)
        c = self.nodes[nearest_node].coordinates
        return np.math.sqrt( (c[0] - coords[0]) **2 + (c[1] - coords[1]) ** 2 )

    def deleteUnconnectedAndSelfConnectedNodes(self, except_nodes = {}, also_delete_single_connected_nodes = True):
        """ removes all nodes and edges that lead to a connection with itself or are without any connection 
        nodes with the is_stop_only attribute or registered as boarding point or depot cant be deleted
        :param except_nodes: dict node_index -> 1 for additional nodes the are not allowed to be deleted
        :param also_delete_single_connected_nodes: bool if True: also nodes with only one connected edge are deleted
        """
        self.updateClassIndices()
        for i, node in enumerate(self.nodes):
            self_connected = False
            for key, edge in node.outgoing_edges.items():
                if edge.end_node.index == node.index:
                    self_connected = True
                    print("self connected ", i, key, edge.end_node.index, edge.source_edge_id)
            if self_connected:
                to_del_edges = []
                for key, edge in node.outgoing_edges.items():
                    if edge.end_node.index == node.index:
                        to_del_edges.append(key)
                for key in to_del_edges:
                    del node.outgoing_edges[key]
                    del node.incoming_edges[key]
                    del self.edge_id_to_edge[key]
        if also_delete_single_connected_nodes:
            while True:
                reduceable_nodes = []
                print("number nodes before {}".format(len(self.nodes)))
                for i, node in enumerate(self.nodes):
                    if node.is_stop_only:
                        continue
                    if except_nodes.get(i):
                        continue
                    if len(node.outgoing_edges.keys()) == 0 or len(node.incoming_edges.keys()) == 0:
                        allready_flag = False
                        for key, edge in node.outgoing_edges.items():
                            if edge.end_node_index in reduceable_nodes:
                                allready_flag = True
                                break
                        if not allready_flag:
                            for key, edge in node.incoming_edges.items():
                                if edge.start_node_index in reduceable_nodes:
                                    allready_flag = True
                                    break
                        if not allready_flag:
                            reduceable_nodes.append(i)
                print("number wrong connected nodes {}".format(len(reduceable_nodes)))
                #self.plotNetwork(highlight_nodes=reduceable_nodes)
                if len(reduceable_nodes) == 0:
                    print("all nodes updated!")
                    break
                for i in reduceable_nodes:
                    node = self.nodes[i]
                    if len(node.outgoing_edges.keys()) == 0 or len(node.incoming_edges.keys()) == 0:
                        self.deleteNode(i)
                self.updateClassIndices()
                print("number of nodes after: {}".format(len(self.nodes)))

    def checkEnds(self):
        high_light = []
        for node in self.nodes:
            if node.is_stop_only:
                if len(node.outgoing_edges) == 1 and len(node.incoming_edges) == 1:
                    prev_node = list(node.incoming_edges.values())[0].end_node
                    next_node = list(node.outgoing_edges.values())[0].start_node
                    if len(prev_node.outgoing_edges) == 1 or len(next_node.incoming_edges) == 1:
                        high_light.append(node.index)
                        turn_edge = NetworkEdge(prev_node.index, next_node.index, 0.01, 0.0, roadtype="Turn", polyline=LineString([prev_node.coordinates, next_node.coordinates]))
                        self.addEdge(turn_edge)
                        print("turn edge added at stop node {}".format(node.index))
        self.addNodeAttributesToClasses()
        #self.plotNetwork(highlight_nodes=high_light)

    def checkDeadEnds(self):
        from src.routing.NetworkForPreprocessing import NetworkForPreprocessing
        last_errors = 0
        while True:
            nw = NetworkForPreprocessing()
            for node in self.nodes:
                nw.add_node(node.index, is_stop_only=node.is_stop_only)
            for edge in self.edge_id_to_edge.values():
                nw.add_edge(edge.start_node_index, edge.end_node_index, edge.travel_time, edge.travel_distance)
            h = []
            for node in nw.nodes:
                if node.must_stop():
                    prev_pairs = node.get_prev_node_edge_pairs()
                    next_pairs = node.get_next_node_edge_pairs()
                    conn_error = False
                    for prev_node, prev_edge in prev_pairs:
                        o_pos = nw.return_node_position(prev_node.node_index)
                        d_posses = [nw.return_node_position(next_node.node_index) for next_node, _ in next_pairs]
                        if len(d_posses) > 0:
                            res = nw.return_travel_costs_1toX(o_pos, d_posses)
                            if len(res) == 0:
                                conn_error = True
                                for next_node, _ in next_pairs:
                                    polyline=LineString([self.nodes[prev_node.node_index].coordinates, self.nodes[next_node.node_index].coordinates])
                                    turn_edge = NetworkEdge(prev_node.node_index, next_node.node_index, 0.01, 0.0, roadtype="Turn", polyline=polyline)
                                    self.addEdge(turn_edge)
                                    #print("turn edge added at stop node {}".format(node.node_index))
                    if conn_error:
                        #print("conn error for {} | in {} -> out {}".format(node.node_index, len(prev_pairs), len(next_pairs)))
                        #print("  in: {} | out: {}".format([str(x) for _,x in prev_pairs], [str(x) for _,x in next_pairs]))
                        h.append(node.node_index)
                        #if len(prev_pairs) > 0 and len(next_pairs) > 0:
            print("errors: {}".format(len(h)))
            self.addNodeAttributesToClasses()
            self.plotNetwork(highlight_nodes=h)
            if last_errors == len(h) or len(h) == 0:
                break
            last_errors = len(h)

    def evaluateConnectivity(self, except_nodex = {}):
        from src.routing.NetworkForPreprocessing import NetworkForPreprocessing
        import random
        del_org_node_ids = []
        while True:
            nw = NetworkForPreprocessing()
            for node in self.nodes:
                nw.add_node(node.index, is_stop_only=node.is_stop_only)
            for edge in self.edge_id_to_edge.values():
                nw.add_edge(edge.start_node_index, edge.end_node_index, edge.travel_time, edge.travel_distance)
            r_targets = [nw.return_node_position(n.node_index) for n in nw.nodes]
            target = random.choice(r_targets)
            res1 = nw.return_travel_costs_1toX(target, r_targets)
            #print("{}/{} targets reach by routing 1toX from {}".format(len(res1), len(nw.nodes), r_targets[0]))
            res2 = nw.return_travel_costs_Xto1(r_targets,target)
            #print("{}/{} targets reach by routing Xto1 from {}".format(len(res2), len(nw.nodes), r_targets[0]))
            tts = 0
            ttd = 0
            c = 0
            unconnected = {}
            connected_fw = {}
            connected_bw = {}
            for a, _, tt, dis in res1:
                if tt == float("inf"):
                    c += 1
                    unconnected[a[0]] = 1
                    continue
                tts += tt
                ttd += dis
                connected_fw[a[0]] = 1
            for a, _, tt, dis in res2:
                if tt == float("inf"):
                    c += 1
                    unconnected[a[0]] = 1
                    continue
                tts += tt
                ttd += dis
                connected_bw[a[0]] = 1
            for i in range(len(nw.nodes)):
                if connected_fw.get(i) is None or connected_bw.get(i) is None:
                    unconnected[i] = 1
            print("average vel found: {} m/s".format(ttd/tts))
            if len(unconnected.keys()) > len(nw.nodes)/2:
                print("{}/{} not reached. seems like nodes itself is ureachable!".format(len(unconnected.keys()), len(nw.nodes)))
                continue
            print("{}/{} unreached nodes found".format(len(unconnected.keys()), len(nw.nodes)))
            #self.plotNetwork(highlight_nodes=[n for n in unconnected.keys()])
            if len(unconnected.keys()) == 0:
                print("no more unconnected nodes found")
                break
            del_node_ids = []
            for n in unconnected.keys():
                if self.nodes[n].is_stop_only:
                    print("warning: deleting a must stop nodes! {}".format(n))
                    del_node_ids.append(n)
                    del_org_node_ids.append(self.original_node_index_to_node[n].index)
                self.deleteNode(n)
            self.updateClassIndices()
            print("del_nodes: {}".format(",".join([str(x) for x in del_node_ids])))
        return del_org_node_ids

    def reduceNetwork(self, except_nodes = {}):
        """ reduces the network by removing iterativly all nodes with exactly one incoming and one outgoing edge
        and merges these edges
        except_nodes: dictionary node_id -> 1 this nodes will not be removed
        """
        self.deleteUnconnectedAndSelfConnectedNodes(except_nodes=except_nodes, also_delete_single_connected_nodes=False)
        self.updateClassIndices()
        while True:
            last_time_number_nodes = len(self.nodes)
            reduceable_nodes = []
            print("number nodes before {}".format(len(self.nodes)))
            for i, node in enumerate(self.nodes):
                if node.is_stop_only:
                    continue
                if except_nodes.get(i):
                    continue
                if (len(node.depot_info.keys()) > 0 or len(node.boarding_point_info.keys()) > 0):
                    continue
                if (len(node.outgoing_edges.keys()) < 2 and len(node.incoming_edges.keys()) < 2) or len(node.outgoing_edges.keys()) == 0 or len(node.incoming_edges.keys()) == 0:
                    allready_flag = False
                    for key, edge in node.outgoing_edges.items():
                        if edge.end_node_index in reduceable_nodes:
                            allready_flag = True
                            break
                    if not allready_flag:
                        for key, edge in node.incoming_edges.items():
                            if edge.start_node_index in reduceable_nodes:
                                allready_flag = True
                                break
                    if not allready_flag:
                        reduceable_nodes.append(i)
            print("number reducable nodes {}".format(len(reduceable_nodes)))
            if len(reduceable_nodes) == 0:
                print("all nodes reduced!")
                break
            for i in reduceable_nodes:
                node = self.nodes[i]
                if len(node.outgoing_edges.keys()) == 0 or len(node.incoming_edges.keys()) == 0:
                    # print("out")
                    # for edge in node.outgoing_edges.values():
                    #     print(edge.end_node.incoming_edges.keys())
                    # print("in")
                    # for edge in node.incoming_edges.values():
                    #     print(edge.end_node.outgoing_edges.keys())
                    # print("")
                    self.deleteNode(i)
                else:
                    incoming_edge = list(node.incoming_edges.values())[0]
                    outgoing_edge = list(node.outgoing_edges.values())[0]
                    new_start = incoming_edge.start_node
                    new_end = outgoing_edge.end_node
                    new_dis = incoming_edge.travel_distance + outgoing_edge.travel_distance
                    new_tt = incoming_edge.travel_time + outgoing_edge.travel_time
                    source_edge_id = "{}-{}".format(incoming_edge.source_edge_id, outgoing_edge.source_edge_id)
                    roadtype = incoming_edge.roadtype
                    if incoming_edge.roadtype != outgoing_edge.roadtype:
                        if incoming_edge.roadtype == "Turn":
                            roadtype = outgoing_edge.roadtype
                        elif outgoing_edge.roadtype == "Turn":
                            roadtype = incoming_edge.roadtype
                        else:
                            print("Warning for reducing network: inconistent roadtypes -> incoming edge roadtype chosen! {} -> {}".format(incoming_edge.roadtype, outgoing_edge.roadtype))
                    polyline = incoming_edge.polyline + outgoing_edge.polyline
                    times = set(list(incoming_edge.time_dependent_tt.keys()) + list(outgoing_edge.time_dependent_tt.keys()))
                    new_time_dependent_tt = {}
                    for t in times:
                        new_time_dependent_tt[t] = incoming_edge.time_dependent_tt.get(t, incoming_edge.travel_time) + outgoing_edge.time_dependent_tt.get(t, outgoing_edge.travel_time)
                    new_edge = NetworkEdge(new_start.index, new_end.index, new_tt, new_dis, source_edge_id=source_edge_id, polyline=polyline, roadtype=roadtype)
                    new_edge.start_node = new_start
                    new_edge.end_node = new_end
                    new_edge.time_dependent_tt = new_time_dependent_tt
                    #print(len(new_edge.start_node.outgoing_edges.items()))
                    if self.edge_id_to_edge.get(new_edge.id):
                        print("WARNING SIMILAR EDGE ALLREADY HERE -> CONTINUE! {} <-> {}".format(new_edge, self.edge_id_to_edge.get(edge.id)))
                        continue
                    self.addEdge(new_edge)
                    self.deleteNode(i)
                    #print(len(new_edge.start_node.outgoing_edges.items()))
            
            self.updateClassIndices()
            print("number of nodes after: {}".format(len(self.nodes)))
            if len(self.nodes) == last_time_number_nodes:
                print("break becouse no number didnt change! {}".format(len(self.nodes)))
                break

        self.deleteUnconnectedAndSelfConnectedNodes(except_nodes=except_nodes, also_delete_single_connected_nodes=False)

    def deleteNodesWithoutZoneSystem(self, instant_remove = True):
        """ this function looks for nodes without entries in node.zoning_info and deletes them 
        instant remove False: nodes are only marked as deleted but not removed until self.updateClassIndices() is called
        """
        c = 0
        for node in self.nodes:
            if len(node.zoning_info.keys()) == 0:
                self.deleteNode(node.index)
                c += 1
        print("deleted {}/{} nodes without a zone system".format(c, len(self.nodes)))
        if instant_remove:
            self.updateClassIndices()
            self.deleteUnconnectedAndSelfConnectedNodes()

    def sort_for_priority_nodes(self, list_priority_nodes = []):
        """ this method resorts the node and the corresponding node indices
        thereby the nodes in the list_prority nodes (current indices) will get the indices [0, len(list_priority_nodes) -1]
        all nodes with the is_stop_only attribute will be at indices [len(list_priority_nodes), ...]
        this can be usefull for non-complete travel-time matrices when using the script
        create_partially_preprocessed_travel_time_tables.py
        following node order is applied:
            first node_indices: nodes from the list list_priority_nodes (same order)
            next node_indices: registered as depot/boarding point
            next node_indices: remaining nodes with the is_stop_only attribute
            next node_indices: all other remaining nodes
        """
        new_node_list = []
        done = {}
        for node_index in list_priority_nodes:
            new_node_list.append(self.nodes[node_index])
            done[node_index] = 1
        for node in self.nodes:
            if (len(node.depot_info.keys()) > 0 or len(node.boarding_point_info.keys()) > 0) and done.get(node.index) is None:
                new_node_list.append(node)
                done[node.index] = 1
        for node in self.nodes:
            if node.is_stop_only and done.get(node.index) is None:
                new_node_list.append(node)
                done[node.index] = 1
        for node in self.nodes:
            if done.get(node.index) is None:
                new_node_list.append(node)
        print("number priority nodes: {}".format(len(done)))
        self.nodes = new_node_list
        self.updateClassIndices()
        
    def create_zone_centriods(self, zone_system_name, mode="single", must_stop_only=True):
        """
        :param zone_system_name: zone system name
        :param mode: either "single" or "all" (single takes center (minimizing quadratic distances))
        :param is_stop_only: only nodes that are stops_only
        """
        zone_id = 0
        while True:
            nodes_in_zone = []
            for n in self.nodes:
                if n.zoning_info.get(zone_system_name, -1) == zone_id:
                    if must_stop_only:
                        if n.is_stop_only:
                            nodes_in_zone.append(n)
                    else:
                        nodes_in_zone.append(n)
            if len(nodes_in_zone) == 0:
                break
            if mode == "single":
                center_distance = float("inf")
                center_node = None
                for n1 in nodes_in_zone:
                    c_dis = 0
                    c1 = n1.coordinates
                    for n2 in nodes_in_zone:
                        c2 = n2.coordinates
                        dis_sq = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
                        c_dis += dis_sq
                    if c_dis < center_distance:
                        center_distance = c_dis
                        center_node = n1
                if center_node is not None:
                    # print("center for zone ", zone_id, " found")
                    center_node.is_centroid = True
            elif mode == "all":
                for n in nodes_in_zone:
                    n.is_centroid = True
            else:
                raise EnvironmentError(f"mode {mode} not defined!")
            zone_id += 1

    def transform_demand_files(self, demand_folder, demand_name, output_network_name):
        """ this method transforms all demand files found in the input_demand_folder and translates
        start node index and end node index of the traveler trips in case node indices have changed
        the resulting demand files are stored in the output_demand_folder
        :param input_demand_folder: (\data\demand\{demand_name}\matched\{old_network_name})
        :param output_demand_folder: (\data\demand\{demand_name}\matched\{new_network_name}) # TODO # automatic
        """
        def translate_start_end(rq_row):
            org = rq_row["start"]
            new_node = self.original_node_index_to_node[org]
            if new_node.is_deleted:
                raise KeyError("this node has been deleted in a preprocessing step {}".format(new_node))
            new_node_index = new_node.index
            rq_row["start"] = new_node_index

            org = rq_row["end"]
            new_node = self.original_node_index_to_node[org]
            if new_node.is_deleted:
                raise KeyError("this node has been deleted in a preprocessing step {}".format(new_node))
            new_node_index = new_node.index
            rq_row["end"] = new_node_index
            return rq_row
        input_demand_folder = os.path.join(demand_folder, demand_name, "matched", self.network_name)
        output_demand_folder = os.path.join(demand_folder, demand_name, "matched", output_network_name)
        if not os.path.isdir(output_demand_folder):
            os.mkdir(output_demand_folder)
        for file in os.listdir(input_demand_folder):
            file_path = os.path.join(input_demand_folder, file)
            if os.path.isfile(file_path):
                print("transforming {}".format(file))
                rq_df = pd.read_csv(file_path)
                rq_df = rq_df.apply(translate_start_end, axis=1)
                rq_df.to_csv(os.path.join(output_demand_folder, file), index = False)

    def transform_boarding_point_files(self, input_bp_file, output_bp_file):
        new_bp_list = []
        with open(input_bp_file, "r") as f:
            lines = f.read()
            for l in lines.split("\n"):
                if not l:
                    continue
                node_index = int(l)
                new_node = self.original_node_index_to_node[node_index]
                if new_node.is_deleted:
                    raise KeyError("this node has been deleted in a preprocessing step {}".format(new_node))
                new_node_index = new_node.index
                new_bp_list.append(str(new_node_index))
        with open(output_bp_file, "w") as f:
            f.write("\n".join(new_bp_list))

    def transform_visum_id_matching_files(self, input_folder, output_folder, match_deleted_nearest = False):
        """ looks for the files "visum_edge_id_to_edge_index.csv" and "visum_node_id_to_node_index.csv" to transform node_indices of the preprocessed network
        and stores the corresponding files in the ouput_folder """
        edge_df = pd.read_csv(os.path.join(input_folder, "visum_edge_id_to_edge_index.csv"))
        node_df = pd.read_csv(os.path.join(input_folder, "visum_node_id_to_node_index.csv"))

        new_node_df_list = []
        v_id_to_node_index = {}
        for _, entries in node_df.iterrows():
            v_id = entries["visum_node_id"]
            n_id_old = entries["node_index"]
            new_node = self.original_node_index_to_node.get(n_id_old)
            if new_node is None or new_node.is_deleted:
                if match_deleted_nearest:
                    new_nearest_node = self.get_nearest_node(new_node.coordinates, only_must_stop=True)
                    v_id_to_node_index[v_id] = self.nodes[new_nearest_node].index
                    new_node_df_list.append( {"visum_node_id" : v_id, "node_index" : self.nodes[new_nearest_node].index})
                continue
            v_id_to_node_index[v_id] = new_node.index
            new_node_df_list.append( {"visum_node_id" : v_id, "node_index" : new_node.index})
        new_node_df = pd.DataFrame(new_node_df_list)
        print("transformed visum node translater: {}/{} entries kept".format(new_node_df.shape[0], node_df.shape[0]))

        new_edge_df_list = []
        for _, entries in edge_df.iterrows():
            #visum_edge_id,edge_id,visum_start_node_id,visum_end_node_id
            if v_id_to_node_index.get(entries["visum_start_node_id"]) and v_id_to_node_index.get(entries["visum_end_node_id"]):
                new_edge_df_list.append({"visum_edge_id" : entries["visum_edge_id"], "visum_start_node_id" : entries["visum_start_node_id"], "visum_end_node_id" : entries["visum_end_node_id"], "edge_id" : ""})
        new_edge_df = pd.DataFrame(new_edge_df_list)
        # new_edge_df_list = []
        # for edge_id, edge in self.edge_id_to_edge.items():
        #     if type(edge.source_edge_id) == str:
        #         visum_edge_ids = edge.source_edge_id.split("-")
        #     else:
        #         visum_edge_ids = [edge.source_edge_id]
        #     start = edge.start_node.index
        #     end = edge.end_node.index
        #     e_id = "{};{}".format(start, end)
        #     for v_id in visum_edge_ids:
        #         if v_id is not None:
        #             new_edge_df_list.append({"visum_edge_id" : v_id, "edge_id" : e_id, , "visum_start_node_id" : start_node, "visum_end_node_id" : end_node})
        # new_edge_df = pd.DataFrame(new_edge_df_list)
        print("transformed visum edge translator: {}/{} entries kept".format(new_edge_df.shape[0], edge_df.shape[0]))

        new_edge_df.to_csv(os.path.join(output_folder, "visum_edge_id_to_edge_index.csv"), index = False)
        new_node_df.to_csv(os.path.join(output_folder, "visum_node_id_to_node_index.csv"), index = False)



        

if __name__ == "__main__":
    network_path = r'C:\Users\ge37ser\Documents\Projekte\TEMPUS\AP7\APP\rpp-backend\src\main\resources\python\data\networks\osmn_munich_test_area'
    nw = FullNetwork(network_path)
    x = nw.get_nearest_node( (11.568688, 48.149828), only_must_stop=False )
    print(x)
    # nw.plotNetwork()
    # nw.storeNewFullNetwork(r'C:\Users\ge37ser\Documents\Projekte\TEMPUS\AP7\APP\rpp-backend\src\main\resources\python\data\networks', "maxvorstadt_rpp_19072022")



        