import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

SPEED_LIMIT = 55 / 3.6
NW_CATEGORIES = {}  # category_str > (line_width, opacity)
NW_CATEGORIES["Autobahn"] = (1.5, 0.4)
NW_CATEGORIES["Hauptverkehrsstrassen"] = (1.2, 0.3)
NW_CATEGORIES["On/Off Ramp"] = (1.0, 0.3)
NW_CATEGORIES["Strassen"] = (1.0, 0.2)
NW_CATEGORIES["Nebenstrassen"] = (0.7, 0.1)
NW_CATEGORIES["Anwohnerstrassen"] = (0.5, 0.1)
NW_CATEGORIES["Turn"] = (0.5, 0.1)
NW_CATEGORIES["Stop_Connector"] = (0.5, 0.1)
OD_MAX_WIDTH = 2.5


def load_latex_conform_type1_fonts():
    """Changes to a font that recognized LaTeX maths notiation and is a type1 fonts (relevant for some Journals)."""
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True
    print("LaTeX conform Type1 fonts loaded.")


def reset_latex_conform_type1_fonts():
    """LaTeX fonts are not working on every system (require explicit LaTeX installation)."""
    matplotlib.rcParams['ps.useafm'] = False
    matplotlib.rcParams['pdf.use14corefonts'] = False
    matplotlib.rcParams['text.usetex'] = False
    print("Default fonts loaded.")


def _create_link(edge_row, node_df):
    n1 = node_df.loc[edge_row["from_node"]]
    n2 = node_df.loc[edge_row["to_node"]]
    return LineString([(n1["pos_x"], n1["pos_y"]), (n2["pos_x"], n2["pos_y"])])


def plot_network_bg(nw_base_dir, ax, filter_fast_roads=False, print_progress=True):
    """This function plots the network on a given axis.

    :param nw_base_dir: network base directory
    :param ax: pyplot axis
    :param filter_fast_roads: only pick vehicles with speed limit > 55 km/h
    :param print_progress: print progress to shell
    :return: ax with network plotted on it
    """
    if print_progress:
        print("plot_network_bg():")
    geojson_f = os.path.join(nw_base_dir, "edges_all_infos.geojson")
    if os.path.isfile(geojson_f):
        # use geojson file if it is available
        if print_progress:
            print("\t ... loading geojson file")
        gdf = gpd.read_file(geojson_f)
        gdf["ff_speed"] = gdf["distance"] / gdf["travel_time"]
        if filter_fast_roads:
            gdf = gdf[gdf["ff_speed"] > SPEED_LIMIT]
    else:
        # build geojson file if it is not available
        if print_progress:
            print("\t ... building geojson file")
        node_f = os.path.join(nw_base_dir, "nodes.csv")
        node_df = pd.read_csv(node_f, index_col=0)
        edge_f = os.path.join(nw_base_dir, "edges.csv")
        edge_df = pd.read_csv(edge_f)
        edge_df["ff_speed"] = edge_df["distance"] / edge_df["travel_time"]
        if filter_fast_roads:
            edge_df = edge_df[edge_df["ff_speed"] > SPEED_LIMIT]
        edge_df["geometry"] = edge_df.apply(_create_link, args=(node_df,), axis=1)
        gdf = gpd.GeoDataFrame(edge_df)
    if print_progress:
        print("\t ... plotting network")
    if "road_type" in gdf.columns:
        for rt, rt_df in gdf.groupby("road_type"):
            lw,a = NW_CATEGORIES.get(rt, (1.5, 0.5))
            rt_df.plot(ax=ax, color="grey", linewidth=lw, alpha=a)
    else:
        gdf.plot(ax=ax, color="grey", linewidth=1.0, alpha=0.2)
    return ax


def _plot_arrow(row, ax1):
    x0 = row["x0"]
    y0 = row["y0"]
    x1 = row["x1"]
    y1 = row["y1"]
    line_width = row["lw"]
    line_color = row["col"]
    if line_width > 0:
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx ** 2 + dy ** 2)
        hw = max(line_width, 1) * 0.1 * length
        hl = 1.5*hw
        ax1.arrow(x0, y0, dx, dy, width=line_width, head_width=hw, head_length=hl, color=line_color,
                  length_includes_head=True)


def plot_od_arrows(ax, od_df, point_geometry_series, o_kw, d_kw, v_kw, mappable=None):
    """This function creates origin destination arrow plots.

    :param ax: pyplot axis
    :param od_df: data-frame containing origin-destination information
    :param point_geometry_series: GeoSeries index > Point
    :param o_kw: column name of origin in od_df
    :param d_kw: column name of destination in od_df
    :param v_kw: column name of value in od_df
    :param mappable: pyplot mappable
    :return: pyplot axis ax with OD arrows
    """
    od_df["x0"] = od_df.apply(lambda x: point_geometry_series[x[o_kw]].x, axis=1)
    od_df["y0"] = od_df.apply(lambda x: point_geometry_series[x[o_kw]].y, axis=1)
    od_df["x1"] = od_df.apply(lambda x: point_geometry_series[x[d_kw]].x, axis=1)
    od_df["y1"] = od_df.apply(lambda x: point_geometry_series[x[d_kw]].y, axis=1)
    #
    if mappable is not None:
        max_v = mappable.norm.vmax
        od_df["col"]  = od_df.apply(lambda x: mappable.to_rgba(x[v_kw]), axis=1, result_type="reduce")
    else:
        max_v = od_df[v_kw].max()
        od_df["col"] = "k"
    od_df["lw"] = OD_MAX_WIDTH * od_df[v_kw] / max_v
    #
    od_df.apply(_plot_arrow, axis=1, args=(ax,))
    return ax