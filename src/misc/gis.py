import numpy as np

def returnEuclidianDistance(x1, y1, x2, y2):
    diff = np.array([x2-x1, y2-y1])
    return np.sqrt(np.dot(diff, diff))

# TODO # after ISTTT: transform_df_crs() not working properly and shift to preprocessing
# def transform_df_crs(df, orig_epsg, co1_col_name, co2_col_name,
#                      transform_epsg, co1_col_out_name=None, co2_col_out_name=None):
#     gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[co1_col_name], df[co2_col_name]),
#                            crs=from_epsg(orig_epsg))
#     tgdf = gdf.to_crs(from_epsg(transform_epsg))
#     return tgdf
