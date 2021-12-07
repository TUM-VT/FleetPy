import os
from multiprocessing import Process
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import typing as tp
from pathlib import Path
import contextily as ctx
from pyproj import Transformer
from datetime import datetime


FIG_SIZE = (15,10)
# Number of historical points to be displayed on the x-axis
PLOT_LENGTH = 200
# Delay between frames in milliseconds
REALTIME_UPDATE_INTERVAL = 200
VEHICLE_POINT_SIZE = 12
CTX_PROVIDER = ctx.providers.Stamen.TonerLite
BOARDER_SIZE = 1000


class PyPlot(Process):

    def __init__(self, nw_dir, shared_dict: dict, plot_folder: str = None, plot_extent=None):
        """ Class for plotting real time information

        :param nw_dir:      network directory, where the background map is/will be saved
        :param shared_dict: a dictionary object for sharing real time information
        :param plot_folder: full path of the folder for saving the animation. If provided, each frame is saved
                            in the location.
        :param plot_extent: Tuple of (lon_1, lon_2, lat_1, lat_2) marking the left bottom and top right boundary of
                            the map
        """

        super().__init__()
        self.bg_map_path = os.path.join(nw_dir, "downloaded_map.tif")
        self.shared_dict: dict = shared_dict
        self.plot_folder: Path = Path(plot_folder) if plot_folder else None

        self.fig, self.grid_spec, self.axes = None, None, None
        self.plot_extent = plot_extent
        x, y = self.convert_lat_lon(plot_extent[2:4], plot_extent[0:2])
        x[0], y[0] = x[0] - BOARDER_SIZE, y[0] - BOARDER_SIZE
        x[1], y[1] = x[1] + BOARDER_SIZE, y[1] + BOARDER_SIZE
        self.plot_extent_3857 = x + y
        # Download the map image using the extent
        _, bbox = ctx.bounds2raster(x[0], y[0], x[1], y[1], path=self.bg_map_path, source=CTX_PROVIDER)
        self.plot_extent_3857 = bbox
        #exit()

    def convert_lat_lon(self, lats: list, lons: list, to_epsg: str = "epsg:3857"):
        proj_transformer = Transformer.from_proj('epsg:4326', to_epsg)
        x, y = proj_transformer.transform(lats, lons)
        return list(x), list(y)

    def generate_plot_axes(self):
        fig = plt.figure(1, figsize=FIG_SIZE)
        gs = gridspec.GridSpec(3, 3, figure=fig)
        gs.update(wspace=0.025, hspace=0.5)
        axes = [plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), plt.subplot(gs[2, 2]), plt.subplot(gs[:, 0:2])]
        return fig, gs, axes

    def draw_plots(self):
        axes = self.axes
        # Plot the data on the map
        axes[3].axis(self.plot_extent_3857)
        axes[3].set_xlim(self.plot_extent_3857[:2])
        axes[3].set_ylim(self.plot_extent_3857[2:])
        ctx.add_basemap(axes[3], source=self.bg_map_path)

        vehicle_df = self.shared_dict["veh_coord_status_df"]
        possible_status = self.shared_dict["possible_status"]
        for status in possible_status:
            mask = vehicle_df["status"] == status
            coords = vehicle_df[mask]["coordinates"].to_list()
            x, y = [], []
            if coords:
                lons, lats = list(zip(*coords))
                x, y = self.convert_lat_lon(lats, lons)
            axes[3].scatter(x, y, s=VEHICLE_POINT_SIZE, label=status)
        axes[3].legend(loc="upper left")
        axes[3].axis('off')
        axes[3].set_title(str(self.shared_dict["simulation_time"]))

    def save_single_plot(self, datetime_stamp: tp.Union[str, datetime]):
        if self.fig is None:
            self.fig, self.grid_spec, self.axes = self.generate_plot_axes()
        [ax.clear() for ax in self.axes]
        self.draw_plots()
        if self.plot_folder.exists() is False:
            self.plot_folder.mkdir()
        if type(datetime_stamp) == datetime:
            file_name = "plot_{}.png".format(datetime_stamp.strftime("%d-%b-%Y %H-%M-%S"))
        else:
            file_name = "plot_{}.png".format(datetime_stamp)
        plt.savefig(str(self.plot_folder.joinpath(file_name)), bbox_inches = 'tight')

    def __animate(self, i):
        [ax.clear() for ax in self.axes]
        try:
            self.draw_plots()
            if self.grid_spec is not None:
                self.grid_spec.tight_layout(self.fig)
        except KeyError:
            pass

    def run(self):
        def frame():
            i = 0
            while True:
                if self.shared_dict.get("stop", False) is False:
                    i = i + 1
                    yield i
                else:
                    return

        self.fig, self.grid_spec, self.axes = self.generate_plot_axes()
        ani = animation.FuncAnimation(self.fig, self.__animate, interval=REALTIME_UPDATE_INTERVAL)
        plt.show()
