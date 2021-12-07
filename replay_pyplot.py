import os
import sys
from src.ReplayFromResult import ReplayPyPlot


def main(output_dir, sim_seconds_per_real_second, start_time_in_seconds = None, end_time_in_seconds = None, plot_extend = None, live_plot = True, create_images = True):
    """This function uses pyplot to visualize the fleet operation.

    :param output_dir: path to result directory
    :param sim_seconds_per_real_second: determines the speed of the replay
    :param start_time_in_seconds: determines simulation time when replay is started
    :param end_time_in_seconds: determines simulation time when replay ends
    :param plot_extent: Tuple of (lon_1, lon_2, lat_1, lat_2) marking the left bottom and top right boundary of
                        the map (EPSG_WGS = 4326); if None -> extend of loaded network chosen
    :param live_plot: if True: plots directly shown; else: figures stored in ouputdir/plots
    :return:
    """
    if not os.path.isdir(output_dir):
        raise IOError(f"Result directory {output_dir} not found!")
    sim_seconds_per_real_second = float(sim_seconds_per_real_second)
    replay = ReplayPyPlot(live_plot=live_plot, create_images = create_images)
    if start_time_in_seconds is not None and end_time_in_seconds is not None:
        replay.load_scenario(output_dir, start_time_in_seconds=int(start_time_in_seconds), end_time_in_seconds = int(end_time_in_seconds), plot_extend=plot_extend)
    elif start_time_in_seconds is not None:
        replay.load_scenario(output_dir, start_time_in_seconds=int(start_time_in_seconds), plot_extend=plot_extend)
    elif end_time_in_seconds is not None:
        replay.load_scenario(output_dir, end_time_in_seconds=int(end_time_in_seconds), plot_extend=plot_extend)
    else:
        replay.load_scenario(output_dir, plot_extend=plot_extend)
    replay.set_time_step(sim_seconds_per_real_second)
    replay.start()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], start_time_in_seconds = sys.argv[3])
    else:
        print(main.__doc__)
        raise IOError("Incorrect number of arguments!")
