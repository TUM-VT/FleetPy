import os
import sys
import argparse

from src.ReplayFromResult import ReplayPyPlot


def main(output_dir, sim_seconds_per_real_second, start_time_in_seconds = None, end_time_in_seconds = None,
          plot_extend = None, live_plot = True, create_images = True,parcels = False,passengers = False,
          status_type = 1,color_list = False,plot_args=[1,1,1,0,0,0]):
    """This function uses pyplot to visualize the fleet operation.

    :param output_dir: path to result directory
    :param sim_seconds_per_real_second: determines the speed of the replay
    :param start_time_in_seconds: determines simulation time when replay is started
    :param end_time_in_seconds: determines simulation time when replay ends
    :param plot_extent: Tuple of (lon_1, lon_2, lat_1, lat_2) marking the left bottom and top right boundary of
                        the map (EPSG_WGS = 4326); if None -> extend of loaded network chosen
    :param live_plot: if True: plots directly shown; else: figures stored in ouputdir/plots    :param parcels: if True: plots parcel data
    :param passengers: if True: plots passenger data
    :param status_type: if 1: occupancy, if 2: vehicle status
    :param color_list: if True: color list is used for plotting
    :param plot_args: list of integers that determine which plots are created
    
    :return:
    """
    if not os.path.isdir(output_dir):
        raise IOError(f"Result directory {output_dir} not found!")
    sim_seconds_per_real_second = float(sim_seconds_per_real_second)
    replay = ReplayPyPlot(live_plot=live_plot, create_images = create_images,
                          parcels = parcels,passengers = passengers,
                          status_type = status_type,plot_args = plot_args,
                          color_list = color_list)
    if start_time_in_seconds is not None and end_time_in_seconds is not None:
        replay.load_scenario(output_dir, start_time_in_seconds=int(start_time_in_seconds)
                             , end_time_in_seconds = int(end_time_in_seconds), plot_extend=plot_extend)
    elif start_time_in_seconds is not None:
        replay.load_scenario(output_dir, start_time_in_seconds=int(start_time_in_seconds)
                             , plot_extend=plot_extend)
    elif end_time_in_seconds is not None:
        replay.load_scenario(output_dir, end_time_in_seconds=int(end_time_in_seconds)
                             , plot_extend=plot_extend)
    else:
        replay.load_scenario(output_dir, plot_extend=plot_extend)
    replay.set_time_step(sim_seconds_per_real_second)
    replay.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay a simulation result.')
    parser.add_argument('output_dir', type=str, help='path to result directory')
    parser.add_argument('sim_seconds_per_real_second', type=float, help='determines the speed of the replay')
    parser.add_argument('--start_time_in_seconds', type=int, help='determines simulation time when replay is started')
    parser.add_argument('--end_time_in_seconds', type=int, help='determines simulation time when replay ends')
    parser.add_argument('--save_figs', action='store_true', help='if set, figures are stored in ouputdir/plots. Otherwise, plots are shown live')
    parser.add_argument('--plot_1',type=str,default="status_count",help='determines the first plot: available options are: \n' + 
                        "status_count, occupancy_average, occupancy_stack_chart, waiting_time_average, ride_time_average, detour_time_average")
    parser.add_argument('--plot_2',type=str,default="occupancy_average",
                        help='determines the second plot: available options are: \n' + 
                        "status_count, occupancy_average, occupancy_stack_chart, waiting_time_average, ride_time_average, detour_time_average")
    parser.add_argument('--plot_3',type=str,default="occupancy_stack_chart",help='determines the third plot: available options are: \n' + 
                        "status_count, occupancy_average, occupancy_stack_chart, waiting_time_average, ride_time_average, detour_time_average")
    """
    parser.add_argument('--plots',type=str,
                        help='determine the type of plots to be created. Add one for each plot you want to be created. \n'+
                        '1- Status count plot \n'
                        +'2- Occupancy average \n'
                        +'3- Occupancy stack chart \n'
                        +'4- average waiting time \n'
                        +'5- average ride time \n'
                        +'6- average detour time. \n'
                        + 'input example: 101001 for 1,3,6', default="111000"
                        )
                        """
    # add argument called status_type that takes two values passenger and parcel
    parser.add_argument('--map_plot', type=str, 
                        help='determines the type of status to plot Either "occupancy" or "vehicle_status"',default="occupancy")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--parcels',action='store_true',help='if set, plots parcel data',default=False)
    group.add_argument('--passengers',action='store_true',help='if set, plots passenger data',default=False)

    args = parser.parse_args()
    #plot_args = args.plots
    #plot_args.split()
    #plot_args = [int(i) for i in plot_args]
    plot_args = [0,0,0,0,0,0]
    possible_plots = ["status_count","occupancy_average","occupancy_stack_chart","waiting_time_average","ride_time_average","detour_time_average"]
    chosen_plots = [args.plot_1,args.plot_2,args.plot_3]
    plot_args[possible_plots.index(chosen_plots[0])] = 1
    plot_args[possible_plots.index(chosen_plots[1])] = 1
    plot_args[possible_plots.index(chosen_plots[2])] = 1
    #plot_args = [int(i) for i in plot_args]
    if args.map_plot != "occupancy" and args.map_plot != "vehicle_status":
        raise IOError("Incorrect map plot type!")
    args.map_plot = "1" if args.map_plot == "occupancy" else "2"

    if len(plot_args) != 6:
        raise IOError("Incorrect number of plots!")
    if sum(plot_args) == 0:
        raise IOError("No plots selected!")
    if sum(plot_args) != 3:
        raise IOError("Incorrect number of plots! You should select 3 plots!")
    main(args.output_dir, args.sim_seconds_per_real_second, start_time_in_seconds = args.start_time_in_seconds,
         end_time_in_seconds = args.end_time_in_seconds, parcels = args.parcels, passengers = args.passengers,
         status_type = int(args.map_plot),live_plot=not args.save_figs,color_list = False,plot_args=plot_args)
    



