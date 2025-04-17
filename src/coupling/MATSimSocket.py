import os
import zmq
import json
import traceback
import datetime
import pandas as pd

to_del = []
for p in os.sys.path:
    if "FleetPy" in p:
        to_del.append(p)
for p in to_del:
    os.sys.path.remove(p)
os.sys.path.append(r"C:\Users\ge37ser\Documents\Coding\FleetPy")

from src.misc.globals import *
from src.coupling.misc import *
from src.coupling.MATSimSimulationClass import MATSimSimulationClass
from src.FleetSimulationBase import build_operator_attribute_dicts

STAT_INT = 60
ENCODING = "utf-8"
LOG_COMMUNICATION = True
LARGE_INT = 100000

class MATSimSocket:
    """
    A class to handle communication with a MATSim server using sockets.
    """
    def __init__(self, host: str, port: int, scenario_parameters, log_communication: bool = LOG_COMMUNICATION):
        self.server_ip = host
        self.server_port = port
        self.log_communication = log_communication
        scenario_parameters["matsim_iteration"] = 0
        self.scenario_parameters = scenario_parameters
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.server_ip}:{self.server_port}")
        
        # build list of operator dictionaries  # TODO: this could be eliminated with a new YAML-based config system
        self.list_op_dicts = build_operator_attribute_dicts(scenario_parameters, scenario_parameters[G_NR_OPERATORS],
                                                                              prefix="op_")
        self.list_ch_op_dicts  = build_operator_attribute_dicts(scenario_parameters, scenario_parameters.get(G_NR_CH_OPERATORS, 0),
                                                                                 prefix="ch_op_")
        
        self.dir_names = get_directory_dict(scenario_parameters, self.list_op_dicts)
        self.scenario_parameters: dict = scenario_parameters
        
        self.matsim_edge_to_fp_edge, self.fp_edge_to_matsim_edge = self._create_fleetpy_network(scenario_parameters["matsim_network_path"])
        
        self.matsim_to_fleetpy_vid = {}
        self.fleetpy_to_matsim_vid = {}
        
        self.fs_obj = MATSimSimulationClass(scenario_parameters)
        self.dir_names = self.fs_obj.dir_names
        
        # create communication log
        self._output_dir = self.fs_obj.dir_names[G_DIR_OUTPUT]
        self.log_f = os.path.join(self._output_dir, "00_socket_com.txt")
        self.last_stat_report_time = datetime.datetime.now()
        with open(self.log_f, "w") as fh_touch:
            fh_touch.write(f"{self.last_stat_report_time}: Opening socket communication ...\n")
                
    def log_com(self, msg):
        print("SOCKET SENDING: ", msg)
        with open(self.log_f, "a") as fhout:
            fhout.write(msg)
            
    def format_object_and_send_msg(self, obj):
        json_content = json.dumps(obj)
        msg = json_content + "\n"
        if LOG_COMMUNICATION:
            prt_str = f"sending: {msg} to {self.socket}\n" + "-" * 20 + "\n"
            self.log_com(prt_str)
        byte_msg = msg.encode(ENCODING)
        self.socket.send(byte_msg)

    def keep_socket_alive(self):
        if LOG_COMMUNICATION:
            prt_str = f"run client mode\n" + "-" * 20 + "\n"
            self.log_com(prt_str)
            
        #
        print("starting socket communication")
        init_obj = {"type": 0} # "message", "content": f"init status: {self.init_status}"}
        self.format_object_and_send_msg(init_obj)
            
        full_msg = None
        current_msg = ""

        retry = True
        stay_online = True
        while stay_online:
            if LOG_COMMUNICATION:
                prt_str = f"{datetime.datetime.now()}: connection from :{self.socket}\n" + "-" * 20 + "\n"
                self.log_com(prt_str)
            #
            if retry:
                retry = False
                continue
            # TODO # think about error status != 0 in init
            await_response = True
            while await_response:
                # listen to server connection
                byte_stream_msg = self.socket.recv()
                time_now = datetime.datetime.now()
                if time_now - self.last_stat_report_time > datetime.timedelta(seconds=STAT_INT):
                    self.last_stat_report_time = time_now
                    if LOG_COMMUNICATION:
                        prt_str = f"time:{time_now}\ncurrent_msg:{current_msg}\nbyte_stream_msg:{byte_stream_msg}\n" \
                                  + "-" * 20 + "\n"
                        self.log_com(prt_str)
                if not byte_stream_msg:
                    continue
                full_msg = byte_stream_msg.decode(ENCODING)
                response_obj = json.loads(full_msg)
                self._treat_matsim_response(response_obj)
                
                # c_stream_msg = byte_stream_msg.decode(ENCODING)
                # if "\n" in c_stream_msg:
                #     tmp = c_stream_msg.split("\n")
                #     full_msg = current_msg + tmp[0]
                #     current_msg = tmp[1]
                #     if LOG_COMMUNICATION:
                #         prt_str = f"full_msg:{full_msg}\ncurrent_msg:{current_msg}\n" + "-"*20 + "\n"
                #         self.log_com(prt_str)
                # else:
                #     full_msg = None
                #     current_msg += c_stream_msg
                # # full_msg can be longer than msg-len!!!
                # if full_msg is not None:
                #     # full message received
                #     response_obj = json.loads(full_msg)
                #     self._treat_matsim_response(response_obj)
                    
    def _treat_matsim_response(self, response_obj):
        """
        Process the response from MATSim.
        """
        if response_obj["type"] == "start_simulation":
            self._start_simulation(response_obj)
        elif response_obj["type"] == "new_iteration":
            self._new_iteration(response_obj)
        elif response_obj["type"] == "new_time_step":
            self._new_time_step(response_obj)
        elif response_obj["type"] == "new_edge_traveltimes":
            self._new_edge_traveltimes(response_obj)
        elif response_obj["type"] == "end_simulation":
            self._end_simulation(response_obj)
        elif response_obj["type"] == "error":
            self._handle_error(response_obj)

    def _start_simulation(self, response_obj):
        list_vehicle_attributes = response_obj["vehicle_attributes"]
        
        self._initialize_vehicles(list_vehicle_attributes)
            
        response = {"type": "start_simulation", "status": 0}
        self.format_object_and_send_msg(response)
        
    def _initialize_vehicles(self, list_vehicle_attributes):
        self.matsim_to_fleetpy_vid = {}
        self.fleetpy_to_matsim_vid = {}
        
        for vehicle_attributes in list_vehicle_attributes:
            matsim_vehicle_id = int(vehicle_attributes["vehicle_id"])
            vehicle_start_pos = self.from_matsim_to_fleetpy_position(int(vehicle_attributes["position"]))
            vehicle_capacity = int(vehicle_attributes["capacity"])

            vehicle_id = self.fs_obj.add_vehicle(0, vehicle_capacity, vehicle_start_pos[0])
            
            self.matsim_to_fleetpy_vid[matsim_vehicle_id] = vehicle_id
            self.fleetpy_to_matsim_vid[vehicle_id] = matsim_vehicle_id
        
    def _new_iteration(self, response_obj):
        """
        Handle new iteration request from MATSim.
        """
        # end FP simulation
        sim_time = response_obj["sim_time"] # maybe no sim_time here
        self.fs_obj.terminate(sim_time)
        
        scenario_parameters["matsim_iteration"] = response_obj["iteration"]
        self.fs_obj = MATSimSimulationClass(self.scenario_parameters)
        self.fs_obj.dir_names = self.dir_names
        
        list_vehicle_attributes = response_obj["vehicle_attributes"]
        
        self._initialize_vehicles(list_vehicle_attributes)
        
        response = {"type": "new_iteration", "status": 0}
        self.format_object_and_send_msg(response)
        
    def _new_time_step(self, response_obj):
        """
        Handle new time step request from MATSim.
        """
        new_sim_time = response_obj["sim_time"]
        list_requests = response_obj["list_requests"]
        list_vehicle_states = response_obj["vehicle_states"]
        
        for veh_state in list_vehicle_states:
            vid = int(veh_state["vehicle_id"])
            veh_pos = self.from_matsim_to_fleetpy_position(veh_state["position"])
            list_ob_rids = veh_state["ob_rids"]
            status = veh_state["status"]
            first_diverge_link = veh_state["first_diverge_link"] # TODO
            self.fs_obj.update_veh_state(vid, veh_pos, list_ob_rids, status, first_diverge_link)
        
        for rq_entry in list_requests:
            rq_info_dict = {G_RQ_ID: int(rq_entry["request_id"]), 
                            G_RQ_ORIGIN: self.from_matsim_to_fleetpy_position(int(rq_entry["origin"])), 
                            G_RQ_DESTINATION: self.from_matsim_to_fleetpy_position(int(rq_entry["destination"])), 
                            G_RQ_TIME: new_sim_time,
                            G_RQ_EPT: int(rq_entry["ept"]), # TODO optional
                            G_RQ_LPT: int(rq_entry["lpt"]), # TODO optional
                            G_RQ_PAX: 1} # TODO to add?
            rq_series = pd.Series(rq_info_dict)
            rq_series.name = rq_info_dict[G_RQ_ID]
            self.fs_obj.add_request(rq_series)
            
        self.fs_obj.step(new_sim_time)
        
        new_assignments = self.fs_obj.get_current_assignments() # dict (op_id, vid) -> VehPlan
        
        assignment_message = self._create_assignment_message(new_assignments)
        self.format_object_and_send_msg(assignment_message)
    
    def _create_assignment_message(self, new_assignments):
        """
        Create a message with the new assignments for MATSim.
        """
        assignment_message = {"type": "new_assignments", "assignments": []}
        
        for (op_id, veh_id), veh_plan in new_assignments.items():
            matsim_vehicle_id = self.fleetpy_to_matsim_vid[(op_id, veh_id)]
            matsim_edge = self.fp_edge_to_matsim_edge[veh_plan.get_current_edge()]
            assignment_message["assignments"].append({"vehicle_id": matsim_vehicle_id, "edge": matsim_edge})
        
        return assignment_message    

    def _create_fleetpy_network(self, matsim_network_path):
        """
        Create FleetPy network based on MATSim network.
        """
        # Example conversion logic (to be replaced with actual logic)
        fleetpy_data_path = self.dir_names[G_DIR_DATA]
        network_name = self.scenario_parameters[G_NETWORK_NAME]
        matsim_edge_to_fp_edge, fp_edge_to_matsim_edge = create_fleetpy_network_from_matsim(matsim_network_path, fleetpy_data_path, network_name)     
        return matsim_edge_to_fp_edge, fp_edge_to_matsim_edge
            
    def from_matsim_to_fleetpy_position(self, matsim_link, remaining_time=None):
        """
        Convert MATSim position to FleetPy position.
        """
        if remaining_time is None:
            fp_edge = self.matsim_edge_to_fp_edge[int(matsim_link)]
            return (fp_edge[0], None, None)
        else:
            raise NotImplementedError("remaining_time is not None, but not implemented yet")
    
    def from_fleetpy_to_matsim_position(self, fleetpy_position):
        """
        Convert FleetPy position to MATSim position.
        """
        # TODO think about this
        if fleetpy_position[-1] is None:
            any_target = list(self.fp_edge_to_matsim_edge[fleetpy_position[0]].keys())[0]
            matsim_edge = self.fp_edge_to_matsim_edge[fleetpy_position[0]][any_target]
            return matsim_edge
        else:
            matsim_edge = self.fp_edge_to_matsim_edge[fleetpy_position[0]][fleetpy_position[1]]
            return matsim_edge
    
    def from_matsim_to_fleetpy_route(self, matsim_route):
        """
        Convert MATSim route to FleetPy route.
        """
        # Example conversion logic (to be replaced with actual logic)
        raise NotImplementedError("from_matsim_to_fleetpy_route is not implemented yet")
    
    def from_fleetpy_to_matsim_route(self, fleetpy_route):
        """
        Convert FleetPy route to MATSim route.
        """
        # Example conversion logic (to be replaced with actual logic)
        matsim_route = []
        for i in range(len(fleetpy_route) - 1):
            matsim_edge = self.fp_edge_to_matsim_edge[fleetpy_route[i]][fleetpy_route[i + 1]]
            matsim_route.append(matsim_edge)
        return matsim_route
    
    
if __name__ == "__main__":
    # Example usage of MATSimSocket class
    scenario_parameters = {}
    host = "localhost"
    port = 9000
    
    from src.misc.config import ConstantConfig, ScenarioConfig
    
    matsim_network_path = r"C:\Users\ge37ser\Documents\Projekte\MINGA\AP5\IRTSystemX\KopplungMATSimFleetPy\matsim-fleetpy\scenario\network.xml\network.xml"
    
    const_cfg = ConstantConfig(r"C:\Users\ge37ser\Documents\Coding\FleetPy\studies\test_matsim_coupling\scenarios\constant_config_pool.csv")
    print(const_cfg)
    scenarios_cfg = ScenarioConfig(r"C:\Users\ge37ser\Documents\Coding\FleetPy\studies\test_matsim_coupling\scenarios\example_pool.csv")
    print(scenarios_cfg)
    
    whole_config = const_cfg + scenarios_cfg[0]
    whole_config["matsim_network_path"] = matsim_network_path
    whole_config["study_name"] = "test_matsim_coupling"
    whole_config["log_level"] = "info"
    whole_config["n_cpu_per_sim"] = 1
    
    matsim_socket = MATSimSocket(host, port, whole_config, log_communication=True)
    matsim_socket.keep_socket_alive()