import os
import socket
import json
import traceback
import datetime

from src.misc.globals import *

STAT_INT = 0
ENCODING = "utf-8"
LOG_COMMUNICATION = True
LARGE_INT = 100000

# TODO # think about use of global variables
class MobiToppSocket:
    def __init__(self, fs_obj, init_status):
        """This method initiates the communcation socket between mobitopp and the python fleet simulation module.

        :param fs_obj: instance of FleetSimulation; fs_obj.scenario_parameters contains all scenario parameters
        """
        # get server localhost ip and port: (127.0.0.1, 6666)
        self.server_ip = fs_obj.scenario_parameters.get("HOST", "localhost")
        self.server_port = fs_obj.scenario_parameters["socket"]
        self.fleet_sim = fs_obj
        # create server connection: AF_INET (IPv4), SOCK_STREAM (TCP)
        self.server_connection = socket.socket()
        # self.server_connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_connection.bind((self.server_ip, self.server_port))
        self.client_connection = None
        self.init_status = init_status
        self.last_stat_report_time = datetime.datetime.now()
        # create communication log
        self.log_f = os.path.join(fs_obj.dir_names[G_DIR_OUTPUT], "00_socket_com.txt")
        with open(self.log_f, "w") as fh_touch:
            fh_touch.write(f"{self.last_stat_report_time}: Opening socket communication ...\n")

        self.list_arrivals_rq = []
        self.delayed_arrivals = {}  # arrival times with egress time -> list answer_dicts

        # initialize simulation times
        self.prev_simulation_time = 0
        self.next_simulation_time = 0

        self.await_response = True
        self.stay_online = True

    def log_com(self, msg):
        with open(self.log_f, "a") as fhout:
            fhout.write(msg)

    def format_object_and_send_msg(self, obj):
        json_content = json.dumps(obj)
        msg = json_content + "\n"
        if LOG_COMMUNICATION:
            prt_str = f"sending: {msg} to {self.client_connection}\n" + "-" * 20 + "\n"
            self.log_com(prt_str)
        byte_msg = msg.encode(ENCODING)
        self.client_connection.send(byte_msg)

    def keep_socket_alive(self):
        if LOG_COMMUNICATION:
            prt_str = f"run server mode\n" + "-" * 20 + "\n"
            self.log_com(prt_str)

        current_msg = ""
        # currently, Java connects twice to socket without sending a shutdown signal
        retry = True
        while self.stay_online:
            self.server_connection.listen()
            self.client_connection, client_address = self.server_connection.accept()
            if LOG_COMMUNICATION:
                prt_str = f"{datetime.datetime.now()}: connection from :{client_address}\n" + "-" * 20 + "\n"
                self.log_com(prt_str)
            #
            init_obj = {"type": "message", "content": f"init status: {self.init_status}"}
            self.format_object_and_send_msg(init_obj)
            #
            if retry:
                retry = False
                continue
            # TODO # think about error status != 0 in init
            while self.await_response:
                print("await_response")
                # listen to server connection
                byte_stream_msg = self.client_connection.recv(1024)
                time_now = datetime.datetime.now()
                if time_now - self.last_stat_report_time > datetime.timedelta(seconds=STAT_INT):
                    self.last_stat_report_time = time_now
                    if LOG_COMMUNICATION:
                        prt_str = f"time:{time_now}\ncurrent_msg:{current_msg}\nbyte_stream_msg:{byte_stream_msg}\n" \
                                  + "-" * 20 + "\n"
                        self.log_com(prt_str)
                if not byte_stream_msg:
                    continue
                c_stream_msg = byte_stream_msg.decode(ENCODING)
                # Process if there's a newline character in the received stream
                while "\n" in c_stream_msg:
                    tmp = c_stream_msg.split("\n", 1)
                    full_msg = current_msg + tmp[0]  # Combine previously stored message with the new part
                    current_msg_2 = tmp[1]  # Store remaining part after the newline

                    if LOG_COMMUNICATION:
                        prt_str = f"full_msg:{full_msg}\ncurrent_msg:{current_msg}\n" + "-" * 20 + "\n"
                        self.log_com(prt_str)

                    # Only try to parse if we have a full message
                    try:
                        response_obj = json.loads(full_msg)
                        self.handle_message(response_obj)
                    except json.JSONDecodeError as e:
                        if LOG_COMMUNICATION:
                            self.log_com(f"JSON decode error: {e}")
                    full_msg = None  # Reset after successful processing
                    c_stream_msg = current_msg_2
                # Handle incomplete messages (no newline yet)
                if "\n" not in c_stream_msg:
                    current_msg += c_stream_msg  # Append incomplete message part to the current buffer
        # shut down server connection at the end of the simulation
        self.server_connection.close()

    def handle_message(self, response_obj):
        if response_obj["type"] == "comunicate_bookings":
            booking_list = []
            # recieve bookings -> non-blocking call
            try:
                # unpack JSON message
                bookings = response_obj["bookings"]
                for booking in bookings:
                    agent_id = booking["agent_id"]
                    booking_list.append(agent_id)
                    earliest_pickup_time = int(booking["time"])
                    number_passengers = int(booking["nr_pax"])
                    o_node = booking["origin"]
                    d_node = booking["destination"]
                    # register request in TransmoveFleetControl through mobitopp_fleet_sim
                    self.fleet_sim.register_request(agent_id,
                                                    o_node,
                                                    d_node,
                                                    self.next_simulation_time,
                                                    earliest_pickup_time, number_passengers)
                # pack JSON message
                send_obj = {"type": "confirm_recieving_bookings"}
                send_obj["success"] = True
                self.format_object_and_send_msg(send_obj)
            except:
                error_str = traceback.format_exc()
                error_obj = {"type": "fs_error", "content": error_str}
                self.format_object_and_send_msg(error_obj)
                raise EnvironmentError(error_str)
        elif response_obj["type"] == "get_current_fleet_state":
            # request for current fleet state -> blocking call with response
            try:
                # unpack JSON message
                self.next_simulation_time = response_obj["time"]
                # optimize fleet control at self.prev_simulation_time
                self.fleet_sim.optimize_fleet()
                # increase previous simulation time to current simulation time
                self.prev_simulation_time = self.fleet_sim.fs_time
                # check: update network at self.next_simulation_time
                self.fleet_sim.update_network(self.next_simulation_time)
                # update state: increase the simulation time from current simulation time to self.next_simulation_time
                self.list_arrivals_rq = self.fleet_sim.update_state(self.next_simulation_time)
                # get fleet state at self.next_simulation_time/current simulation time
                fleet_state = self.fleet_sim.get_current_fleet_state(self.next_simulation_time)
                # pack JSON message
                send_obj = {"type": "fleet_state"}
                send_obj["time"] = int(self.next_simulation_time)
                send_obj["vehicleStates"] = fleet_state['vehicleStates']
                self.format_object_and_send_msg(send_obj)
            except:
                error_str = traceback.format_exc()
                error_obj = {"type": "fs_error", "content": error_str}
                self.format_object_and_send_msg(error_obj)
                raise EnvironmentError(error_str)
        elif response_obj["type"] == "get_customers_arriving":
            # request for customers arriving -> blocking call with response
            # return list of agents that end their trip in the new time step
            # arrivals follow this specification: arrival = {} with following keys
            # agent_id | int |
            # t_access | int |
            # t_wait | int |
            # t_drive | int |
            # t_egress | int |
            # car_id | int |
            try:
                list_arrivals_dict = []
                for t in range(self.prev_simulation_time + 1, self.next_simulation_time + 1):
                    prev_arrival_dict_list = self.delayed_arrivals.get(t)
                    if prev_arrival_dict_list is not None:
                        for entry in prev_arrival_dict_list:
                            list_arrivals_dict.append(entry)
                        del self.delayed_arrivals[t]
                for rq_obj in self.list_arrivals_rq:
                    t_access = rq_obj.t_access
                    t_egress = rq_obj.t_egress
                    t_wait = rq_obj.pu_time - rq_obj.rq_time - t_access
                    agent_arrival = {"agent_id": int(rq_obj.rid),
                                     "t_drive": int(rq_obj.do_time - rq_obj.pu_time - self.fleet_sim.bt),
                                     "car_id": rq_obj.service_vid, "t_wait": int(t_wait),
                                     "t_access": int(t_access), "t_egress": int(t_egress)}
                    t_arrival = int(rq_obj.do_time + t_egress)
                    if t_arrival <= self.next_simulation_time:
                        list_arrivals_dict.append(agent_arrival)
                    else:
                        try:
                            self.delayed_arrivals[t_arrival].append(agent_arrival)
                        except:
                            self.delayed_arrivals[t_arrival] = [agent_arrival]
                send_obj = {"type": "customers_arriving"}
                # pack JSON message
                send_obj["time"] = self.next_simulation_time
                send_obj["list_arrivals"] = list_arrivals_dict
                self.format_object_and_send_msg(send_obj)
            except:
                error_str = traceback.format_exc()
                error_obj = {"type": "fs_error", "content": error_str}
                self.format_object_and_send_msg(error_obj)
                raise EnvironmentError(error_str)
        elif response_obj["type"] == "end_of_simulation":
            # end of simulation
            self.await_response = False
            self.stay_online = False
        elif response_obj["type"] == "mT_error":
            # error in Java simulation
            self.await_response = False
            self.stay_online = False
        else:
            # raise Warning/Error for unknown message
            pass

