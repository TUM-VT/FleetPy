#pragma once
#include <vector>
#include <string>
#include <queue>
#include <utility>
#include <tuple>
#include "Node.h"
#include "Edge.h"  

struct Resultstruct {
	int target;
	double traveltime;
	double traveldistance;
	double travel_time_var;
	double cost_function_value;
};

class Network {
private:
	std::vector<Node> nodes;
	std::vector<int> current_targets;
	int dijkstra_number = 0;
	double getEdgeCostValue(Edge& edge, std::string mode);
	void updateEdgeTravelTime(int start_node_index, int end_node_index, double edge_travel_time,double edge_var, double edge_cfv);
	void setTargets(const std::vector<int>& targets);
	int dijkstraForward(int start_node_index, double time_range, int max_targets,std::string mode); 
	void dijkstraStepForward_(std::priority_queue<std::pair<double, int>>& current_pq, Node& current_node, double current_cost,string mode);
	int dijkstraBackward(int start_node_index, double time_range = -1, int max_targets = -1,std::string mode="edge_tt");
	void dijkstraStepBackward_(std::priority_queue<std::pair<double, int>>& current_pq, Node& current_node, double current_cost,string mode);
	//double Network::getCostValue(std::vector<double> cost_vector, string mode);
	double getCostValue(std::vector<double> cost_vector, std::string mode);  // Fixed extra qualification

	//std::pair<double, double> dijkstraBidirectional(int start_node_index, int end_node_index, int* meeting_node_index);
	std::tuple<double, double, double, double> dijkstraBidirectional(int start_node_index, int end_node_index,string mode, int* meeting_node_index);

	std::vector<int> _last_found_route_fw;
	std::vector<int> _last_found_route_bw;

public:
	Network(std::string node_path, std::string edge_path);
	void updateEdgeTravelTimes(std::string file_path);
	unsigned int getNumberNodes();
	std::vector<Resultstruct> computeTravelCosts1toX(int start_node_index, const std::vector<int>& targets, double time_range = -1, int max_targets = -1,std::string mode="edge_tt");
	std::vector<Resultstruct> computeTravelCostsXto1(int start_node_index, const std::vector<int>& targets, double time_range = -1, int max_targets = -1,std::string mode="edge_tt");
	int computeTravelCosts1ToXpy(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis, double time_range = -1, int max_targets = -1,string mode="edge_tt");
	int computeTravelCostsXTo1py(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis,double* reached_target_std,double* reached_target_cfv, double time_range, int max_targets,string mode);
	void computeTravelCosts1To1py(int start_node_index, int end_node_index, std::string mode, double* tt, double* dis, double* var, double* cfv);
	int computeRouteSize1to1(int start_node_index, int end_node_index,std::string mode);
	void writeRoute(int* output_array);
};