#pragma once
#include <vector>
#include <string>
#include <queue>
#include <utility>
#include "Node.h"

struct Resultstruct {
	int target;
	double traveltime;
	double traveldistance;
};

class Network {
private:
	std::vector<Node> nodes;
	std::vector<int> current_targets;
	int dijkstra_number = 0;

	void updateEdgeTravelTime(int start_node_index, int end_node_index, double edge_travel_time);

	void setTargets(const std::vector<int>& targets);
	int dijkstraForward(int start_node_index, double time_range = -1, int max_targets = -1);
	void dijkstraStepForward_(std::priority_queue<std::pair<double, int>>& current_pq, Node& current_node, double current_cost);
	int dijkstraBackward(int start_node_index, double time_range = -1, int max_targets = -1);
	void dijkstraStepBackward_(std::priority_queue<std::pair<double, int>>& current_pq, Node& current_node, double current_cost);
	std::pair<double, double> dijkstraBidirectional(int start_node_index, int end_node_index, int* meeting_node_index);

	std::vector<int> _last_found_route_fw;
	std::vector<int> _last_found_route_bw;

public:
	Network(std::string node_path, std::string edge_path);
	void updateEdgeTravelTimes(std::string file_path);
	unsigned int getNumberNodes();
	std::vector<Resultstruct> computeTravelCosts1toX(int start_node_index, const std::vector<int>& targets, double time_range = -1, int max_targets = -1);
	std::vector<Resultstruct> computeTravelCostsXto1(int start_node_index, const std::vector<int>& targets, double time_range = -1, int max_targets = -1);
	int computeTravelCosts1ToXpy(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis, double time_range = -1, int max_targets = -1);
	int computeTravelCostsXTo1py(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis, double time_range = -1, int max_targets = -1);
	void computeTravelCosts1To1py(int start_node_index, int end_node_index, double* tt, double* dis);
	int computeRouteSize1to1(int start_node_index, int end_node_index);
	void writeRoute(int* output_array);
};