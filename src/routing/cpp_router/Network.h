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
	void updateEdgeTravelTimeLayer(int start_node_index, int end_node_index, double edge_travel_time, int layer);
	void loadTravelTimeFile_(std::string file_path, int layer);
	// Seconds one forecast layer covers. <= 0 means the search is static, which
	// is the state every existing scenario runs in.
	double layer_seconds_ = -1.0;
	// Seconds between the bin the layers describe and the query being answered.
	// The layers describe absolute clock intervals from the bin's start, so a
	// query issued 240 s into a 300 s bin reaches the second layer after only
	// 60 s of driving. Without this the whole route is priced as if it had
	// departed at the bin boundary.
	double query_offset_ = 0.0;

	void setTargets(const std::vector<int>& targets);
	int dijkstraForward(int start_node_index, double time_range = -1, int max_targets = -1);
	void dijkstraStepForward_(std::priority_queue<std::pair<double, int>>& current_pq, Node& current_node, double current_cost);
	int dijkstraBackward(int start_node_index, double time_range = -1, int max_targets = -1);
	void dijkstraStepBackward_(std::priority_queue<std::pair<double, int>>& current_pq, Node& current_node, double current_cost);
	std::pair<double, double> dijkstraBidirectional(int start_node_index, int end_node_index, int* meeting_node_index);
	// Forward-only search to one target. A bidirectional search cannot carry a
	// departure time -- its backward half would have to know the arrival time it
	// is solving for -- so time-dependent queries take this instead.
	bool dijkstraForwardTo_(int start_node_index, int end_node_index);

	std::vector<int> _last_found_route_fw;
	std::vector<int> _last_found_route_bw;

public:
	Network(std::string node_path, std::string edge_path);
	void updateEdgeTravelTimes(std::string file_path);
	// Load one forecast layer. Layer 0 is the ordinary table; layers 1..n-1 are
	// what the same export predicts for later bins.
	void updateEdgeTravelTimesLayer(std::string file_path, int layer);
	// Turn departure-time-dependent search on (layer_seconds > 0) or off.
	void setLayerSeconds(double layer_seconds);
	void setQueryOffset(double query_offset);
	double getQueryOffset();
	// Drop every edge's layers. Called before a bin's horizons are loaded: the
	// base table is refreshed for the links a bin lists and the layers must not
	// outlive it, or a link the exporter drops from one horizon file keeps the
	// previous bin's forecast beside this bin's base value.
	void clearAllLayers();
	double getLayerSeconds();
	unsigned int getNumberNodes();
	std::vector<Resultstruct> computeTravelCosts1toX(int start_node_index, const std::vector<int>& targets, double time_range = -1, int max_targets = -1);
	std::vector<Resultstruct> computeTravelCostsXto1(int start_node_index, const std::vector<int>& targets, double time_range = -1, int max_targets = -1);
	int computeTravelCosts1ToXpy(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis, double time_range = -1, int max_targets = -1);
	int computeTravelCostsXTo1py(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis, double time_range = -1, int max_targets = -1);
	void computeTravelCosts1To1py(int start_node_index, int end_node_index, double* tt, double* dis);
	int computeRouteSize1to1(int start_node_index, int end_node_index);
	void writeRoute(int* output_array);
};