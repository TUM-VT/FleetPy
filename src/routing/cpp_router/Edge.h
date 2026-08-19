#pragma once
#include <vector>

// An edge carries one travel time per forecast layer. Layer 0 is the value the
// static router has always used, so an edge with no further layers behaves
// exactly as before. Layer k is what the table predicts for the k-th bin after
// the routing query, which is what makes a departure-time-dependent search
// possible: the search knows the elapsed travel time when it reaches an edge,
// and that elapsed time selects the layer.
class Edge {
private:
	int start_node_;
	int end_node_;
	double travel_time_;
	double travel_distance_;
	std::vector<double> tt_layers_;
public:
	Edge(int start_node, int end_node, double travel_time, double travel_distance);
	int getStartNode();
	int getEndNode();
	double getTravelTime();
	// Travel time for a vehicle entering this edge `elapsed_s` after the query.
	// Falls back to layer 0 whenever no layer covers that offset, so a partially
	// populated table degrades to the static behaviour rather than to zero.
	double getTravelTimeAt(double elapsed_s, double layer_seconds);
	double getTravelDistance();
	void setNewTravelTime(double travel_time);
	void setLayerTravelTime(int layer, double travel_time);
	void clearLayers();
};