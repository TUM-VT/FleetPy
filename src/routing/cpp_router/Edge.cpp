#include <iostream>
#include "Edge.h"

Edge::Edge(int start_node, int end_node, double travel_time, double travel_distance) {
	start_node_ = start_node;
	end_node_ = end_node;
	travel_time_ = travel_time;
	travel_distance_ = travel_distance;
	//std::cout << "new edge: " << start_node_ << " " << end_node_ << " " << travel_time_ << " " << travel_distance << std::endl;
}

int Edge::getStartNode() {
	return start_node_;
}

int Edge::getEndNode() {
	return end_node_;
}

double Edge::getTravelDistance() {
	return travel_distance_;
}

double Edge::getTravelTime() {
	return travel_time_;
}

void Edge::setNewTravelTime(double travel_time) {
	this->travel_time_ = travel_time;
}
double Edge::getTravelTimeAt(double elapsed_s, double layer_seconds) {
	if (tt_layers_.empty() || layer_seconds <= 0.0) {
		return travel_time_;
	}
	int layer = 0;
	if (elapsed_s > 0.0) {
		layer = (int)(elapsed_s / layer_seconds);
	}
	if (layer < 0) {
		layer = 0;
	}
	// Beyond the last exported horizon the forecast has nothing further to say,
	// so the last layer is held rather than extrapolated.
	if (layer >= (int)tt_layers_.size()) {
		layer = (int)tt_layers_.size() - 1;
	}
	double tt = tt_layers_[layer];
	// A layer nobody wrote stays at -1 and must not be routed on.
	return tt >= 0.0 ? tt : travel_time_;
}

void Edge::setLayerTravelTime(int layer, double travel_time) {
	if (layer < 0) {
		return;
	}
	if ((int)tt_layers_.size() <= layer) {
		tt_layers_.resize(layer + 1, -1.0);
	}
	tt_layers_[layer] = travel_time;
}

void Edge::clearLayers() {
	tt_layers_.clear();
}
