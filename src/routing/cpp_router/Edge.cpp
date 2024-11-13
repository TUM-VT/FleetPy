#include <iostream>
#include "Edge.h"

Edge::Edge(int start_node, int end_node, double travel_time, double travel_distance) {
	start_node_ = start_node;
	end_node_ = end_node;
	travel_time_ = travel_time;
	travel_distance_ = travel_distance;
	edge_var_ = 0;
	edge_cfv_ = travel_time;
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

double Edge::getTravelTimeVar() {
	return edge_var_;
}

double Edge::getCostFunctionValue() {
	return edge_cfv_;
}

void Edge::setNewTravelTimeVar(double edge_var) {
	this->edge_var_ = edge_var;
}

void Edge::setNewTravelTime(double travel_time) {
	this->travel_time_ = travel_time;
}

void Edge::setNewEdgeCfv(double edge_cfv) {
	this->edge_cfv_ = edge_cfv;
}