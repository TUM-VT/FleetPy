#pragma once

class Edge {
private:
	int start_node_;
	int end_node_;
	double travel_time_;
	double travel_distance_;
public:
	Edge(int start_node, int end_node, double travel_time, double travel_distance);
	int getStartNode();
	int getEndNode();
	double getTravelTime();
	double getTravelDistance();
	void setNewTravelTime(double travel_time);
};