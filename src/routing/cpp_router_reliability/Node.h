#pragma once
#include <tuple>
#include <vector>
#include <utility>
#include <string>
#include "Edge.h"

class Node {
private:
	int index_;
	std::vector<Edge> incoming_edges_;
	std::vector<Edge> outgoing_edges_;
	double x_coord_;
	double y_coord_;
	bool is_stop_only_ = false;
	bool is_target_ = false;

	int settled_fw_index_ = 0;
	int visit_fw_index_ = 0;
	std::vector<double> cost_fw_ = {0.0, 0.0, 0.0, 0.0};
	int prev_ = -1;

	int settled_bw_index_ = 0;
	int visit_bw_index_ = 0;
	std::vector<double> cost_bw_ = {0.0, 0.0, 0.0, 0.0};
	int next_ = -1;

public:
	Node(int index, double x_coord, double y_coord, bool is_stop_only);
	std::vector<Edge>& getIncomingEdges();
	std::vector<Edge>& getOutgoingEdges();
	void addIncomingEdge(Edge& edge);
	void addOutgoingEdge(Edge& edge);
	int getIndex();
	std::string getStr();
	bool mustStop();

	void setTarget();
	bool isTarget();
	void unsetTarget();

	void setSettledFw(int dijkstra_number);
	bool isSettledFw(int dijkstra_number);
	std::vector<double> getCostFw();
	void setPrev(int new_prev);
	int getPrev();
	void setVisitFw(int dijkstra_number);
	bool isVisitedFw(int dijkstra_number);
    void setCostFw(std::vector<double> cost);

    void setSettledBw(int dijkstra_number);
	bool isSettledBw(int dijkstra_number);
	void setCostBw(std::vector<double> cost);
	std::vector<double> getCostBw();
	void setNext(int new_next);
	int getNext();
	void setVisitBw(int dijkstra_number);
	bool isVisitedBw(int dijkstra_number);

};