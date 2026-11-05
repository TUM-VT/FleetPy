#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <sstream>
#include "Node.h"
#include "Edge.h"

using namespace std;

Node::Node(int index, double x_coord, double y_coord, bool is_stop_only) {
	index_ = index;
	x_coord_ = x_coord;
	y_coord_ = y_coord;
	is_stop_only_ = is_stop_only;
	//cout << "created node " << index_ << " " << x_coord_ << " " << y_coord_ << " " << is_stop_only << endl;
}

vector<Edge>& Node::getIncomingEdges() {
	return incoming_edges_;
}

vector<Edge>& Node::getOutgoingEdges() {
	return outgoing_edges_;
}

int Node::getIndex() {
	return index_;
}

string Node::getStr() {
	stringstream s;
	s << "Node " << index_ << " is settled: " << settled_fw_index_ << " " << settled_bw_index_ << " is target: " << is_target_;
	return s.str();
}

void Node::addIncomingEdge(Edge& e) {
	if (e.getEndNode() != index_) {
		cout << "ERROR" << endl;
	}
	else {
		incoming_edges_.push_back(e);
	}
	//cout << "node " << index_ << " now has " << incoming_edges_.size() << " incoming edges and " << outgoing_edges_.size() << " outgoing edges!" << endl;
}

void Node::addOutgoingEdge(Edge& e) {
	if (e.getStartNode() != index_) {
		cout << "EERORR" << endl;
	}
	else {
		outgoing_edges_.push_back(e);
	}
	//cout << "node " << index_ << " now has " << incoming_edges_.size() << " incoming edges and " << outgoing_edges_.size() << " outgoing edges!" << endl;
}

bool Node::mustStop() {
	return is_stop_only_;
}

//===============================================================================================================
void Node::setTarget() {
	is_target_ = true;
}

bool Node::isTarget() {
	return is_target_;
}

void Node::unsetTarget() {
	is_target_ = false;
}

//===============================================================================================================
//FORWARD DIJKSTRA
void Node::setSettledFw(int dijkstra_number) {
	//cout << "im settled: " << index_ << endl;
	settled_fw_index_ = dijkstra_number;
}
bool Node::isSettledFw(int dijkstra_number) {
	//cout << "am i settled? " << index_ << " " << settled_ << endl;
	return dijkstra_number == settled_fw_index_;
}
void Node::setVisitFw(int dijkstra_number) {
	visit_fw_index_ = dijkstra_number;
}
bool Node::isVisitedFw(int dijkstra_number) {
	return visit_fw_index_ == dijkstra_number;
}
void Node::setCostFw(pair<double, double> cost) {
	//cout << "set cost" << cost.first << " " << cost.second << endl;
	cost_fw_ = cost;
}
pair<double, double> Node::getCostFw() {
	return cost_fw_;
}
void Node::setPrev(int new_prev) {
	prev_ = new_prev;
}
int Node::getPrev() {
	return prev_;
}

//===============================================================================================================
//BACKWARD DIJKSTRA
void Node::setSettledBw(int dijkstra_number) {
	//cout << "im settled: " << index_ << endl;
	settled_bw_index_ = dijkstra_number;
}
bool Node::isSettledBw(int dijkstra_number) {
	//cout << "am i settled? " << index_ << " " << settled_ << endl;
	return dijkstra_number == settled_bw_index_;
}
void Node::setVisitBw(int dijkstra_number) {
	visit_bw_index_ = dijkstra_number;
}
bool Node::isVisitedBw(int dijkstra_number) {
	return visit_bw_index_ == dijkstra_number;
}
void Node::setCostBw(pair<double, double> cost) {
	//cout << "set cost" << cost.first << " " << cost.second << endl;
	cost_bw_ = cost;
}
pair<double, double> Node::getCostBw() {
	return cost_bw_;
}
void Node::setNext(int new_next) {
	next_ = new_next;
}
int Node::getNext() {
	return next_;
}