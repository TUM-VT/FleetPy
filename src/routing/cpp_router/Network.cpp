#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <queue>
#include <utility>
#include "Network.h"
#include "Node.h"
#include "Edge.h"

using namespace std;

template <class Container>
void split(const std::string& str, Container& cont, char delim = ' '){
    stringstream ss(str);
    string token;
    while (std::getline(ss, token, delim)) {
        cont.push_back(token);
    }
}


Network::Network(string node_path, string edge_path) {

    //string node_path = network_path + "\\base\\nodes.csv";
    //string node_path = network_path + "\\GraphFull_0_0.txt";
    cout << node_path << endl;
    ifstream file(node_path);

    if (file.is_open()) {
        cout << "is open!" << endl;

        int node_index_col = -1;
        int x_coord_col = -1;
        int y_coord_col = -1;
        int stop_onyl_col = -1;

        int row_counter = 0;
        while (!file.eof()) {
            string a;
            getline(file, a);
            vector<string> linesplits = {};
            split(a, linesplits, ',');
            int column_counter = 0;

            if (row_counter == 0) {
                for (const auto& entry : linesplits) {
                    if (entry == "node_index") {
                        node_index_col = column_counter;
                    }
                    else if (entry == "pos_x") {
                        x_coord_col = column_counter;
                    }
                    else if (entry == "pos_y") {
                        y_coord_col = column_counter;
                    }
                    else if (entry == "is_stop_only") {
                        stop_onyl_col = column_counter;
                    }
                    column_counter++;
                }
                row_counter++;
                continue;
            }
            int index;
            double x_coord;
            double y_coord;
            bool is_stop_only = false;
            for (const auto &entry : linesplits) {
                //cout << " ,   " << entry;
                if (column_counter == node_index_col) {
                    index = stoi(entry);
                }
                else if (column_counter == x_coord_col) {
                    x_coord = stod(entry);
                }
                else if (column_counter == y_coord_col) {
                    y_coord = stod(entry);
                }
                else if (column_counter == stop_onyl_col) {
                    if (entry == "True") {
                        is_stop_only = true;
                    }
                }
                column_counter++;
            }
            //cout << endl;
            if (column_counter >= 4) {
                nodes.push_back(Node(index, x_coord, y_coord, is_stop_only));
            }
            //cout << " -> " << nodes.size() << endl;
            row_counter++;
        }
        cout << nodes.size() << " nodes loaded!" << endl;
        file.close();
    }
    else {
        cout << "Die Datei konnte nicht geoeffnet werden. : " << node_path << endl;
        return;
    }
    //=============================================================
    //string edge_path = network_path + "\\base\\edges.csv";
    //string node_path = network_path + "\\GraphFull_0_0.txt";
    cout << edge_path << endl;
    ifstream file2(edge_path);

    int from_node_col = -1;
    int to_node_col = -1;
    int tt_col = -1;
    int dis_col = -1;

    if (file2.is_open()) {
        cout << "is open!" << endl;
        int row_counter = 0;
        while (!file2.eof()) {
            string a;
            getline(file2, a);
            vector<string> linesplits = {};
            split(a, linesplits, ',');
            //from_node,to_node,distance,travel_time,source_edge_id
            int column_counter = 0;
            if (row_counter == 0) {
                for (const auto& entry : linesplits) {
                    if (entry == "from_node") {
                        from_node_col = column_counter;
                    }
                    else if (entry == "to_node") {
                        to_node_col = column_counter;
                    }
                    else if (entry == "travel_time") {
                        tt_col = column_counter;
                    }
                    else if (entry == "distance") {
                        dis_col = column_counter;
                    }
                    column_counter++;
                }
                row_counter++;
                continue;
            }
            //cout << a << endl;
            int start_node;
            int end_node;
            double distance;
            double travel_time;
            for (const auto& entry : linesplits) {
                //cout << " ,   " << entry;
                if (column_counter == from_node_col) {
                    start_node = stoi(entry);
                }
                else if (column_counter == to_node_col) {
                    end_node = stoi(entry);
                }
                else if (column_counter == dis_col) {
                    distance = stod(entry);
                }
                else if (column_counter == tt_col) {
                    travel_time = stod(entry);
                }
                column_counter++;
            }
            //cout << endl;
            if (column_counter >= 4) {
                Edge e(start_node, end_node, travel_time, distance);
                nodes[start_node].addOutgoingEdge(e);
                nodes[end_node].addIncomingEdge(e);
            }
            row_counter++;
        }

        file2.close();
    }
    else {
        cout << "Die Datei konnte nicht geoeffnet werden. : " << edge_path << endl;
        return;
    }

}

void Network::updateEdgeTravelTimes(std::string file_path) {
    cout << "c++ update tts " << file_path << endl;
    ifstream file(file_path);
    if (file.is_open()) {
        cout << "is open!" << endl;
        int row_counter = 0;

        int from_node_col = -1;
        int to_node_col = -1;
        int tt_col = -1;

        while (!file.eof()) {
            string a;
            getline(file, a);
            vector<string> linesplits = {};
            split(a, linesplits, ',');
            int column_counter = 0;
            if (row_counter == 0) {
                for (const auto& entry : linesplits) {
                    if (entry == "from_node") {
                        from_node_col = column_counter;
                    }
                    else if (entry == "to_node") {
                        to_node_col = column_counter;
                    }
                    else if (entry == "edge_tt") {
                        tt_col = column_counter;
                    }
                    column_counter++;
                }
                row_counter++;
                continue;
            }
            //from_node,to_node,edge_tt
            int from_node_index;
            int to_node_index;
            double edge_tt;
            for (const auto& entry : linesplits) {
                //cout << " ,   " << entry;
                if (column_counter == from_node_col) {
                    from_node_index = stoi(entry);
                }
                else if (column_counter == to_node_col) {
                    to_node_index = stoi(entry);
                }
                else if (column_counter == tt_col) {
                    edge_tt = stod(entry);
                }
                column_counter++;
            }
            //cout << endl;
            if (column_counter >= 3) {
                updateEdgeTravelTime(from_node_index, to_node_index, edge_tt);
            }
            //cout << row_counter << " " << column_counter << endl;
            //cout << " -> " << nodes.size() << endl;
            row_counter++;
        }
        cout << "c++ travel times updated!" << endl;
        file.close();
    }
    else {
        cout << "Die Datei konnte nicht geoeffnet werden. : " << file_path << endl;
        return;
    }
}

void Network::updateEdgeTravelTime(int start_node_index, int end_node_index, double edge_travel_time) {
    //cout << "update edge" << start_node_index << " " << end_node_index << " " << edge_travel_time << endl;
    bool fw_found = false;
    for (Edge &edge : nodes[start_node_index].getOutgoingEdges()) {
        if (edge.getEndNode() == end_node_index) {
            edge.setNewTravelTime(edge_travel_time);
            fw_found = true;
            break;
        }
    }
    bool bw_found = false;
    for (Edge& edge : nodes[end_node_index].getIncomingEdges()) {
        if (edge.getStartNode() == start_node_index) {
            //cout << "other edge: " << edge.getTravelTime() << endl;
            edge.setNewTravelTime(edge_travel_time);
            bw_found = true;
            break;
        }
    }
    if (!fw_found || !bw_found) {
        cout << "C++ ERROR: edge not found: " << start_node_index << " " << end_node_index << endl;
    }
    //cout << "C++ ERROR: edge not found: " << start_node_index << " " << end_node_index << endl;
}

unsigned int Network::getNumberNodes() {
    return nodes.size();
}

void Network::setTargets(const vector<int>& targets) {
    for (const auto& target : current_targets) {
        nodes[target].unsetTarget();
    }
    for (const auto& target : targets) {
        nodes[target].setTarget();
    }
    current_targets = targets;
}

std::vector<Resultstruct> Network::computeTravelCosts1toX(int start_node_index, const std::vector<int>& targets, double time_range, int max_targets) {
    setTargets(targets);

    int targets_reached = dijkstraForward(start_node_index, time_range = time_range, max_targets = max_targets);

    vector<Resultstruct> return_vector(targets_reached);
    int i = 0;
    for (int target : targets) {
        if (nodes[target].isSettledFw(dijkstra_number)) {
            //cout << "reached " << target << " max: " << targets_reached << endl;
            pair<double, double> costs = nodes[target].getCostFw();
            Resultstruct target_results;
            target_results.target = target;
            target_results.traveltime = nodes[target].getCostFw().first;
            target_results.traveldistance = nodes[target].getCostFw().second;
            return_vector[i] = target_results;
            i++;
        }
    }
    //cout << "1toX: " << targets_reached << " " << i << endl;
    return return_vector;
}

int Network::computeTravelCosts1ToXpy(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis, double time_range, int max_targets) {
    vector<int> vec_targets(number_targets);
    for (int i = 0; i < number_targets;++i) {
        //cout << targets + i << " " << * (targets + i) << endl;
        vec_targets[i] = *(targets + i);
    }

    vector<Resultstruct> return_vector = computeTravelCosts1toX(start_node_index, vec_targets, time_range = time_range, max_targets = max_targets);

    for (unsigned int i = 0; i < return_vector.size(); ++i) {
        *(reached_targets + i) = return_vector[i].target;
        *(reached_target_tts + i) = return_vector[i].traveltime;
        *(reached_target_dis + i) = return_vector[i].traveldistance;
    }
    return return_vector.size();
}

int Network::dijkstraForward(int start_node_index, double time_range, int max_targets) {
    dijkstra_number++;
    priority_queue<pair<double, int>> pq = {};

    int targets_reached = 0;

    Node& start_node = nodes[start_node_index];
    start_node.setPrev(start_node_index);
    start_node.setCostFw(pair<double, double>(0.0, 0.0));
    double current_cost = 0.0;

    pq.push(pair<double, int>(current_cost, start_node_index));
    
    while (!pq.empty()) {
        pair<double, int> current_pair = pq.top();
        pq.pop();
        //cout << endl;
        //cout << "FW " << pq.size() << " " << current_pair.first << " " << current_pair.second << endl;
        Node& current_node = nodes[current_pair.second];
        if (current_node.isSettledFw(dijkstra_number)) {
            continue;
        }
        else if ((time_range >= 0) &(-current_pair.first > time_range)) {
            break;
        }
        current_node.setSettledFw(dijkstra_number);
        if (current_node.isTarget()) {
            targets_reached++;
            if (current_targets.size() == targets_reached) {
                break;
            }
            else if (targets_reached == max_targets) {
                break;
            }
        }
        if (current_node.mustStop() & (current_node.getIndex() != start_node_index)) {
            continue;
        }
        dijkstraStepForward_(pq, current_node, -current_pair.first);
    
    }
    return targets_reached;
}

void Network::dijkstraStepForward_(std::priority_queue<std::pair<double, int>>& current_pq, Node& current_node, double current_cost) {
    double next_cost;
    //cout << "djijstra step " << current_node.getStr() << endl;
    for (Edge& edge : current_node.getOutgoingEdges()) {
        Node& next_node = nodes[edge.getEndNode()];
        //cout << current_node.getStr() << " " << next_node.getStr() << endl;
        if (!next_node.isSettledFw(dijkstra_number)) {
            next_cost = current_cost + edge.getTravelTime();
            if (!next_node.isVisitedFw(dijkstra_number)) {
                next_node.setPrev(current_node.getIndex());
                next_node.setCostFw(pair<double, double>(next_cost, current_node.getCostFw().second + edge.getTravelDistance()));
                next_node.setVisitFw(dijkstra_number);
                current_pq.push(pair<double, int>(-next_cost, next_node.getIndex()));
            }
            else {
                if (next_node.getCostFw().first > next_cost) {
                    next_node.setPrev(current_node.getIndex());
                    next_node.setCostFw(pair<double, double>(next_cost, current_node.getCostFw().second + edge.getTravelDistance()));
                    current_pq.push(pair<double, int>(-next_cost, next_node.getIndex()));
                }
            }
        }
        

    }
}

std::vector<Resultstruct> Network::computeTravelCostsXto1(int start_node_index, const std::vector<int>& targets, double time_range, int max_targets) {
    setTargets(targets);

    int targets_reached = dijkstraBackward(start_node_index, time_range = time_range, max_targets = max_targets);

    vector<Resultstruct> return_vector(targets_reached);
    int i = 0;
    for (int target : targets) {
        if (nodes[target].isSettledBw(dijkstra_number)) {
            pair<double, double> costs = nodes[target].getCostBw();
            Resultstruct target_results;
            target_results.target = target;
            target_results.traveltime = nodes[target].getCostBw().first;
            target_results.traveldistance = nodes[target].getCostBw().second;
            return_vector[i] = target_results;
            i++;
        }
    }
    //cout << "Xto1: " << targets_reached << " " << i << endl;
    return return_vector;
}

int Network::computeTravelCostsXTo1py(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis, double time_range, int max_targets) {
    vector<int> vec_targets(number_targets);
    for (int i = 0; i < number_targets;++i) {
        //cout << targets + i << " " << * (targets + i) << endl;
        vec_targets[i] = *(targets + i);
    }

    vector<Resultstruct> return_vector = computeTravelCostsXto1(start_node_index, vec_targets, time_range = time_range, max_targets = max_targets);

    for (unsigned int i = 0; i < return_vector.size(); ++i) {
        *(reached_targets + i) = return_vector[i].target;
        *(reached_target_tts + i) = return_vector[i].traveltime;
        *(reached_target_dis + i) = return_vector[i].traveldistance;
    }
    return return_vector.size();
}

int Network::dijkstraBackward(int start_node_index, double time_range, int max_targets) {
    dijkstra_number++;
    priority_queue<pair<double, int>> pq = {};

    int targets_reached = 0;

    Node& start_node = nodes[start_node_index];
    start_node.setNext(start_node_index);
    start_node.setCostBw(pair<double, double>(0.0, 0.0));
    double current_cost = 0.0;

    pq.push(pair<double, int>(current_cost, start_node_index));

    while (!pq.empty()) {
        pair<double, int> current_pair = pq.top();
        pq.pop();
        //cout << endl;
        //cout << "BW " << pq.size() << " " << current_pair.first << " " << current_pair.second << endl;
        Node& current_node = nodes[current_pair.second];
        if (current_node.isSettledBw(dijkstra_number)) {
            continue;
        }
        else if ((time_range >= 0) & (-current_pair.first > time_range)) {
            break;
        }
        current_node.setSettledBw(dijkstra_number);
        //current_node.isSettled();
        if (current_node.isTarget()) {
            targets_reached++;
            //cout << "reached target" << current_node.getStr() << endl;
            if (current_targets.size() == targets_reached) {
                break;
            }
            else if (targets_reached == max_targets) {
                break;
            }
        }
        if (current_node.mustStop() & (current_node.getIndex() != start_node_index)) {
            continue;
        }
        //cout << "dijkstra  " << current_node.getStr() << endl;
        dijkstraStepBackward_(pq, current_node, -current_pair.first);

    }
    return targets_reached;
}

void Network::dijkstraStepBackward_(std::priority_queue<std::pair<double, int>>& current_pq, Node& current_node, double current_cost) {
    double next_cost;
    //cout << "djijstra step " << current_node.getStr() << endl;
    for (Edge& edge : current_node.getIncomingEdges()) {
        Node& next_node = nodes[edge.getStartNode()];
        //cout << current_node.getStr() << " " << next_node.getStr() << endl;
        if (!next_node.isSettledBw(dijkstra_number)) {
            next_cost = current_cost + edge.getTravelTime();
            if (!next_node.isVisitedBw(dijkstra_number)) {
                next_node.setNext(current_node.getIndex());
                next_node.setCostBw(pair<double, double>(next_cost, current_node.getCostBw().second + edge.getTravelDistance()));
                next_node.setVisitBw(dijkstra_number);
                current_pq.push(pair<double, int>(-next_cost, next_node.getIndex()));
            }
            else {
                if (next_node.getCostBw().first > next_cost) {
                    next_node.setNext(current_node.getIndex());
                    next_node.setCostBw(pair<double, double>(next_cost, current_node.getCostBw().second + edge.getTravelDistance()));
                    current_pq.push(pair<double, int>(-next_cost, next_node.getIndex()));
                }
            }
        }


    }
}

void Network::computeTravelCosts1To1py(int start_node_index, int end_node_index, double* tt, double* dis) {
    int meeting_node = 1;
    pair<double, double> result = dijkstraBidirectional(start_node_index, end_node_index, &meeting_node);
    *tt = result.first;
    *dis = result.second;
}

pair<double, double> Network::dijkstraBidirectional(int start_node_index, int end_node_index, int* meeting_node_index) {
    dijkstra_number++;
    *meeting_node_index = -1;

    priority_queue<pair<double, int>> fw_pq = {};
    Node& start_node = nodes[start_node_index];
    start_node.setPrev(start_node_index);
    start_node.setCostFw(pair<double, double>(0.0, 0.0));
    start_node.setVisitFw(dijkstra_number);
    double current_fw_cost = 0.00000001;
    fw_pq.push(pair<double, int>(current_fw_cost, start_node_index));

    priority_queue<pair<double, int>> bw_pq = {};
    Node& end_node = nodes[end_node_index];
    end_node.setNext(end_node_index);
    end_node.setCostBw(pair<double, double>(0.0, 0.0));
    end_node.setVisitBw(dijkstra_number);
    double current_bw_cost = 0.0000001;
    bw_pq.push(pair<double, int>(current_bw_cost, end_node_index));

    int common_node = -1;

    while (true) {
        //cout << "fw cost " << current_fw_cost << " bw cost " << current_bw_cost << endl;
        if ((current_fw_cost <= current_bw_cost || current_bw_cost < -0.9) && current_fw_cost >= -0.9) {
            if (fw_pq.empty()) {
                current_fw_cost = -1.0;
                continue;
            }
            pair<double, int> current_pair = fw_pq.top();
            fw_pq.pop();
            Node& current_node = nodes[current_pair.second];
            if (current_node.isSettledFw(dijkstra_number)) {
                continue;
            }
            current_node.setSettledFw(dijkstra_number);
            if (current_node.mustStop() & (current_node.getIndex() != start_node_index)) {
                //cout << "skip a " << current_node.getIndex() << endl;
                continue;
            }
            //cout << "dijkstra  " << current_node.getStr() << endl;
            current_fw_cost = -current_pair.first;
            if (current_node.isSettledBw(dijkstra_number)) {
                common_node = current_node.getIndex();
                break;
            }
            //cout << "forward " << current_pair.first << " " << current_pair.second << endl;
            dijkstraStepForward_(fw_pq, current_node, -current_pair.first);
        }
        else if (current_bw_cost >= -0.9) {
            if (bw_pq.empty()) {
                current_bw_cost = -1.0;
                continue;
            }
            pair<double, int> current_pair = bw_pq.top();
            bw_pq.pop();
            Node& current_node = nodes[current_pair.second];
            if (current_node.isSettledBw(dijkstra_number)) {
                continue;
            }
            current_node.setSettledBw(dijkstra_number);
            if (current_node.mustStop() & (current_node.getIndex() != end_node_index)) {
                //cout << "skip b " << current_node.getIndex() << endl;
                continue;
            }
            //cout << "dijkstra  " << current_node.getStr() << endl;
            current_bw_cost = -current_pair.first;
            if (current_node.isSettledFw(dijkstra_number)) {
                common_node = current_node.getIndex();
                break;
            }
            //cout << "backward " << current_pair.first << " " << current_pair.second << endl;
            dijkstraStepBackward_(bw_pq, current_node, -current_pair.first);
        }
        else {
            //cout << "TODO NO ROUTE FOUND!" << endl;
            return pair<double, double>(-1.0, -1.0);
        }
    }
    //cout << "common node found: " << common_node << endl;
    pair<double, double> best_value;
    best_value.first = nodes[common_node].getCostFw().first + nodes[common_node].getCostBw().first;
    best_value.second = nodes[common_node].getCostFw().second + nodes[common_node].getCostBw().second;
    *meeting_node_index = common_node;
    while (!fw_pq.empty()) {
        pair<double, int> current_pair = fw_pq.top();
        fw_pq.pop();
        Node& current_node = nodes[current_pair.second];
        if (current_node.mustStop() || current_node.isSettledFw(dijkstra_number)) {
            if ((current_node.getIndex() != start_node_index) & (current_node.getIndex() != end_node_index)) {
                continue;
            }
        }
        if (current_node.isVisitedBw(dijkstra_number)) {
            double tt = current_node.getCostFw().first + current_node.getCostBw().first;
            if (tt < best_value.first) {
                //cout << "rest a " << current_node.getIndex() << " " << tt << endl;
                best_value.first = tt;
                best_value.second = current_node.getCostFw().second + current_node.getCostBw().second;
                *meeting_node_index = current_node.getIndex();
            }
        }
    }
    while (!bw_pq.empty()) {
        pair<double, int> current_pair = bw_pq.top();
        bw_pq.pop();
        Node& current_node = nodes[current_pair.second];
        if (current_node.mustStop() || current_node.isSettledBw(dijkstra_number)) {
            if ((current_node.getIndex() != start_node_index) & (current_node.getIndex() != end_node_index)) {
                continue;
            }
        }
        if (current_node.isVisitedFw(dijkstra_number)) {
            double tt = current_node.getCostFw().first + current_node.getCostBw().first;
            if (tt < best_value.first) {
                //cout << "rest b " << current_node.getIndex() << " " << tt << endl;
                best_value.first = tt;
                best_value.second = current_node.getCostFw().second + current_node.getCostBw().second;
                *meeting_node_index = current_node.getIndex();
            }
        }
    }
    return best_value;
}

int Network::computeRouteSize1to1(int start_node_index, int end_node_index) {
    int meeting_node_index = -1;
    pair<double, double> result = dijkstraBidirectional(start_node_index, end_node_index, &meeting_node_index);
    if (meeting_node_index >= 0) {
        _last_found_route_fw = {};
        _last_found_route_fw.push_back(meeting_node_index);
        if ((nodes[meeting_node_index].getPrev() >= 0) & (nodes[meeting_node_index].getPrev() != meeting_node_index)) {
            //Node& c_node = nodes[common_node.getPrev()];
            int next_index = nodes[meeting_node_index].getPrev();
            while ((nodes[next_index].getPrev() >= 0) & (nodes[next_index].getPrev() != next_index)) {
                _last_found_route_fw.push_back(next_index);
                next_index = nodes[next_index].getPrev();
            }
            _last_found_route_fw.push_back(next_index);
        }

        _last_found_route_bw = {};
        if ((nodes[meeting_node_index].getNext() >= 0) & (nodes[meeting_node_index].getNext() != meeting_node_index)) {
            int next_index = nodes[meeting_node_index].getNext();
            while ((nodes[next_index].getNext() >= 0) & (nodes[next_index].getNext() != next_index)) {
                _last_found_route_bw.push_back(next_index);
                next_index = nodes[next_index].getNext();
            }
            _last_found_route_bw.push_back(next_index);
        }

        return _last_found_route_fw.size() + _last_found_route_bw.size();
    }
    else {
        return -1;
    }
}

void Network::writeRoute(int* output_array) {
    int c = 0;
    for (int i = _last_found_route_fw.size() - 1; i >= 0; i--) {
        output_array[c] = _last_found_route_fw[i];
        c++;
    }
    for (auto n : _last_found_route_bw) {
        output_array[c] = n;
        c++;
    }
}