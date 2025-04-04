/**
 * @file Stop.cpp
 * @brief Stop class implementation
 *
 * This file contains the implementation of the Stop class, which represents
 * a stop in the GTFS dataset.
 *
 * @author Maria
 * @date 11/20/2024
 * 
 * Modified by:
 * @author Chenhao Ding
 * @date 2025-02-28
 */

#include "Stop.h"

void Stop::addRouteKey(const std::pair<std::string, std::string> &route_key) {
  routes_keys.insert(route_key);
}

void Stop::addFootpath(const std::string &target_id, int duration) {
  footpaths[target_id] = duration;
}

const std::unordered_set<std::pair<std::string, std::string>, pair_hash> &Stop::getRouteKeys() const {
  return routes_keys;
}

int Stop::getFootpathTime(const std::string &target_id) const {
  if (!hasFootpath(target_id))
    throw std::runtime_error("No footpath to " + target_id); 
  return footpaths.at(target_id);
}

bool Stop::hasFootpath(const std::string &target_id) const {
  return footpaths.find(target_id) != footpaths.end();
}

const std::unordered_map<std::string, int> &Stop::getFootpaths() const {
  return footpaths;
}

