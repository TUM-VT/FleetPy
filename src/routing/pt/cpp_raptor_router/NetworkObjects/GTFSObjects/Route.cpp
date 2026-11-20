/**
 * @file Route.cpp
 * @brief Route class implementation
 *
 * This file contains the implementation of the Route class, which represents
 * a route in the GTFS dataset.
 *
 * @author Maria
 * @date 11/20/2024
 * 
 * @version fleetpy-1.0
 * @author Chenhao Ding
 * @date 12/12/2025
 */

#include "Route.h"
#include <iostream>

void Route::addTripId(const std::string& trip_id) {
  trips_ids.push_back(trip_id);
}

void Route::addStopId(const std::string& stop_id) {
  stops_ids.insert(stop_id);
}

void Route::sortTrips(const std::function<bool(const std::string &, const std::string &)> &comparator) {
  std::sort(trips_ids.begin(), trips_ids.end(), comparator);
}

const std::vector<std::string> &Route::getTripsIds() const {
  return trips_ids;
}

const std::unordered_set<std::string> &Route::getStopsIds() const {
  return stops_ids;
}


