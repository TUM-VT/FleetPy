/**
 * @file Raptor.cpp
 * @brief Raptor class implementation
 *
 * This file contains the implementation of the Raptor class, which represents
 * the Round-Based Public Transit Routing algorithm, for journey planning.
 *
 * @author Maria
 * @date 10/28/2024
 * 
 * @version fleetpy-1.0
 * @author Chenhao Ding
 * @date 12/12/2025
 */

#include "Raptor.h"
#include <limits>
#include <stdexcept>

namespace {
int checkedSeconds(long long seconds) {
  if (seconds < std::numeric_limits<int>::min() || seconds > std::numeric_limits<int>::max()) {
    throw std::overflow_error("Journey time exceeds the supported integer range");
  }
  return static_cast<int>(seconds);
}
}

Raptor::Raptor(const std::unordered_map<std::string, Agency> &agencies,
               const std::unordered_map<std::string, Service> &services,
               const std::unordered_map<std::string, Stop> &stops,
               const std::unordered_map<std::pair<std::string, std::string>, Route, pair_hash> &routes,
               const std::unordered_map<std::string, Trip> &trips)
        : agencies_(agencies), services_(services), stops_(stops), routes_(routes), trips_(trips) {
  k = 1;
}

void Raptor::setQuery(const Query &query) {
  if (query.max_transfers < 0) {
    throw std::invalid_argument("max_transfers must be non-negative");
  }
  query_ = query;

  max_transfers_ = query_.max_transfers;
  
  included_source_ids_.clear();
  included_target_ids_.clear();
  
  for (const auto &source : query_.included_sources) {
    // Only include stops that are in stops_
    if (stops_.find(source.first) != stops_.end()) {
      included_source_ids_.insert(source.first);
    } 
  }

  for (const auto &target : query_.included_targets) {
    // Only include stops that are in stops_
    if (stops_.find(target.first) != stops_.end()) {
      included_target_ids_.insert(target.first);
    }
  }

  if (included_source_ids_.empty() || included_target_ids_.empty()) {
    throw std::runtime_error("No included source or target stops found");
  }
}

void Raptor::initializeAlgorithm() {
  // Initialize data structures
  arrivals_.clear();
  labels_.clear();
  prev_marked_stops.clear();
  marked_stops.clear();

  // Allocate round 0 only; later rounds are appended when searched.
  StopInfo default_info{};
  arrivals_.reserve(stops_.size());
  for (const auto &[stop_id, stop]: stops_) {
    arrivals_.emplace(stop_id, std::vector<StopInfo>(1, default_info));
  }

  // Initialize the round 0
  k = 0;
  int departure_time = Utils::timeToSeconds(query_.departure_time);
  // Mark all included source stops that are in included_source_ids_
  for (const auto &source : query_.included_sources) {
    if (included_source_ids_.find(source.first) == included_source_ids_.end()) continue;
    int station_stop_transfer_time = source.second;
    int arrival_time = checkedSeconds(static_cast<long long>(departure_time) + station_stop_transfer_time);
    if (earlier(arrival_time, arrivals_.at(source.first)[0].arrival_seconds)) {
      markStop(source.first, arrival_time, std::nullopt, std::nullopt, arrival_time);
    }
  }

  advanceRound(); // k=1

  // Fill active trips for current and next day
  fillActiveTrips(Day::CurrentDay);
  fillActiveTrips(Day::NextDay);
}

void Raptor::markStop(
  const std::string &stop_id, 
  int arrival,
  const std::optional<std::string> &parent_trip_id,
  const std::optional<std::size_t> &parent_label_id,
  int departure_seconds
) {
  std::optional<std::string> parent_stop_id;
  std::size_t vehicle_legs = 0;
  const std::size_t max_vehicle_legs = static_cast<std::size_t>(max_transfers_) + 1;
  if (stops_.find(stop_id) == stops_.end() || arrival < departure_seconds) {
    throw std::logic_error("Invalid stop or time in journey label");
  }
  if (parent_label_id.has_value()) {
    const auto &parent = getLabel(*parent_label_id);
    if (parent.round > k || parent.vehicle_legs > parent.round ||
        parent.vehicle_legs > max_vehicle_legs || departure_seconds < parent.arrival_seconds) {
      throw std::logic_error("Journey label has an inconsistent predecessor");
    }
    parent_stop_id = parent.stop_id;
    vehicle_legs = parent.vehicle_legs;
    if (parent_trip_id.has_value()) {
      if (k == 0 || parent.round >= k) {
        throw std::logic_error("Vehicle predecessor must come from a previous round");
      }
      ++vehicle_legs;
    } else if (departure_seconds != parent.arrival_seconds ||
               (parent.parent_label_id.has_value() && !parent.trip_id.has_value())) {
      throw std::logic_error("Walking predecessor does not match the frozen arrival");
    }
  } else if (k != 0 || parent_trip_id.has_value() || departure_seconds != arrival ||
             included_source_ids_.find(stop_id) == included_source_ids_.end()) {
    throw std::logic_error("Journey root must be an included source in round zero");
  }
  if (vehicle_legs > k || vehicle_legs > max_vehicle_legs) {
    throw std::logic_error("Journey label exceeds the vehicle-leg limit");
  }

  // Copy all predecessor data before push_back can reallocate the label pool.
  const std::size_t label_id = labels_.size();
  labels_.push_back({stop_id, arrival, parent_label_id, parent_trip_id,
                     departure_seconds, k, vehicle_legs});
  Day day = arrival > MIDNIGHT ? Day::NextDay : Day::CurrentDay;
  setMinArrivalTime(stop_id, {arrival, parent_trip_id, parent_stop_id, day, label_id});
  marked_stops.insert(stop_id);
}

void Raptor::setMinArrivalTime(
  const std::string &stop_id, 
  StopInfo stop_info
  ) {
  arrivals_.at(stop_id).at(k) = std::move(stop_info);
}

const JourneyLabel &Raptor::getLabel(std::size_t label_id) const {
  if (label_id >= labels_.size()) {
    throw std::logic_error("Invalid journey label ID");
  }
  return labels_[label_id];
}

std::size_t Raptor::getArrivalLabel(const std::string &stop_id, std::size_t round) const {
  const auto &arrival = arrivals_.at(stop_id).at(round);
  if (!arrival.label_id.has_value() || !arrival.arrival_seconds.has_value()) {
    throw std::logic_error("Reachable stop has no journey label");
  }
  const auto &label = getLabel(*arrival.label_id);
  if (label.stop_id != stop_id || label.arrival_seconds != *arrival.arrival_seconds || label.round > round) {
    throw std::logic_error("Arrival slot and journey label do not match");
  }
  return *arrival.label_id;
}

void Raptor::fillActiveTrips(Day day) {
  Date target_date = (day == Day::CurrentDay) ? query_.date : Utils::addOneDay(query_.date);
  int target_date_int = Utils::dateToInt(target_date);

  // Check if the active_trips_by_day_ map already has the trips for the target date
  if (current_date_int_ == target_date_int) {
    return;
  }

  // Iterates over all trips
  for (auto &[trip_id, trip]: trips_) {
    // Get service calendar for the trip
    const Service &service = services_.at(trip.getField("service_id"));

    if (service.isActive(target_date)) {
      trip.setActive(day, true);
    } else {
      trip.setActive(day, false);
    }
  }
}

std::vector<Journey> Raptor::findJourneys() {
  
  std::vector<Journey> journeys;

  initializeAlgorithm();

  while (true) {
    // std::cout << std::endl << "Round " << k << std::endl;

    // Use the minimum arrival time from the previous round as the base for the current round
    setUpperBound();

    prev_marked_stops = marked_stops;
    marked_stops.clear();

    // Accumulate routes serving marked stops from previous round --> routes_stops_set: ((route_id, direction_id), stop_id)
    std::unordered_set<std::pair<std::pair<std::string, std::string>, std::string>, nested_pair_hash> routes_stops_set = accumulateRoutesServingStops();
    // std::cout << "Accumulated " << routes_stops_set.size() << " routes serving stops." << std::endl;

    // Traverse each route --> find earliest trip on each route --> traverse trip --> update arrival times
    traverseRoutes(routes_stops_set);
    // std::cout << "Traversed routes. " << marked_stops.size() << " stop(s) improved." << std::endl;

    // Look for footpaths --> find possible walking connections of marked stops --> update arrival times
    handleFootpaths();
    // std::cout << "Handled footpaths. " << marked_stops.size() << " stop(s) improved." << std::endl;

    // Stopping criterion: if no stops are marked, then stop
    if (marked_stops.empty()) break;   
    
    // Check if any included target stop has been improved
    for (const auto& [target_id, station_stop_transfer_time] : query_.included_targets) {
      if (marked_stops.find(target_id) != marked_stops.end()) {
        Journey journey = reconstructJourney(target_id, station_stop_transfer_time);
        
        if (isValidJourney(journey)) {
          journeys.push_back(journey);
        }
      }
    }

    // Check if the maximum number of transfers is reached
    if (k > static_cast<std::size_t>(max_transfers_)) {
      // std::cout << "Reached maximum number of transfers (" << max_transfers_ << "). Stopping searching." << std::endl;
      break;
    }

    advanceRound();
  }
  return journeys;
}

std::optional<Journey> Raptor::findOptimalJourney() {
  std::optional<Journey> optimal_journey = std::nullopt;
  int optimal_target_station_arrival_secs = std::numeric_limits<int>::max();

  initializeAlgorithm();

  while (true) {
    // std::cout << std::endl << "Round " << k << std::endl;

    // Use the minimum arrival time from the previous round as the base for the current round
    setUpperBound();

    prev_marked_stops = marked_stops;
    marked_stops.clear();

    // Accumulate routes serving marked stops from previous round --> routes_stops_set: ((route_id, direction_id), stop_id)
    std::unordered_set<std::pair<std::pair<std::string, std::string>, std::string>, nested_pair_hash> routes_stops_set = accumulateRoutesServingStops();
    // std::cout << "Accumulated " << routes_stops_set.size() << " routes serving stops." << std::endl;

    // Traverse each route --> find earliest trip on each route --> traverse trip --> update arrival times
    traverseRoutes(routes_stops_set);
    // std::cout << "Traversed routes. " << marked_stops.size() << " stop(s) improved." << std::endl;

    // Look for footpaths --> find possible walking connections of marked stops --> update arrival times
    handleFootpaths();
    // std::cout << "Handled footpaths. " << marked_stops.size() << " stop(s) improved." << std::endl;

    // Stopping criterion: if no stops are marked, then stop
    if (marked_stops.empty()) break;   
    
    for (const auto& [target_id, target_station_stop_transfer_time] : query_.included_targets) {
      if (marked_stops.find(target_id) != marked_stops.end()) {
        int target_station_arrival_secs = arrivals_[target_id][k].arrival_seconds.value() + target_station_stop_transfer_time;
        // TODO: add more criteria for optimality if needed, such as number of transfers, walking time, etc.
        if (target_station_arrival_secs < optimal_target_station_arrival_secs) {
          Journey possible_journey = reconstructJourney(target_id, target_station_stop_transfer_time);
          if (isValidJourney(possible_journey)) {
            optimal_target_station_arrival_secs = target_station_arrival_secs;
            optimal_journey = possible_journey;
          }
        }
      }
    }

    // Check if the maximum number of transfers is reached
    if (k > static_cast<std::size_t>(max_transfers_)) {
      // std::cout << "Reached maximum number of transfers (" << max_transfers_ << "). Stopping searching." << std::endl;
      break;
    }

    advanceRound();
  }
  return optimal_journey;
}

void Raptor::setUpperBound() {
  for (auto &[stop_id, stop_arrivals]: arrivals_) {
    if (stop_arrivals.size() != k) {
      throw std::logic_error("Arrival labels do not match the RAPTOR round");
    }
    StopInfo previous_round = stop_arrivals.back();
    stop_arrivals.push_back(std::move(previous_round));
  }
}

void Raptor::advanceRound() {
  if (k == std::numeric_limits<std::size_t>::max()) {
    throw std::length_error("RAPTOR round index exceeds the supported range");
  }
  ++k;
}

std::unordered_set<std::pair<std::pair<std::string, std::string>, std::string>, nested_pair_hash> 
Raptor::accumulateRoutesServingStops() {
  std::unordered_set<std::pair<std::pair<std::string, std::string>, std::string>, nested_pair_hash> routes_stops_set;

  // For each previously marked stop p
  for (const auto &marked_stop_id: prev_marked_stops) {
    // If the stop is one of the included target stops, skip it
    if (included_target_ids_.find(marked_stop_id) != included_target_ids_.end()) continue;
              
    // For each route r serving p: Route key: (route_id, direction_id)
    for (const auto &route_key: stops_[marked_stop_id].getRouteKeys()) {
      routes_stops_set.insert({route_key, marked_stop_id});
    }
  }
  return routes_stops_set;
}

void Raptor::traverseRoutes(
  std::unordered_set<std::pair<std::pair<std::string, std::string>, std::string>, nested_pair_hash> routes_stops_set
) {
  while (!routes_stops_set.empty()) {
    auto route_stop = routes_stops_set.begin();
    const auto &[route_key, p_stop_id] = *route_stop;

    auto et = findEarliestTrip(p_stop_id, route_key);
    if (et.has_value()) {
      traverseTrip(et->first, et->second, p_stop_id);
    }
      
    routes_stops_set.erase(route_stop);
  }
}

std::optional<std::pair<std::string, Day>> Raptor::findEarliestTrip(
  const std::string &p_stop_id, 
  const std::pair<std::string, std::string> &route_key
) {
  // Get all trips of the route
  const auto& route = routes_.at(route_key);
  const auto& trips = route.getTripsIds();

  // Get earliest trip on current day
  for (const auto& trip_id: trips) {
    if (trips_[trip_id].isActive(Day::CurrentDay)) {
      // Get departure time of the trip at p_stop_id
      const auto& stop_time_record = trips_[trip_id].getStopTimeRecord(p_stop_id);
      if (!stop_time_record) continue;
      const int& departure_seconds = stop_time_record->departure_seconds;
      if (isValidTrip(p_stop_id, departure_seconds))
        return std::make_pair(trip_id, Day::CurrentDay);
    }
  }

  // Get earliest trip on next day
  for (const auto& trip_id: trips) {
    if (trips_[trip_id].isActive(Day::NextDay)) {
      // Get departure time of the trip at p_stop_id
      const auto& stop_time_record = trips_[trip_id].getStopTimeRecord(p_stop_id);
      if (!stop_time_record) continue;
      const int& departure_seconds = stop_time_record->departure_seconds + MIDNIGHT;
      if (isValidTrip(p_stop_id, departure_seconds))
        return std::make_pair(trip_id, Day::NextDay);
    }
  }

  return std::nullopt;
}

bool Raptor::isValidTrip(
  const std::string &p_stop_id,
  const int &departure_seconds
) {
  std::optional<int> stop_prev_arrival_seconds = arrivals_[p_stop_id][k - 1].arrival_seconds;

  if (earlier(departure_seconds, stop_prev_arrival_seconds)) return false;

  // Check if the target stop's arrival time is earlier than the departure time
  for (const auto &target_id : included_target_ids_) {
    std::optional<int> target_arrival_seconds = arrivals_[target_id][k].arrival_seconds;
    if (earlier(departure_seconds, target_arrival_seconds))
      return true;
  }

  return false;
}

void Raptor::traverseTrip(
  const std::string &et_id, 
  const Day &et_day, 
  const std::string &p_stop_id
) {
  Trip et = trips_[et_id];
  const std::size_t boarding_label_id = getArrivalLabel(p_stop_id, k - 1);
  const auto *boarding_record = et.getStopTimeRecord(p_stop_id);
  if (boarding_record == nullptr) {
    throw std::logic_error("Boarding stop is missing from the chosen trip");
  }
  const int day_offset = et_day == Day::CurrentDay ? 0 : MIDNIGHT;
  const int departure_seconds = checkedSeconds(static_cast<long long>(boarding_record->departure_seconds) + day_offset);

  // Find all stop time records of the trip after p_stop_id
  const auto& stop_time_records = et.getStopTimeRecordsAfter(p_stop_id);

  // Traverse remaining stops on the trip to update arrival times
  for (const auto& stop_time_record: stop_time_records) {
    const auto& next_stop_arrival_seconds = stop_time_record.arrival_seconds;

    // Access arrival seconds at next_stop_id for trip et_id, according to the day
    int arr_secs = checkedSeconds(static_cast<long long>(next_stop_arrival_seconds) + day_offset);

    // If arrival time can be improved, update Tk(pj) using et
    if (improvesArrivalTime(arr_secs, stop_time_record.stop_id))
      markStop(stop_time_record.stop_id, arr_secs, et_id, boarding_label_id, departure_seconds);
  }
}

bool Raptor::earlier(
  int secondsA, 
  std::optional<int> secondsB
) {
  if (!secondsB.has_value()) return true; // if still not set, then any value is better
  return secondsA < secondsB.value();
}

bool Raptor::improvesArrivalTime(
  int arrival, 
  const std::string &dest_id
) {
  // Check if the arrival time at the destination stop can be improved
  if (!earlier(arrival, arrivals_[dest_id][k].arrival_seconds))
    return false;
  
  // Check if the arrival time at any included target stop can be improved
  for (const auto &target_id : included_target_ids_) {
    if (earlier(arrival, arrivals_[target_id][k].arrival_seconds))
      return true;
  }
  // If no improvement was found for any target, return false
  return false;
}

void Raptor::handleFootpaths() {
  // Copy the marked stops
  auto current_marked_stops = marked_stops;
  
  // For each stop p that is marked by methods traverseTrip()
  while (!current_marked_stops.empty()) {
    auto it = current_marked_stops.begin();
    std::string stop_id = *it;
    current_marked_stops.erase(it);
    
    // Preserve the existing walking propagation rules.
    if (isFootpath(arrivals_[stop_id][k - 1])) continue;
    if (isFootpath(arrivals_[stop_id][k])) continue;
    if (included_target_ids_.find(stop_id) != included_target_ids_.end()) continue;

    // Freeze both the ID and its time before any destination update.
    const std::size_t walking_label_id = getArrivalLabel(stop_id, k);
    const int p_arrival = getLabel(walking_label_id).arrival_seconds;
    for (const auto &[dest_id, duration]: stops_[stop_id].getFootpaths()) {
      if (included_target_ids_.find(dest_id) != included_target_ids_.end()) continue;
      if (included_source_ids_.find(dest_id) != included_source_ids_.end()) continue;

      int new_arrival = checkedSeconds(static_cast<long long>(p_arrival) + duration);
      if (improvesArrivalTime(new_arrival, dest_id)) {
        markStop(dest_id, new_arrival, std::nullopt, walking_label_id, p_arrival);
      }
    }
  } // end each marked stop p
}

bool Raptor::isFootpath(const StopInfo &stop_info) {
  return stop_info.parent_stop_id.has_value() && !stop_info.parent_trip_id.has_value();
}

Journey Raptor::reconstructJourney(
  const std::string &target_id,
  const int target_station_stop_transfer_time
) {
  Journey journey{};
  const std::size_t target_label_id = getArrivalLabel(target_id, k);
  const auto &target_label = getLabel(target_label_id);
  const std::size_t max_vehicle_legs = static_cast<std::size_t>(max_transfers_) + 1;
  std::size_t current_label_id = target_label_id;
  std::size_t vehicle_legs = 0;

  // IDs decrease along the chain, even when a valid journey revisits a stop.
  while (true) {
    const auto &label = getLabel(current_label_id);
    if (label.round > k || label.vehicle_legs > label.round ||
        label.vehicle_legs > max_vehicle_legs) {
      throw std::logic_error("Journey label has an invalid round or vehicle count");
    }
    if (!label.parent_label_id.has_value()) {
      if (label.round != 0 || label.vehicle_legs != 0 || label.trip_id.has_value() ||
          included_source_ids_.find(label.stop_id) == included_source_ids_.end()) {
        throw std::logic_error("Journey chain does not end at an included source");
      }
      break;
    }
    if (*label.parent_label_id >= current_label_id) {
      throw std::logic_error("Journey predecessor IDs must decrease");
    }
    const auto &parent = getLabel(*label.parent_label_id);
    if (parent.round > label.round || parent.vehicle_legs > parent.round ||
        parent.vehicle_legs > max_vehicle_legs ||
        label.departure_seconds < parent.arrival_seconds ||
        label.arrival_seconds < label.departure_seconds) {
      throw std::logic_error("Journey predecessor has inconsistent times or rounds");
    }
    const bool vehicle_leg = label.trip_id.has_value();
    if ((vehicle_leg && parent.round >= label.round) ||
        label.vehicle_legs != parent.vehicle_legs + (vehicle_leg ? 1 : 0)) {
      throw std::logic_error("Journey edge has an inconsistent vehicle count");
    }
    if (!vehicle_leg && (label.departure_seconds != parent.arrival_seconds ||
                        (parent.parent_label_id.has_value() && !parent.trip_id.has_value()))) {
      throw std::logic_error("Journey walking edge has an inconsistent predecessor");
    }
    if (vehicle_leg) ++vehicle_legs;

    Day day = label.arrival_seconds > MIDNIGHT ? Day::NextDay : Day::CurrentDay;
    journey.steps.push_back({label.trip_id, std::nullopt,
                             &stops_.at(parent.stop_id), &stops_.at(label.stop_id),
                             label.departure_seconds, day,
                             checkedSeconds(static_cast<long long>(label.arrival_seconds) - label.departure_seconds),
                             label.arrival_seconds});
    current_label_id = *label.parent_label_id;
  }

  if (vehicle_legs != target_label.vehicle_legs || vehicle_legs > k ||
      vehicle_legs > max_vehicle_legs) {
    throw std::logic_error("Reconstructed journey exceeds its vehicle-leg budget");
  }
  std::reverse(journey.steps.begin(), journey.steps.end());
  if (journey.steps.empty()) return journey;

  const auto &source_label = getLabel(current_label_id);
  journey.source_station_departure_secs = Utils::timeToSeconds(query_.departure_time);
  journey.source_station_departure_day = Day::CurrentDay;
  journey.source_transfer_time = checkedSeconds(static_cast<long long>(source_label.arrival_seconds) - journey.source_station_departure_secs);
  journey.source_waiting_time = checkedSeconds(static_cast<long long>(journey.steps.front().departure_secs) - source_label.arrival_seconds);
  if (journey.source_transfer_time < 0 || journey.source_waiting_time < 0 || target_station_stop_transfer_time < 0) {
    throw std::logic_error("Journey access, waiting, and egress times must be non-negative");
  }
  journey.target_transfer_time = target_station_stop_transfer_time;
  journey.target_station_arrival_secs = checkedSeconds(static_cast<long long>(target_label.arrival_seconds) + journey.target_transfer_time);
  journey.target_station_arrival_day = journey.target_station_arrival_secs > MIDNIGHT ? Day::NextDay : Day::CurrentDay;
  journey.duration = checkedSeconds(static_cast<long long>(journey.target_station_arrival_secs) - journey.source_station_departure_secs);
  journey.trip_time = checkedSeconds(static_cast<long long>(target_label.arrival_seconds) - journey.steps.front().departure_secs);
  journey.num_transfers = vehicle_legs == 0 ? 0 : static_cast<int>(vehicle_legs - 1);
  return journey;
}

bool Raptor::isValidJourney(Journey journey) const {
  if (journey.steps.empty())
    return false;

  // Get the starting stop ID of the journey
  std::string start_stop_id = journey.steps.front().src_stop->getField("stop_id");

  // Check if the starting stop is one of the included source IDs
  if (included_source_ids_.find(start_stop_id) == included_source_ids_.end()) {
    return false;
  }

  // DRT Constraint: First leg must be a vehicle trip (not walking)
  if (!journey.steps.front().trip_id.has_value()) {
    return false;
  }

  // DRT Constraint: Last leg must be a vehicle trip (not walking)
  if (!journey.steps.back().trip_id.has_value()) {
    return false;
  }

  return true;
}
