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
 * Modified by:
 * @author Chenhao Ding
 * @date 2025-03-19
 */

#include "Raptor.h"

Raptor::Raptor(const std::unordered_map<std::string, Agency> &agencies,
               const std::unordered_map<std::string, Service> &services,
               const std::unordered_map<std::string, Stop> &stops,
               const std::unordered_map<std::pair<std::string, std::string>, Route, pair_hash> &routes,
               const std::unordered_map<std::string, Trip> &trips)
        : agencies_(agencies), services_(services), stops_(stops), routes_(routes), trips_(trips) {
  k = 1;

  // TODO: Remove this part
  std::cout << "CPP PT Router: "
            << "Raptor initialized with "
            << agencies_.size() << " agencies, "
            << services_.size() << " services, "
            << stops.size() << " stops, "
            << routes.size() << " routes, and "
            << trips.size() << " trips." << std::endl;
}

void Raptor::setQuery(const Query &query) {
  query_ = query;

  max_transfers_ = (query_.max_transfers < 0) ? std::numeric_limits<int>::max() : query_.max_transfers;
  
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
  prev_marked_stops.clear();
  marked_stops.clear();

  // Initialize arrival times for all stops for each k
  StopInfo default_info = {std::nullopt, std::nullopt, std::nullopt, std::nullopt};
  arrivals_.reserve(stops_.size());
  for (const auto &[stop_id, stop]: stops_) {
    arrivals_.emplace(stop_id, std::vector<StopInfo>(max_transfers_+2, default_info));  // +2 for k=0 and k=max_transfers_+1
  }

  // Initialize the round 0
  k = 0;
  int departure_time = Utils::timeToSeconds(query_.departure_time);
  // Mark all included source stops that are in included_source_ids_
  for (const auto &source : query_.included_sources) {
    if (included_source_ids_.find(source.first) == included_source_ids_.end()) continue;
    int station_stop_transfer_time = source.second;
    int arrival_time = departure_time + station_stop_transfer_time;
    markStop(source.first, arrival_time, std::nullopt, std::nullopt);
  }

  k++; // k=1

  // Fill active trips for current and next day
  fillActiveTrips(Day::CurrentDay);
  fillActiveTrips(Day::NextDay);
}

void Raptor::markStop(
  const std::string &stop_id, 
  int arrival,
  const std::optional<std::string> &parent_trip_id,
  const std::optional<std::string> &parent_stop_id
) {
  Day day = arrival > MIDNIGHT ? Day::NextDay : Day::CurrentDay;
  setMinArrivalTime(stop_id, {arrival, parent_trip_id, parent_stop_id, day});
  marked_stops.insert(stop_id);
}

void Raptor::setMinArrivalTime(
  const std::string &stop_id, 
  StopInfo stop_info
  ) {
  arrivals_[stop_id][k] = std::move(stop_info);  // Only keep one stop info per stop per round
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
    if (k > max_transfers_) {
      // std::cout << "Reached maximum number of transfers (" << max_transfers_ << "). Stopping searching." << std::endl;
      break;
    }

    k++;
  }
  return journeys;
}

std::optional<Journey> Raptor::findOptimalJourney() {
  Journey optimal_journey;
  int optimal_arrival_secs = std::numeric_limits<int>::max();

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
    
    for (const auto& [target_id, station_stop_transfer_time] : query_.included_targets) {
      if (marked_stops.find(target_id) != marked_stops.end()) {
        int arrival_secs = arrivals_[target_id][k].arrival_seconds.value();
        if (arrival_secs < optimal_arrival_secs) {
          Journey possible_journey = reconstructJourney(target_id, station_stop_transfer_time);
          if (isValidJourney(possible_journey)) {
            optimal_arrival_secs = arrival_secs;
            optimal_journey = possible_journey;
          }
        }
      }
    }

    // Check if the maximum number of transfers is reached
    if (k > max_transfers_) {
      // std::cout << "Reached maximum number of transfers (" << max_transfers_ << "). Stopping searching." << std::endl;
      break;
    }

    k++;
  }
  return optimal_journey;
}

void Raptor::setUpperBound() {
  for (const auto &[stop_id, stop]: stops_)
    setMinArrivalTime(stop_id, arrivals_[stop_id][k - 1]);
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

      try {
        auto et = findEarliestTrip(p_stop_id, route_key);

        if (et.has_value()) {
          try {
            std::string et_id = et.value().first;
            Day et_day = et.value().second;
            traverseTrip(et_id, et_day, p_stop_id);
          } catch (const std::exception& e) {
            std::cerr << "Exception while processing trip: " << e.what() << std::endl;
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "Exception in findEarliestTrip for stop " << p_stop_id << " and route " << route_key.first << ": " << e.what() << std::endl;
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

  // Find all stop time records of the trip after p_stop_id
  const auto& stop_time_records = et.getStopTimeRecordsAfter(p_stop_id);

  // Traverse remaining stops on the trip to update arrival times
  for (const auto& stop_time_record: stop_time_records) {
    const auto& next_stop_arrival_seconds = stop_time_record.arrival_seconds;

    // Access arrival seconds at next_stop_id for trip et_id, according to the day
    int arr_secs = et_day == Day::CurrentDay ? next_stop_arrival_seconds
                                            : next_stop_arrival_seconds + MIDNIGHT;

    // If arrival time can be improved, update Tk(pj) using et
    if (improvesArrivalTime(arr_secs, stop_time_record.stop_id))
      markStop(stop_time_record.stop_id, arr_secs, et_id, p_stop_id);
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
    
    try {
      // If parent step in the previous round is a footpath, skip it to avoid chaining footpaths
      if (isFootpath(arrivals_[stop_id][k - 1])) continue;
      
      // If parent step in the current round is from a footpath, skip it to avoid chaining footpaths
      if (isFootpath(arrivals_[stop_id][k])) continue;
      
      // If the stop is one of the included target stops, skip it
      if (included_target_ids_.find(stop_id) != included_target_ids_.end()) continue;

      // Get the arrival time at the marked stop in this round
      int p_arrival = arrivals_[stop_id][k].arrival_seconds.value();

      // For each footpath (p, p')
      for (const auto &[dest_id, duration]: stops_[stop_id].getFootpaths()) {       
        int new_arrival = p_arrival + duration;
        if (improvesArrivalTime(new_arrival, dest_id)) {
          markStop(dest_id, new_arrival, std::nullopt, stop_id);
        }
      } // end each footpath (p, p')
    } catch (const std::exception& e) {
      std::cerr << "Exception in handleFootpaths for stop " << stop_id << ": " << e.what() << std::endl;
    }
  } // end each marked stop p
}

bool Raptor::isFootpath(const StopInfo &stop_info) {
  return stop_info.parent_stop_id.has_value() && !stop_info.parent_trip_id.has_value();
}

Journey Raptor::reconstructJourney(
  const std::string &target_id,
  const int station_stop_transfer_time
  ) {
  Journey journey;
  std::string current_stop_id = target_id;

  try {
    while (true) {
      const std::optional<std::string> parent_trip_id_opt = arrivals_[current_stop_id][k].parent_trip_id;
      // TODO: add agency info into journey step
      const std::optional<std::string> parent_agency_name_opt = std::nullopt;

      const std::optional<std::string> parent_stop_id_opt = arrivals_[current_stop_id][k].parent_stop_id;

      if (!parent_stop_id_opt.has_value()) break;  // No parent stop means it is a source stop
      const std::string &parent_stop_id = parent_stop_id_opt.value();

      int departure_seconds, duration, arrival_seconds;
      if (!parent_trip_id_opt.has_value()) { // Walking leg
        arrival_seconds = arrivals_[current_stop_id][k].arrival_seconds.value();
        
        const auto& footpaths = stops_[parent_stop_id].getFootpaths();
        duration = footpaths.at(current_stop_id);
        departure_seconds = arrival_seconds - duration;
      } else { // PT leg
        const std::string &parent_trip_id = parent_trip_id_opt.value();
        
        std::string route_id = trips_[parent_trip_id].getField("route_id");

        bool found_route = false;
        for (const auto &[key, route]: routes_) {
          if (key.first == route_id) {
            found_route = true;
            break;
          }
        }

        if (!found_route) {
          break;
        }

        // Get departure time of the trip at parent_stop_id
        const auto& stop_time_record = trips_[parent_trip_id].getStopTimeRecord(parent_stop_id);
        departure_seconds = stop_time_record->departure_seconds;
        
        arrival_seconds = arrivals_[current_stop_id][k].arrival_seconds.value();
        duration = arrival_seconds - departure_seconds;
      }

      Day day = arrival_seconds > MIDNIGHT ? Day::NextDay : Day::CurrentDay;
      JourneyStep step = {parent_trip_id_opt, parent_agency_name_opt, &stops_[parent_stop_id], &stops_[current_stop_id],
                          departure_seconds, day, duration, arrival_seconds};

      journey.steps.push_back(step);

      // Update to the previous stop boarded
      current_stop_id = parent_stop_id;
    }

    // Reverse the journey to obtain the correct sequence
    std::reverse(journey.steps.begin(), journey.steps.end());

    // If no steps are found, return an empty journey
    if (journey.steps.empty()) {
      return journey;
    }

    // Set journey departure time and day
    journey.departure_secs = Utils::timeToSeconds(query_.departure_time);
    journey.departure_day = Day::CurrentDay;

    // Set journey arrival time and day
    journey.arrival_secs = journey.steps.back().arrival_secs + station_stop_transfer_time;
    journey.arrival_day = journey.steps.back().day;

    // Set journey duration, source transfer time, waiting time, trip time and transfer numbers
    journey.duration = journey.arrival_secs - journey.departure_secs;
    
    // Get station stop transfer time from Query
    Stop src_stop = *journey.steps.front().src_stop;
    for (const auto &source : query_.included_sources) {
      if (source.first == src_stop.getField("stop_id")) {
        journey.source_transfer_time = source.second;
        break;
      }
    }

    journey.waiting_time = journey.steps.front().departure_secs - journey.departure_secs - journey.source_transfer_time;
    journey.trip_time = journey.duration - journey.waiting_time;
    journey.num_transfers = journey.steps.size() - 1;
  } catch (const std::exception& e) {
    std::cerr << "Exception in reconstructJourney: " << e.what() << std::endl;
  }

  return journey;
}

bool Raptor::isValidJourney(Journey journey) const {
  // TODO: add more checks?

  if (journey.steps.empty()) 
    return false;
  
  // Get the starting stop ID of the journey
  std::string start_stop_id = journey.steps.front().src_stop->getField("stop_id");
    
  // Check if the starting stop is one of the included source IDs
  if (included_source_ids_.find(start_stop_id) == included_source_ids_.end()) {
    return false;
  }

  return true;
}