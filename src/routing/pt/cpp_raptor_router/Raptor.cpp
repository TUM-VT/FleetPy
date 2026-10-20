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
#include <tuple>

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
  int max_departure_seconds = 0;
  for (const auto &[trip_id, trip] : trips_) {
    for (const auto &record : trip.getStopTimeRecords()) {
      max_departure_seconds = std::max(max_departure_seconds, record.departure_seconds);
    }
  }
  service_lookback_days_ = max_departure_seconds / MIDNIGHT;
}

void Raptor::setQuery(const Query &query) {
  if (query.max_transfers < 0) {
    throw std::invalid_argument("max_transfers must be non-negative");
  }
  if (query.departure_microseconds < 0 || query.departure_microseconds > 999999) {
    throw std::invalid_argument("departure_microseconds must be between 0 and 999999");
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
  RoundArrival default_info{};
  arrivals_.reserve(stops_.size());
  for (const auto &[stop_id, stop]: stops_) {
    arrivals_.emplace(stop_id, std::vector<RoundArrival>(1, default_info));
  }

  // Initialize the round 0
  k = 0;
  int departure_time = queryDepartureSeconds();
  // Mark all included source stops that are in included_source_ids_
  for (const auto &source : query_.included_sources) {
    if (included_source_ids_.find(source.first) == included_source_ids_.end()) continue;
    int station_stop_transfer_time = source.second;
    int arrival_time = checkedSeconds(static_cast<long long>(departure_time) + station_stop_transfer_time);
    markStop(source.first, arrival_time, std::nullopt, std::nullopt, arrival_time);
  }

  advanceRound(); // k=1

  prepareServiceWindow();
}

int Raptor::queryDepartureSeconds() const {
  return checkedSeconds(static_cast<long long>(Utils::timeToSeconds(query_.departure_time)) +
                        (query_.departure_microseconds > 0 ? 1 : 0));
}

void Raptor::prepareServiceWindow() {
  constexpr int lookahead_days = 1;
  const auto &date = query_.date;
  if (service_window_.has_value() &&
      service_window_->query_date.year == date.year &&
      service_window_->query_date.month == date.month &&
      service_window_->query_date.day == date.day &&
      service_window_->query_date.weekday == date.weekday &&
      service_window_->lookback_days == service_lookback_days_ &&
      service_window_->lookahead_days == lookahead_days) {
    return;
  }

  // Keep an old complete window intact if preparing any new service date fails.
  ServiceWindow prepared{date, service_lookback_days_, lookahead_days, {}};
  Date service_date = Utils::addDays(date, 0);
  int first_offset = 0;
  while (first_offset > -service_lookback_days_) {
    if (service_date.year == 1 && service_date.month == 1 && service_date.day == 1) break;
    service_date = Utils::addDays(service_date, -1);
    --first_offset;
  }

  std::unordered_set<std::string> required_service_ids;
  for (const auto &[trip_id, trip] : trips_) {
    required_service_ids.insert(trip.getField("service_id"));
  }
  for (int offset = first_offset; offset <= lookahead_days; ++offset) {
    ActiveServiceDay active_day{offset, {}};
    for (const auto &service_id : required_service_ids) {
      if (services_.at(service_id).isActive(service_date)) {
        active_day.active_service_ids.insert(service_id);
      }
    }
    prepared.days.push_back(std::move(active_day));
    if (offset < lookahead_days) service_date = Utils::addDays(service_date, 1);
  }
  service_window_ = std::move(prepared);
  occurrence_cache_.clear();
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
  const ArrivalKind kind = parent_trip_id.has_value() ? ArrivalKind::Vehicle : ArrivalKind::NonVehicle;
  if (!improvesArrivalTime(arrival, stop_id, kind)) return;
  const bool improves_boarding = earlier(arrival, arrivals_.at(stop_id).at(k).boarding.arrival_seconds);
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
  Day day = arrival >= MIDNIGHT ? Day::NextDay : Day::CurrentDay;
  if (improves_boarding) {
    setMinArrivalTime(stop_id, {arrival, parent_trip_id, parent_stop_id, day, label_id});
  }
  if (parent_trip_id.has_value()) {
    arrivals_.at(stop_id).at(k).vehicle_label_id = label_id;
  }
  marked_stops.insert(stop_id);
}

void Raptor::setMinArrivalTime(
  const std::string &stop_id, 
  StopInfo stop_info
  ) {
  arrivals_.at(stop_id).at(k).boarding = std::move(stop_info);
}

const JourneyLabel &Raptor::getLabel(std::size_t label_id) const {
  if (label_id >= labels_.size()) {
    throw std::logic_error("Invalid journey label ID");
  }
  return labels_[label_id];
}

std::size_t Raptor::getArrivalLabel(const std::string &stop_id, std::size_t round) const {
  const auto &arrival = arrivals_.at(stop_id).at(round).boarding;
  if (!arrival.label_id.has_value() || !arrival.arrival_seconds.has_value()) {
    throw std::logic_error("Reachable stop has no journey label");
  }
  const auto &label = getLabel(*arrival.label_id);
  if (label.stop_id != stop_id || label.arrival_seconds != *arrival.arrival_seconds || label.round > round) {
    throw std::logic_error("Arrival slot and journey label do not match");
  }
  return *arrival.label_id;
}

std::optional<std::size_t> Raptor::getVehicleArrivalLabel(const std::string &stop_id, std::size_t round) const {
  const auto &arrival = arrivals_.at(stop_id).at(round);
  if (!arrival.vehicle_label_id.has_value()) return std::nullopt;
  const auto &label = getLabel(*arrival.vehicle_label_id);
  if (label.stop_id != stop_id || !label.trip_id.has_value() || label.round > round ||
      label.vehicle_legs == 0 || label.vehicle_legs > round ||
      label.vehicle_legs > static_cast<std::size_t>(max_transfers_) + 1 ||
      !arrival.boarding.arrival_seconds.has_value() ||
      label.arrival_seconds < *arrival.boarding.arrival_seconds) {
    throw std::logic_error("Vehicle arrival slot and journey label do not match");
  }
  return arrival.vehicle_label_id;
}

std::vector<Journey> Raptor::findJourneys() {
  
  std::vector<Journey> journeys;

  initializeAlgorithm();
  // The same source and target platform set does not require a PT round trip.
  if (included_source_ids_ == included_target_ids_) return journeys;

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
      const auto target_label_id = getVehicleArrivalLabel(target_id, k);
      if (target_label_id.has_value() &&
          target_label_id != arrivals_.at(target_id).at(k - 1).vehicle_label_id) {
        Journey journey = reconstructJourney(target_id, *target_label_id, station_stop_transfer_time);
        
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
  // Keep same-station queries from offering a vehicle loop back to their origin.
  if (included_source_ids_ == included_target_ids_) return optimal_journey;

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
      const auto target_label_id = getVehicleArrivalLabel(target_id, k);
      if (target_label_id.has_value() &&
          target_label_id != arrivals_.at(target_id).at(k - 1).vehicle_label_id) {
        int target_station_arrival_secs = checkedSeconds(
          static_cast<long long>(getLabel(*target_label_id).arrival_seconds) + target_station_stop_transfer_time);
        // TODO: add more criteria for optimality if needed, such as number of transfers, walking time, etc.
        if (target_station_arrival_secs < optimal_target_station_arrival_secs) {
          Journey possible_journey = reconstructJourney(target_id, *target_label_id, target_station_stop_transfer_time);
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
    RoundArrival previous_round = stop_arrivals.back();
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
    // Target platforms can also be intermediate boarding stops.
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

    // Keep route-set iteration order, but do not rescan an unchanged prefix.
    // Vehicle-only improvements already propagated walking in their own round.
    if (k > 1 &&
        arrivals_.at(p_stop_id).at(k - 1).boarding.label_id ==
        arrivals_.at(p_stop_id).at(k - 2).boarding.label_id) {
      routes_stops_set.erase(route_stop);
      continue;
    }

    const std::size_t boarding_label_id = getArrivalLabel(p_stop_id, k - 1);
    const auto &occurrences = findBoardableOccurrences(p_stop_id, route_key);
    const int readiness = getLabel(boarding_label_id).arrival_seconds;
    auto occurrence = std::lower_bound(occurrences.begin(), occurrences.end(), readiness,
      [](const auto &candidate, int ready) { return candidate.departure_seconds < ready; });
    for (; occurrence != occurrences.end(); ++occurrence) {
      // Departures increase while target bounds only tighten within this round.
      if (!isValidTrip(p_stop_id, occurrence->departure_seconds)) break;
      traverseTrip(*occurrence, boarding_label_id);
    }
      
    routes_stops_set.erase(route_stop);
  }
}

const std::vector<Raptor::TripOccurrence> &Raptor::findBoardableOccurrences(
  const std::string &p_stop_id, 
  const std::pair<std::string, std::string> &route_key
) {
  if (!service_window_.has_value()) {
    throw std::logic_error("Service-date window has not been prepared");
  }
  const auto cached_route = occurrence_cache_.find(route_key);
  if (cached_route != occurrence_cache_.end()) {
    const auto cached_stop = cached_route->second.find(p_stop_id);
    if (cached_stop != cached_route->second.end()) return cached_stop->second;
  }
  const auto& route = routes_.at(route_key);
  std::vector<TripOccurrence> occurrences;
  for (const auto &trip_id : route.getTripsIds()) {
    const auto &trip = trips_.at(trip_id);
    const auto service_id = trip.getField("service_id");
    const auto &records = trip.getStopTimeRecords();
    for (const auto &day : service_window_->days) {
      if (day.active_service_ids.find(service_id) == day.active_service_ids.end()) continue;
      const long long day_offset = static_cast<long long>(day.service_day_offset) * MIDNIGHT;
      for (std::size_t index = 0; index < records.size(); ++index) {
        if (records[index].stop_id != p_stop_id || index + 1 == records.size()) continue;
        const long long departure = static_cast<long long>(records[index].departure_seconds) + day_offset;
        occurrences.push_back({trip_id, day.service_day_offset, index, departure});
      }
    }
  }
  std::sort(occurrences.begin(), occurrences.end(), [](const auto &a, const auto &b) {
    return std::tie(a.departure_seconds, a.trip_id, a.service_day_offset, a.boarding_stop_index) <
           std::tie(b.departure_seconds, b.trip_id, b.service_day_offset, b.boarding_stop_index);
  });
  return occurrence_cache_[route_key].emplace(p_stop_id, std::move(occurrences)).first->second;
}

bool Raptor::isValidTrip(
  const std::string &p_stop_id,
  long long departure_seconds
) {
  const auto stop_prev_arrival_seconds = arrivals_.at(p_stop_id).at(k - 1).boarding.arrival_seconds;

  if (stop_prev_arrival_seconds.has_value() && departure_seconds < *stop_prev_arrival_seconds) return false;

  // Check if the target stop's arrival time is earlier than the departure time
  for (const auto &target_id : included_target_ids_) {
    const auto target_label_id = arrivals_.at(target_id).at(k).vehicle_label_id;
    if (!target_label_id.has_value() || departure_seconds < getLabel(*target_label_id).arrival_seconds)
      return true;
  }

  return false;
}

void Raptor::traverseTrip(
  const TripOccurrence &occurrence,
  std::size_t boarding_label_id
) {
  const auto &records = trips_.at(occurrence.trip_id).getStopTimeRecords();
  const std::string boarding_stop = getLabel(boarding_label_id).stop_id;
  const int boarding_arrival = getLabel(boarding_label_id).arrival_seconds;
  const long long day_offset = static_cast<long long>(occurrence.service_day_offset) * MIDNIGHT;
  if (occurrence.boarding_stop_index >= records.size() ||
      records[occurrence.boarding_stop_index].stop_id != boarding_stop ||
      static_cast<long long>(records[occurrence.boarding_stop_index].departure_seconds) + day_offset != occurrence.departure_seconds) {
    throw std::logic_error("Trip occurrence does not match its frozen boarding stop");
  }
  const int departure_seconds = checkedSeconds(occurrence.departure_seconds);
  if (departure_seconds < boarding_arrival) {
    throw std::logic_error("Trip occurrence departs before its boarding label");
  }

  // Traverse remaining stops on the trip to update arrival times
  for (std::size_t index = occurrence.boarding_stop_index + 1; index < records.size(); ++index) {
    const auto &record = records[index];
    int arr_secs = checkedSeconds(static_cast<long long>(record.arrival_seconds) + day_offset);

    if (arr_secs < departure_seconds) {
      throw std::logic_error("Invalid stop or time in journey label");
    }
    // Filter before converting the trip ID to an optional string for markStop.
    if (improvesArrivalTime(arr_secs, record.stop_id, ArrivalKind::Vehicle)) {
      markStop(record.stop_id, arr_secs, occurrence.trip_id, boarding_label_id, departure_seconds);
    }
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
  const std::string &dest_id,
  ArrivalKind kind
) {
  const auto &current = arrivals_.at(dest_id).at(k);
  if (kind == ArrivalKind::Vehicle) {
    if (current.vehicle_label_id.has_value() &&
        arrival >= getLabel(*current.vehicle_label_id).arrival_seconds) return false;
  } else if (!earlier(arrival, current.boarding.arrival_seconds)) {
    return false;
  }

  // Only a vehicle arrival is a valid terminal bound under the endpoint policy.
  for (const auto &target_id : included_target_ids_) {
    const auto target_label_id = arrivals_.at(target_id).at(k).vehicle_label_id;
    if (!target_label_id.has_value() || arrival < getLabel(*target_label_id).arrival_seconds)
      return true;
  }
  // If no improvement was found for any target, return false
  return false;
}

void Raptor::handleFootpaths() {
  // Freeze vehicle arrivals before walking updates can replace other seeds.
  std::vector<std::size_t> walking_label_ids;
  walking_label_ids.reserve(marked_stops.size());
  for (const auto &stop_id : marked_stops) {
    const auto label_id = getVehicleArrivalLabel(stop_id, k);
    if (label_id.has_value() && label_id != arrivals_.at(stop_id).at(k - 1).vehicle_label_id) {
      walking_label_ids.push_back(*label_id);
    }
  }

  for (const std::size_t walking_label_id : walking_label_ids) {
    // Copy label fields before markStop can grow the immutable label pool.
    const std::string stop_id = getLabel(walking_label_id).stop_id;
    const int p_arrival = getLabel(walking_label_id).arrival_seconds;
    for (const auto &[dest_id, duration]: stops_[stop_id].getFootpaths()) {
      if (included_source_ids_.find(dest_id) != included_source_ids_.end()) continue;

      // Walking to a target enables boarding there; only vehicle labels can finish.
      int new_arrival = checkedSeconds(static_cast<long long>(p_arrival) + duration);
      if (new_arrival < p_arrival) {
        throw std::logic_error("Invalid stop or time in journey label");
      }
      if (improvesArrivalTime(new_arrival, dest_id, ArrivalKind::NonVehicle)) {
        markStop(dest_id, new_arrival, std::nullopt, walking_label_id, p_arrival);
      }
    }
  }
}

bool Raptor::isFootpath(const StopInfo &stop_info) {
  return stop_info.parent_stop_id.has_value() && !stop_info.parent_trip_id.has_value();
}

Journey Raptor::reconstructJourney(
  const std::string &target_id,
  std::size_t target_label_id,
  const int target_station_stop_transfer_time
) {
  Journey journey{};
  const auto &target_label = getLabel(target_label_id);
  if (target_label.stop_id != target_id || !target_label.trip_id.has_value()) {
    throw std::logic_error("Journey must end at the selected target vehicle label");
  }
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

    Day day = label.arrival_seconds >= MIDNIGHT ? Day::NextDay : Day::CurrentDay;
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
  journey.source_station_departure_secs = queryDepartureSeconds();
  journey.source_station_departure_day = journey.source_station_departure_secs >= MIDNIGHT ? Day::NextDay : Day::CurrentDay;
  journey.source_transfer_time = checkedSeconds(static_cast<long long>(source_label.arrival_seconds) - journey.source_station_departure_secs);
  journey.source_waiting_time = checkedSeconds(static_cast<long long>(journey.steps.front().departure_secs) - source_label.arrival_seconds);
  if (journey.source_transfer_time < 0 || journey.source_waiting_time < 0 || target_station_stop_transfer_time < 0) {
    throw std::logic_error("Journey access, waiting, and egress times must be non-negative");
  }
  journey.target_transfer_time = target_station_stop_transfer_time;
  journey.target_station_arrival_secs = checkedSeconds(static_cast<long long>(target_label.arrival_seconds) + journey.target_transfer_time);
  journey.target_station_arrival_day = journey.target_station_arrival_secs >= MIDNIGHT ? Day::NextDay : Day::CurrentDay;
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
