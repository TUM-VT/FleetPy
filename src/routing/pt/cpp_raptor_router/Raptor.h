/**
 * @file Raptor.h
 * @brief Defines the Raptor class for finding Pareto-optimal journeys in a transit network.
 *
 * This header file declares the Raptor class, which implements the
 * Round-Based Public Transit Routing algorithm.
 *
 * The main method involve finding journeys.
 *
 * The class also contains several private methods for initializing the algorithm,
 * traversing routes, and reconstructing journeys.
 *
 * @author Maria
 * @date 10/28/2024
 * 
 * @version fleetpy-1.0
 * @author Chenhao Ding
 * @date 12/12/2025
 */

#ifndef RAPTOR_RAPTOR_H
#define RAPTOR_RAPTOR_H

#include <cstddef>
#include <iostream>
#include <vector>
#include <iomanip>  // for setw
#include "Parser.h"
#include "Utils.h"

/**
 * @class Raptor
 * @brief Implements the RAPTOR algorithm for finding Pareto-optimal journeys.
 *
 * The Raptor class provides methods to set a query, find Pareto-optimal journeys,
 * and print journey steps. It uses various data structures to store information
 * about agencies, calendars, stops, routes, trips, and stop times.
 */
class Raptor {
public:

  /**
   * @brief Default constructor for the Raptor class.
   *
   * Initializes the Raptor object with empty data structures.
   */
  Raptor() = default;

  /**
   * @brief Parameterized constructor for Raptor.
   *
   * Initializes the Raptor object with provided agency, calendar, stop, route, trip, and stop time data.
   *
   * @param[in] agencies_ A map of agency IDs to Agency objects.
   * @param[in] services_ A map of service IDs to Service objects.
   * @param[in] stops A map of stop IDs to Stop objects.
   * @param[in] routes A map of pairs of route IDs and direction IDs to Route objects.
   * @param[in] trips A map of trip IDs to Trip objects.
   */
  Raptor(const std::unordered_map<std::string, Agency> &agencies_,
         const std::unordered_map<std::string, Service> &services_,
         const std::unordered_map<std::string, Stop> &stops,
         const std::unordered_map<std::pair<std::string, std::string>, Route, pair_hash> &routes,
         const std::unordered_map<std::string, Trip> &trips);

  /**
   * @brief Sets the query for the Raptor algorithm.
   *
   * @param[in] query The query containing the parameters for journey search.
   */
  void setQuery(const Query &query);

  /**
   * @brief Finds all Pareto-optimal journeys.
   *
   * This function uses the RAPTOR algorithm to compute all optimal journeys based on the provided query.
   *
   * @return A vector of Journey objects representing the Pareto-optimal journeys.
   */
  std::vector<Journey> findJourneys();

  /**
   * @brief Finds the journey with the earliest arrival time.
   *
   * @return The earliest valid journey, or std::nullopt if no valid journey is found.
   */
  std::optional<Journey> findOptimalJourney();
  
  /**
   * @brief Validates if the given journey is valid.
   *
   * Checks whether the given journey meets the required criteria.
   *
   * @param[in] journey The Journey object to be validated.
   * @return True if the journey is valid, false otherwise.
   */
  bool isValidJourney(Journey journey) const;

private:

  struct TripOccurrence {
    std::string trip_id;
    int service_day_offset;
    std::size_t boarding_stop_index;
    long long departure_seconds;
  };

  struct ActiveServiceDay {
    int service_day_offset;
    std::unordered_set<std::string> active_service_ids;
  };

  struct ServiceWindow {
    Date query_date;
    int lookback_days;
    int lookahead_days;
    std::vector<ActiveServiceDay> days;
  };

  struct RoundArrival {
    StopInfo boarding; ///< Earliest legal arrival available for the next vehicle.
    std::optional<std::size_t> vehicle_label_id; ///< Earliest vehicle arrival eligible for walking or finishing.
  };

  enum class ArrivalKind { NonVehicle, Vehicle };

  std::unordered_map<std::string, Agency> agencies_; ///< Map of agency IDs to Agency objects.
  std::unordered_map<std::string, Service> services_; ///< Map of service IDs to Service objects.
  std::unordered_map<std::string, Stop> stops_; ///< Map of stop IDs to Stop objects.
  std::unordered_map<std::pair<std::string, std::string>, Route, pair_hash> routes_; ///< Map of route keys to Route objects.
  std::unordered_map<std::string, Trip> trips_; ///< Map of trip IDs to Trip objects.
  int service_lookback_days_{}; ///< Derived from the largest static stop departure time.
  std::optional<ServiceWindow> service_window_{}; ///< Fully prepared service dates for one query-date window.
  std::unordered_map<std::pair<std::string, std::string>,
                     std::unordered_map<std::string, std::vector<TripOccurrence>>, pair_hash>
    occurrence_cache_; ///< Complete sorted candidates for each route-stop in the prepared window.

  Query query_; ///< The current query for the RAPTOR algorithm.
  int max_transfers_; ///< Non-negative maximum transfers; N permits N + 1 vehicle legs.
  std::unordered_set<std::string> included_source_ids_; ///< Set of included source stop IDs.
  std::unordered_set<std::string> included_target_ids_; ///< Set of included target stop IDs.
  std::unordered_map<std::string, std::vector<RoundArrival>> arrivals_; ///< Boarding and vehicle arrivals for each stop and round.
  std::vector<JourneyLabel> labels_; ///< Append-only path labels for the current query; references use IDs.
  std::unordered_set<std::string> prev_marked_stops; ///< Set of previously marked stops.
  std::unordered_set<std::string> marked_stops; ///< Set of currently marked stops.
  std::size_t k{}; ///< The current round, equal to the maximum vehicle legs searched.

  /**
   * @brief Initializes the algorithm by setting required parameters.
   */
  void initializeAlgorithm();

  /** @brief Returns the first integer second not earlier than the query time. */
  int queryDepartureSeconds() const;

  // TODO: Remove this function
  /**
   * @brief Sets the minimum arrival time for a given stop.
   *
   * @param[in] stop_id The ID of the stop.
   * @param[in] stop_info The stop info containing the arrival time, parent trip ID, and parent stop ID.
   * 
   * @deprecated This function is deprecated and will be removed in the future.
   */
  void setMinArrivalTime(const std::string &stop_id, StopInfo stop_info);

  const JourneyLabel &getLabel(std::size_t label_id) const;
  std::size_t getArrivalLabel(const std::string &stop_id, std::size_t round) const;
  std::optional<std::size_t> getVehicleArrivalLabel(const std::string &stop_id, std::size_t round) const;

  /** @brief Prepares the complete service-date window before publishing it. */
  void prepareServiceWindow();

  /**
   * @brief Appends current-round labels initialized from the previous round.
   */
  void setUpperBound();

  /**
   * @brief Advances the round index without overflowing its size type.
   */
  void advanceRound();

  /**
   * @brief Accumulates routes serving each stop.
   *
   * @return A set of routes that serve stops.
   */
  std::unordered_set<std::pair<std::pair<std::string, std::string>, std::string>, nested_pair_hash>
  accumulateRoutesServingStops();

  /**
   * @brief Traverses the routes serving each stop.
   *
   * @param[in] routes_stops_set The set of routes and stops to be traversed.
   */
  void traverseRoutes(
    std::unordered_set<std::pair<std::pair<std::string, std::string>, std::string>, nested_pair_hash> routes_stops_set
  );

  /**
   * @brief Returns cached active trip occurrences for a stop and route.
   *
   * @param[in] pi_stop_id The ID of the stop.
   * @param[in] route_key The key consisting of route and direction.
   * @return Complete candidates sorted by normalized departure, without assuming FIFO trips.
   * The reference remains valid until a different service-date window is published.
   * Readiness and target bounds are applied by the caller and are never cached.
   */
  const std::vector<TripOccurrence> &findBoardableOccurrences(
    const std::string &pi_stop_id, 
    const std::pair<std::string, std::string> &route_key
  );

  /**
   * @brief Checks if a trip is valid based on the route and stop time.
   *
   * @param[in] p_stop_id The stop ID.
   * @param[in] departure_seconds The departure time in seconds.
   * @return True if the trip is valid, false otherwise.
   */
  bool isValidTrip(
    const std::string &p_stop_id,
    long long departure_seconds
  );

  /**
   * @brief Checks if the service is active based on the calendar and date.
   *
   * @param[in] service The service object containing service dates.
   * @param[in] date The date to check.
   * @return True if the service is active on the given date, false otherwise.
   */
  static bool isServiceActive(const Service &service, const Date &date);

  /**
   * @brief Traverses a specific trip.
   *
   * @param[in] occurrence The service date and exact boarding index of the trip.
   * @param[in] boarding_label_id The frozen preceding-round path used to board.
   */
  void traverseTrip(
    const TripOccurrence &occurrence,
    std::size_t boarding_label_id
  );

  /**
   * @brief Compares two arrival times to determine which is earlier.
   *
   * @param[in] secondsA The first arrival time in seconds.
   * @param[in] secondsB The second arrival time in seconds.
   * @return True if the first arrival time is earlier, false otherwise.
   */
  static bool earlier(int secondsA, std::optional<int> secondsB);

  /**
   * @brief Checks if a step improves the arrival time for a destination.
   *
   * @param[in] arrival The arrival time.
   * @param[in] dest_id The destination stop ID.
   * @param[in] kind Whether the candidate ends with a vehicle edge.
   * @return True if the relevant arrival improves a useful search state.
   */
  bool improvesArrivalTime(int arrival, const std::string &dest_id, ArrivalKind kind);

  /**
   * @brief Appends an improving immutable label and updates its current-round arrival channels.
   *
   * @param[in] stop_id The ID of the stop.
   * @param[in] arrival The arrival time at the stop.
   * @param[in] parent_trip_id The ID of the parent trip.
   * @param[in] parent_label_id The exact path used to reach the boarding/walking stop.
   * @param[in] departure_seconds Departure time on the query day's absolute time axis.
   */
  void markStop(
    const std::string &stop_id, 
    int arrival,
    const std::optional<std::string> &parent_trip_id, 
    const std::optional<std::size_t> &parent_label_id,
    int departure_seconds
  );

  // TODO: Remove this function
  /**
   * @brief Handles footpath logic during traversal.
   * 
   * @deprecated This function is deprecated and will be removed in the future.
   */
  void handleFootpaths();

  // TODO: Remove this function
  /**
   * @brief Checks if the given stop info represents a footpath.
   *
   * @param[in] stop_info The stop info to be checked.
   * @return True if the stop is a footpath, false otherwise.
   * 
   * @deprecated This function is deprecated and will be removed in the future.
   */
  static bool isFootpath(const StopInfo &stop_info);

  /**
   * @brief Reconstructs the journey from a given stop.
   *
   * @param[in] target_id The ID of the target stop for the journey.
   * @param[in] target_label_id The exact vehicle label selected at the target.
   * @param[in] station_stop_transfer_time The target station stop transfer time for the journey.
   * @return A Journey object representing the reconstructed journey.
   */
  Journey reconstructJourney(
    const std::string &target_id,
    std::size_t target_label_id,
    const int station_stop_transfer_time
  );
};

#endif //RAPTOR_RAPTOR_H
