/**
 * @file DataStructures.h
 * @brief Defines core data structures and utility classes for the RAPTOR project.
 *
 * This header file includes declarations for structs like `Query`, `StopInfo`, `JourneyStep`,
 * and `Journey`, which are used to represent transit queries, stop information, and journey details.
 * It also provides hash functions for specific pair-based keys.
 */

#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include "../DateTime.h"

class Stop;

/**
 * @struct Query
 * @brief Represents a transit query.
 *
 * This structure is used to define a user's query for transit planning,
 * including source and target stops, the desired date, and departure time.
 */
struct Query {
  std::vector<std::pair<std::string, int>> included_sources;   ///< Stops in the source station with its station_stop_transfer_time (in seconds).
  std::vector<std::pair<std::string, int>> included_targets;   ///< Stops in the target station with its station_stop_transfer_time (in seconds).
  Date date;               ///< Date of the journey (year, month, day).
  Time departure_time;     ///< Desired departure time in the source station (in seconds from midnight).
  int max_transfers;  ///< Maximum number of transfers.
};

/**
 * @struct StopInfo
 * @brief Represents information about a transit stop during a journey.
 *
 * This structure holds details about a stop's arrival time, the trip and stop
 * it depends on, and the day of operation. Values are optional to handle
 * cases where a stop is unreachable or is a starting point.
 */
struct StopInfo {
  std::optional<int> arrival_seconds;          ///< Arrival time in seconds, or `std::nullopt` if unreachable.
  std::optional<std::string> parent_trip_id;   ///< ID of the parent trip, or `std::nullopt` for footpaths.
  std::optional<std::string> parent_stop_id;   ///< ID of the parent stop, or `std::nullopt` for first stops.
  std::optional<Day> day;                      ///< Day of arrival, or `std::nullopt` if unreachable.
};

/**
 * @struct JourneyStep
 * @brief Represents a single step in a journey.
 *
 * A journey step can correspond to a trip or a footpath. It contains information
 * about the source and destination stops, departure and arrival times, and duration.
 */
struct JourneyStep {
  std::optional<std::string> trip_id;      ///< ID of the trip, or `std::nullopt` for footpaths.
  std::optional<std::string> agency_name;  ///< Name of the agency, or `std::nullopt` for footpaths.
  Stop *src_stop{};                        ///< Pointer to the source stop.
  Stop *dest_stop{};                       ///< Pointer to the destination stop.

  int departure_secs{};                    ///< Departure time in seconds from midnight of the query day.
  Day day{};                               ///< Day of the journey step.
  int duration{};                          ///< Duration of the step in seconds.
  int arrival_secs{};                      ///< Arrival time in seconds from midnight of the query day.
};

/**
 * @struct Journey
 * @brief Represents an entire journey consisting of multiple steps.
 *
 * The `Journey` structure contains details about all steps in the journey,
 * as well as overall departure and arrival times and durations.
 */
struct Journey {
  std::vector<JourneyStep> steps;          ///< Steps making up the journey.
  int departure_secs;                      ///< Overall departure time in seconds from midnight of the query day at source station.
  Day departure_day;                       ///< Departure day of the journey at source station.

  int arrival_secs;                        ///< Overall arrival time in seconds from midnight of the query day at target station.
  Day arrival_day;                         ///< Arrival day of the journey at target station.

  int duration;                            ///< Total duration of the journey in seconds.
  int source_transfer_time;                ///< Transfer time from source station to source station stop in seconds.
  int waiting_time;                        ///< Waiting time at the source station in seconds.
  int trip_time;                           ///< Trip time from source station stops to target station in seconds.
  
  int num_transfers;                       ///< Number of transfers in the journey.
};

/**
 * @struct StopTimeRecord
 * @brief Represents a stop time record in the GTFS data.
 *
 * This struct stores information about a stop time record, including the stop ID,
 * arrival time, departure time, and stop sequence.
 */
struct StopTimeRecord {
  std::string stop_id; ///< The ID of the stops
  int arrival_seconds; ///< The arrival time in seconds
  int departure_seconds; ///< The departure time in seconds
  int stop_sequence; ///< The sequence number of the stop in the trip

  bool operator<(const StopTimeRecord& other) const {
    return stop_sequence < other.stop_sequence;
  }
};

/**
 * @struct pair_hash
 * @brief Hash function for a pair of strings.
 *
 * Provides a custom hash implementation for pairs of strings,
 * used in unordered containers like `std::unordered_map` and `std::unordered_set`.
 */
struct pair_hash {
  /**
   * @brief Computes the hash value for a pair of strings.
   * @param pair The pair of strings to hash.
   * @return The computed hash value.
   */
  std::size_t operator()(const std::pair<std::string, std::string> &pair) const {
    return std::hash<std::string>()(pair.first) ^ std::hash<std::string>()(pair.second);
  }
};

/**
 * @struct nested_pair_hash
 * @brief Hash function for nested pairs of strings.
 *
 * Provides a custom hash implementation for nested pairs of strings,
 * used in unordered containers for hierarchical keys.
 */
struct nested_pair_hash {
  /**
   * @brief Computes the hash value for a nested pair of strings.
   * @param nested_pair The nested pair to hash.
   * @return The computed hash value.
   */
  std::size_t operator()(const std::pair<std::pair<std::string, std::string>, std::string> &nested_pair) const {
    std::size_t hash1 = pair_hash{}(nested_pair.first);  // Hash of internal part
    std::size_t hash2 = std::hash<std::string>{}(nested_pair.second);
    return hash1 ^ (hash2 << 1);
  }
};

#endif //DATASTRUCTURES_H
