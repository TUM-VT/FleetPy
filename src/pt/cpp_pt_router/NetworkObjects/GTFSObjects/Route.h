/**
 * @file Route.h
 * @brief Defines the Route class, representing a route in the GTFS dataset.
 *
 * This header file declares the Route class, which inherits from GTFSObject.
 * The class serves as a representation of the GTFS "route.txt" file, storing
 * information about a route.
 *
 * @author Maria
 * @date 11/20/2024
 */

#ifndef RAPTOR_ROUTE_H
#define RAPTOR_ROUTE_H

#include "GTFSObject.h"

/**
 * @class Route
 * @brief Represents a route in the GTFS data.
 *
 * This class inherits from GTFSObject and manages trip and stop information
 * for a specific route. It provides methods for adding trip and stop IDs,
 * retrieving sorted data, and defining custom sorting mechanisms.
 *
 */
class Route : public GTFSObject {
public:
  /**
   * @brief Adds a trip ID to the route.
   * @param trip_id The ID of the trip to add.
   */
  void addTripId(const std::string &trip_id);

  /**
   * @brief Adds a stop ID to the route.
   * @param stop_id The ID of the stop to add.
   */
  void addStopId(const std::string &stop_id);

  /**
   * @brief Sorts the trips using a custom comparator.
   * @param comparator A function defining the sorting criteria.
   */
  void sortTrips(const std::function<bool(const std::string &, const std::string &)> &comparator);

  /**
   * @brief Retrieves the list of trip IDs.
   * @return A constant reference to the vector of trip IDs.
   */
  const std::vector<std::string> &getTripsIds() const;

  /**
   * @brief Retrieves the list of stop IDs.
   * @return A constant reference to the set of stop IDs.
   */
  const std::unordered_set<std::string> &getStopsIds() const;

private:
  std::vector<std::string> trips_ids; ///< Vector of trip IDs, sorted by earliest arrival time
  std::unordered_set<std::string> stops_ids; ///< Set of stop IDs
};

#endif //RAPTOR_ROUTE_H
