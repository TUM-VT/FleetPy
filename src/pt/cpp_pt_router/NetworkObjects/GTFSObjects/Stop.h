/**
 * @file Stop.h
 * @brief Defines the Stop class, representing a stop in the GTFS dataset.
 *
 * This header file declares the Stop class, which inherits from GTFSObject.
 * The class serves as a representation of the GTFS "stop.txt" file, storing
 * information about a stop.
 *
 * @author Maria
 * @date 11/20/2024
 * 
 * Modified by:
 * @author Chenhao Ding
 * @date 2025-02-28
 */

#ifndef RAPTOR_STOP_H
#define RAPTOR_STOP_H

#include "GTFSObject.h"
#include "../DataStructures.h"

/**
 * @class Stop
 * @brief Represents a stop in the GTFS data.
 *
 * This class inherits from GTFSObject and manages stop time and route information
 * for a specific stop. It provides methods for adding stop time and route IDs,
 * retrieving sorted data, and defining custom sorting mechanisms.
 *
 */
class Stop : public GTFSObject {
public:
  /**
   * @brief Adds a route key (route_id, direction_id) to the stop.
   * @param route_key A pair representing the route key.
   */
  void addRouteKey(const std::pair<std::string, std::string> &route_key);

  /**
   * @brief Adds a footpath/transfer to another stop.
   * @param target_id The ID of the other stop.
   * @param duration The duration of the footpath/transfer in seconds.
   */
  void addFootpath(const std::string &target_id, int duration);

  /**
  * @brief Retrieves the set of route keys.
  * @return A constant reference to the unordered set of route keys.
    */
  const std::unordered_set<std::pair<std::string, std::string>, pair_hash> &getRouteKeys() const;

  /**
   * @brief Retrieves the transfer time to another stop.
   * @param target_id The ID of the other stop.
   * @return The transfer time in seconds.
   */
  int getFootpathTime(const std::string &target_id) const;

  /**
   * @brief Checks if there is a transfer to another stop.
   * @param target_id The ID of the other stop.
   * @return True if there is a transfer, false otherwise.
   */
  bool hasFootpath(const std::string &target_id) const;

  /**
   * @brief Retrieves the map of footpaths.
   * @return A constant reference to the map of footpaths.
   */
  const std::unordered_map<std::string, int> &getFootpaths() const;

private:
  std::unordered_set<std::pair<std::string, std::string>, pair_hash> routes_keys; ///< Set of route keys (route_id, direction_id)
  std::unordered_map<std::string, int> footpaths; ///< Map of footpaths to other stops
};

#endif //RAPTOR_STOP_H
