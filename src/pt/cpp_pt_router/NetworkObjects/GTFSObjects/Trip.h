/**
 * @file Trip.h
 * @brief Defines the Trip class, representing a trip in the GTFS dataset.
 *
 * This header file declares the Trip class, which inherits from GTFSObject.
 * The class serves as a representation of the GTFS "trip.txt" file, storing
 * information about a trip.
 *
 * @author Maria
 * @date 11/20/2024
 */

#ifndef RAPTOR_TRIP_H
#define RAPTOR_TRIP_H

#include "GTFSObject.h"
#include "../DataStructures.h"

/**
 * @class Trip
 * @brief Represents a trip in the GTFS data.
 *
 * This class inherits from GTFSObject and manages stop time information
 * for a specific trip. It provides methods for adding stop time records,
 * retrieving sorted data, and defining custom sorting mechanisms.
 *
 */
class Trip : public GTFSObject {
public:
  /**
   * @brief Adds a stop time record to the trip.
   * @param record The stop time record to add.
   */
  void addStopTimeRecord(const StopTimeRecord &record);

  /**
   * @brief Retrieves the stop time record for a specific stop.
   * @param stop_id The ID of the stop to retrieve the stop time record for.
   * @return A constant pointer to the stop time record for the specified stop. If the stop is not found, returns nullptr.
   */
  const StopTimeRecord* getStopTimeRecord(const std::string &stop_id) const;

  /**
   * @brief Retrieves the stop time records.
   * @return A constant reference to the stop time records.
   */
  const std::vector<StopTimeRecord> &getStopTimeRecords() const;

  /**
   * @brief Sorts the stop time records.
   */
  void sortStopTimeRecords();

  /**
   * @brief Retrieves the stop time records after a specific stop.
   * @param stop_id The ID of the stop to retrieve the stop time records after.
   * @return A vector of stop time records after the specified stop.
   */
  std::vector<StopTimeRecord> getStopTimeRecordsAfter(const std::string &stop_id) const;

  /**
   * @brief Sets the active status for a specific day.
   * @param day
   * @param is_active
   */
  void setActive(Day day, bool is_active);

  /**
   * @brief Checks if a specific day is active.
   * @param day
   * @return True if the day is active, false otherwise.
   */
  bool isActive(Day day) const;

  /**
   * @brief Checks if the stop time records are sorted.
   * @return True if the stop time records are sorted, false otherwise.
   */
  bool isSorted() const;

private:
  std::unordered_map<Day, bool> active_days_; ///< Map of active days for the trip
  std::vector<StopTimeRecord> stop_time_records_; ///< Vector of stop-time records, sorted by stopTime's sequence
  bool is_sorted_; ///< Whether the stop time records are sorted
};

#endif //RAPTOR_TRIP_H
