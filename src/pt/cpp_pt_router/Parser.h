/**
 * @file Parser.h
 * @brief Provides the Parser class for parsing GTFS data files.
 *
 * This header file declares the Parser class, which is responsible for parsing
 * GTFS data files and associating the data to construct a transit network.
 * 
 * @author Maria
 * @date 11/20/2024
 * 
 * Modified by:
 * @author Chenhao Ding
 * @date 2025-02-28
 */

#ifndef PARSE_H
#define PARSE_H

#include <fstream> // for file input
#include <sstream> // for string stream
#include <iostream> // for input and output
#include <chrono> // for timing

#include "Utils.h" // for hash functions
#include "DateTime.h" // for Date and Time

#include "NetworkObjects/GTFSObjects/GTFSObject.h" // for GTFSObject
#include "NetworkObjects/DataStructures.h" // for DataStructures
#include "NetworkObjects/GTFSObjects/Agency.h" // for Agency
#include "NetworkObjects/GTFSObjects/Service.h" // for Calendar and CalendarDate
#include "NetworkObjects/GTFSObjects/Route.h" // for Route
#include "NetworkObjects/GTFSObjects/Stop.h" // for Stop
#include "NetworkObjects/GTFSObjects/Trip.h" // for Trip

/**
 * @class Parser
 * @brief Class for parsing GTFS data files and organizing the information.
 *
 * This class is responsible for parsing various GTFS data files such as agencies, calendars, stops, routes,
 * trips, and stop times. It stores the parsed data in appropriate data structures and allows access to the
 * parsed information.
 */
class Parser {
private:

  std::string inputDirectory; /**< Directory where the input files are located. */

  /**
   * Maps to store parsed data.
   */
  std::unordered_map<std::string, Agency> agencies_; ///< A map from agency IDs to Agency objects.
  std::unordered_map<std::string, Service> services_; ///< A map from service IDs to Service objects.
  std::unordered_map<std::string, Stop> stops_; ///< A map from stop IDs to Stop objects.
  std::unordered_map<std::pair<std::string, std::string>, Route, pair_hash> routes_; ///< A map from (route_id, direction_id) to Route objects.
  std::unordered_map<std::string, Trip> trips_; ///< A map from trip IDs to Trip objects.

  /**
   *@brief Parses the GTFS data files and stores the results in the appropriate maps.
   */
  void parseData();

  /**
   * @brief Parses the agencies file and stores the results in the agencies_ map.
   */
  void parseAgencies();

  /**
   * @brief Parses the calendars file and the calendar_dates file,
   * stores the results in the services_ map.
   */
  void parseServices();

  /**
   * @brief Parses the stop times file and stores the results in the stop_times_ map.
   */
  void parseStopTimes();

  /**
   * @brief Parses the trips file and stores the results in the trips_ map.
   */
  void parseTrips();

  /**
   * @brief Parses the routes file and stores the results in the routes_ map.
   */
  void parseRoutes();

  /**
   * @brief Parses the stops file and stores the results in the stops_ map.
   */
  void parseStops();

  /**
   * @brief Parses the transfers file and stores the results in the transfers_ map.
   */
  void parseTransfers();

  /**
   * @brief Associates data across various GTFS components (routes, trips, stops, etc.).
   *
   * This method processes the data from different GTFS files and associates the relevant information
   * such as matching trips with corresponding stops and stop times.
   */
  void associateData();

public:

  /**
   * @brief Constructor for the Parser class.
   *
   * Initializes the parser with the specified directory containing the GTFS data files.
   *
   * @param[in] directory Path to the directory containing the GTFS files.
   */
  explicit Parser(std::string directory);

  /**
   * @brief Gets the parsed agencies data.
   *
   * @return A map of agency IDs to Agency objects.
   */
  [[nodiscard]] std::unordered_map<std::string, Agency> getAgencies();

  /**
   * @brief Gets the parsed services data.
   *
   * @return A map of service IDs to Service objects.
   */
  [[nodiscard]] std::unordered_map<std::string, Service> getServices();


  /**
   * @brief Gets the parsed stops data.
   *
   * @return A map of stop IDs to Stop objects.
   */
  [[nodiscard]] std::unordered_map<std::string, Stop> getStops();

  /**
   * @brief Gets the parsed routes data.
   *
   * @return A map of (route_id, direction_id) pairs to Route objects.
   */
  [[nodiscard]] std::unordered_map<std::pair<std::string, std::string>, Route, pair_hash> getRoutes();

  /**
   * @brief Gets the parsed trips data.
   *
   * @return A map of trip IDs to Trip objects.
   */
  [[nodiscard]] std::unordered_map<std::string, Trip> getTrips();
};

#endif //PARSE_H
