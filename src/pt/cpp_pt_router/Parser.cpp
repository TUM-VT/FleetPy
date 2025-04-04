/**
 * @file Parser.cpp
 * @brief Implementation of the Parser class
 *
 * This file contains the implementation of the Parser class,
 * which is responsible for parsing GTFS data.
 * 
 * @author Maria
 * @date 11/20/2024
 * 
 * Modified by:
 * @author Chenhao Ding
 * @date 2025-02-28
 */

#include "Parser.h"

Parser::Parser(std::string directory) : inputDirectory(std::move(directory)) {
  // // Record the start time
  // auto start_time = std::chrono::high_resolution_clock::now();

  // std::cout << "Parsing GTFS data from " << inputDirectory << "..." << std::endl;
  parseData();

  // std::cout << "Associating data..." << std::endl;
  associateData();

  // // Record the end time
  // auto end_time = std::chrono::high_resolution_clock::now();

  // // Calculate the duration
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // std::cout << std::fixed << std::setprecision(2) 
  //           << "Parsing completed in " << (duration.count() / 1000.0) << " seconds." << std::endl;
}

void Parser::parseData() {
  //The order of parsing is important due
  // std::cout << "Parsing agencies..." << std::endl;
  parseAgencies();

  // std::cout << "Parsing calendars and calendar dates..." << std::endl;
  parseServices();

  // std::cout << "Parsing stop times..." << std::endl;
  parseStopTimes();

  // std::cout << "Parsing trips..." << std::endl;
  parseTrips();

  // std::cout << "Parsing routes..." << std::endl;
  parseRoutes();

  // std::cout << "Parsing stops..." << std::endl;
  parseStops();

  // std::cout << "Parsing transfers..." << std::endl;
  parseTransfers();
}

void Parser::parseAgencies() {
  std::ifstream file(inputDirectory + "/agency_fp.txt");

  if (!file.is_open())
    throw std::runtime_error("Could not open agency_fp.txt");

  std::string line;
  std::getline(file, line);
  Utils::clean(line);
  std::vector<std::string> fields = Utils::split(line, ',');

  while (std::getline(file, line)) {
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    if (tokens.size() != fields.size())
      throw std::runtime_error("Mismatched number of tokens and fields");

    Agency agency;

    for (size_t i = 0; i < fields.size(); ++i)
      agency.setField(fields[i], tokens[i]);

    agencies_[agency.getField("agency_id")] = agency;
  }
}

void Parser::parseServices() {
  // 1. Parse calendar_fp.txt
  std::ifstream file(inputDirectory + "/calendar_fp.txt");
  if (!file.is_open())
    throw std::runtime_error("Could not open calendar_fp.txt");

  std::string line;
  std::getline(file, line);
  Utils::clean(line);
  std::vector<std::string> fields = Utils::split(line, ',');

  while (std::getline(file, line)) {
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    if (tokens.size() != fields.size())
      throw std::runtime_error("Mismatched number of tokens and fields");

    Service service;

    for (size_t i = 0; i < fields.size(); ++i)
      service.setField(fields[i], tokens[i]);

    // Register active_weekdays: 0-6, 0 = monday, 6 = sunday
    for (int i = 0; i < 7; ++i) {
      std::string weekdays_name = weekdays_names[i];
      if (service.getField(weekdays_name) == "1") {
        service.addActiveWeekday(i);
      }
    }

    services_[service.getField("service_id")] = service;
  }

  // 2. Parse calendar_dates.txt
  std::ifstream dates_file(inputDirectory + "/calendar_dates_fp.txt");
  if (!dates_file.is_open())
    // This file is an optional file, so we don't throw an error if it's not found
    return;

  std::getline(dates_file, line);
  Utils::clean(line);
  fields = Utils::split(line, ',');

  while (std::getline(dates_file, line)) {
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    if (tokens.size() != fields.size())
      throw std::runtime_error("Mismatched number of tokens and fields");

    GTFSObject calendar_date;

    for (size_t i = 0; i < fields.size(); ++i)
      calendar_date.setField(fields[i], tokens[i]);

    std::string service_id = calendar_date.getField("service_id");

    if (services_.find(service_id) != services_.end()) {
      int date = std::stoi(calendar_date.getField("date"));
      int type = std::stoi(calendar_date.getField("exception_type"));

      services_[service_id].addExceptionDate(date, type);
    }
  }
}

void Parser::parseStopTimes() {
  std::ifstream file(inputDirectory + "/stop_times_fp.txt");
  if (!file.is_open())
    throw std::runtime_error("Could not open stop_times_fp.txt");

  std::string line;
  std::getline(file, line);
  Utils::clean(line);
  std::vector<std::string> fields = Utils::split(line, ',');

  while (std::getline(file, line)) {
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    if (tokens.size() != fields.size())
      throw std::runtime_error("Mismatched number of tokens and fields");

    GTFSObject stop_time;

    for (size_t i = 0; i < fields.size(); ++i)
      stop_time.setField(fields[i], tokens[i]);

    std::string trip_id = stop_time.getField("trip_id");
    std::string stop_id = stop_time.getField("stop_id");
    int stop_sequence = std::stoi(stop_time.getField("stop_sequence"));
    // Convert arrival_time and departure_time to seconds
    // If the departure_time is on the next day, add 24 hours to it
    int arrival_seconds = Utils::timeToSeconds(stop_time.getField("arrival_time"));
    int departure_seconds = Utils::timeToSeconds(stop_time.getField("departure_time"));
    if (departure_seconds < arrival_seconds) {
      departure_seconds += MIDNIGHT;
    }

    // 1. Register active trip ids and corresponding stop time records
    StopTimeRecord record = {stop_id, arrival_seconds, departure_seconds, stop_sequence};  

    if (trips_.find(trip_id) == trips_.end()) {
      Trip trip;
      trips_[trip_id] = trip;
    }
    trips_[trip_id].addStopTimeRecord(record);

    // 2. Register active stop ids
    if (stops_.find(stop_id) == stops_.end()) {
      Stop stop;
      stops_[stop_id] = stop;
    }
  }
}

void Parser::parseTrips() {
  std::ifstream file(inputDirectory + "/trips_fp.txt");
  if (!file.is_open())
    throw std::runtime_error("Could not open trips_fp.txt");

  std::string line;
  std::getline(file, line);
  Utils::clean(line);
  std::vector<std::string> fields = Utils::split(line, ',');

  while (std::getline(file, line)) {
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    if (tokens.size() != fields.size())
      throw std::runtime_error("Mismatched number of tokens and fields");

    Trip trip;

    for (size_t i = 0; i < fields.size(); ++i)
      trip.setField(fields[i], tokens[i]);

    // Only parse trips that are effective
    std::string trip_id = trip.getField("trip_id");
    if (trips_.find(trip_id) != trips_.end()) {
      std::string route_id = trip.getField("route_id");
      std::string direction_id = trip.getField("direction_id");

      trips_[trip_id].setField("route_id", route_id);
      trips_[trip_id].setField("direction_id", direction_id);
      trips_[trip_id].setField("service_id", trip.getField("service_id"));

      // Register effective routes
      auto route_key = std::make_pair(route_id, direction_id);
      if (routes_.find(route_key) == routes_.end()) {
        Route route;
        routes_[route_key] = route;
      }
      routes_[route_key].addTripId(trip_id);
    }
  }
}

void Parser::parseRoutes() {
  std::ifstream file(inputDirectory + "/routes_fp.txt");

  if (!file.is_open())
    throw std::runtime_error("Could not open routes_fp.txt");

  std::string line;
  std::getline(file, line);
  Utils::clean(line);
  std::vector<std::string> fields = Utils::split(line, ',');

  while (std::getline(file, line)) {
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    if (tokens.size() != fields.size())
      throw std::runtime_error("Mismatched number of tokens and fields");

    Route route;
    for (size_t i = 0; i < fields.size(); ++i)
      route.setField(fields[i], tokens[i]);

    // If there is only one agency, agency_id field is optional
    if (!route.hasField("agency_id"))
      route.setField("agency_id", agencies_.begin()->second.getField("agency_id"));

    // Iterate through all existing (route_id, direction_id) pairs in routes_
    for (auto &[key, r]: routes_) {
      auto [route_id, direction_id] = key;
      if (route_id == route.getField("route_id")) {
        // Merge two routes with the same route_id, keep the original route
        r.merge(route, false);
      }
    }
  }
}

void Parser::parseStops() {
  std::ifstream file(inputDirectory + "/stops_fp.txt");

  if (!file.is_open())
    throw std::runtime_error("Could not open stops_fp.txt");

  std::string line;
  std::getline(file, line);
  Utils::clean(line);
  std::vector<std::string> fields = Utils::split(line, ',');

  while (std::getline(file, line)) {
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    if (tokens.size() != fields.size())
      throw std::runtime_error("Mismatched number of tokens and fields");

    Stop stop;
    for (size_t i = 0; i < fields.size(); ++i)
      stop.setField(fields[i], tokens[i]);

    std::string stop_id = stop.getField("stop_id");

    if (stops_.find(stop_id) != stops_.end()) {
      stops_[stop_id].merge(stop, false);
    }
  }
}

void Parser::parseTransfers() {
  std::ifstream file(inputDirectory + "/transfers_fp.txt");
  if (!file.is_open())
    throw std::runtime_error("Could not open transfers_fp.txt");

  std::string line;
  std::getline(file, line);
  Utils::clean(line);
  std::vector<std::string> fields = Utils::split(line, ',');

  while (std::getline(file, line)) {
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');  

    if (tokens.size() != fields.size())
      throw std::runtime_error("Mismatched number of tokens and fields");

    GTFSObject transfer;
    for (size_t i = 0; i < fields.size(); ++i)
      transfer.setField(fields[i], tokens[i]);

    int min_transfer_time = static_cast<int>(std::stof(transfer.getField("min_transfer_time")));

    std::string from_stop_id = transfer.getField("from_stop_id");
    std::string to_stop_id = transfer.getField("to_stop_id");

    if (stops_.find(from_stop_id) != stops_.end() && 
        stops_.find(to_stop_id) != stops_.end()) {
      stops_[from_stop_id].addFootpath(to_stop_id, min_transfer_time);
    }
  }
}

void Parser::associateData() {
  // 0. Check data effectiveness of all data structures


  // 1. Sort trip time records in stop_times_
  for (auto &[key, trip]: trips_) {
    trip.sortStopTimeRecords();
  }

  // 2. Update routes_ with trips_
  for (auto &[route_key, route]: routes_) {
    // Sort trips in the route by arrival time of the first stop time record
    route.sortTrips(
      [&](const std::string &a, const std::string &b) {
        const Trip &tripA = trips_.at(a);
        const Trip &tripB = trips_.at(b);

        // Find the first stop_time record of the trip
        const StopTimeRecord &stopTimeA = tripA.getStopTimeRecords().front();
        const StopTimeRecord &stopTimeB = tripB.getStopTimeRecords().front();

        int timeA = stopTimeA.arrival_seconds;
        int timeB = stopTimeB.arrival_seconds;

        return timeA < timeB;
      }
    );

    // Find all possible stops ids in the route
    for (const auto &trip_id: route.getTripsIds()) {
      const Trip &trip = trips_[trip_id];
      for (const auto &stop_time_record: trip.getStopTimeRecords()) {
        route.addStopId(stop_time_record.stop_id);
      }
    }
  }

  // routes_[{"R01", "0"}] = {
  //     trips_ids: ["T101", "T102", "T103"],
  //     stops_ids: ["S001", "S002", "S004", "S005"]
  // };

  // 3. Associate routes to stops
  for (auto &[route_key, route]: routes_) {
    for (const auto &stop_id: route.getStopsIds()) {
      stops_[stop_id].addRouteKey(route_key);
    }
  }
}

std::unordered_map<std::string, Agency> Parser::getAgencies() {
  return agencies_;
}

std::unordered_map<std::string, Service> Parser::getServices() {
  return services_;
}

std::unordered_map<std::string, Stop> Parser::getStops() {
  return stops_;
}

std::unordered_map<std::pair<std::string, std::string>, Route, pair_hash> Parser::getRoutes() {
  return routes_;
}

std::unordered_map<std::string, Trip> Parser::getTrips() {
  return trips_;
}