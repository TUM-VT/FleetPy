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
 * @author Maria
 * @date 11/20/2024
 * 
 * @version fleetpy-1.0
 * @author Chenhao Ding
 * @date 12/12/2025
 */

#include "Parser.h"
#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {
std::vector<std::string> readRequiredHeader(
  std::istream &file, const std::vector<std::string> &required, const std::string &filename
) {
  std::string line;
  if (!std::getline(file, line)) throw std::invalid_argument(filename + ": missing header");
  Utils::clean(line);
  if (line.empty()) throw std::invalid_argument(filename + ": empty header");
  auto fields = Utils::split(line, ',');
  for (const auto &field : required) {
    if (std::count(fields.begin(), fields.end(), field) != 1) {
      throw std::invalid_argument(filename + ": expected exactly one '" + field + "' column");
    }
  }
  return fields;
}

template <typename ParseValue>
int parseField(const GTFSObject &row, const std::string &field, const std::string &context, ParseValue parse) {
  try {
    return parse(row.getField(field));
  } catch (const std::invalid_argument &error) {
    throw std::invalid_argument(context + ": " + field + ": " + error.what());
  } catch (const std::overflow_error &error) {
    throw std::overflow_error(context + ": " + field + ": " + error.what());
  }
}
}

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

  std::vector<std::string> required = {"service_id", "start_date", "end_date"};
  for (int i = 0; i < 7; ++i) required.emplace_back(weekdays_names[i]);
  auto fields = readRequiredHeader(file, required, "calendar_fp.txt");
  std::string line;
  std::size_t line_number = 1;

  while (std::getline(file, line)) {
    ++line_number;
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    try {
      if (tokens.size() != fields.size()) throw std::invalid_argument("Mismatched number of tokens and fields");
      Service service;
      for (size_t i = 0; i < fields.size(); ++i) service.setField(fields[i], tokens[i]);
      const std::string service_id = service.getField("service_id");
      if (service_id.find_first_not_of(" \t\r\n") == std::string::npos) {
        throw std::invalid_argument("Empty service_id");
      }
      const int start = parseField(service, "start_date", "service '" + service_id + "'", Utils::parseServiceDate);
      const int end = parseField(service, "end_date", "service '" + service_id + "'", Utils::parseServiceDate);
      if (start > end) throw std::invalid_argument("start_date is after end_date for service '" + service_id + "'");

      // Validate weekly rows even when all weekdays are inactive.
      for (int i = 0; i < 7; ++i) {
        const std::string weekday = weekdays_names[i];
        const std::string active = service.getField(weekday);
        if (active != "0" && active != "1") throw std::invalid_argument("Expected 0 or 1 for '" + weekday + "'");
        if (active == "1") service.addActiveWeekday(i);
      }
      if (!services_.emplace(service_id, std::move(service)).second) {
        throw std::invalid_argument("Duplicate service_id '" + service_id + "'");
      }
    } catch (const std::invalid_argument &error) {
      throw std::invalid_argument("calendar_fp.txt line " + std::to_string(line_number) + ": " + error.what());
    }
  }

  // 2. Parse the optional calendar_dates_fp.txt, including new service IDs.
  const std::string dates_path = inputDirectory + "/calendar_dates_fp.txt";
  std::ifstream dates_file(dates_path);
  if (!dates_file.is_open()) {
    if (std::filesystem::exists(dates_path)) throw std::runtime_error("Could not open calendar_dates_fp.txt");
    return;
  }

  fields = readRequiredHeader(dates_file, {"service_id", "date", "exception_type"}, "calendar_dates_fp.txt");
  line_number = 1;

  while (std::getline(dates_file, line)) {
    ++line_number;
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    try {
      if (tokens.size() != fields.size()) throw std::invalid_argument("Mismatched number of tokens and fields");
      GTFSObject calendar_date;
      for (size_t i = 0; i < fields.size(); ++i) calendar_date.setField(fields[i], tokens[i]);
      const std::string service_id = calendar_date.getField("service_id");
      if (service_id.find_first_not_of(" \t\r\n") == std::string::npos) throw std::invalid_argument("Empty service_id");
      const int date = parseField(calendar_date, "date", "service '" + service_id + "'", Utils::parseServiceDate);
      const std::string type = calendar_date.getField("exception_type");
      if (type != "1" && type != "2") throw std::invalid_argument("Expected exception_type 1 or 2");

      auto [service, inserted] = services_.try_emplace(service_id);
      if (inserted) service->second.setField("service_id", service_id);
      service->second.addExceptionDate(date, type == "1" ? 1 : 2);
    } catch (const std::invalid_argument &error) {
      throw std::invalid_argument("calendar_dates_fp.txt line " + std::to_string(line_number) + ": " + error.what());
    }
  }
}

void Parser::parseStopTimes() {
  std::ifstream file(inputDirectory + "/stop_times_fp.txt");
  if (!file.is_open())
    throw std::runtime_error("Could not open stop_times_fp.txt");

  auto fields = readRequiredHeader(file, {"trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"},
                                   "stop_times_fp.txt");
  std::string line;
  std::size_t line_number = 1;

  while (std::getline(file, line)) {
    ++line_number;
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');

    const std::string context = "stop_times_fp.txt line " + std::to_string(line_number);
    if (tokens.size() != fields.size()) {
      throw std::invalid_argument(context + ": Mismatched number of tokens and fields");
    }

    GTFSObject stop_time;

    for (size_t i = 0; i < fields.size(); ++i)
      stop_time.setField(fields[i], tokens[i]);

    std::string trip_id = stop_time.getField("trip_id");
    std::string stop_id = stop_time.getField("stop_id");
    const std::string record_context = context + ": trip '" + trip_id + "', stop '" + stop_id + "'";
    int stop_sequence = parseField(stop_time, "stop_sequence", record_context, Utils::parseNonNegativeInt);
    // Convert arrival_time and departure_time to seconds
    // If the departure_time is on the next day, add 24 hours to it
    const auto parse_time = [](const std::string &value) { return Utils::timeToSeconds(value); };
    int arrival_seconds = parseField(stop_time, "arrival_time", record_context, parse_time);
    int departure_seconds = parseField(stop_time, "departure_time", record_context, parse_time);
    if (departure_seconds < arrival_seconds) {
      if (departure_seconds > std::numeric_limits<int>::max() - MIDNIGHT) {
        throw std::overflow_error(record_context + ": departure_time exceeds the supported range after midnight adjustment");
      }
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

  auto fields = readRequiredHeader(file, {"from_stop_id", "to_stop_id", "min_transfer_time"}, "transfers_fp.txt");
  std::string line;
  std::size_t line_number = 1;

  while (std::getline(file, line)) {
    ++line_number;
    Utils::clean(line);

    if (line.empty()) continue;

    std::vector<std::string> tokens = Utils::split(line, ',');  

    const std::string context = "transfers_fp.txt line " + std::to_string(line_number);
    if (tokens.size() != fields.size()) {
      throw std::invalid_argument(context + ": Mismatched number of tokens and fields");
    }

    GTFSObject transfer;
    for (size_t i = 0; i < fields.size(); ++i)
      transfer.setField(fields[i], tokens[i]);

    std::string from_stop_id = transfer.getField("from_stop_id");
    std::string to_stop_id = transfer.getField("to_stop_id");
    int min_transfer_time = parseField(transfer, "min_transfer_time",
                                      context + ": from stop '" + from_stop_id + "', to stop '" + to_stop_id + "'",
                                      Utils::parseNonNegativeInt);

    if (stops_.find(from_stop_id) != stops_.end() && 
        stops_.find(to_stop_id) != stops_.end()) {
      stops_[from_stop_id].addFootpath(to_stop_id, min_transfer_time);
    }
  }
}

void Parser::associateData() {
  // 0. Check data effectiveness of all data structures
  for (const auto &[trip_id, trip] : trips_) {
    if (!trip.hasField("service_id") || trip.getField("service_id").find_first_not_of(" \t\r\n") == std::string::npos) {
      throw std::invalid_argument("trips_fp.txt: trip '" + trip_id + "' has no service_id");
    }
    const std::string service_id = trip.getField("service_id");
    if (services_.find(service_id) == services_.end()) {
      throw std::invalid_argument("trips_fp.txt: trip '" + trip_id + "' references undefined service_id '" + service_id + "'");
    }
  }

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
