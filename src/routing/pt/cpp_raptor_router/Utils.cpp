/**
 * @file Utils.cpp
 * @brief Provides utility functions for the RAPTOR application.
 *
 * This file contains utility functions for the RAPTOR application,
 * including functions for calculating distances, durations, and time conversions.
 *
 * @author Maria
 * @date 10/28/2024
 * 
 * @version fleetpy-1.0
 * @author Chenhao Ding
 * @date 12/12/2025
 */

#include "Utils.h"
#include <limits>
#include <stdexcept>

std::string Utils::secondsToTime(std::optional<int> seconds) {

  if (!seconds.has_value()) return "INF";

  int seconds_value = seconds.value();

  int hours = seconds_value / 3600 % 24;
  int minutes = (seconds_value % 3600) / 60;
  int secs = seconds_value % 60;

  std::ostringstream oss;
  oss << std::setw(2) << std::setfill('0') << hours << ":"
      << std::setw(2) << std::setfill('0') << minutes << ":"
      << std::setw(2) << std::setfill('0') << secs;
  return oss.str();
}

int Utils::timeToSeconds(const std::string &timeStr) {
  int hours, minutes, seconds;
  char colon;

  std::istringstream iss(timeStr);
  iss >> hours >> colon >> minutes >> colon >> seconds;

  return hours * 3600 + minutes * 60 + seconds;
}

int Utils::timeToSeconds(const Time &time) {
  return time.hours * 3600 + time.minutes * 60 + time.seconds;
}

int Utils::dateToInt(const Date &date) {
  return date.year * 10000 + date.month * 100 + date.day;
}

std::vector<std::string> Utils::split(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);

  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }

  if (str.back() == delimiter) {
    tokens.emplace_back("");  // Add empty token if the last character is the delimiter
  }

  return tokens;
}


std::string Utils::getFirstWord(const std::string &str) {
  return str.substr(0, str.find(' '));
}

void Utils::clean(std::string &input) {
  size_t first = input.find_first_not_of(" \t\n\r");
  size_t last = input.find_last_not_of(" \t\n\r");
  input = (first == std::string::npos) ? "" : input.substr(first, (last - first + 1));
}


bool Utils::isNumber(const std::string &str) {
  return !str.empty() && std::all_of(str.begin(), str.end(), [](char c) { return std::isdigit(c); });
}

int Utils::daysInMonth(int year, int month) {
  static const int daysInMonths[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month == 2 && ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0))) {
    return 29; // February in a leap year
  }
  return daysInMonths[month - 1];
}

bool Utils::dateWithinRange(const Date &date, const std::string &start_date, const std::string &end_date) {
  // Convert strings YYYYMMDD to Date objects
  Date start = {std::stoi(start_date.substr(0, 4)), std::stoi(start_date.substr(4, 2)),
                std::stoi(start_date.substr(6, 2))};
  Date end = {std::stoi(end_date.substr(0, 4)), std::stoi(end_date.substr(4, 2)), std::stoi(end_date.substr(6, 2))};

  // Check if the date is earlier than the start date
  if ((date.year < start.year)
      || (date.year == start.year && date.month < start.month)
      || (date.year == start.year && date.month == start.month && date.day < start.day))
    return false;

  // Check if the date is later than the end date
  if ((date.year > end.year)
      || (date.year == end.year && date.month > end.month)
      || (date.year == end.year && date.month == end.month && date.day > end.day))
    return false;

  return true;
}

Date Utils::addOneDay(Date date) {
  if (date.year < 1 || date.month < 1 || date.month > 12 ||
      date.weekday < 0 || date.weekday > 6) {
    throw std::invalid_argument("Invalid calendar date or weekday");
  }
  const int month_days = daysInMonth(date.year, date.month);
  if (date.day < 1 || date.day > month_days) {
    throw std::invalid_argument("Invalid day in calendar date");
  }

  // Advance a Gregorian calendar date without consulting the host time zone.
  date.weekday = (date.weekday + 1) % 7;
  if (date.day < month_days) {
    ++date.day;
  } else {
    date.day = 1;
    if (date.month < 12) {
      ++date.month;
    } else {
      if (date.year == std::numeric_limits<int>::max()) {
        throw std::overflow_error("Calendar year exceeds the supported integer range");
      }
      date.month = 1;
      ++date.year;
    }
  }
  return date;
}

Date Utils::addDays(Date date, int days) {
  if (date.year < 1 || date.month < 1 || date.month > 12 ||
      date.weekday < 0 || date.weekday > 6) {
    throw std::invalid_argument("Invalid calendar date or weekday");
  }
  if (date.day < 1 || date.day > daysInMonth(date.year, date.month)) {
    throw std::invalid_argument("Invalid day in calendar date");
  }
  if (days == 0) return date;

  // Count from 0001-01-01 with wide arithmetic, including the year upper bound.
  const auto days_before_year = [](long long year) {
    const long long previous = year - 1;
    return 365 * previous + previous / 4 - previous / 100 + previous / 400;
  };
  long long ordinal = days_before_year(date.year) + date.day - 1;
  for (int month = 1; month < date.month; ++month) {
    ordinal += daysInMonth(date.year, month);
  }
  ordinal += days;
  const long long max_year = std::numeric_limits<int>::max();
  if (ordinal < 0 || ordinal >= days_before_year(max_year + 1)) {
    throw std::overflow_error("Calendar year exceeds the supported positive integer range");
  }

  // Binary search avoids work proportional to a potentially large signed offset.
  long long first = 1, last = max_year;
  while (first < last) {
    const long long middle = first + (last - first + 1) / 2;
    if (days_before_year(middle) <= ordinal) first = middle;
    else last = middle - 1;
  }
  const int year = static_cast<int>(first);
  long long day_of_year = ordinal - days_before_year(year);
  int month = 1;
  while (day_of_year >= daysInMonth(year, month)) {
    day_of_year -= daysInMonth(year, month);
    ++month;
  }
  return {year, month, static_cast<int>(day_of_year) + 1,
          (date.weekday + days % 7 + 7) % 7};
}

std::string Utils::dayToString(Day day) {
  return (day == Day::CurrentDay) ? "current" : "next";
}

