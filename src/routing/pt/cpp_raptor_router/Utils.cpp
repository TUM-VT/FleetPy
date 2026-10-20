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
  const auto colon = timeStr.find(':');
  if (colon == std::string::npos || colon == 0 || timeStr.size() - colon != 6 ||
      timeStr[colon + 3] != ':') {
    throw std::invalid_argument("Expected H:MM:SS with nonnegative hours, got '" + timeStr + "'");
  }
  return timeToSeconds(Time{parseNonNegativeInt(timeStr.substr(0, colon)),
                            parseNonNegativeInt(timeStr.substr(colon + 1, 2)),
                            parseNonNegativeInt(timeStr.substr(colon + 4, 2))});
}

int Utils::timeToSeconds(const Time &time) {
  if (time.hours < 0 || time.minutes < 0 || time.minutes > 59 || time.seconds < 0 || time.seconds > 59) {
    throw std::invalid_argument("Time requires nonnegative hours and minutes/seconds in 0..59");
  }
  const long long seconds = static_cast<long long>(time.hours) * 3600 + time.minutes * 60 + time.seconds;
  if (seconds > std::numeric_limits<int>::max()) {
    throw std::overflow_error("Time in seconds exceeds the supported integer range");
  }
  return static_cast<int>(seconds);
}

int Utils::parseNonNegativeInt(const std::string &value) {
  if (!isNumber(value)) {
    throw std::invalid_argument("Expected a nonnegative integer, got '" + value + "'");
  }
  int result = 0;
  for (const char digit : value) {
    if (result > (std::numeric_limits<int>::max() - (digit - '0')) / 10) {
      throw std::overflow_error("Integer exceeds the supported range: '" + value + "'");
    }
    result = result * 10 + (digit - '0');
  }
  return result;
}

int Utils::parseServiceDate(const std::string &value) {
  if (value.size() != 8 || !isNumber(value)) {
    throw std::invalid_argument("Expected a YYYYMMDD service date, got '" + value + "'");
  }
  return dateToInt(Date{parseNonNegativeInt(value.substr(0, 4)),
                        parseNonNegativeInt(value.substr(4, 2)),
                        parseNonNegativeInt(value.substr(6, 2))});
}

int Utils::dateToInt(const Date &date) {
  if (date.day < 1 || date.day > daysInMonth(date.year, date.month)) {
    throw std::invalid_argument("Invalid day in calendar date");
  }
  const long long value = static_cast<long long>(date.year) * 10000 + date.month * 100 + date.day;
  if (value > std::numeric_limits<int>::max()) {
    throw std::overflow_error("Calendar date exceeds the supported integer range");
  }
  return static_cast<int>(value);
}

std::vector<std::string> Utils::split(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);

  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }

  if (!str.empty() && str.back() == delimiter) {
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
  return !str.empty() && std::all_of(str.begin(), str.end(), [](char c) { return c >= '0' && c <= '9'; });
}

int Utils::daysInMonth(int year, int month) {
  if (year < 1 || month < 1 || month > 12) {
    throw std::invalid_argument("Invalid calendar year or month");
  }
  static const int daysInMonths[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month == 2 && ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0))) {
    return 29; // February in a leap year
  }
  return daysInMonths[month - 1];
}

bool Utils::dateWithinRange(const Date &date, const std::string &start_date, const std::string &end_date) {
  const int start = parseServiceDate(start_date);
  const int end = parseServiceDate(end_date);
  if (start > end) throw std::invalid_argument("start_date is after end_date");
  const int value = dateToInt(date);
  return start <= value && value <= end;
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

