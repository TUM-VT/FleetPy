/**
 * @file DateTime.h
 * @brief Provides data structures for representing dates and times in the RAPTOR application.
 *
 * This header defines the Date and Time structures, along with auxiliary enums and constants,
 * for handling and manipulating temporal data.
 */

#ifndef RAPTOR_DATETIME_H
#define RAPTOR_DATETIME_H

#include <optional>

/**
 * @brief Represents midnight in seconds (24 hours * 3600 seconds per hour).
 */
static constexpr int MIDNIGHT = 24 * 3600;

/**
 * @brief Names of the weekdays starting from Monday.
 */
constexpr const char* weekdays_names[] = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"};

/**
 * @struct Date
 * @brief Represents a specific date in the Gregorian calendar.
 *
 * Includes fields for the year, month, day, and the day of the week.
 */
struct Date {
  int year;       ///< Year of the date (e.g., 2024).
  int month;      ///< Month of the date (1 = January, ..., 12 = December).
  int day;        ///< Day of the month (1-31).
  int weekday;    ///< Day of the week (0 = Monday, 1 = Tuesday, ..., 6 = Sunday).
};

/**
 * @enum Day
 * @brief Represents the current or the next day for calculations.
 */
enum class Day {
  CurrentDay, ///< Refers to the current day.
  NextDay     ///< Refers to the next day.
};

/**
 * @struct Time
 * @brief Represents a specific time of day in hours, minutes, and seconds.
 */
struct Time {
  int hours;   ///< Hours component of the time (0-23).
  int minutes; ///< Minutes component of the time (0-59).
  int seconds; ///< Seconds component of the time (0-59).
};

#endif //RAPTOR_DATETIME_H
