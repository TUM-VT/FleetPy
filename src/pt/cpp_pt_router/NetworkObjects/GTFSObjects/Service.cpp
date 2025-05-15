/**
 * @file Service.cpp
 * @brief Implements the Service class.
 *
 * This file contains the implementation of the Service class, which represents
 * active days of a service in the GTFS dataset.
 *
 * @author Chenhao Ding
 * @date 2025-03-21
 */

#include "Service.h"

bool Service::isActive(Date date) const {
    int date_int = Utils::dateToInt(date);
    int start_date = std::stoi(getField("start_date"));
    int end_date = std::stoi(getField("end_date"));

    if (included_dates.find(date_int) != included_dates.end()) {
        return true;
    }

    if (excluded_dates.find(date_int) != excluded_dates.end()) {
        return false;
    }

    if (date_int < start_date || date_int > end_date) {
        return false;
    }

    return active_weekdays.find(date.weekday) != active_weekdays.end();
}

void Service::addActiveWeekday(int weekday) {
    active_weekdays.insert(weekday);
}

void Service::addExceptionDate(int date_int, int type) {
    if (type == 1) {
        included_dates.insert(date_int);
    } else if (type == 2) {
        excluded_dates.insert(date_int);
    }
}

