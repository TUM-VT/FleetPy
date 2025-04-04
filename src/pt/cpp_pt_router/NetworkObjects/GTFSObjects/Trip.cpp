/**
 * @file Trip.cpp
 * @brief Trip class implementation
 *
 * This file contains the implementation of the Trip class, which represents
 * a trip in the GTFS dataset.
 *
 * @autor Maria
 * @date 11/20/2024
 */

#include "Trip.h"

void Trip::addStopTimeRecord(const StopTimeRecord &record) {
  stop_time_records_.push_back(record);
}

const StopTimeRecord* Trip::getStopTimeRecord(const std::string &stop_id) const {
  auto it = std::find_if(stop_time_records_.begin(), stop_time_records_.end(), 
                       [&](const StopTimeRecord &record) {
                         return record.stop_id == stop_id;
                       });
  if (it == stop_time_records_.end()) {
    return nullptr;
  }
  return &(*it);
}

const std::vector<StopTimeRecord> &Trip::getStopTimeRecords() const {
  return stop_time_records_;
}

void Trip::sortStopTimeRecords() {
  std::sort(stop_time_records_.begin(), stop_time_records_.end());
  is_sorted_ = true;
}

std::vector<StopTimeRecord> Trip::getStopTimeRecordsAfter(const std::string &stop_id) const {
  auto it = std::find_if(stop_time_records_.begin(), stop_time_records_.end(), [&](const StopTimeRecord &record) {
    return record.stop_id == stop_id;
  });

  if (it == stop_time_records_.end()) {
    return {};
  }
  
  return std::vector<StopTimeRecord>(it + 1, stop_time_records_.end());
}

bool Trip::isActive(Day day) const {
  return active_days_.at(day);
}

void Trip::setActive(Day day, bool is_active) {
  active_days_[day] = is_active;
}

bool Trip::isSorted() const {
  return is_sorted_;
}
