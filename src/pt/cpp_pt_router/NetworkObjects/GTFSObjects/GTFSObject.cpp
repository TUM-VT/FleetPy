/**
 * @file GTFSObject.cpp
 * @brief Implements the GTFSObject class.
 *
 * This file contains the implementation of the GTFSObject class, which represents
 * a generic GTFS object.
 *
 * @autor Maria
 * @date 11/20/2024
 */

#include "GTFSObject.h"

void GTFSObject::setField(const std::string &field, const std::string &value) {
  fields[field] = value;
}

std::string GTFSObject::getField(const std::string &field) const {
  auto it = fields.find(field);
  if (it == fields.end())
    throw std::runtime_error("Field not found: " + field);
  return it->second;
}

const std::unordered_map<std::string, std::string> &GTFSObject::getFields() const {
  return fields;
}

bool GTFSObject::hasField(const std::string& field) const {
  return fields.find(field) != fields.end();
}

void GTFSObject::merge(const GTFSObject &other, bool override) {
  for (const auto &[key, value]: other.getFields()) {
    if (hasField(key) && !override) {
      // Keep the original value
      continue;
    } else {
      fields[key] = value;
    }
  }
}
