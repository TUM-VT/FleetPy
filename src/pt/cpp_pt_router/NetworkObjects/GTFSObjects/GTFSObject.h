/**
 * @file GTFSObject.h
 * @brief Defines the GTFSObject class, representing a generic GTFS object.
 *
 * This header file declares the GTFSObject class,
 * which serves as a base class for all GTFS objects.
 *
 * @author Maria
 * @date 11/20/2024
 */

#ifndef RAPTOR_GTFSOBJECT_H
#define RAPTOR_GTFSOBJECT_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <stdexcept>
#include <functional>

#include "../../Utils.h"

/**
 * @class GTFSObject
 * @brief Represents a generic GTFS object.
 *
 * This class serves as a base class for all GTFS objects.
 * It provides a generic interface for setting and getting field values.
 */
class GTFSObject {
public:
  /**
 * @brief Sets the value of a field.
 * @param field The name of the field.
 * @param value The value to assign to the field.
 */
  void setField(const std::string &field, const std::string &value);

  /**
   * @brief Retrieves the value of a field.
   * @param field The name of the field to retrieve.
   * @return The value of the specified field.
   * @throws std::runtime_error If the field does not exist.
   */
  std::string getField(const std::string &field) const;

  /**
   * @brief Gets all fields as an unordered map.
   * @return A reference to the map of fields.
   */
  const std::unordered_map<std::string, std::string> &getFields() const;

  /**
   * @brief Checks if a field exists.
   * @param field The name of the field to check.
   * @return True if the field exists, false otherwise.
   */
  bool hasField(const std::string &field) const;

  /**
   * @brief Merges two GTFS objects.
   * @param other The GTFS object to merge with.
   * @param override Whether to override existing fields.
   */
  void merge(const GTFSObject &other, bool override = false);

protected:
  std::unordered_map<std::string, std::string> fields; ///< Map of field names and values.

};

#endif //RAPTOR_GTFSOBJECT_H
