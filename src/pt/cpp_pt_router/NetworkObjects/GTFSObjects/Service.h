/**
 * @file Service.h
 * @brief Defines the Service class, representing active days for a service.
 *
 * This header file declares the Service class, which inherits from GTFSObject.
 * The class serves as a representation of the GTFS "calendar.txt" file and "calendar_dates.txt" file, storing
 * information about active days of a service.
 *
 * @author Chenhao Ding
 * @date 2025-03-21
 */

#ifndef RAPTOR_SERVICE_H
#define RAPTOR_SERVICE_H

#include "GTFSObject.h"

/**
 * @class Service
 * @brief Represents active days for a service in the GTFS data.
 *
 * This class inherits from GTFSObject and encapsulates the details of active days for a service.
 */
class Service : public GTFSObject  {
public:
    bool isActive(Date date) const;
    void addActiveWeekday(int weekday);
    void addExceptionDate(int date_int, int type);

private:
    std::unordered_set<int> active_weekdays;  // 0-6, 0 = monday, 6 = sunday
    std::unordered_set<int> included_dates;  // exception type 1
    std::unordered_set<int> excluded_dates;  // exception type 2
};

#endif //RAPTOR_SERVICE_H

