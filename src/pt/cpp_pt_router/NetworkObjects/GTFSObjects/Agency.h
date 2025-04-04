/**
 * @file Agency.h
 * @brief Defines the Agency class, representing transit agencies in the GTFS dataset.
 *
 * This header file declares the Agency class, which inherits from GTFSObject.
 * The class serves as a representation of the GTFS "agency.txt" file, storing
 * information about transit agencies.
 *
 * @author Maria
 * @date 11/20/2024
 */
#ifndef RAPTOR_AGENCY_H
#define RAPTOR_AGENCY_H

#include "GTFSObject.h"

/**
 * @class Agency
 * @brief Represents a transit agency in the GTFS data.
 *
 * This class inherits from GTFSObject and encapsulates the details of a transit agency.
 *
 * * @note This class currently acts as a placeholder and can be extended
 * with specific attributes and methods relevant to transit agencies.
 */
class Agency : public GTFSObject  {

};


#endif //RAPTOR_AGENCY_H
