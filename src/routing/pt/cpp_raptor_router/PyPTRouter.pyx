# cython: language_level=3
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp.optional cimport optional
import json
from datetime import datetime

cdef class PyPTRouter:
    cdef Raptor* raptor_ptr  # Hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, str input_directory):
        """Initialize the RAPTOR router
        
        This is a Cython-specific constructor that is called first during object creation.
        It initializes the C++ RAPTOR router by loading and processing input data.

        Args:
            input_directory (str): Directory path containing GTFS data.
                                 The directory should contain necessary GTFS files
                                 (e.g., stops.txt, routes.txt, etc.)

        Notes:
            - This method is automatically called during object creation
            - If raptor_ptr already exists, it will be deleted before creating a new instance
            - The input path will be converted to UTF-8 encoded C++ string
        """
        # Convert Python string to C++ string
        cdef string cpp_directory = input_directory.encode('utf-8')

        # Initialize data containers for GTFS data
        cdef unordered_map[string, Agency] agencies
        cdef unordered_map[string, Service] services
        cdef unordered_map[string, Trip] trips
        cdef unordered_map[pair[string, string], Route, pair_hash] routes
        cdef unordered_map[string, Stop] stops

        # Process directory
        cdef Parser* parser = new Parser(cpp_directory)
        
        agencies = parser.getAgencies()
        services = parser.getServices()
        trips = parser.getTrips()
        routes = parser.getRoutes()
        stops = parser.getStops()

        del parser

        # Clean up existing instance if any
        if self.raptor_ptr != NULL:
            del self.raptor_ptr
            
        # Create new RAPTOR instance with data
        self.raptor_ptr = new Raptor(agencies, services, stops, routes, trips)

    def __dealloc__(self):
        """Deallocate the RAPTOR router

        This is a Cython-specific destructor that is called when the object is garbage collected.
        It ensures proper cleanup of C++ resources to prevent memory leaks.

        Notes:
            - Automatically called during garbage collection
            - Safely deletes the C++ Raptor object if it exists
            - Sets raptor_ptr to NULL after deletion is handled by C++
        """
        if self.raptor_ptr != NULL:
            del self.raptor_ptr

    def construct_query(
        self,
        arrival_datetime,
        list included_sources, list included_targets, int max_transfers=-1,
    ):
        """Construct query information.

        Args:
            arrival_datetime (datetime): Arrival datetime at the source station
            included_sources (list): List of source stop IDs and their station stop transfer times
            included_targets (list): List of target stop IDs and their station stop transfer times
            max_transfers (int): Maximum number of transfers allowed

        Returns:
            query (Query)
        """
        # Calculate day of week using Python's datetime
        cdef int year = arrival_datetime.year
        cdef int month = arrival_datetime.month
        cdef int day = arrival_datetime.day
        cdef int weekday = arrival_datetime.weekday()
        cdef int hours = arrival_datetime.hour
        cdef int minutes = arrival_datetime.minute
        cdef int seconds = arrival_datetime.second
        
        # Create date, time objects for C++
        cdef Date date = Date(year, month, day, weekday)
        cdef Time departure_time = Time(hours, minutes, seconds)
        
        cdef Query query

        cdef vector[pair[string, int]] src_vec
        cdef vector[pair[string, int]] tgt_vec

        src_vec = vector[pair[string, int]]()
        for inc_src in included_sources:
            if isinstance(inc_src, tuple) and len(inc_src) == 2 and isinstance(inc_src[0], str) and isinstance(inc_src[1], int):
                src_vec.push_back(pair[string, int](inc_src[0].encode('utf-8'), inc_src[1]))
            else:
                raise TypeError(f"Expected (string, int) tuple, got {type(inc_src)}")
        
        tgt_vec = vector[pair[string, int]]()
        for inc_tgt in included_targets:
            if isinstance(inc_tgt, tuple) and len(inc_tgt) == 2 and isinstance(inc_tgt[0], str) and isinstance(inc_tgt[1], int):
                tgt_vec.push_back(pair[string, int](inc_tgt[0].encode('utf-8'), inc_tgt[1]))
            else:
                raise TypeError(f"Expected (string, int) tuple, got {type(inc_tgt)}")

        return Query(src_vec, tgt_vec, date, departure_time, max_transfers)

    def return_pt_journeys_1to1(
        self,
        arrival_datetime,
        list included_sources, list included_targets, int max_transfers=-1,
        bool detailed=False,
    ):
        """Find the best public transport journey from source to target

        This method queries the RAPTOR router to find the optimal journey between two stops
        at a specified departure time.

        Args:
            arrival_datetime (datetime): Arrival datetime at the source station
            included_sources (list): List of source stop IDs and their station stop transfer times
            included_targets (list): List of target stop IDs and their station stop transfer times
            max_transfers (int): Maximum number of transfers allowed (-1 for unlimited)
            detailed (bool): Whether to return the detailed journey plan.

        Returns:
            dict: A dictionary containing journey details, or None if no journey is found.
                 The dictionary includes:
                 - duration: Total journey duration in seconds
        """
        if self.raptor_ptr == NULL:
            raise RuntimeError("RAPTOR router not initialized. Please initialize first.")  

        query = self.construct_query(
            arrival_datetime, included_sources, included_targets, max_transfers,
        )
            
        # Set query and find journeys
        self.raptor_ptr.setQuery(query)
        cdef vector[Journey] journeys = self.raptor_ptr.findJourneys()

        # Check if any journeys were found
        if journeys.size() == 0:
            return None

        # Convert all journeys to Python list of dictionaries
        journeys_list = []
        for i in range(journeys.size()):
            journey_dict = self._convert_journey_to_dict(journeys[i], detailed)
            journeys_list.append(journey_dict)
        
        return journeys_list

    def return_fastest_pt_journey_1to1(
        self,
        arrival_datetime,
        list included_sources, list included_targets, int max_transfers=-1,
        bool detailed=False,
    ):
        """Find the fastest public transport journey from source to target

        This method queries the RAPTOR router to find the optimal journey between two stops
        at a specified departure time.

        Args:
            arrival_datetime (datetime): Arrival datetime at the source station
            included_sources (list): List of source stop IDs and their station stop transfer times
            included_targets (list): List of target stop IDs and their station stop transfer times
            max_transfers (int): Maximum number of transfers allowed (-1 for unlimited)
            detailed (bool): Whether to return the detailed journey plan.

        Returns:
            dict: A dictionary containing journey details, or None if no journey is found.
                 The dictionary includes:
                 - duration: Total journey duration in seconds
        """
        if self.raptor_ptr == NULL:
            raise RuntimeError("RAPTOR router not initialized. Please initialize first.")      
        
        query = self.construct_query(
            arrival_datetime, included_sources, included_targets, max_transfers,
        )
        
        # Set query and find journeys
        self.raptor_ptr.setQuery(query)
        cdef optional[Journey] journey_opt = self.raptor_ptr.findOptimalJourney()

        # Check if journey was found
        if not journey_opt.has_value():
            return None

        # Get the actual journey from optional
        cdef Journey journey = journey_opt.value()
        
        # Convert journey to Python dictionary
        journey_dict = self._convert_journey_to_dict(journey, detailed)
        
        return journey_dict

    cdef _convert_journey_to_dict(self, Journey journey, bool detailed):
        """Convert a Journey object to a Python dictionary
        
        This method takes a C++ Journey object and converts it to a Python dictionary
        with all the relevant journey information.
        
        Args:
            journey (Journey): The C++ Journey object to convert
            detailed (bool): Whether to include the detailed journey plan
            
        Returns:
            dict: A dictionary containing journey details
        """
        journey_dict = {
            # Overall journey information
            "duration": journey.duration,
            "source_transfer_time": journey.source_transfer_time,
            "waiting_time": journey.waiting_time,
            "trip_time": journey.trip_time,
            "num_transfers": journey.num_transfers,
            
            # Departure information
            "departure_time": journey.departure_secs,
            "departure_day": self._day_to_str(journey.departure_day),
            
            # Arrival information
            "arrival_time": journey.arrival_secs,
            "arrival_day": self._day_to_str(journey.arrival_day),
            
            # Journey steps
            "steps": []
        }

        if not detailed:
            return journey_dict

        # Convert each journey step
        for i in range(journey.steps.size()):
            step_dict = self._convert_journey_step(journey.steps[i])
            journey_dict["steps"].append(step_dict)
    
        return journey_dict

    cdef str _day_to_str(self, Day day):
        """Convert Day enum to string"""
        if day == Day.CurrentDay:
            return "current_day"
        else:
            return "next_day"

    cdef _convert_journey_step(self, JourneyStep step):
        """Convert a JourneyStep to Python dictionary"""
        
        cdef string stop_id_str = string(b"stop_id")
        
        step_dict = {
            # Basic information
            "duration": step.duration,
            
            # Time information
            "departure_time": step.departure_secs,
            "arrival_time": step.arrival_secs,
            "day": self._day_to_str(step.day),
            
            # Stop information
            "from_stop_id": step.src_stop.getField(stop_id_str).decode('utf-8'),
            "to_stop_id": step.dest_stop.getField(stop_id_str).decode('utf-8'),
        }
        
        # Handle optional fields
        if step.trip_id.has_value():
            step_dict["trip_id"] = step.trip_id.value().decode('utf-8')
        else:
            step_dict["trip_id"] = "walking"
            
        if step.agency_name.has_value():
            step_dict["agency_name"] = step.agency_name.value().decode('utf-8')
        else:
            step_dict["agency_name"] = "Unknown"
            
        return step_dict