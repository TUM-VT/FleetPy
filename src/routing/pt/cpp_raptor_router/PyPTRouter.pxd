# cython: language_level=3
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp.optional cimport optional
from libcpp cimport bool

cdef extern from "DateTime.h":
    cdef enum class Day:
        CurrentDay
        NextDay
    
    cdef struct Date:
        int year
        int month
        int day
        int weekday
    
    cdef struct Time:
        int hours
        int minutes
        int seconds

cdef extern from "NetworkObjects/DataStructures.h":
    cdef struct pair_hash:
        pass
    
    cdef struct nested_pair_hash:
        pass
    
    cdef struct Query:
        vector[pair[string, int]] included_sources
        vector[pair[string, int]] included_targets
        Date date
        Time departure_time
        int max_transfers
    
    cdef struct StopInfo:
        optional[int] arrival_seconds
        optional[string] parent_trip_id
        optional[string] parent_stop_id
        optional[Day] day
    
    cdef struct JourneyStep:
        optional[string] trip_id
        optional[string] agency_name
        Stop* src_stop
        Stop* dest_stop
        int departure_secs
        Day day
        int duration
        int arrival_secs
    
    cdef struct Journey:
        vector[JourneyStep] steps
        int departure_secs
        Day departure_day
        int arrival_secs
        Day arrival_day
        int duration
        int source_transfer_time
        int waiting_time
        int trip_time
        int num_transfers

cdef extern from "NetworkObjects/GTFSObjects/GTFSObject.h":
    cdef cppclass GTFSObject:
        void setField(const string& field, const string& value)
        string getField(const string& field) const
        const unordered_map[string, string]& getFields() const
        bool hasField(const string& field) const

cdef extern from "NetworkObjects/GTFSObjects/Stop.h":
    cdef cppclass Stop(GTFSObject):
        void addRouteKey(const pair[string, string]& route_key)
        void addFootpath(const string& target_id, int duration)
        const unordered_map[string, int]& getFootpaths() const
        int getFootpathTime(const string& target_id) const
        bool hasFootpath(const string& target_id) const

cdef extern from "NetworkObjects/GTFSObjects/Agency.h":
    cdef cppclass Agency(GTFSObject):
        pass

cdef extern from "NetworkObjects/GTFSObjects/Service.h":
    cdef cppclass Service(GTFSObject):
        pass

cdef extern from "NetworkObjects/GTFSObjects/Route.h":
    cdef cppclass Route(GTFSObject):
        pass

cdef extern from "NetworkObjects/GTFSObjects/Trip.h":
    cdef cppclass Trip(GTFSObject):
        pass

cdef extern from "Parser.h":
    cdef cppclass Parser:
        Parser(string directory)
        unordered_map[string, Agency] getAgencies()
        unordered_map[string, Service] getServices()
        unordered_map[string, Stop] getStops()
        unordered_map[pair[string, string], Route, pair_hash] getRoutes()
        unordered_map[string, Trip] getTrips()

cdef extern from "Raptor.h":
    cdef cppclass Raptor:
        Raptor()
        Raptor(const unordered_map[string, Agency]& agencies_,
               const unordered_map[string, Service]& services_,
               const unordered_map[string, Stop]& stops,
               const unordered_map[pair[string, string], Route, pair_hash]& routes,
               const unordered_map[string, Trip]& trips)
        void setQuery(const Query& query)
        vector[Journey] findJourneys()
        optional[Journey] findOptimalJourney()
        @staticmethod
        void showJourney(const Journey& journey)
        const unordered_map[string, Stop]& getStops() const
        bool isValidJourney(Journey journey) const