from enum import Enum


class COMCODE(Enum):
    """ communication codes for the (call_id, code, args, kwargs) messages sent on a
    FleetControlProcessProxy's request queue, and the (call_id, code, payload) messages
    sent back on its response queue. mirrors the (comm_code, args) pattern already used
    by src/fleetctrl/pooling/batch/AlonsoMora/AlonsoMoraParallelization.py. """

    USER_REQUEST = 1
    GET_OFFER = 2
    TIME_TRIGGER = 3
    RECEIVE_STATUS_UPDATE = 4
    ACK_BOARDING = 5
    ACK_ALIGHTING = 6
    CONFIRM_BOOKING = 7
    CANCEL_REQUEST = 8
    INFORM_TT_UPDATE = 9
    RECORD_STATS = 10
    ADD_INIT = 11
    SYNC_VEHICLES = 12  # replace local vehicle snapshots with a fresh set
    KILL = 99

    # response codes
    ACK = 100  # call handled, no return value expected
    RESULT = 101  # call handled, payload carries the return value
    ERROR = 102  # an exception was raised while handling the call; payload carries (repr(exc), traceback string)
