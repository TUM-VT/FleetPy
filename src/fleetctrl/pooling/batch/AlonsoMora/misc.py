from __future__ import annotations

from typing import Callable, Dict, List, Any, Tuple, TYPE_CHECKING
from functools import cmp_to_key
if TYPE_CHECKING:
    from src.fleetctrl.planning.VehiclePlan import VehiclePlan

# help functions
# --------------
def comp_key_entries(entry1 : Any, entry2 : Any) -> int:
    """ this function is used to sort keys with different data types (int, tuple, str)
    """
    if type(entry1) == type(entry2):
        if type(entry1) == tuple:
            if len(entry1) < len(entry2):
                return -1
            elif len(entry1) > len(entry2):
                return 1
            else:
                for x, y in zip(entry1, entry2):
                    c = comp_key_entries(x, y)
                    if c != 0:
                        return c
                return 0
        else:
            if entry1 < entry2:
                return -1
            elif entry1 > entry2:
                return 1
            else:
                return 0
    else:
        if type(entry1) == str:
            return -1
        elif type(entry2) == str:
            return 1
        else:
            if type(entry1) == int:
                return -1
            elif type(entry2) == int:
                return 1
    raise EnvironmentError("compare keys {} <-> {} : a new datatype within? -> not comparable".format(entry1, entry2))


def deleteRidFromRtv(rid : Any, rtv_key : tuple) -> tuple:
    """This method returns
    - rtv_key without rid        in case rids are left
    - None                       in case no rid is left
    """
    if rtv_key is None:
        return None
    vid = getVidFromRTVKey(rtv_key)
    list_rids = list(getRidsFromRTVKey(rtv_key))
    list_rids.remove(rid)
    if list_rids:
        return createRTVKey(vid, list_rids)
    else:
        return None


def getRRKey(rid1 : Any, rid2 : Any) -> tuple:
    """ this function returns an rr-key (ordered request_id pair)
    :param rid1: request_id of plan_request 1
    :param rid2: request_id of plan_request 2
    :return: ordered tuple of rid1 and rid2 """
    return tuple(sorted((rid1, rid2), key = cmp_to_key(comp_key_entries)))


def getRidsFromRTVKey(rtv_key) -> List[Any]:
    """ this function returns a list of plan_request_ids corresponding to the rtv_key
    :param rtv_key: rtv_key corresponding to an v2rb-obj
    :return: list of planrequest_ids """
    if rtv_key is None:
        return []
    return rtv_key[1:]


def getVidFromRTVKey(rtv_key : tuple) -> Any:
    """ this function returns the vehicle_id corresponding to the rtv_key
    :param rtv_key: rtv_key corresponding to an v2rb-obj
    :return: vehicle_id """
    return rtv_key[0]


def createRTVKey(vid : int, rid_list : List[Any]) -> tuple:
    """ this functions creates a new rtv_key from a vehicle_id and a list of plan_request_ids
    :param vid: vehicle id
    :param rid_list: list of plan_request_ids
    :return: type tuple : rtv_key """
    if len(rid_list) == 0:
        return None
    sorted_rid_list = tuple(sorted(rid_list, key = cmp_to_key(comp_key_entries)))
    return (vid, ) + sorted_rid_list

def get_full_assigned_tree(rtv_key : tuple, r_ob ):
    """ this functions computes all lower rtv_keys that must be existent for rtv_key to exist
    assumes that r_ob are part of the key!
    :param rtv_key: rtv_key of v2rb_obj
    :param r_ob: list of request_ids currently on board of the corresponding vehicle
    :return: dict number of requests -> rtv_key -> 1 of all necessary key of rtv_key with r_ob on board """
    #print(rtv_key, r_ob)
    if not rtv_key:
        return
    vid = getVidFromRTVKey(rtv_key)
    ass_rids = getRidsFromRTVKey(rtv_key)
    if len(r_ob) != 0:
        key_tree = {len(r_ob) : {createRTVKey(vid, r_ob) : 1}}
    else:
        key_tree = {}
    for rid in ass_rids:
        if rid not in r_ob:
            for i in range(len(r_ob), len(ass_rids)):
                for key in key_tree.get(i, {}).keys():
                    rids = getRidsFromRTVKey(key)
                    if rid not in rids:
                        new_key = createRTVKey(vid, list(rids) + [rid])
                        try:
                            key_tree[i+1][new_key] = 1
                        except KeyError:
                            key_tree[i+1] = {new_key : 1}
        if len(r_ob) == 0:
            new_key = createRTVKey(vid, [rid])
            try:
                key_tree[1][new_key] = 1
            except KeyError:
                key_tree[1] = {new_key : 1}
    return key_tree

def createListLowerLevelKeys(build_key : tuple, new_rid : Any, do_not_remove_for_lower_keys : List[tuple]) -> List[tuple]:
    """This function creates keys from build_key that are one level lower than build_key
    by removing one of the rids.
    If one of the rids is in do_not_remove_for_lower_keys, no key is created for this rid.
    :param build_key: rtv_key of v2rbs to build on
    :param new_rid: plan_request_id of new request
    :param do_not_remove_for_lower_key: list of plan_request_ids that must be part of the lower key
    :return: list of lower rtv_keys"""
    return_keys = []
    vid = getVidFromRTVKey(build_key)
    list_rids = list(getRidsFromRTVKey(build_key))
    if new_rid in list_rids:
        return []
    for rid in list_rids:
        if rid not in do_not_remove_for_lower_keys:
            copy_of_list = list_rids[:]
            copy_of_list.remove(rid)
            if len(copy_of_list) == 0:
                return []
            copy_of_list.append(new_rid)
            new_key = createRTVKey(vid, copy_of_list)
            return_keys.append(new_key)
    return return_keys


def getNecessaryKeys(rtv_key : tuple, r_ob : List[Any]) -> List[tuple]:
    """ this functions computes all rtv_keys that must be existent for rtv_key to exist
    :param rtv_key: rtv_key of v2rb_obj
    :param r_ob: list of request_ids currently on board of the corresponding vehicle
    :return: iterator of necessary keys """
    if not rtv_key:
        return []
    vid = getVidFromRTVKey(rtv_key)
    ass_rids = getRidsFromRTVKey(rtv_key)
    v_r_ob = []
    for rid in ass_rids:
        if rid in r_ob:
            v_r_ob.append(rid)
    rid_combs = [v_r_ob]
    for rid in ass_rids:
        if rid in v_r_ob or rid in r_ob:
            continue
        #new_combs = []
        for comb in rid_combs:
            new_comb = comb[:]
            new_comb.append(rid)
            yield createRTVKey(vid, new_comb)
    #         new_combs.append(new_comb)
    #     rid_combs += new_combs
    # nec_key = [createRTVKey(vid, comb) for comb in rid_combs if len(comb) > 0]
    # return nec_key


def getRTVkeyFromVehPlan(veh_plan : VehiclePlan) -> tuple:
    """ creates the rtv_key based on a vehicle plan
    :param veh_plan: vehicle plan object in question
    :return: rtv_key
    """
    if veh_plan is None:
        return None
    rids = {}
    vid = veh_plan.vid
    for pstop in veh_plan.list_plan_stops:
        for rid in pstop.get_list_boarding_rids():
            rids[rid] = 1
        for rid in pstop.get_list_alighting_rids():
            rids[rid] = 1
    if len(rids.keys()) == 0:
        return None
    return createRTVKey(vid, rids.keys())