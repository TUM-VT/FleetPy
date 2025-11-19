import math


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the bearing (angle) between two points.

    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2

    Returns:
        Bearing in degrees from point 1 to point 2
    """
    # Convert to radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    # Calculate bearing
    x = math.cos(lat2) * math.sin(lon2 - lon1)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * \
        math.cos(lat2) * math.cos(lon2 - lon1)
    bearing = math.atan2(x, y)

    # Convert to degrees
    return math.degrees(bearing)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in kilometers.

    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers

    # Convert to radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def is_peak_hour(time_of_day: float) -> bool:
    """Check if a given time is during peak hours.

    Args:
        time_of_day: Time in minutes since midnight

    Returns:
        True if the time is during peak hours, False otherwise
    """
    # Convert time to hours (assuming time is in minutes)
    hour = (time_of_day / 60.0) % 24
    # Morning peak: 7-10 AM, Evening peak: 4-7 PM
    return (7 <= hour <= 10) or (16 <= hour <= 19)
