import numpy as np

# radius of earth in meters
EARTH_RADIUS_M = 6371.0088e3

def haversine(lat1, lon1, lat2, lon2):
    # https://stackoverflow.com/a/4913653
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.asin(np.sqrt(a))

    return EARTH_RADIUS_M * c
