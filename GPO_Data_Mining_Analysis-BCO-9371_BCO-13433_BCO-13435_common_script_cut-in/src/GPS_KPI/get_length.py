"""
@author: Filip Ulmaniec (gjl6gk)
last update: 09.11.2024

Function get_length gets gpkg file as a parametr, it calculates the length of each object. As a return value we
receive a list of all objects with their lengths (float).

"""

import geopandas as gpd

#loading gpkg file
def get_osm_data(osm_file_name):
    try:
        return gpd.read_file(osm_file_name)
    except Exception:
        return gpd.GeoDataFrame()

def get_length(gpkg_data):
    #calculates length of each object in gpkg file
    gpkg_data_length = gpkg_data.to_crs(3035).length * 0.000621371  # meters -> milage
    return gpkg_data_length

#Example use
if __name__ == '__main__':
    osm_file_name = "C:\\Users\\gjl6gk\\Desktop\\gpkg\\gpkg_states\\gcc-states-latest_barrier.gpkg"
    gpkg_data = get_osm_data(osm_file_name)
    x = get_length(gpkg_data)
    print(x)
