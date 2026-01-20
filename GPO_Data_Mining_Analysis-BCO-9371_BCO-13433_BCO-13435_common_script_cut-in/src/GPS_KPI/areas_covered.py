"""
@author: Filip Ulmaniec (gjl6gk)
last update: 09.11.2024

Function check_area gets gpkg file as a parametr, it calculates the area where all objects in given file are located. As a return value we
receive 4 coordinates minx, miny, maxx, maxy (list).

"""

import geopandas as gpd

#loading gpkg file
def get_osm_data(osm_file_name):
    try:
        return gpd.read_file(osm_file_name)
    except Exception:
        return gpd.GeoDataFrame()


def check_area(gpkg_data):
    # calculates area of gpkg file
    bounds = (gpkg_data.total_bounds.tolist())
    return bounds

#Example use
if __name__ == '__main__':
    osm_file_name = "C:\\Users\\gjl6gk\\Desktop\\gpkg\\gpkg_states\\gcc-states-latest_barrier.gpkg"
    gpkg_data = get_osm_data(osm_file_name)
    x = check_area(gpkg_data)
    print(x)