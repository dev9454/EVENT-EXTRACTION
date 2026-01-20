"""
@author: Filip Ulmaniec (gjl6gk)
last update: 09.17.2024

Function merge_objects gets gpkg file as a parametr, it merges objects whose start and end coordinates are the same and type matches.
As a return value we receive reduced number of rows by those merged.

"""


import geopandas as gpd
from shapely.geometry import LineString

#loading gpkg file
def get_osm_data(osm_file_name):
    try:
        return gpd.read_file(osm_file_name)
    except Exception:
        return gpd.GeoDataFrame()
def merge_objects(gdf):

    gdf['original_index'] = gdf.index # add new column with an index

    # filter tunnel and bridges object names that are LineStrings
    filtered_gdf = gdf[(gdf['object_name'].isin(['tunnel', 'bridges'])) & (gdf.geometry.type == 'LineString')]

    # we check None values and creates 2 new columns with start and end coordinates
    filtered_gdf['start_coords'] = filtered_gdf.geometry.apply(lambda geom: geom.coords[0] if geom is not None else None)
    filtered_gdf['end_coords'] = filtered_gdf.geometry.apply(lambda geom: geom.coords[-1] if geom is not None else None)

    while True:
        # matches coordinates checking also type and road type (highway)
        matches = filtered_gdf.merge(
            filtered_gdf,
            left_on=['end_coords', 'type', 'highway'],
            right_on=['start_coords', 'type', 'highway'],
            suffixes=('', '_match')
        )

        # if there is no more matches it breaks
        if matches.empty:
            break

        # updates geometry
        def merge_geometries(row):
            geom1 = row['geometry'] # first geometry
            geom2 = row['geometry_match'] # second geometry
            # we check None geometries before connecting
            if geom1 is None and geom2 is None:
                return geom1 if geom1 is not None else geom2
            return LineString(list(geom1.coords) + list(geom2.coords)[1:]) # we connect both coordinates, omitting the first coordinate of the second coordiantes

        matches['new_geometry'] = matches.apply(merge_geometries, axis=1) # create new column with new merged coordiates

        # update new coordinates in orignal index (first coordinates)
        filtered_gdf.loc[matches['original_index'], 'geometry'] = matches['new_geometry']

        # delete row with (second coordinates)
        filtered_gdf = filtered_gdf.drop(matches['original_index_match'])

        # update start_coords and end_coords
        filtered_gdf['start_coords'] = filtered_gdf.geometry.apply \
            (lambda geom: geom.coords[0] if geom is not None else None)
        filtered_gdf['end_coords'] = filtered_gdf.geometry.apply \
            (lambda geom: geom.coords[-1] if geom is not None else None)

    # return dataframe with merged coordinates
    return filtered_gdf.drop(columns=['start_coords', 'end_coords', 'original_index'])


#Example use
if __name__ == '__main__':
    osm_file_name = "C:\\Users\\gjl6gk\\Desktop\\gpkg\\gpkg_states\\gcc-states-latest_tunnel.gpkg"
    gpkg_data = get_osm_data(osm_file_name)
    x = merge_objects(gpkg_data)
    print(x)