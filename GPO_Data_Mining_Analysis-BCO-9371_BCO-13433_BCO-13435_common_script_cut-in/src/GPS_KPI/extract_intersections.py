"""
@author Wiktor Wiernasiewcz
last update: Wiktor Wiernasiewicz 23.10.2024

Function that processes geometry intersections in a large spatial dataset.

Input:
- gdf: GeoDataFrame containing the geometry and optionally additional columns with data.
- n_jobs: Number of CPU cores to use for parallel processing (default -1, which means using all available cores).
- tolerance: Tolerance value for caching the geometry to account for possible precision errors (default 1e-6).
- batch_size: Size of the batch of data to process in one thread (default 1000).

Action:
1. Reads data from a GeoPackage file into a GeoDataFrame object.
2. Splits the data into smaller batches to facilitate parallel processing.
3. For each pair of geometry in the batch, calculates the intersections and angles between the lines.
4. Collects information about the intersections, including angle, geometry type, and data from the original geometries.
5. Combines the results from all batches and writes them to a new GeoPackage file.

Output:
- A GeoDataFrame containing information about the geometry intersections, including the intersection angle, the type of the intersection geometry, and additional data from the original geometries.

"""
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, MultiLineString, MultiPoint
from shapely.strtree import STRtree
import pandas as pd
from joblib import Parallel, delayed
import math
import data_acquisition_utils

# A function that calculates the angle of intersection of two lines
def calculate_intersection_angle(line1, line2):
    def vector_angle(v1, v2):
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return np.nan  # Return NaN for zero-length vectors
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1, 1)  # Avoid precision issues
        return np.degrees(np.arccos(cos_angle))
    
    # Calculating vectors for both lines
    vector1 = np.array(line1.coords[-1]) - np.array(line1.coords[0])
    vector2 = np.array(line2.coords[-1]) - np.array(line2.coords[0])
    angle = vector_angle(vector1, vector2)
    
    # Exclude 0° and 180° angles
    if angle == 0 or angle == 180:
        return np.nan
    return min(angle, 180 - angle)  # Return the smaller angle


# Function to process a batch of geometries
def process_batch(batch, gdf, tolerance):
    intersections = []  # List to store intersection information
    geometries = []     # List to store intersection geometries
    
    # Create spatial index for the entire dataset
    spatial_index = STRtree(gdf.geometry)

    # Iterate over the geometries in the batch
    for i, geom1 in batch.iterrows():
        possible_matches_idx = spatial_index.query(geom1.geometry.buffer(tolerance))  # Query for possible matches

        for idx in possible_matches_idx:
            geom2 = gdf.geometry.iloc[idx]  # Get the second geometry

            # Checking if the geometries are different and intersect
            if geom1.geometry != geom2 and geom1.geometry.buffer(tolerance).intersects(geom2.buffer(tolerance)): 
                # Check if the layers are the same
                layer1 = geom1.get('layer', 0)
                layer2 = gdf.iloc[idx].get('layer', 0)
                if layer1 != layer2:
                    continue  # Skip intersections of roads on different layers

                intersection = geom1.geometry.intersection(geom2)

                if not intersection.is_empty and isinstance(intersection, (Point, MultiPoint)):
                    if intersection in set(geometries):
                        continue  # Skip duplicates

                    angle = calculate_intersection_angle(geom1.geometry, geom2)

                    # Check if the angle is valid (not NaN) and exclude 0 and 180 degrees
                    if not np.isnan(angle):
                        data = {
                            'angle': angle,
                            'type': type(intersection).__name__,
                            'geometry': intersection,  # Store the intersection point
                            'geometry_1': geom1.geometry,
                            'geometry_2': geom2
                        }
                        # Add all relevant info from both intersecting segments
                        for col in gdf.columns:
                            if col != 'geometry':
                                data[f"{col}_1"] = geom1[col]  # Geometry information 1
                                data[f"{col}_2"] = gdf.iloc[idx][col]  # Geometry information 2

                        intersections.append(data)
                        geometries.append(intersection)

    # Return the result as GeoDataFrame
    return gpd.GeoDataFrame(intersections, geometry='geometry')


# Function to parallelize the process
def find_intersections_parallel(gdf, n_jobs=-1, tolerance=1e-4, batch_size=5000):
    # Split the GeoDataFrame into batches
    length = len(gdf)
    chunk_size = length // math.ceil(length / batch_size)
    batches = [gdf[i:i + chunk_size] for i in range(0, length, chunk_size)]

    # Run the process_batch function in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch, gdf, tolerance) for batch in batches
    )

    # Concatenate all results into one GeoDataFrame
    intersection_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), geometry='geometry').set_crs(4326)

    
    intersection_gdf = intersection_gdf[intersection_gdf.geometry.type.isin(['Point', 'MultiPoint'])]

    
    columns_to_drop = ['geometry_1', 'geometry_2']
    intersection_gdf = intersection_gdf.drop(columns=columns_to_drop, errors='ignore')

    return intersection_gdf