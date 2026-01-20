from pathlib import Path
from data_acquisition_utils import Config, read_file, create_map, create_pptx, save_file, get_intersection, \
    merge_gdf_information
import sys
import geopandas as gpd
import shapely
import numpy as np


class TrafficSigns:
    """
    Class that extracts HOV_lanes scenarios from OSM geopackage file
    """

    def __init__(self, output_path: Path) -> None:
        self.output_directory = output_path

    def filter_yield(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts filter sign rows from geodataframe

        Args:
        gdf (gpd.DataFrame)

        Returns
        output_gdf (gpd.DataFrame) Dataframe with HOV_lane filtered rows
        """
        output_gdf = gdf.loc[gdf['highway'] == 'give_way']
        if len(output_gdf) == 0:
            print("No no yield signs found...")
        return output_gdf.reset_index(drop=True)

    def filter_stop(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts stop sign rows from geodataframe

        Args:
        gdf (gpd.DataFrame)

        Returns
        output_gdf (gpd.DataFrame) Dataframe with HOV_lane filtered rows
        """
        output_gdf = gdf.loc[gdf['highway'] == 'stop']
        if len(output_gdf) == 0:
            print("No stop signs found...")
            sys.exit()
        return output_gdf.reset_index(drop=True)

    def filter_no_entry(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts no entry sign rows from geodataframe

        Args:
        gdf (gpd.DataFrame)

        Returns
        output_gdf (gpd.DataFrame) Dataframe with HOV_lane filtered rows
        """
        oneway = gdf[gdf["oneway"] == 'yes'].reset_index(drop=True)
        twoway = gdf[gdf["oneway"] != 'yes'].reset_index(drop=True)
        intersects = get_intersection(oneway, twoway)
        oneway['geometry'] = shapely.get_point(oneway['geometry'], -1)

        # Use only intersections of oneway with twoway
        oneway = oneway[oneway['geometry'].isin(intersects["inter_geometry"])]

        # Only streets defined as 'tertiary','residential','living_street'
        oneway = oneway[oneway["highway"].isin(['tertiary', 'residential', 'living_street', ])]
        output_gdf = get_intersection(oneway, twoway).set_geometry("geometry")
        output_gdf.columns = output_gdf.columns.str.replace("_left", "")

        # Only leave geometry column as the only geometry dtype
        output_gdf = output_gdf[
            output_gdf.columns[(output_gdf.dtypes != 'geometry') | (output_gdf.columns == 'geometry')]]
        if len(output_gdf) == 0:
            print("No no entry signs found...")
            sys.exit()
        return output_gdf.reset_index(drop=True)

    def run(self, config: Config) -> None:
        """
        Runs filter function and then saves filtered dataframe to .html map, pptx presentation
        and .gpkg geopackage file
        """
        self.config = config
        yield_output_directory = self.output_directory / '51_Yield_Sign'
        stop_output_directory = self.output_directory / '51_Stop_Sign'
        no_entry_output_directory = self.output_directory / '51_No_Entry_Sign'

        traffic_signs = read_file(config.traffic_sign['raw_file_path'])
        road_type = read_file(config.road_type['processed_file_path'])
        # Select roads that are accesible
        road_type = road_type[road_type["access"].isin([np.nan, ""])]

        signs_roads_merged = merge_gdf_information(traffic_signs, road_type, max_distance=5)

        # Remove road_type columns and only keep traffic signs columns
        signs_roads_merged = signs_roads_merged[
            signs_roads_merged.columns[~signs_roads_merged.columns.str.contains("right")]]
        signs_roads_merged.columns = signs_roads_merged.columns.str.replace("_left", "")
        signs_roads_merged = signs_roads_merged[traffic_signs.columns]

        comments = ['highway', 'direction']
        # Yield sign logic
        yield_sign_df = self.filter_yield(signs_roads_merged)
        if len(yield_sign_df) == 0:
            print("No yield sign events found...")
        else:
            yield_output_directory.mkdir(parents=True, exist_ok=True)
            create_map(yield_sign_df, yield_output_directory / '51_Yield_Sign.html')
            create_pptx(yield_sign_df, yield_output_directory / '51_Yield_Sign.pptx', comments)
            save_file(yield_sign_df, yield_output_directory / '51_Yield_Sign.gpkg')

        # Stop sign logic
        stop_sign_df = self.filter_stop(signs_roads_merged)
        if len(stop_sign_df) == 0:
            print("No stop sign events found...")
        else:
            stop_output_directory.mkdir(parents=True, exist_ok=True)
            create_map(stop_sign_df, stop_output_directory / '51_Stop_Sign.html')
            create_pptx(stop_sign_df, stop_output_directory / '51_Stop_Sign.pptx', comments)
            save_file(stop_sign_df, stop_output_directory / '51_Stop_Sign.gpkg')

        # No entry sign logic
        comments = ['length', 'highway', 'surface', 'lanes', 'maxspeed']
        no_entry_sign_df = self.filter_no_entry(road_type)
        if len(no_entry_sign_df) == 0:
            print("No no entry sign events found...")
        else:
            no_entry_output_directory.mkdir(parents=True, exist_ok=True)
            create_map(no_entry_sign_df, no_entry_output_directory / '51_No_Entry_Sign.html')
            create_pptx(no_entry_sign_df, no_entry_output_directory / '51_No_Entry_Sign.pptx', comments)
            save_file(no_entry_sign_df, no_entry_output_directory / '51_No_Entry_Sign.gpkg')
