import data_acquisition_utils
import geopandas as gpd
import pathlib
import json
import sys


class RailwayCrossings:
    """
    Class that extracts railway crossings from OSM geopackage files
    """

    def __init__(self, output_path):
        self.scenario = '36_Railway_Crossings'
        self.output_directory = output_path / self.scenario

    @staticmethod
    def filter(railway_crossing: gpd.GeoDataFrame, road_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts railway crossings from railway_crossing adds information about the road from road_type
        Args:
        railway_crossing (gpd.GeoDataFrame)
        road_type (gpd.GeoDataFrame)

        Returns
        output_gdf (gpd.DataFrame) GeoDataframe with railway crossings filtered rows with road information
        """
        return data_acquisition_utils.merge_gdf_information(railway_crossing, road_type,
                                                            gdf_a_name='railway_crossing',
                                                            gdf_b_name='road_type')

    def run(self, config: data_acquisition_utils.Config) -> None:
        """
        Runs filter function and then saves filtered dataframe to .html map, pptx presentation
        and .gpkg geopackage file
        """
        railway_crossing = data_acquisition_utils.read_file(config.railway_crossing['processed_file_path'])
        road_type = data_acquisition_utils.read_file(config.road_type['processed_file_path'])

        gdf_filtered = self.filter(railway_crossing, road_type)

        if len(gdf_filtered) == 0:
            print("No events found...")
        else:
            # Keep only one geometry dtype
            for col in gdf_filtered.columns:
                if col != gdf_filtered.geometry.name and gdf_filtered[col].dtype == 'geometry':
                    gdf_filtered[col] = gdf_filtered[col].astype(str)

            comments = ['object_name_railway_crossing', 'railway']  # TODO: Add comments as required

            self.output_directory.mkdir(parents=True, exist_ok=True)

            data_acquisition_utils.create_map(
                gdf_filtered.drop(['object_timestamp_railway_crossing', 'object_timestamp_road_type'], axis=1),
                self.output_directory / f'{self.scenario}.html')
            data_acquisition_utils.create_pptx(gdf_filtered, self.output_directory / f'{self.scenario}.pptx', comments)
            data_acquisition_utils.save_file(gdf_filtered, self.output_directory / f'{self.scenario}.gpkg')
