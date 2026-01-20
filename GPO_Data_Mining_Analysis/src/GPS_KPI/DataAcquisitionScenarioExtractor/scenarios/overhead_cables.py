import data_acquisition_utils
import geopandas as gpd
import pathlib
import json
import sys


class OverheadCables:
    """
    Class that extracts intersections of the power lines with the roads from OSM geopackage files
    """

    def __init__(self, output_path):
        self.scenario = '38_Overhead_Cables'
        self.output_directory = output_path / self.scenario

    @staticmethod
    def filter(road_type: gpd.GeoDataFrame, power_lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts powerline crossings from geodataframes
        Args:
        road_type (gpd.GeoDataFrame)
        power_lines (gpd.GeoDataFrame)

        Returns
        output_gdf (gpd.DataFrame) GeoDataframe with intersections of the power lines with the roads
        """
        return data_acquisition_utils.get_intersection(road_type, power_lines)

    def run(self, config: data_acquisition_utils.Config) -> None:
        """
        Runs filter function and then saves filtered dataframe to .html map, pptx presentation
        and .gpkg geopackage file
        """
        gdf_power_lines = data_acquisition_utils.read_file(config.power_line['processed_file_path'])
        gdf_road_type = data_acquisition_utils.read_file(config.road_type['processed_file_path'])

        gdf_crossings = self.filter(gdf_road_type, gdf_power_lines)

        # Keep only one geometry dtype
        for col in gdf_crossings.columns:
            if col != gdf_crossings.geometry.name and gdf_crossings[col].dtype == 'geometry':
                gdf_crossings[col] = gdf_crossings[col].astype(str)

        if len(gdf_crossings) == 0:
            print("No events found...")
        else:
            comments = []  # TODO: Add comments as required
            self.output_directory.mkdir(parents=True, exist_ok=True)
            data_acquisition_utils.create_map(
                gdf_crossings.drop(['object_timestamp_left', 'object_timestamp_right'], axis=1),
                self.output_directory / f'{self.scenario}.html')
            data_acquisition_utils.create_pptx(gdf_crossings, self.output_directory / f'{self.scenario}.pptx', comments)
            data_acquisition_utils.save_file(gdf_crossings, self.output_directory / f'{self.scenario}.gpkg')

