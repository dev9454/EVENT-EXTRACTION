from pathlib import Path
from data_acquisition_utils import Config, read_file, create_map, create_pptx, save_file
from .bridge import Bridge
import sys
import geopandas as gpd


class MetalBridge:
    """
    Class that extracts metal bridges from OSM geopackage file or from National Bridge Inventory for US
    """

    def __init__(self, output_path: Path) -> None:
        self.output_directory = output_path / "59_66_Feature_Metal_Bridge"

    def filter(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts metal bridges rows from geodataframe

        Args:
        gdf (gpd.DataFrame)

        Returns
        output_gdf (gpd.DataFrame) Dataframe with HOV_lane filtered rows
        """
        region = self.config.region

        # To be implemented
        # if region == "US":
        #    pass

        output_gdf = gdf[gdf["surface"] == 'metal']

        return output_gdf

    def run(self, config: Config) -> None:
        """
        Runs filter function and then saves filtered dataframe to .html map, pptx presentation
        and .gpkg geopackage file
        """
        self.config = config
        bridge = read_file(config.bridges['processed_merged_file_path'])

        # Extract raw bridges from Bridge class
        bridge_filtered = Bridge.filter(bridge)

        metal_bridges = self.filter(bridge_filtered)

        if len(metal_bridges) == 0:
            print("No events found...")
        else:
            comments = ['length', 'highway', 'surface', 'lanes', 'maxspeed']
            self.output_directory.mkdir(parents=True, exist_ok=True)
            create_map(metal_bridges, self.output_directory / '59_66_Feature_Metal_Bridge.html')
            create_pptx(metal_bridges, self.output_directory / '59_66_Feature_Metal_Bridge.pptx', comments)
            save_file(metal_bridges, self.output_directory / '59_66_Feature_Metal_Bridge.gpkg')
