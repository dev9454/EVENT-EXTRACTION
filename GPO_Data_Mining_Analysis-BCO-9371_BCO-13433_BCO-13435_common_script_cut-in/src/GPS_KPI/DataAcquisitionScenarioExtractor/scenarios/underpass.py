from pathlib import Path
from data_acquisition_utils import Config, read_file, create_map, create_pptx, save_file, get_intersection
import geopandas as gpd
import numpy as np
import shapely
import pandas as pd

class Underpass:
    """
    Class that extracts underpasses from OSM geopackage file.
    Underpasses are roads under bridges or short tunnels
    """

    def __init__(self, output_path: Path) -> None:
        self.output_directory = output_path / "23_Underpass"

    def filter(self, bridge: gpd.GeoDataFrame, tunnel: gpd.GeoDataFrame, road_type: gpd.GeoDataFrame ) -> gpd.GeoDataFrame:
        """
        Extracts underpasses rows from geodataframe

        Args:
        bridge (gpd.DataFrame) Dataframe containing bridges
        tunnel (gpd.DataFrame) Dataframe containing tunnels
        road_type (gpd.DataFrame) Dataframe containing road_types

        Returns
        output_gdf (gpd.DataFrame) Dataframe with underpass filtered rows
        """

        road_type = road_type[road_type.access.isin([np.nan, ""])]
        bridges = bridge[bridge["length"] > 0.02]

        #Tunnels filtering
        tunnels = tunnel[tunnel.access.isin([np.nan, ""])]
        tunnels = tunnel[tunnel["length"] <= 0.05]

        #Road with bridge intersection filtering
        intersections = get_intersection(road_type, bridges)

        #Exclude intersections of roads with themselves
        intersections = intersections[intersections["el_id_left"] != intersections["el_id_right"]].reset_index(
            drop=True)

        intersections["road_start"] = shapely.get_point(intersections['geometry'], 0)
        intersections["road_end"] = shapely.get_point(intersections['geometry'], -1)
        intersections["bridge_start"] = shapely.get_point(intersections['b_geometry'], 0)
        intersections["bridge_end"] = shapely.get_point(intersections['b_geometry'], -1)

        #Exclude cases where a road becomes a bridge and vice versa
        intersections = intersections[(intersections["road_end"] != intersections["bridge_start"]) & (
                    intersections["bridge_end"] != intersections["road_start"]) & (
                                                  intersections["road_end"] != intersections["bridge_end"]) & (
                                                  intersections["road_start"] != intersections["bridge_start"])]

        #Select intersections with only one bridge
        intersections_one_bridge = intersections[
            ~((intersections["bridge_left"] == "yes") & (intersections["bridge_right"] == "yes"))].copy()

        #Select intersections of two bridges
        intersections_two_bridge = intersections[
            (intersections["bridge_left"] == "yes") & (intersections["bridge_right"] == "yes")].copy()


        #Assigning the lower bridge as a geometry
        intersections_two_bridge["geometry"] = np.where(
            intersections_two_bridge["layer_left"] < intersections_two_bridge["layer_right"],
            intersections_two_bridge["geometry"],
            intersections_two_bridge["b_geometry"]
        )


        overpass = pd.concat([intersections_one_bridge, intersections_two_bridge], ignore_index=True)
        overpass.columns = overpass.columns.str.replace("_left", "")
        output_gdf = pd.concat([overpass, tunnels], ignore_index=True)

        return output_gdf.set_geometry("geometry")

    def run(self, config: Config) -> None:
        """
        Runs filter function and then saves filtered dataframe to .html map, pptx presentation
        and .gpkg geopackage file
        """
        self.config = config
        tunnel = read_file(config.tunnel['processed_merged_file_path'])
        bridge = read_file(config.bridges['processed_file_path'])
        road_type = read_file(config.road_type['processed_file_path'])

        underpass = self.filter(bridge,tunnel,road_type)

        if len(underpass) == 0:
            print("No events found...")
            return
        else:
            # Keep only one geometry dtype
            for col in underpass.columns:
                if col != underpass.geometry.name and underpass[col].dtype == 'geometry':
                    underpass[col] = underpass[col].astype(str)
            comments = ['length', 'highway', 'surface', 'lanes', 'maxspeed']
            self.output_directory.mkdir(parents=True, exist_ok=True)
            create_map(underpass, self.output_directory / '23_Underpass.html')
            create_pptx(underpass, self.output_directory / '23_Underpass.pptx', comments)
            save_file(underpass, self.output_directory / '23_Underpass.gpkg')
