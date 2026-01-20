'''@author  : Wiktor Wiernasiewicz
    
'''

import geopandas as gpd
from pathlib import Path
import data_acquisition_utils
import sys


class PedestrianCyclistCrossing:
    

    def __init__(self, output_path):
        self.scenario = '40_Pedestrian_Cyclist_Crossing'
        self.output_directory = Path(output_path) / self.scenario

    @staticmethod
    def filter(road_type, crossing):
        """
        Filter function extract crossing scenarios that intersects with road_type. The scenarios contain roads that are accessible, it means access tag is empty.
        :param road_type: GeoDataFrame as road_type
        :param crossing: GeoDataFrame as crossing
        :return: Filtered GeoDataFrame with crossing scenarios 
        """
   
        road_type = road_type[road_type['access:lanes'] == '']

        return data_acquisition_utils.get_intersection(road_type,crossing)
       


    def run(self, config) -> None:
        """
        Executes the filtering process, creates map, saves files in HTML, GPKG, and PPTX formats.

        Args:
        config: Configuration object containing paths and settings.
        """
        
        crossing = data_acquisition_utils.read_file(config.crossing['processed_file_path'])
        road_type = data_acquisition_utils.read_file(config.road_type['processed_file_path'])
        

        filtered = self.filter(road_type, crossing)
        if len(filtered) == 0:
            print("No events found...")
        else:
            # Keep only one geometry dtype
            for col in filtered.columns:
                if col != filtered.geometry.name and filtered[col].dtype == 'geometry':
                    filtered[col] = filtered[col].astype(str)
            self.output_directory.mkdir(parents=True, exist_ok=True)      
                   
            comments = [] # TODO: Add comments as required
            
            data_acquisition_utils.create_map(filtered, self.output_directory / f'{self.scenario}.html')
            data_acquisition_utils.create_pptx(filtered, self.output_directory / f'{self.scenario}.pptx', comments)
            data_acquisition_utils.save_file(filtered, self.output_directory / f'{self.scenario}.gpkg')
