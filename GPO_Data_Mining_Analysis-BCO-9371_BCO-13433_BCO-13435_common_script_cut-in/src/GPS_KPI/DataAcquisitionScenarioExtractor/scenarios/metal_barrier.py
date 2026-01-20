'''@author  : wjb0cf
    
'''

import geopandas as gpd
from pathlib  import Path
import sys
from data_acquisition_utils import Config, read_file, create_map, create_pptx, save_file
from .poles_guardrails import PolesGuardrails
 
class MetalBarrier:
    """
    Class that extracts metal barriers from OSM geopackage file
    """

    def __init__(self, output_path: Path) -> None:
        self.scenario = '37_Metal_Barrier'
        self.output_directory = output_path / self.scenario

    def run(self, config) -> None:
        """
        Executes the filtering process then saves filtered dataframe to .html map, pptx presentation
        and .gpkg geopackage file

        Args:
        config: Configuration object containing paths and settings.
        """
        
        road_type = read_file(config.road_type['processed_file_path'])
        barrier = read_file(config.barrier['processed_file_path'])
        tags = ['cable_barrier','fence', 'guard_rail','chain']

        filtered = PolesGuardrails.filter_guardrail(road_type, barrier, tags)
        if len(filtered) == 0:
            print("No events found...")
        else:
            # Keep only one geometry dtype
            for col in filtered.columns:
                if col != filtered.geometry.name and filtered[col].dtype == 'geometry':
                    filtered[col] = filtered[col].astype(str)
            self.output_directory.mkdir(parents=True, exist_ok=True)      
                   
            comments = ['covered', 'highway', 'incline', 'lanes', 'layer']
            
            create_map(filtered, self.output_directory / f'{self.scenario}.html')
            save_file(filtered, self.output_directory / f'{self.scenario}.gpkg')
            create_pptx(filtered, self.output_directory / f'{self.scenario}.pptx', comments)

