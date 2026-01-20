'''@author  : Wiktor Wiernasiewicz
    
'''

import geopandas as gpd
from pathlib  import Path
import sys
from data_acquisition_utils import Config, read_file, create_map, create_pptx, save_file
from .poles_guardrails import PolesGuardrails
 
class ConcreteBarrier:
    

    def __init__(self, output_path):
        self.scenario = '39_Concrete_Barrier'
        self.output_directory = Path(output_path) / self.scenario

  

    def run(self, config) -> None:
        """
        Executes the filtering process, creates map, saves files in HTML, GPKG, and PPTX formats.

        Args:
        config: Configuration object containing paths and settings.
        """
        
        # road_type = read_file(config.road_type['processed_file_path'])
        # barrier = read_file(config.barrier['processed_file_path'])
        # tags = ['city_wall','wall', 'jersey_barrier']

        # filtered = PolesGuardrails.filter_guardrail(road_type, barrier, tags)
        # self.output_directory.mkdir(parents=True, exist_ok=True)      
               
        # comments = ['covered', 'highway', 'incline', 'lanes', 'layer']
        
        # create_map(filtered, self.output_directory / f'{self.scenario}.html')
        # create_pptx(filtered, self.output_directory / f'{self.scenario}.pptx', comments)
        # save_file(filtered, self.output_directory / f'{self.scenario}.gpkg')