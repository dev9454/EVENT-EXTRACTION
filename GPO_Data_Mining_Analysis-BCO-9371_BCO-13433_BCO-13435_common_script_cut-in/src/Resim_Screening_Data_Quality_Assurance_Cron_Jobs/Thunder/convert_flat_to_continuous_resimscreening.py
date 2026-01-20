import os
from datetime import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

year = 2024
month = "March"
start_date = 26
end_date = 31
create_backup = False

parallel_execution = True

if end_date < start_date:
    raise ValueError("End date must be greater than start date.")

directory = "/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/"


def process_directory(i):
    temp_flatList = os.path.join(directory, f"{month}_{i}_{year}",'Delta','KX_W5U_RWUP_AEB_NA_Mat_flat_delta_dma')
    temp_cont_list = os.path.join(directory, f"{month}_{i}_{year}",'Delta','KX_W5U_RWUP_AEB_NA_Mat_cont_delta_dma')
    command = f"/mnt/usmidet/users/mjxk2s/STLA_THUNDER/Scripts/resim_filelist_util.py -l {temp_flatList} -o {temp_cont_list} -op convert_cont --log_number_string_size 4"
    subprocess.run(command, shell=True)

if parallel_execution : 
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(start_date, end_date):
            futures.append(executor.submit(process_directory, i))
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update(1)
else : 
    process_directory(start_date)