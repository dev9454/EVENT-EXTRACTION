import multiprocessing
import sys
import os

def create_matFileslist(input_file,is_dext=False):
    """
    Creates a dma matgen thunder output mat files list based on the input MF4 File list, if the files exist.
    """
    output_file = input_file.replace("_MF4_", "_Mat_", 1) + "_dma"
    # Empty output file if it already exists
    open(output_file, 'w').close()
    # Processed in batches. Change the batch size if needed.
    with open(input_file, 'r') as file:
        while True:
            batch = file.read()
            if not batch:
                break
            new_paths = []
            for line in batch.splitlines():
                if '_bus_' in line and not is_dext:
                    new_paths.append(line.replace('_bus_', '_dma_').replace('.MF4', '.mat'))
                else:
                    new_paths.append(line.replace('.MF4', '.mat'))
            for mat_path in new_paths:
                if os.path.exists(mat_path):
                    with open(output_file, 'a') as output:
                        output.write('')
                        output.write(mat_path + '\n')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-i':
        input_files = {
            "Input_File_Name": sys.argv[2]
        }
    else:
        input_files = {
            "Chicago_Overall" : "/mnt/usmidet/projects/STLA-THUNDER/8-Users/AlgoGroup/Filelist/Chicago_dataset_July_Aug_2024/Chicago_Data_MF4_NoSWFilter"
            # "Boston_Overall": "/mnt/usmidet/projects/STLA-THUNDER/8-Users/AlgoGroup/Filelist/EastCoast_dataset/MF4_Filelist_Flat_NoSWVerFilter"
            # "Mule_AEB" : "/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/July_24_2024/WL_MULE_RWUP_AEB_NA/WL_MULE_RWUP_AEB_NA_MF4_flat",
            # "HAS_LC" : "/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/July_24_2024/WL_MULE_RWUP_HAS_LC_NA/WL_MULE_RWUP_HAS_LC_NA_MF4_flat",

        }

    is_dext = False

    input_file_list = list(input_files.values())
    pool = multiprocessing.Pool()
    pool.map(create_matFileslist, input_file_list)
    pool.close()
    pool.join()
    # process_file(input_file_list[0])

#   To make continuous filelist
#   source /mnt/usmidet/projects/STLA-THUNDER/7-Tools/hpc_tools_py_venv/bin/activate
#   python /mnt/usmidet/users/mjxk2s/STLA_THUNDER/Scripts/resim_filelist_util.py -l temp_flatList -o temp_cont_list -op convert_cont  --log_number_string_size 4


        
        

