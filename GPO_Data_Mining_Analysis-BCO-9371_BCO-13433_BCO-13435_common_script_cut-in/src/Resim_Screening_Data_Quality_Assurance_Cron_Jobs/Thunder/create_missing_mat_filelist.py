import os
import multiprocessing



def create_matFileslist(input_file,is_dext=False,single_op_dir=False,output_dir=None):
    
    output_file = input_file.replace("_MF4_", "_Mat_", 1) + "_dma"
    if single_op_dir:
        output_file = os.path.join(output_dir, os.path.basename(output_file))
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
                else:
                    readme_file = os.path.join(os.path.dirname(output_file), 'Missing_Matfile_list')
                    with open(readme_file, 'a+') as readme:
                        readme.write(f'{mat_path}\n')


if __name__ == '__main__':
    input_files = {
    "KX_RWUP" : "/mnt/usmidet/projects/STLA-THUNDER/7-Tools/ResimScreening/filelists_by_data_set/KX_W5U_RWUP_AEB_NA/KX_W5U_RWUP_AEB_NA_MF4_flat",
    "WL_Mule" : "/mnt/usmidet/projects/STLA-THUNDER/7-Tools/ResimScreening/filelists_by_data_set/WL_MULE_RWUP_AEB_NA/WL_MULE_RWUP_AEB_NA_MF4_flat"
    }
        
    is_dext = False
    single_op_dir = True

    input_file_list = list(input_files.values())
    # Output Directory to be updated only if single_op_dir is True
    output_dir  = "/mnt/usmidet/projects/STLA-THUNDER/2-Sim/USER_DATA/ioiny8/Tasks/Comparison_output/"

    pool = multiprocessing.Pool()
    pool.starmap(create_matFileslist, [(input_file, is_dext, single_op_dir, output_dir) for input_file in input_file_list])
    pool.close()
    pool.join()
    # process_file(input_file_list[0])

    # Merge the filelists
    # Find all files in output_dir that ends with _dma
    # Merge all files into a single file
    def merge_filelists(output_dir):
        filelists = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('_dma')]
        with open(os.path.join(output_dir, 'merged_filelist'), 'w') as merged_filelist:
            for filelist in filelists:
                with open(filelist, 'r') as file:
                    merged_filelist.write(file.read()) 


        
        

