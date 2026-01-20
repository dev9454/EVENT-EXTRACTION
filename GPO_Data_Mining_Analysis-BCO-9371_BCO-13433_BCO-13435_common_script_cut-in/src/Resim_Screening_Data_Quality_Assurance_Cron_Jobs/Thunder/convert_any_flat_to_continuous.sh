#!/bin/bash

source /mnt/usmidet/projects/STLA-THUNDER/7-Tools/hpc_tools_py_venv/bin/activate

temp_flatList="/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/July_19_2024/WL_MULE_RWUP_HAS_LC_NA/WL_MULE_RWUP_HAS_LC_NA_Mat_flat_dma
/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/July_19_2024/WL_MULE_RWUP_AEB_NA/WL_MULE_RWUP_AEB_NA_Mat_flat_dma
/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/July_19_2024/LB_RWUP_AEB_NA/LB_RWUP_AEB_NA_Mat_flat_dma
/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/July_19_2024/KX_W5U_RWUP_AEB_NA/KX_W5U_RWUP_AEB_NA_Mat_flat_dma"

temp_cont_list=$(echo "$temp_flatList" | sed 's/flat/cont/g')

echo "Converting flat file list to continuous file list..."
for file in $temp_flatList; do
    python /mnt/usmidet/users/mjxk2s/STLA_THUNDER/Scripts/resim_filelist_util.py -l "$file" -o "$(echo "$file" | sed 's/flat/cont/g')" -op convert_cont --log_number_string_size 4
done

echo "Conversion complete. Continuous file created at $temp_cont_list."

