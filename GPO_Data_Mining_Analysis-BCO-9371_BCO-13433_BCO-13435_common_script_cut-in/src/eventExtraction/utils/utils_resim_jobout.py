# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:56:06 2024

@author: mfixlz
"""

import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def resim_jobout_df(radar_path: os.path.join,
                    vision_path: os.path.join,
                    radar_vision_aurix_path: os.path.join,
                    radar_aurix_path: os.path.join = None,
                    req_col_keyword: str = 'Radar_vision_Aurix',
                    ):

    df_vision = pd.read_excel(vision_path, sheet_name='Output')
    df_radar_aurix = pd.read_excel(radar_aurix_path, sheet_name='Output')
    df_radar = pd.read_excel(radar_path, sheet_name='Output')
    df_radar_vision_aurix = pd.read_excel(
        radar_vision_aurix_path, sheet_name='Output')
    df = pd.concat([df_radar,
                    df_radar_aurix,
                    df_radar_vision_aurix,
                    df_vision], axis=0, ignore_index=True)

    assert set(['Resim_status', 'rtag',
                'Resim_log_path', 'Resim_log_name',
                'Resim_exit_code', 'Error_msg',
                'log_path', 'log_name',
                'Resim_type',]).issubset(set(df.columns)), \
        'check whether excel files have all needed columns'

    df_orig = df.copy(deep=True)

    df = df.pivot_table(values=['Resim_status', 'rtag',
                                'Resim_log_path', 'Resim_log_name',
                                'Resim_exit_code', 'Error_msg'],
                        index=['log_path', 'log_name', ],
                        columns=['Resim_type', ], aggfunc=lambda x: x)

    df.columns = df.columns.map('_'.join).str.strip('_')

    considered_cols = [col for col in df.columns
                       if 'rtag' in col or 'Resim_log' in col]

    drop_cols = [col for col in considered_cols if req_col_keyword not in col]
    keep_cols = [col for col in considered_cols if req_col_keyword in col]

    df.columns = df.columns.str.replace('Resim_status_', '')
    col_rename_dict = {col: col.replace(
        '_'+req_col_keyword, '') for col in keep_cols}

    df = df.drop(columns=drop_cols).rename(mapper=col_rename_dict,
                                           axis=1)

    df = df.reset_index()
    df.loc[df.query('Radar == "Failed"').index,
           'Error_msg_Radar_Aurix'] = 'NA'

    df.loc[df.query('Radar == "Failed"').index,
           'Error_msg_Radar_vision_Aurix'] = 'NA'

    # df3_3.loc[df3_3.query('Radar_Aurix == "Failed"').index,'Error_msg_Radar_vision_Aurix'] = np.nan

    df.loc[df.query('Radar == "Failed"').index,
           'Resim_exit_code_Radar_Aurix'] = 'NA'

    df.loc[df.query('Radar == "Failed"').index,
           'Resim_exit_code_Radar_vision_Aurix'] = 'NA'

    # df3_3.loc[df3_3.query('Radar_Aurix == "Failed"').index,'Resim_exit_code_Radar_vision_Aurix'] = np.nan

    df.loc[df.query('Vision == "Failed"').index,
           'Error_msg_Radar_vision_Aurix'] = 'NA'

    df.loc[df.query('Vision == "Failed"').index,
           'Resim_exit_code_Radar_vision_Aurix'] = 'NA'

    return df, df_orig


def create_mysql_engine():
    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@{ip}/{db}".format(user="aptiv_db_algo_team",
                                                       pw="DKwVd5Le",
                                                       ip="10.192.229.101",
                                                       db="aptiv_production",

                                                       ), echo=False
    )

    if isinstance(engine, str):
        print('Connection with database cannot be established')
        sys.exit(-1)
    else:
        return engine


if __name__ == '__main__':
    import time
    from functools import reduce
    import psutil
    import warnings
    import contextlib
    import joblib
    from joblib import Parallel, delayed, parallel_backend, parallel_config
    from tqdm import tqdm
    import scipy as sp
    warnings.filterwarnings("ignore")

    def secondsToStr(t):
        return "%d:%02d:%02d.%03d" % \
            reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                   [(t*1000,), 1000, 60, 60])

    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss, mem_info.vms

    root_path = os.path.join(r'C:\Users\mfixlz\Downloads')
    vision_path = os.path.join(root_path, 'Vision_output_3.xlsx')
    radar_aurix_path = os.path.join(root_path, 'Radar_Aurix_output_3.xlsx')
    radar_path = os.path.join(root_path, 'Radar_output_3.xlsx')
    radar_vision_aurix_path = os.path.join(
        root_path, 'Radar_vision_Aurix_output_3.xlsx')
    req_col_keyword = 'Radar_vision_Aurix'

    table_name = 'resim_jobout_mining_sample_3'

    start_time = time.time()
    mem_before_phy, mem_before_virtual = process_memory()

    df, df_orig = resim_jobout_df(radar_path=radar_path,
                                  vision_path=vision_path,
                                  radar_vision_aurix_path=radar_vision_aurix_path,
                                  radar_aurix_path=radar_aurix_path,
                                  req_col_keyword=req_col_keyword
                                  )

    # engine = create_mysql_engine()

    # table_write_out = df.to_sql(name=table_name,
    #                             con=engine,
    #                             index=False,
    #                             if_exists='append'
    #                             )
    # with engine.connect() as conn:
    #     count_ = pd.read_sql(
    #         f"select count(*) from aptiv_production.{table_name}", engine)

    mem_after_phy, mem_after_virtual = process_memory()

    end_time = time.time()

    elapsed_time = secondsToStr(end_time-start_time)
    consumed_memory_phy = (mem_after_phy - mem_before_phy)*1E-6
    consumed_memory_virtual = (
        mem_after_virtual - mem_before_virtual)*1E-6

    print(
        f'&&&&&&&&&&&& Elapsed time is {elapsed_time} %%%%%%%%%%%%%%%%')
    print(
        '&&&&&&&&&&&& Consumed physical memory MB is ',
        f'{consumed_memory_phy} %%%%%%%%%%%%%%%%')

    print(
        '&&&&&&&&&&&& Consumed virtual memory MB is ',
        f'{consumed_memory_virtual} %%%%%%%%%%%%%%%%')
