# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:11:58 2025

@author: mfixlz
"""
from collections import defaultdict
import os
from xml.etree import cElementTree as ElementTree
from decord import cpu, gpu
from decord import VideoReader
import numpy as np
import pandas as pd

import cv2  # still used to save images out

from moviepy import VideoFileClip

print('Entering Image extraction')


def crop_image(frame, ratio=0.5):

    frame_borders = (ratio*frame.shape[0],
                     ratio*frame.shape[1])

    frame_square = \
        frame.copy()[
            : int(frame_borders[0]),
            int(frame_borders[1]/2):
            int(frame.shape[1]/2) +
            int(frame_borders[1]/2),
            :]

    return frame_square


def _rect_2_square(frame, crop_ratio):

    Border_column = (max(frame.shape[:2]) -
                     min(frame.shape[:2]))//2
    frame_square = \
        frame.copy()[
            :min(frame.shape[:2]),
            Border_column:
            min(frame.shape[:2]) + Border_column,
            :] \
        if frame.shape[0] < frame.shape[1]\
        else \
        frame.copy()[
            Border_column:
            min(frame.shape[:2] + Border_column),
            :min(frame.shape[:2]),
            :]

    if crop_ratio != 1:

        return crop_image(frame_square, crop_ratio)

    else:

        return frame_square


def extract_frames(video_path, frames_dir, crop_ratio,
                   overwrite=False, start=-1,
                   end=-1, every=1):
    """
    Extract frames from a video using decord's VideoReader
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """
    # OS (Windows) compatible path
    # video_path = os.path.normpath(video_path)

    # OS (Windows) compatible path
    # frames_dir = os.path.normpath(frames_dir)

    # get the video path and filename from the path
    video_dir, video_filename = os.path.split(video_path)

    assert os.path.exists(video_path)
    # assert the video file exists

    # load the VideoReader
    # can set to cpu or gpu. ctx=gpu(0)
    vr = VideoReader(video_path, ctx=cpu(0))

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the
        # end of the video
        end = len(vr)

    frames_list = list(range(start, end, every))
    saved_count = 0

    # this is faster for every > 25 frames and can \
    # fit in memory
    if every > 25 and len(frames_list) < 1000:
        frames = vr.get_batch(frames_list).asnumpy()

        # lets loop through the frames until the end
        for index, frame in zip(frames_list, frames):

            # create the save path
            save_path = os.path.join(frames_dir,
                                     # video_filename,
                                     "{:05d}.jpg".format(index)
                                     )

            # if it doesn't exist or we want to
            # overwrite anyways
            if not os.path.exists(save_path) or overwrite:

                # save the extracted image
                frame_square = _rect_2_square(frame, crop_ratio)

                # save the extracted image
                cv2.imwrite(save_path,
                            cv2.cvtColor(frame_square,
                                         cv2.COLOR_RGB2BGR))
                saved_count += 1  # increment our
                # counter by one

    else:  # this is faster for every <25 and consumes
        # small memory

        # lets loop through the frames until the end
        for index in range(start, end):
            frame = vr[index].asnumpy()  # read an image from
            # the capture

            # if this is a frame we want to write out based
            # on the 'every' argument
            if index % every == 0:

                # create the save path
                save_path = os.path.join(frames_dir,
                                         # video_filename,
                                         "{:05d}.jpg".format(index)
                                         )

                # if it doesn't exist or we want to
                # overwrite anyways
                if not os.path.exists(save_path) \
                        or overwrite:

                    frame_square = _rect_2_square(frame, crop_ratio)

                    # save the extracted image
                    cv2.imwrite(save_path,
                                cv2.cvtColor(frame_square,
                                             cv2.COLOR_RGB2BGR))
                    saved_count += 1
                    # increment our counter by one

    # and return the count of the images we saved
    return saved_count


def video_to_frames(video_path, frames_dir,
                    overwrite=False, every=1, crop_ratio=1):
    """
    Extracts the frames from a video
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :return: path to the directory where the frames 
            were saved, 
            or None if fails
    """
    # OS (Windows) compatible path
    video_path = os.path.normpath(video_path)

    # OS (Windows) compatible path
    frames_dir = os.path.normpath(frames_dir)

    # get the video path and filename from the path
    video_dir, video_filename = os.path.split(video_path)

    # make directory to save frames, its a sub dir in the
    # frames_dir with the video name
    os.makedirs(os.path.join(frames_dir,
                             # video_filename
                             ),
                exist_ok=True)

    print("Extracting frames from {}".format(video_filename))

    # let's now extract the frames
    Image_count = extract_frames(video_path,
                                 frames_dir, crop_ratio,
                                 every=every)

    print('####################################')

    # when done return the directory containing the frames
    return (Image_count, os.path.join(frames_dir,
                                      # video_filename
                                      ))


def _video_to_images(video_path,
                     output_frames_dir,
                     req_fps=None,
                     image_extension: str = 'jpg'
                     ):

    video_dir, video_filename = os.path.split(video_path)

    print("Extracting frames from {}".format(video_filename))

    if not image_extension in ['jpeg', 'png', 'jpg']:

        image_extension = 'jpg'

    name_format = os.path.join(output_frames_dir,
                               'frame%04d.' + image_extension)

    os.makedirs(os.path.join(output_frames_dir,

                             ),
                exist_ok=True)

    clip_obj = VideoFileClip(video_path)

    fps = clip_obj.fps

    if req_fps is not None and req_fps > fps:

        req_fps = fps

    output_image_list = clip_obj.write_images_sequence(
        name_format=name_format,
        fps=req_fps)

    Image_count = len(output_image_list)

    return (Image_count, output_frames_dir)


def _mapping_frames_avi_cTimes(cTime_array,
                               frame_count,
                               log_path,
                               xml_files_path: str,
                               image_extension: str = 'jpg',
                               point_of_flip: float = 0.5
                               ):

    if point_of_flip > 0.75 or point_of_flip < 0:

        point_of_flip = 0.5

    def find_closest(first_cTime_array,
                     second_cTime_array):
        insertion_indices = np.searchsorted(
            0.5*(second_cTime_array[1:]
                 + second_cTime_array[:-1]),
            first_cTime_array,
            side='right')
        return second_cTime_array[insertion_indices]

    ideal_cTime_grid = np.linspace(cTime_array[0],
                                   cTime_array[-1],
                                   frame_count)

    frame_time_array = find_closest(ideal_cTime_grid, cTime_array)

    mapping_dict = {f'{image_num:04}': time_
                    for image_num, time_ in
                    zip(range(frame_count), frame_time_array)
                    }

    xml_files_path = [os.path.join(xml_files_path, item)
                      for item in os.listdir(xml_files_path)
                      if item.endswith('.xml')]

    # print(xml_files_path)

    list_of_dict = []

    cTime_for_xml_file_list = []
    cTime_indices_for_xml_file_list = []
    categories_xml_file_list = []
    extra_comments_xml_file_list = []

    for xml_file in xml_files_path:

        print(xml_file)
        out_dict = xml_etree_to_dict(xml_file)['annotation']

        cTime_for_xml_file = mapping_dict[
            out_dict['filename'].split('.')[0]]
        cTime_for_xml_file_list.append(cTime_for_xml_file)

        cTime_indices_for_xml_file = int(out_dict['filename'].split('.')[0])
        cTime_indices_for_xml_file_list.append(cTime_indices_for_xml_file)

        if isinstance(out_dict['object'], list):

            categories_xml_file = [item['name']
                                   for item in out_dict['object']
                                   ]

            extra_comments_xml_file = [item['extra']
                                       for item in out_dict['object']
                                       ]
        else:

            categories_xml_file = [out_dict['object']['name']]
            extra_comments_xml_file = [out_dict['object']['extra']]

        categories_xml_file_list.append(categories_xml_file)

        extra_comments_xml_file_list.append(extra_comments_xml_file)

        # list_of_dict.append(out_dict)

    log_path_only, log_name = os.path.split(log_path)

    out_table_dict = {
        'log_path': log_path_only,
        'log_name': log_name,
        'cTime': cTime_for_xml_file_list,
        'cTime_index': cTime_indices_for_xml_file_list,
        'category': categories_xml_file_list,
        'comments': extra_comments_xml_file_list,
    }

    print(out_table_dict)

    out_table_df = pd.DataFrame(out_table_dict)

    out_table_df = out_table_df.explode(['category',
                                         'comments'],
                                        ignore_index=True)

    return out_table_df


def xml_etree_to_dict(xml_file_path):

    tree = ElementTree.parse(xml_file_path)
    root = tree.getroot()

    def _main(root):

        out_dict = {root.tag: {} if root.attrib else None}
        children = list(root)
        if children:
            dd = defaultdict(list)
            for dc in map(_main, children):
                for k, v in dc.items():
                    dd[k].append(v)
            out_dict = {root.tag: {k: v[0]
                                   if len(v) == 1
                                   else v
                                   for k, v in dd.items()}}

            # out_dict = {root.tag: {k: v
            #                        for k, v in dd.items()}
            #             }
        if root.attrib:
            out_dict[root.tag].update(('@' + k, v)
                                      for k, v in root.attrib.items())
        if root.text:
            text = root.text.strip()
            if children or root.attrib:
                if text:
                    out_dict[root.tag]['#text'] = text
            else:
                out_dict[root.tag] = text

        return out_dict

    return_value = _main(root)

    return return_value


if __name__ == '__main__':
    # test it
    import warnings
    import os
    from pathlib import Path
    warnings.filterwarnings("ignore")

    import time
    from functools import reduce
    import psutil
    from tqdm import tqdm

    def secondsToStr(t):
        return "%d:%02d:%02d.%03d" % \
            reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                   [(t*1000,), 1000, 60, 60])

    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss, mem_info.vms

    start_time = time.time()
    mem_before_phy, mem_before_virtual = process_memory()

    parent_path = r'C:\Users\mfixlz\OneDrive - Aptiv\Documents\DM_A\PO_Vinay\Projects\2025\BCO-14362'
    video_absolute_path = os.path.join(parent_path, 'data', 'video',
                                       )
    video_extensions = ['webm', 'mp4', 'avi', 'tavi']
    videos_list = [os.path.join(video_absolute_path, file)
                   for file in os.listdir(video_absolute_path)
                   if os.path.split(file)[-1].split('.')[-1] in video_extensions
                   ]

    image_output_dir = [os.path.join(Path(file).parent.parent,
                                     'images3',
                                     os.path.split(file)[-1].split('.')[0])
                        for file in videos_list]

    data_out_list = []
    # for video_path, image_dir in tqdm(zip(videos_list,
    #                                       image_output_dir),
    #                                   desc='Video to image progress'):
    # for video_path, image_dir in zip(videos_list,
    #                                  image_output_dir):

    #     data_out = video_to_frames(video_path=video_path,
    #                                frames_dir=image_dir, overwrite=False,
    #                                every=1)
    #     data_out_list.append(data_out)

    # for video_path, image_dir in zip(videos_list,
    #                                  image_output_dir):

    #     data_out = _video_to_images(video_path,
    #                                 image_dir,
    #                                 )
    #     data_out_list.append(data_out)

    cTime_array = np.linspace(0, 1, 1200)
    frame_count = 1800
    xml_files_path = r"C:\Users\mfixlz\OneDrive - Aptiv\Documents\DM_A\PO_Vinay\Projects\2025\BCO-14362\data\images2\TNDR1_ASHN_20240320_180140_WDC5_v01_0001"
    log_path = r"C:\Users\mfixlz\Downloads\TNDR1_DUCK_20240412_112956_WDC4_dma_0010.mat"

    out_table_df = _mapping_frames_avi_cTimes(cTime_array,
                                              frame_count,
                                              log_path,
                                              xml_files_path
                                              )

    mem_after_phy, mem_after_virtual = process_memory()

    end_time = time.time()

    elapsed_time = secondsToStr(end_time-start_time)
    consumed_memory_phy = (mem_after_phy - mem_before_phy)*1E-6
    consumed_memory_virtual = (
        mem_after_virtual - mem_before_virtual)*1E-6

    print(
        f'&&&&&&&&&&&& Elapsed time is {elapsed_time} %%%%%%%%%%%%%%%%')
    print(
        f'&&&&&&&&&&&& Consumed physical memory MB is {consumed_memory_phy} %%%%%%%%%%%%%%%%')

    print(
        f'&&&&&&&&&&&& Consumed virtual memory MB is {consumed_memory_virtual} %%%%%%%%%%%%%%%%')
