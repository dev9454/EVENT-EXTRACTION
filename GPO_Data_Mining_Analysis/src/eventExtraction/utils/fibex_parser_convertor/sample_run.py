# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:52:37 2024

@author: mfixlz
"""

from dlt.header.header import DltHeader
from dlt.payload.payload import DltPayload
from scipy.io import loadmat as load_mat_scipy
import os
import sys
import numpy as np
import xml.etree.ElementTree


import ctypes
import re
from io import IOBase


if __package__ is None:

    print('Here at none package 1')
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    actual_package_path = to_change_path
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, \n to change 1: {to_change_path}')

    from fibex_parser import FibexParser
    from configuration_to_text import SimpleConfigurationFactory

else:

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    actual_package_path = to_change_path
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, to change 1: {to_change_path}')

    from fibex_parser import FibexParser
    from configuration_to_text import SimpleConfigurationFactory

conf_factory = SimpleConfigurationFactory()
req_file_path_2 = os.path.join(
    r"C:\Users\mfixlz\Downloads\standard_download676a7db2d75ce_13131\examples\flexray\FlexRay_Cluster_Example.xml")
req_file_path = os.path.join(r"C:\Users\mfixlz\Downloads\ip_next\ip_next.xml")

stream_def_path = os.path.join(
    r"C:\Users\mfixlz\Downloads\ip_next\ip_next_str_def.txt")

ecu_name_replacement = None
plugin_file = None
fibex_parser_obj = FibexParser(plugin_file, ecu_name_replacement)
verbose = True


tree = xml.etree.ElementTree.parse(req_file_path)

root = tree.getroot()


fibex_parser_obj.parse_file(conf_factory, req_file_path, verbose=verbose)


all_frames = root.findall('.//fx:FRAMES/fx:FRAME', fibex_parser_obj.__ns__)

all_frames_id_order_dict = {frame.attrib['ID']: idx
                            for idx, frame in enumerate(all_frames)}


frame_data_pairs_3 = {(all_frames_id_order_dict[frame.__id__],
                       frame.__id__,
                       frame.name()):
                      [(pdu_key.pdu().__id__,
                        pdu_key.pdu().__short_name__,
                          pdu_key.pdu().__byte_length__,
                        pdu_key.pdu().signal_instances_sorted_by_bit_position()[
                          0].__signal__.__name__
                        if len(pdu_key.pdu().signal_instances_sorted_by_bit_position()) > 0
                        else None)
                       for pdu_key in frame.pdu_instances().values()]
                      for frame in fibex_parser_obj.__frames__.values()}


all_frames_length_dict = {frame.attrib['ID']: int(frame.find('./fx:BYTE-LENGTH',
                                                             fibex_parser_obj.__ns__).text) for frame in all_frames}

total_data_length = np.sum(list(all_frames_length_dict.values()))


# data_type_mapping = {
#     'UINT8': 'B',
#     'UINT16': 'H',
#     'UINT32': 'I',
#     'UINT64': 'Q',
#     'SINT8': 'b',
#     'SINT16': 'h',
#     'SINT32': 'i',
#     'SINT64': 'q',
#     'FLOA16': 'f2',
#     'FLOA32': 'f',
#     'FLOA64': 'd',
#     'STRG_UTF8': 'U',
# }

data_type_mapping = {
    'UINT8': np.uint8,
    'UINT16': np.uint16,
    'UINT32': np.uint32,
    'UINT64': np.uint64,
    'SINT8': np.int8,
    'SINT16': np.int16,
    'SINT32': np.int32,
    'SINT64': np.int64,
    'FLOA16': np.float16,
    'FLOA32': np.float32,
    'FLOA64': np.float64,
    'STRG_UTF8': np.str_,
}


message_ID_value_pairs = dict()


for key, val in frame_data_pairs_3.items():
    message_ID = int(key[1][3:])
    message_ID_value_pairs[message_ID] = []
    for item in val:

        if item[3] is not None:

            message_ID_value_pairs[message_ID].append((f'{key[2]}.{item[1]}',
                                                       data_type_mapping[item[3][2:]]
                                                       ))

        else:
            continue


req_ID = 150010


def _create_buffer_dtype(req_ID,
                         message_ID_value_pairs,
                         endianness: str = '<'):

    req_value_data = message_ID_value_pairs[req_ID]

    # req_dtype_list = [endianness + dtype[1] for dtype in req_value_data]

    # complete_dtype = ','.join(req_dtype_list)

    complete_dtype = req_value_data

    return complete_dtype


UEH = 0b00000001    # use extended header
MSBF = 0b00000010   # True => Payload BigEndian else Payload LittleEndian
WEID = 0b00000100   # with ECU ID
WSID = 0b00001000   # with Session ID
WTMS = 0b00010000   # with timestamp


class ArrayUploader(IOBase):
    # set this up as a child of IOBase because boto3 wants an object
    # with a read method.
    def __init__(self, array):
        # get the number of bytes from the name of the data type
        # this is a kludge; make sure it works for your case
        dbits = re.search('\d+', str(np.dtype(array.dtype))).group(0)
        dbytes = int(dbits) // 8
        self.nbytes = array.size * dbytes
        self.bufferview = (ctypes.c_char*(self.nbytes)
                           ).from_address(array.ctypes.data)
        self._pos = 0

    def tell(self):
        return self._pos

    def seek(self, pos):
        self._pos = pos

    def read(self, size=-1):
        if size == -1:
            return self.bufferview.raw[self._pos:]
        old = self._pos
        self._pos += size
        return self.bufferview.raw[old:self._pos]


head_obj = DltHeader(
    # msbf = False
)

kwargs_load_mat_scipy = {  # 'struct_as_record' : False,
    # 'squeeze_me' : True,
    'simplify_cells': True,
    'verify_compressed_data_integrity': True
}
data_150010 = load_mat_scipy(os.path.join(r"C:\Users\mfixlz\Downloads\fibex_parser_convertor\150010_t_1.mat"),
                             **kwargs_load_mat_scipy
                             )

yyy = data_150010['yyy']
endianness_dict = {0: False,  # Little Endian
                   1: True  # Big Endian
                   }

header_type = bin(yyy[0])[2:]

# from Log and Trace Protocol Specification AUTOSAR FO R20-11
# endianness = int(header_type[::-1][1]) #big endian like in binary conversion
endianness = int(header_type[1])  # little endian like in binary conversion

header_data = head_obj.create_from(f=ArrayUploader(yyy),
                                   msbf=endianness_dict[endianness])


print('%s', header_data)

buffer_dtype = _create_buffer_dtype(req_ID, message_ID_value_pairs)
# payload_obj = DltPayload.create_from(ArrayUploader(yyy), header=header_data)
# print('%s', payload_obj)
decoded_data_150010 = np.frombuffer(yyy[16:].tobytes(),
                                    dtype=buffer_dtype)

decoded_data_150010 = decoded_data_150010[0]
req_names = [name[0] for name in message_ID_value_pairs[req_ID]]
decoded_data_dict = {key: val for key,
                     val in zip(req_names, decoded_data_150010)}

# text_file_data = []

# for key, val in frame_data_pairs_3.items():
#     for item in val:
#         # text_data_frame = f'{key[2]}.{item[1]} {item[2]} \n'
#         if item[3] is not None:
#             text_data_frame = f'{item[3]} {key[2]}.{item[1]} \n'
#         else:
#             continue
#         text_file_data.append(text_data_frame)


# text_file_data_string = ''.join(text_file_data)


# text_file_data_string = str(total_data_length) + '\n' + text_file_data_string
