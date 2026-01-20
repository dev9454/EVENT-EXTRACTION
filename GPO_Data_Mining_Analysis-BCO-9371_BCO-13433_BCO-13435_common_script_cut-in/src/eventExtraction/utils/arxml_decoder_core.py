from bs4 import BeautifulSoup
from lxml import etree
import numpy
import os
import json
import pandas
import warnings
import bitstruct
import re

AUTOSAR_PDU_HEADER_SIZE_BYTES = 8
UNPACK_E2E_AS_LITTLE_ENDIAN = True


def arxml_to_decode_dict(xml_filepath, force_decode=False):
    # Functions to extract relevant information from the arxml for decoding payloads into a human-readable format
    # Extract signals as intermediate JSON to make the necessary params more human-readable
    dict()
    xml_path, xml_name = os.path.split(xml_filepath)
    xml_name, ext = os.path.splitext(xml_name)
    json_path = os.path.join(xml_path, xml_name + "_extract.json")

    # If ARXML is already decoded, skip execution
    if os.path.exists(json_path) and not force_decode:
        with open(json_path, "r") as read_file:
            pdu_description = json.load(read_file)
        return pdu_description

    with open(xml_filepath, "r", encoding='utf-8') as xml_file:
        arxml_tree = BeautifulSoup(xml_file, features="xml")

    # Extract comm details for PDU
    pdu_comm = dict()
    pdu_parse = arxml_tree.find_all('NETWORK-ENDPOINT')
    for ip_conf in pdu_parse:
        short_name = ip_conf.find("SHORT-NAME").get_text()
        if "NEP" in short_name:
            # Only add if Network Endpoint
            pdu_comm[short_name] = dict()
            pdu_comm[short_name]["default_gateway"] = ip_conf.find(
                "DEFAULT-GATEWAY").get_text()
            pdu_comm[short_name]["ipv4_addr"] = ip_conf.find(
                "IPV-4-ADDRESS").get_text()
            pdu_comm[short_name]["network_mask"] = ip_conf.find(
                "NETWORK-MASK").get_text()
            pdu_comm[short_name]["ports"] = dict()

    pdu_parse = arxml_tree.find_all('APPLICATION-ENDPOINT')
    for port_conf in pdu_parse:
        nw_endpoint = port_conf.find(
            "NETWORK-ENDPOINT-REF").get_text().split("/")[-1]
        if nw_endpoint in pdu_comm.keys():
            pdu_comm[nw_endpoint]["ports"][port_conf.find("SHORT-NAME").get_text()] = port_conf.find(
                "PORT-NUMBER").get_text()

    pdu_parse = arxml_tree.find("SO-AD-CONFIG")
    pdu_parse = pdu_parse.find_all("SOCKET-CONNECTION-IPDU-IDENTIFIER")
    pdu_messages = dict()

    # Extract messages - change multiple finds to etree xpath for speed
    for pdu in pdu_parse:
        pdu_name = pdu.find("PDU-TRIGGERING-REF").get_text().split("/")[-1]
        pdu_messages[pdu_name] = dict()
        pdu_messages[pdu_name]["pdu_id"] = int(pdu.find("HEADER-ID").get_text().split(".")[0],
                                               16 if "0x" in pdu.find("HEADER-ID").get_text() else 10)

    # Extract signals for each message
    pdu_parse = arxml_tree.find_all("I-SIGNAL-I-PDU")
    for pdu in pdu_parse:
        pdu_name = pdu.find("SHORT-NAME").get_text()
        if pdu_name in pdu_messages.keys():
            pdu_etree = etree.fromstring(str(pdu))
            pdu_messages[pdu_name]["msg_length"] = int(pdu_etree.xpath('LENGTH')[0].text.split(".")[0],
                                                       16 if "0x" in pdu_etree.xpath('LENGTH')[0].text else 10)
            periodicity_str = pdu_etree.xpath(
                'I-PDU-TIMING-SPECIFICATIONS/I-PDU-TIMING/TRANSMISSION-MODE-DECLARATION/TRANSMISSION-MODE-TRUE-TIMING'
                '/EVENT-CONTROLLED-TIMING/REPETITION-PERIOD/VALUE')[
                0].text
            pdu_messages[pdu_name]["periodicity"] = int(periodicity_str.split(".")[0],
                                                        16 if "0x" in periodicity_str else 10)
            signals_to_pdu_mapping = BeautifulSoup(etree.tostring(pdu_etree.xpath('I-SIGNAL-TO-PDU-MAPPINGS')[0]),
                                                   features="xml").find_all('I-SIGNAL-TO-I-PDU-MAPPING')
            pdu_messages[pdu_name]["signals"] = dict()
            for signal in signals_to_pdu_mapping:
                signal_etree = etree.fromstring(str(signal))
                if signal.find("I-SIGNAL-REF"):
                    signal_name = signal_etree.xpath(
                        "I-SIGNAL-REF")[0].text.split("/")[-1]
                    pdu_messages[pdu_name]["signals"][signal_name] = dict()
                    # Extract signal params
                    pdu_messages[pdu_name]["signals"][signal_name]["packing_byte_order"] = ""
                    if signal.find("PACKING-BYTE-ORDER"):
                        pdu_messages[pdu_name]["signals"][signal_name]["packing_byte_order"] = \
                            signal_etree.xpath("PACKING-BYTE-ORDER")[0].text
                    # Endianness has no meaning for single byte signals and for single bit (boolean) signals
                    pdu_messages[pdu_name]["signals"][signal_name]["start_position"] = int(
                        signal_etree.xpath(
                            "START-POSITION")[0].text.split(".")[0],
                        16 if "0x" in signal_etree.xpath("START-POSITION")[0].text else 10)

    # Parse I-SIGNAL - Extract params for each signal. Once signal dict is made, combine with message dict
    signals_dict = dict()
    pdu_parse = arxml_tree.find_all("I-SIGNAL")
    for signal in pdu_parse:
        signal_etree = etree.fromstring(str(signal))
        short_name = signal_etree.xpath("SHORT-NAME")[0].text
        signals_dict[short_name] = dict()
        signals_dict[short_name]["length"] = int(signal_etree.xpath("LENGTH")[0].text.split(".")[0],
                                                 16 if "0x" in signal_etree.xpath("LENGTH")[0].text else 10)
        signals_dict[short_name]["default_value"] = \
            signal_etree.xpath(
                "INIT-VALUE/NUMERICAL-VALUE-SPECIFICATION/VALUE")[0].text
        signals_dict[short_name]["datatype"] = signal_etree.xpath(
            "NETWORK-REPRESENTATION-PROPS/SW-DATA-DEF-PROPS-VARIANTS/SW-DATA-DEF-PROPS-CONDITIONAL/BASE-TYPE-REF")[
            0].text.split("/")[-1]

    for message in pdu_messages.keys():
        for signal in pdu_messages[message]['signals'].keys():
            pdu_messages[message]['signals'][signal].update(
                signals_dict[signal])

    pdu_description = {"arxml_name": xml_name,
                       "comm": pdu_comm, "decode": pdu_messages}

    with open(json_path, "w") as write_file:
        json.dump(pdu_description, write_file, indent=4)

    return pdu_description


def add_padding(signals_df, idx, padding_start_pos, total_bits_to_pad, offset, pad_idx_8_init, pad_idx_1_init):
    PAD_8BIT = pandas.Series({
        "signal_name": "PAD_8BIT",
        "big_endian": True,
        "start_position": -1,
        "length": 8,
        "default_value": 0,
        "datatype": "uint8"
    })
    PAD_1BIT = pandas.Series({
        "signal_name": "PAD_1BIT",
        "big_endian": True,
        "start_position": -1,
        "length": 1,
        "default_value": 0,
        "datatype": "Boolean"
    })
    bytes_to_pad = numpy.uint(total_bits_to_pad / 8)
    rem_bits_to_pad = numpy.uint(total_bits_to_pad % 8)
    pad_idx_8 = 0
    pad_idx_1 = 0
    if bytes_to_pad > 0:
        pad_idx_8_init = pad_idx_8_init + 1
        for pad_idx_8 in range(0, bytes_to_pad):
            pad = PAD_8BIT.copy()
            pad["signal_name"] = pad["signal_name"] + \
                "_" + str(pad_idx_8 + pad_idx_8_init)
            pad["start_position"] = padding_start_pos + (pad_idx_8 * 8)
            signals_df.loc[idx + offset] = pad
            # Offset df insertion idx by rows added
            offset = offset + 1
    if rem_bits_to_pad > 0:
        pad_idx_1_init = pad_idx_1_init + 1
        for pad_idx_1 in range(0, rem_bits_to_pad):
            pad = PAD_1BIT.copy()
            pad["signal_name"] = pad["signal_name"] + \
                "_" + str(pad_idx_1 + pad_idx_1_init)
            pad["start_position"] = padding_start_pos + \
                (bytes_to_pad * 8) + pad_idx_1
            signals_df.loc[idx + offset] = pad
            # Offset df insertion idx by rows added
            offset = offset + 1
    return pad_idx_8 + pad_idx_8_init, pad_idx_1 + pad_idx_1_init, offset


def convert_to_bitstruct_string(datatype_dict, E2E_Header=True):
    # Apply endianness
    payload_formats = datatype_dict["formats"]
    endianness = datatype_dict["endianness"]
    signal_names = datatype_dict["signal_names"]
    if E2E_Header:
        # Radar messages have an E2E Profile Header in them
        # This header is characterized by signals with the below regex
        # E2E header signals are assumed to be grouped together at the beginning of the message
        #
        LITTLE_ENDIAN_SIGN = ">"
        E2E_header_signals_regex = "^Profile_[0-9]+"
        E2E_header_signals_alt_regex = "^LRRF_Det_.+?_[0-9]+"
        E2E_indices = [idx for idx, signal in enumerate(signal_names)
                       if re.match(E2E_header_signals_regex, signal)
                       or re.match(E2E_header_signals_alt_regex, signal)]
        last_e2e_sig_index = max(E2E_indices) if E2E_indices else 0
        endianness = [LITTLE_ENDIAN_SIGN
                      if idx <= last_e2e_sig_index
                      else endian_value
                      for idx, endian_value in enumerate(endianness)]
    prev_endian = endianness[0]
    prev_idx = 0
    format_str = ""
    bitstruct_fmt_dict = dict()
    bitstruct_fmt_dict["extents"] = list()
    bitstruct_fmt_dict["formats"] = list()
    for idx, val in enumerate(zip(payload_formats, endianness)):
        fmt, endian = val
        if not prev_endian and endian:
            prev_endian = endian
        if endian != prev_endian:
            # Cut the payload, endianness is reversing
            bitstruct_fmt_dict["extents"].append((prev_idx, idx))
            bitstruct_fmt_dict["formats"].append(format_str + prev_endian)
            prev_idx = idx
            prev_endian = endian
            format_str = fmt
        elif endian == prev_endian or not endian:
            # Use same endianness
            format_str = format_str + fmt
    bitstruct_fmt_dict["extents"].append((prev_idx, len(payload_formats)))
    bitstruct_fmt_dict["formats"].append(format_str + prev_endian)
    return bitstruct_fmt_dict


def arxml_dict_to_numpy_decode_def(arxml_filepath, force_decode=False):
    # Function to turn an arxml decode description dictionary into a numpy datatype that can then be applied to
    # decode payloads

    # If decode description exists, retrieve and send it
    pdu_description = arxml_to_decode_dict(arxml_filepath, force_decode)
    xml_path, xml_name = os.path.split(arxml_filepath)
    xml_name, ext = os.path.splitext(xml_name)
    json_path = os.path.join(xml_path, xml_name + "_decode.json")
    if os.path.exists(json_path) and not force_decode:
        with open(json_path, "r") as read_file:
            decode_message_dict = json.load(read_file)
        return decode_message_dict

    # Create decode description if not present
    numpy_datatype_def = {
        "b1": ["Boolean"],
        'u8': ["uint8"],
        'u16': ["uint16"],
        'u32': ["uint32"],
        'u64': ["uint64"],
        'f32': ["float32"],
        's8': ["sint8"],
        's16': ["sint16"],
        's32': ["sint32"],
        's64': ["sint64"]
    }
    decode_message_dict = dict()
    decode_message_dict["arxml_name"] = xml_name
    decode_message_dict["comm"] = dict()
    decode_message_dict["msg"] = dict()
    for message_id in pdu_description["decode"].keys():
        message = pdu_description["decode"][message_id]
        signals_df = pandas.DataFrame(message["signals"]).transpose()
        signals_df = signals_df.sort_values("start_position")
        signals_df = signals_df.reset_index()
        signals_df = signals_df.rename(columns={"index": "signal_name"})
        # Padding check - insert pad if necessary
        start_positions = signals_df["start_position"].to_numpy().astype(
            numpy.uint16)
        signal_lengths = signals_df["length"].to_numpy().astype(numpy.uint16)
        signals_length_sum = sum(signal_lengths)
        message_length_bits = message["msg_length"] * 8
        padding_required = False
        skip_message = False
        if signals_length_sum < message_length_bits:
            # If signals don't add up to msg length, padding is needed
            padding_required = True
        elif signals_length_sum > message_length_bits:
            skip_message = True
            warning_str = "Message \'" + message_id + "\' length (" + \
                          str(message_length_bits) + \
                          " bits) does not match sum of length of signals (" + \
                          str(signals_length_sum) + \
                          " bits). Please double check extract.json. Skipping message..."
            warnings.warn(warning_str)
        if not skip_message and padding_required:
            # Increment offset every time padding is introduced to understand where to insert data in the array
            offset = 0
            skip_message = False
            pad_idx_8 = 0
            pad_idx_1 = 0
            for idx, (start_pos, sig_len) in enumerate(zip(start_positions, signal_lengths)):
                if idx == len(start_positions) - 1:
                    # No padding required
                    continue
                elif start_pos + sig_len == start_positions[idx + 1]:
                    continue
                else:
                    # padding logic
                    padding_start_pos = start_pos + sig_len
                    total_bits_to_pad = start_positions[idx +
                                                        1] - padding_start_pos
                    # Add padding within the message
                    pad_idx_8, pad_idx_1, offset = add_padding(signals_df, idx, padding_start_pos, total_bits_to_pad,
                                                               offset, 0, 0)

            # Check if padding is needed at end of the message
            signal_lengths = signals_df["length"].to_numpy().astype(
                numpy.uint16)
            signals_length_sum = sum(signal_lengths)
            message_length_bits = message["msg_length"] * 8
            if signals_length_sum < message_length_bits:
                # Add padding at end of the message
                end_padding_length_bits = message_length_bits - signals_length_sum
                add_padding(signals_df, len(signals_df), signals_length_sum,
                            end_padding_length_bits,
                            0, pad_idx_8, pad_idx_1)
        # if earlier length check failed, then skip the message while throwing warning
        if skip_message:
            continue
        # Convert dataframe to numpy datatype
        decode_message = dict()
        decode_message["name"] = message_id
        # Create numpy datatype for decode
        bitstruct_datatype = dict()
        bitstruct_datatype['signal_names'] = list()
        bitstruct_datatype['formats'] = list()
        bitstruct_datatype['endianness'] = list()
        for idx, signal in signals_df.iterrows():
            sig_name = signal["signal_name"]
            datatype = signal["datatype"]
            # Endianness Implementation
            if signal["packing_byte_order"] == "MOST-SIGNIFICANT-BYTE-LAST":
                # Explicit Big Endian
                endianness = "<"
            elif signal["packing_byte_order"] == "MOST-SIGNIFICANT-BYTE-FIRST":
                # Explicit Little Endian
                endianness = ">"
            elif signal["packing_byte_order"] == "OPAQUE":
                # Assume Big Endian
                endianness = "<"
            else:
                endianness = ""
            signal_length = signal["length"]
            bitstruct_datatype['signal_names'].append(sig_name)
            # endian_sign = '>' if big_endian else '<'
            bitstruct_dtype = [key for key in numpy_datatype_def.keys(
            ) if datatype in numpy_datatype_def[key]][0]
            bitstruct_datatype['formats'].append(bitstruct_dtype)
            bitstruct_datatype['endianness'].append(endianness)
            if signal_length != bitstruct.calcsize(bitstruct_dtype):
                if "error" in decode_message.keys():
                    if "length_error" not in decode_message["error"].keys():
                        decode_message["error"]["length_error"] = list()
                else:
                    decode_message["error"] = dict()
                    decode_message["error"]["length_error"] = list()
                decode_message["error"]["length_error"].append(sig_name)
            # IMPROVEMENT - Add error logging for endianness swap
        decode_message["extract_msg_len"] = message["msg_length"]
        decode_message["bitstruct_dtype"] = bitstruct_datatype
        decode_message["bitstruct_string"] = convert_to_bitstruct_string(decode_message["bitstruct_dtype"],
                                                                         E2E_Header=UNPACK_E2E_AS_LITTLE_ENDIAN)
        decode_message_dict["msg"][message["pdu_id"]] = decode_message
    # Add IP and Port list to decode_message_dict for extracting the right data for decode
    decode_message_dict["comm"]["ip_filter"] = list(
        set([pdu_description['comm'][comm]['ipv4_addr'] for comm in pdu_description['comm'].keys()]))
    decode_message_dict["comm"]["udp_port_filter"] = list(
        set([port for comm in pdu_description['comm'].keys() for port in
             pdu_description['comm'][comm]['ports'].values()]))
    with open(json_path, "w") as write_file:
        json.dump(decode_message_dict, write_file, indent=4)
    return decode_message_dict


def test_decode_def(decode_desc):
    # Self test to check if the decode type and the extracted message length are the same
    messages = decode_desc["msg"]
    # Add endianness test to make sure coherent datatype can be created. Copy same implementation for decode
    for msg_id in messages.keys():
        msg = messages[msg_id]
        status = "!!!---FAIL---!!!"
        datatype_size = sum([bitstruct.calcsize(fmt)
                            for fmt in msg["bitstruct_string"]["formats"]])
        if (datatype_size == msg['extract_msg_len'] * 8):
            status = 'Pass'
        print("Verifying: " + str(msg_id) + " " + msg["name"] + " | msg_size: " + str(
            msg['extract_msg_len'] * 8) + " | bitstruct_size: " + str(datatype_size) + " | status: " + status)


def decode_payload(payload_numpy_array, timestamps, msg_decode_schema):
    # Function to decode a payload numpy array given a decode schema.
    # Requirement: Numpy array should have defined size in 2 dimensions
    autosar_pdu_header_size = AUTOSAR_PDU_HEADER_SIZE_BYTES
    msg_extract_len_bytes = msg_decode_schema["extract_msg_len"]
    sample_payload_header = payload_numpy_array[0, 0:autosar_pdu_header_size]
    payload_numpy_array = payload_numpy_array[:,
                                              autosar_pdu_header_size:msg_extract_len_bytes + autosar_pdu_header_size]
    bitstruct_decode_def = msg_decode_schema["bitstruct_string"]
    decode_dict = {"header": dict()}
    pdu_id, dlc = decode_autosar_pdu_header(sample_payload_header)
    decode_dict["header"]["pdu_id"] = numpy.array(pdu_id)
    decode_dict["header"]["dlc"] = numpy.array(dlc)
    decode_dict["header"]["time"] = numpy.array(timestamps)
    payload_processed_slice = numpy.uint(0)
    for extent, fmt in zip(bitstruct_decode_def["extents"], bitstruct_decode_def["formats"]):
        # Cut and process payload as slices based on endianness
        datatype_bitstruct = bitstruct.compile(fmt)
        datatype_byte_span = numpy.ceil(
            datatype_bitstruct.calcsize() / 8).astype(numpy.uint)
        payload_slice = payload_numpy_array[:,
                                            payload_processed_slice:datatype_byte_span + payload_processed_slice]
        signal_names = msg_decode_schema["bitstruct_dtype"]["signal_names"][extent[0]:extent[1]]
        decode_result = numpy.apply_along_axis(func1d=decode_bytes_pdu,
                                               axis=1,
                                               arr=payload_slice,
                                               datatype_in=datatype_bitstruct,
                                               signal_names=signal_names)
        payload_processed_slice = payload_processed_slice + datatype_byte_span
        decode_dict.update({k: [] for k in signal_names})
        [decode_dict[k].append(info[k])
         for info in decode_result for k in info.keys()]
    # Preserve header datatype to get header struct in mat file
    decode_dict = {k: numpy.array(
        decode_dict[k]) if k != "header" else decode_dict[k] for k in decode_dict.keys()}
    return decode_dict


def decode_bytes_pdu(payload, datatype_in, signal_names):
    # Decode a payload given a bitstruct datatype and return a dictionary of values
    signal_names = signal_names
    decode_tuple = datatype_in.unpack(payload.tobytes())
    signal_decode_dict = {name: value for name,
                          value in zip(signal_names, decode_tuple)}
    return signal_decode_dict


def create_query_from_decode_schema(load_decode_schema):
    query = "(((IP_src in (\'" + "\',\'".join(
        load_decode_schema["comm"]["ip_filter"]) + "\')) or (IP_dst in (\'" + "\',\'".join(
        load_decode_schema["comm"]["ip_filter"]) + "\'))) and ((UDP_src_port in (" + ",".join(
        load_decode_schema["comm"]["udp_port_filter"]) + ")) or (UDP_dst_port in (" + ",".join(
        load_decode_schema["comm"]["udp_port_filter"]) + "))))"
    return query


def decode_autosar_pdu_header(data_bytes):
    PDU_ID = hex(numpy.frombuffer(
        buffer=data_bytes[0:4], dtype=numpy.dtype(">u4"))[0])
    DLC = numpy.frombuffer(buffer=data_bytes[4:8], dtype=numpy.dtype(">u4"))[0]
    return PDU_ID, DLC


def decode_eth_channel_by_arxml(dataframe, arxml_path):
    # Load all required data
    # Modify this function as required to adapt with data from matgen
    # Required params - filters for IP/UDP, timestamps, UDP payloads, UDP lengths
    # Required params and dataframe struct borrowed from stream_stats_timing.py
    load_decode_schema = arxml_dict_to_numpy_decode_def(arxml_path)
    query = create_query_from_decode_schema(load_decode_schema)
    autosar_pdu_filter_df = dataframe.query(query).copy()
    autosar_pdu_filter_df = autosar_pdu_filter_df.loc[:, :'UDP_Payload']
    autosar_pdu_header_size = AUTOSAR_PDU_HEADER_SIZE_BYTES
    autosar_pdu_filter_df["autosar_pdu"], autosar_pdu_filter_df["DLC"] = zip(*autosar_pdu_filter_df.apply(
        lambda x: decode_autosar_pdu_header(x["UDP_Payload"]), axis=1))
    autosar_pdu_df_dict = {pdu_id: autosar_pdu_filter_df[autosar_pdu_filter_df["autosar_pdu"] == pdu_id] for pdu_id in
                           autosar_pdu_filter_df["autosar_pdu"].unique()}
    pdu_decode = dict()
    pdu_decode["ARXML"] = load_decode_schema["arxml_name"]
    for pdu_id in autosar_pdu_df_dict.keys():
        # IMPORTANT: Need to segregate data by individual PDUs before feeding to decoder.
        # Decoder expects constant payload length
        # This function covers some typical error checks on the payload
        # Loop for data from each PDU
        payloads = [
            payload for payload in autosar_pdu_df_dict[pdu_id]['UDP_Payload']]
        payloads = numpy.array(payloads)
        time = autosar_pdu_df_dict[pdu_id]['timestamps'].to_numpy()
        payload_len = autosar_pdu_df_dict[pdu_id]["DLC"].to_numpy()
        pdu_id_int = numpy.uint(int(pdu_id, 16))

        # Throw warning if no decode schema present
        if str(pdu_id_int) in load_decode_schema['msg'].keys():
            pdu_decode_schema = load_decode_schema['msg'][str(pdu_id_int)]
        else:
            warning_str = "No definition found for message: " + \
                          str(pdu_id_int) + " " + \
                          load_decode_schema['msg'][str(pdu_id_int)] + \
                          " Skipping decode..."
            warnings.warn(warning_str)
            pdu_decode[str(pdu_id_int)] = warning_str
            continue

        # Throw warnings if size checks are not consistent. Multiplexing is not supported
        if len(set(payload_len)) > 1:
            warning_str = "Multiple DLCs (" + "bytes, ".join([str(sz) for sz in set(set(payload_len))]) + \
                          ") found for message: " + str(pdu_id_int) + " " + pdu_decode_schema["name"] + \
                          ". Skipping decode..."
            warnings.warn(warning_str)
            pdu_decode[pdu_decode_schema["name"]] = warning_str
            continue

        pdu_dlc = max(payload_len)
        msg_extract_len_bytes = pdu_decode_schema["extract_msg_len"]

        # Check if UDP length is greater than expected payload length - if not throw warning and skip message
        if msg_extract_len_bytes != pdu_dlc:
            warning_str = "Expected payload DLC mismatch for message: " + str(pdu_id_int) + " " + \
                          pdu_decode_schema["name"] + " [ARXML DLC: " + str(msg_extract_len_bytes) + ", Payload DLC: " \
                          + str(pdu_dlc) \
                          + "]. Skipping decode..."
            warnings.warn(warning_str)
            pdu_decode[pdu_decode_schema["name"]] = warning_str
            continue

        # Decode logic
        pdu_decode[pdu_decode_schema["name"]] = decode_payload(
            payloads, time, pdu_decode_schema)
    return pdu_decode


if __name__ == '__main__':
    # COMMENT BEFORE INTEGRATION
    # MULTIPLE FILES
    arxml_folder = r"arxmls"
    for xml_filename in os.listdir(arxml_folder):
        arxml_filename = os.path.join(arxml_folder, xml_filename)
        filepath, filename = os.path.split(arxml_filename)
        [base_name, extension] = os.path.splitext(filename)
        if extension == ".arxml" or extension == ".ARXML":
            print("Decoding: "+filename)
            print("\n\n")
            bitstruct_decode_desc = arxml_dict_to_numpy_decode_def(
                arxml_filename)
            test_decode_def(bitstruct_decode_desc)
            print("\n\n")

    # SINGLE FILE - Use these 2 lines to process an ARXML and create a extract and decode json
    # bitstruct_decode_desc = arxml_dict_to_numpy_decode_def(r"arxmls\ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml", force_decode=True)
    # Below is a built in self test to test if the created bitstruct definitions have the same data length as the payload - to catch missing bytes/pads
    # test_decode_def(bitstruct_decode_desc)
