import numpy as np
import pandas


class IPHeader:
    def __init__(self, data_bytes):
        # IP Header is the first 20 bytes of eth payload
        ip_header = data_bytes
        self.version = ip_header[0] >> 4
        # length of IP header is measured in 32 bit words - 32 bit = 4 bytes
        self.header_length = (ip_header[0] & 0xf)*4
        self.tos = ip_header[1]
        self.total_length = ip_header[2:4].view(np.dtype(">H"))[0]
        self.identification = ip_header[4:6].view(np.dtype(">H"))[0]
        self.flags = ip_header[6:8].view(np.dtype(">H"))[0] >> 13
        self.fragment_offset = self.flags & 0x1fff
        self.ttl = ip_header[8]
        self.protocol = ip_header[9]
        self.header_checksum = ip_header[10:12].view(np.dtype(">H"))[0]
        self.source_address = ip_header[12:16].astype(np.uint8)
        self.source_address_str = ".".join(str(byte) for byte in self.source_address)
        self.destination_address = ip_header[16:20].astype(np.uint8)
        self.destination_address_str = ".".join(str(byte) for byte in self.destination_address)
        pass



class UDPHeader:
    def __init__(self, data_bytes):
        # UDP header is 8 bytes after IP header
        udp_header = data_bytes.view(np.dtype(">H"))
        self.source_port = udp_header[0]
        self.destination_port = udp_header[1]
        self.length = udp_header[2]
        self.checksum = udp_header[3]
        pass

class SPIHeader:
    def __init__(self, data_bytes):
        # Send first 13 bytes of SPI payload
        spi_header = data_bytes.view(np.dtype("B"))
        #Throw first 6 bytes
        spi_header = spi_header[6:]
        self.single_check = True if not spi_header[0] else False
        self.frame_num = spi_header[0]
        self.spi_appid = 0
        self.spi_size = 0
        self.spi_transmit_id = 0
        if self.single_check:
            # Single Frame
            self.spi_appid = spi_header[2]
            self.spi_size = spi_header[1]
        elif self.frame_num == 1:
            # First frame of multi frame
            self.spi_transmit_id = spi_header[1]
            self.spi_size = np.frombuffer(spi_header[2:6], dtype=np.dtype("u4"))[0]
            self.spi_appid = spi_header[6]
        else:
            # Consecutive Frames of Multi Frame
            self.spi_transmit_id = spi_header[1]
        pass

    def return_spi_header_info(self):
        return (self.single_check,
        self.frame_num,
        self.spi_appid,
        self.spi_size,
        self.spi_transmit_id)

class AutosarPDUHeader:
    def __init__(self, data_bytes):
        self.pdu_header_id = hex(np.frombuffer(data_bytes, dtype=np.dtype(">u4"))[0])


