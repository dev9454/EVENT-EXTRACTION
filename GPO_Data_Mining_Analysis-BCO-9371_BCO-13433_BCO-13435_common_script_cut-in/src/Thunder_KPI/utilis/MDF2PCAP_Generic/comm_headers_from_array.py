import numpy as np


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
