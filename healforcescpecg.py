#!/usr/bin/env python3

from struct import unpack, unpack_from
from copy import copy
import numpy as np
#from crcmod.predefined import mkCrcFun

class Section:
    def __init__(self, buf):
        self.read(buf)

    def read(self, buf):
        header = unpack('<HHIBB6s', buf[:16])
        self.crc = header[0]
        # crc preset xmodem
        # quirk: bytes may be swapped
        self.id = header[1]
        self.size = header[2]
        self.section_version = header[3]
        self.protocol_version = header[4]
        self.reserved = header[5]
        self.data = buf[16:]
        ## ignore crc
        #crc16 = mkCrcFun('xmodem')
        #self.crc_calc = crc16(buf[2:])
        #print(self.id, hex(self.crc), hex(self.crc_calc))

    def __str__(self):
        data = copy(self.__dict__)
        return self.__class__.__name__ + str(data)


class HealforceSCPECG:
    """ decodes SCP-ECG files from Healforce devices """
    def __init__(self, filename, read_metadata_only=False):
        with open(filename, 'rb') as f:
            crc, _, filesize = unpack('<HHH', f.read(6))
            # quirk: filesize is only 16 bits
            # quirk: crc does not seem to be crc-ccitt
            self._read_section_0(f)
            for section_id in self.section_offsets:
                f.seek(self.section_offsets[section_id])
                buf = f.read(self.section_sizes[section_id])
                if section_id == 1:
                    self._read_section_1(buf)
                # section 2 contains a short huffman table, but the data is not huffman encoded
                if section_id == 3:
                    self._read_section_3(buf)
                    if read_metadata_only:
                        break
                if section_id == 6:
                    self._read_section_6(buf, num_leads=1)
                    # quirk: number of rhythm data streams is always 1,
                    #        >1 streams are interleaved
                if section_id == 9:
                    self._read_section_9(buf)
                else:
                    pass
        if not read_metadata_only:
            self._decompress_healforce_coding()


    def _decompress_healforce_coding(self):
        """
        the rhythm data consists of 12 bit samples (10 bit signed integer + 2 bit beat marker)
        4 12-bit-samples are squeezed into 3 16-bit-words
        """
        sample_bytes = self.compressed_samples[0]
        n_quadruples = len(sample_bytes) // (2*3) # stored in 3 16-bit-words
        samples = np.zeros((n_quadruples,4), dtype=np.int16)
        samples[:,0:3] = np.ndarray(shape=(n_quadruples,3), dtype='<i2', buffer=sample_bytes)
        # for the 4th sample, take 4 bits from each of previous 3 words
        samples[:,3] = ((samples[:,0] & 0x3C00) >> 2) + ((samples[:,1] & 0x3C00) >> 6) + ((samples[:,2] & 0x3C00) >> 10)
        samples = samples.reshape((-1,self.num_leads))
        n_samples = len(samples)
        # heart beat (QRS) detected by hardware
        self.beats = (samples & 0x8000) >> 15
        # irregular heart beat detected by hardware - details are in section 9
        self.marked_beats = (samples & 0x4000) >> 14
        # this converts the 10-bit data into regular 16-bit signed integers
        #  the high byte of the 10-bit data indicate sign or rather range of the sample
        samples = (((((samples & 0x0300) >> 8) + 0xFE) & 0xFF) << 8) + (samples & 0xFF)
        # samples are quantized in 0.01 mV steps, i.e. 100 ~= 1 mV
        self.samples = samples
        

    def _read_section_0(self, f):
        """ SCP ECG Header """
        buf = f.read(8)
        size = unpack('<HHI', buf)[2]
        buf += f.read(size - 8)
        s = Section(buf)
        offset = 0
        self.section_offsets = {}
        self.section_sizes = {}
        while offset < len(s.data):
            entry = unpack_from('<HII', s.data, offset)
            section_id = entry[0]
            self.section_sizes[section_id] = entry[1]
            self.section_offsets[section_id] = entry[2]
            offset += 10


    def _read_section_1(self, buf):
        """ Patient Data """
        s = Section(buf)
        offset = 0
        self.patient_info = {}
        while offset < len(s.data):
            entry = unpack('<BH', s.data[offset:offset+3])
            tag = entry[0]
            size = entry[1]
            if size == 0:
                break
            data = s.data[offset+3:offset+3+size]
            if tag == 2:
                self.patient_info['id'] = data.decode(encoding="utf-8", errors="replace").strip('\x00')
            elif tag == 25:
                self.patient_info['startdate'] = unpack('>HBB', data)
            elif tag == 26:
                self.patient_info['starttime'] = unpack('>BBB', data)
            else:
                self.patient_info[str(tag)] = data
            offset += 3+size


    def _read_section_3(self, buf):
        """ ECG lead definition """
        s = Section(buf)
        self.num_leads = s.data[0]
        self.sample_number_start = unpack('>I', s.data[2:6])
        self.sample_number_end = unpack('>I', s.data[2:6])
        offset = 0


    def _read_section_6(self, buf, num_leads=1):
        """ Rhythm data """
        s = Section(buf)
        header = unpack('<HHBB', s.data[:6])
        self.amplitude_value_multiplier = header[0]
        self.sample_time_interval = header[1]
        self.difference_data_used = header[2]
        self.bimodal_compression_used = header[3]
        compressed_offset = 6+(num_leads*2)
        self.compressed_sizes = unpack('<'+('H'*num_leads), s.data[6:compressed_offset])
        self.compressed_samples = []
        offset = 0
        for size in self.compressed_sizes:
            data = s.data[compressed_offset+offset:compressed_offset+offset+size]
            assert(len(data) == size)
            self.compressed_samples.append(data)
            offset += size


    def _read_section_9(self, buf):
        """ Event data """
        self.low_freq_heart_rate = list(unpack('<HHHHHH', buf[0x44:0x44+12]))
        self.heart_rate = []
        self.other = [] # not yet known
        self.flags = []
        self.irregular_beat_markers = []

        t = unpack('<IHHBBH', buf[0x10:0x1c])
        if t[0] != 0x20000 or t[1] != 0x200:
            print("unusual section 9 (header)")
            print(hex(t[0]), hex(t[1]), t[5])

        page = t[2]
        self.irregular_beat_detected = (t[5] == 0)
        #heart_beats_list_size = t[4]

        offset = 0
        last_heart_rate = 0
        while 0x134+offset+4 < len(buf):
            t = unpack('<BBBB', buf[0x134+offset:0x134+offset+4])

            # averaged heart rate
            #  if you want the beat-to-beat heart rate,
            #  use data in section 6
            last_heart_rate = t[0]
            if last_heart_rate == 0xff:
                break
            self.heart_rate.append(last_heart_rate)

            # not known what this indicates
            self.other.append(t[1])

            # irregular beat markers
            m = t[3]
            # remapping same event types
            if m == 0x0e:
                m = 8
            elif m == 0x0f:
                m = 9
            elif m == 0x13:
                m = 10
            elif m == 0x14:
                m = 11
            self.irregular_beat_markers.append(m)

            # heart beat flags
            f = t[2]
            self.flags.append(f)
            # no idea what most flags mean
            # 0x20 and 0x80 seem to indicate motion

            offset += 4
