# Healforce SCP-ECG
Decoder for Healforce SCP-ECG files

Healforce portable electrocardiogram ECG devices export SCP-ECG files which are not fully compliant to the EN 1064 standard.
The rhythm data is not huffman encoded, multiple leads data is interleaved in one data stream instead of separate streams, and the two MSB of each 10-bit sample have to be mapped to get regular 16-bit signed integers.

This script decodes the ECG data of a .SCP file into numpy arrays.
See [demo.ipynb](demo.ipynb) for example usage.

The decoder is based on experience with data from a Healforce Prince 180D device.
Data from other devices may not be compatible.