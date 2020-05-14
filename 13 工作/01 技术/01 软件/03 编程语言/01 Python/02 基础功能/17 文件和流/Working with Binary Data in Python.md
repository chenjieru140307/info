
# Working with Binary Data in Python







## Hexadecimal

```
# Starting with a hex string you can unhexlify it to bytes
deadbeef = binascii.unhexlify('DEADBEEF')
print(deadbeef)

# Given raw bytes, get an ASCII string representing the hex values
hex_data = binascii.hexlify(b'\x00\xff')  # Two bytes values 0 and 255

# The resulting value will be an ASCII string but it will be a bytes type
# It may be necessary to decode it to a regular string
text_string = hex_data.decode('utf-8')  # Result is string "00ff"
print(text_string)
```





## Format Strings

Format strings can be helpful to visualize or output byte values. Format strings require an integer value so the byte will have to be converted to an integer first.

```
a_byte = b'\xff'  # 255
i = ord(a_byte)   # Get the integer value of the byte

bin = "{0:b}".format(i) # binary: 11111111
hex = "{0:x}".format(i) # hexadecimal: ff
oct = "{0:o}".format(i) # octal: 377

print(bin)
print(hex)
print(oct)
```





## Bitwise Operations

```
# Some bytes to play with
byte1 = int('11110000', 2)  # 240
byte2 = int('00001111', 2)  # 15
byte3 = int('01010101', 2)  # 85

# Ones Complement (Flip the bits)
print(~byte1)

# AND
print(byte1 & byte2)

# OR
print(byte1 | byte2)

# XOR
print(byte1 ^ byte3)

# Shifting right will lose the right-most bit
print(byte2 >> 3)

# Shifting left will add a 0 bit on the right side
print(byte2 << 1)

# See if a single bit is set
bit_mask = int('00000001', 2)  # Bit 1
print(bit_mask & byte1)  # Is bit set in byte1?
print(bit_mask & byte2)  # Is bit set in byte2?
```





## Struct Packing and Unpacking

Packing and unpacking requires a string that defines how the binary data is structured. It needs to know which bytes represent values. It needs to know whether the entire set of bytes represets characters or if it is a sequence of 4-byte integers. It can be structured in any number of ways. The format strings can be simple or complex. In this example I am packing a single four-byte integer followed by two characters. The letters **i** and **c** represent integers and characters.

```
import struct

# Packing values to bytes
# The first parameter is the format string. Here it specifies the data is structured
# with a single four-byte integer followed by two characters.
# The rest of the parameters are the values for each item in order
binary_data = struct.pack("icc", 8499000, b'A', b'Z')
print(binary_data)

# When unpacking, you receive a tuple of all data in the same order
tuple_of_data = struct.unpack("icc", binary_data)
print(tuple_of_data)

# For more information on format strings and endiannes, refer to
# https://docs.Python.org/3.5/library/struct.html
```





## System Byte Order

You might need to know what byte order your system uses. Byte order refers to big endian or little endian. The **sys** module can provide that value.

```
# Find out what byte order your system uses
import sys
print("Native byteorder: ", sys.byteorder)
```





## Examples





```
# diff.py - Do two files match?
# Exercise: Rewrite this code to compare the files part at a time so it
# will not run out of RAM with large files.
import sys

with open(sys.argv[1], 'rb') as file1, open(sys.argv[2], 'rb') as file2:
    data1 = file1.read()
    data2 = file2.read()

if data1 != data2:
    print("Files do not match.")
else:
    print("Files match.")
```





```
#is_jpeg.py - Does the file have a JPEG binary signature?

import sys
import binascii

jpeg_signatures = [
    binascii.unhexlify(b'FFD8FFD8'),
    binascii.unhexlify(b'FFD8FFE0'),
    binascii.unhexlify(b'FFD8FFE1')
]

with open(sys.argv[1], 'rb') as file:
    first_four_bytes = file.read(4)

    if first_four_bytes in jpeg_signatures:
        print("JPEG detected.")
    else:
        print("File does not look like a JPEG.")
```





```
# read_boot_sector.py - Inspect the first 512 bytes of a file

import sys

in_file = open(sys.argv[1], 'rb')  # Provide a path to disk or ISO image
chunk_size = 512
data = in_file.read(chunk_size)
print(data)
```





```
# find_ascii_in_binary.py - Identify ASCII characters in binary files

import sys
from functools import partial

chunk_size = 1
with open(sys.argv[1], 'rb') as in_file:
    for data in iter(partial(in_file.read, chunk_size), b''):
        x = int.from_bytes(data, byteorder='big')
        if (x > 64 and x < 91) or (x > 96 and x < 123) :
            sys.stdout.write(chr(x))
        else:
            sys.stdout.write('.')
```





```
# create_stego_zip_jpg.py - Hide a zip file inside a JPEG

import sys

# Start with a jpeg file
jpg_file = open(sys.argv[1], 'rb')  # Path to JPEG
jpg_data = jpg_file.read()
jpg_file.close()

# And the zip file to embed in the jpeg
zip_file = open(sys.argv[2], 'rb')  # Path to ZIP file
zip_data = zip_file.read()
zip_file.close()

# Combine the files
out_file = open('special_image.jpg', 'wb')  # Output file
out_file.write(jpg_data)
out_file.write(zip_data)
out_file.close()

# The resulting output file appears like a normal jpeg but can also be
# unzipped and used as an archive.
```





```
# extract_pngs.py
# Extract PNGs from a file and put them in a pngs/ directory
import sys
with open(sys.argv[1], "rb") as binary_file:
    binary_file.seek(0, 2)  # Seek the end
    num_bytes = binary_file.tell()  # Get the file size

    count = 0
    for i in range(num_bytes):
        binary_file.seek(i)
        eight_bytes = binary_file.read(8)
        if eight_bytes == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a":  # PNG signature
            count += 1
            print("Found PNG Signature #" + str(count) + " at " + str(i))

            # Next four bytes after signature is the IHDR with the length
            png_size_bytes = binary_file.read(4)
            png_size = int.from_bytes(png_size_bytes, byteorder='little', signed=False)

            # Go back to beginning of image file and extract full thing
            binary_file.seek(i)
            # Read the size of image plus the signature
            png_data = binary_file.read(png_size + 8)
            with open("pngs/" + str(i) + ".png", "wb") as outfile:
                outfile.write(png_data)
```


# 相关

- [Working with Binary Data in Python](https://www.devdungeon.com/content/working-binary-data-Python)
