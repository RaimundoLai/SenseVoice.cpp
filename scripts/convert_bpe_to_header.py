#!/usr/bin/env python3
"""
Convert BPE model binary file to C++ header.
Usage: python convert_bpe_to_header.py <input_file> <output_header>
"""

import sys

def convert_to_header(input_path: str, output_path: str):
    with open(input_path, 'rb') as f:
        data = f.read()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("// Auto-generated. Do NOT edit!\n")
        f.write("// Generated from: chn_jpn_yue_eng_ko_spectok.bpe.model\n")
        f.write("#ifndef SENSE_VOICE_BPE_DATA_H\n")
        f.write("#define SENSE_VOICE_BPE_DATA_H\n\n")
        f.write("#include <cstddef>\n\n")
        f.write(f"const size_t SENSE_VOICE_BPE_DATA_LENGTH = {len(data)};\n\n")
        f.write("const unsigned char SENSE_VOICE_BPE_DATA[] = {\n")
        
        # Write data as hex bytes, 16 per line
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_values = ', '.join(f'0x{b:02X}' for b in chunk)
            if i + 16 < len(data):
                f.write(f"    {hex_values},\n")
            else:
                f.write(f"    {hex_values}\n")
        
        f.write("};\n\n")
        f.write("#endif // SENSE_VOICE_BPE_DATA_H\n")
    
    print(f"Generated {output_path} ({len(data)} bytes)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_header>")
        sys.exit(1)
    
    convert_to_header(sys.argv[1], sys.argv[2])
