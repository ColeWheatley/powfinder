
import struct
import numpy as np
import os

BIN_DIR = 'frontend/piston_viewer/tiles_bin'

def analyze_bins():
    files = [f for f in os.listdir(BIN_DIR) if f.endswith('.bin')]
    print(f"{'Filename':<25} | {'Base':>8} | {'MinRel':>8} | {'AbsMin':>8}")
    print("-" * 60)
    for f in sorted(files):
        path = os.path.join(BIN_DIR, f)
        with open(path, 'rb') as fb:
            data = fb.read()
        if len(data) < 4: continue
        
        base = struct.unpack('f', data[:4])[0]
        hex_data = np.frombuffer(data[4:], dtype=np.half)
        
        if len(hex_data) == 0: continue
        
        # In the bin format, every 7th float16 is z.
        # [z, n0, n1, n2, n3, n4, n5]
        z_values = hex_data[::7]
        
        min_rel = z_values.min()
        abs_min = base + min_rel
        
        print(f"{f:<25} | {base:>8.1f} | {min_rel:>8.1f} | {abs_min:>8.1f}")

if __name__ == "__main__":
    analyze_bins()
