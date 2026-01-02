
import struct
import numpy as np
import os

BIN_DIR = 'frontend/piston_viewer/tiles_bin'

def analyze_bins():
    files = [f for f in os.listdir(BIN_DIR) if f.endswith('.bin')]
    print(f"{'Filename':<25} | {'Base':>8} | {'MinAbs':>8} | {'AvgAbs':>8} | {'MinRel':>8}")
    print("-" * 60)
    for f in sorted(files):
        path = os.path.join(BIN_DIR, f)
        with open(path, 'rb') as fb:
            data = fb.read()
        if len(data) < 4: continue

        base = struct.unpack('<f', data[:4])[0]
        header_size = 4
        min_abs = None
        avg_abs = None

        if (len(data) - 12) % 14 == 0:
            header_size = 12
            min_abs = struct.unpack('<f', data[4:8])[0]
            avg_abs = struct.unpack('<f', data[8:12])[0]
        elif (len(data) - 4) % 14 == 0:
            header_size = 4
        else:
            continue

        hex_data = np.frombuffer(data[header_size:], dtype=np.half)
        
        if len(hex_data) == 0: continue
        
        # In the bin format, every 7th float16 is z.
        # [z, n0, n1, n2, n3, n4, n5]
        z_values = hex_data[::7]
        
        min_rel = z_values.min()
        abs_min = base + min_rel
        if min_abs is None:
            min_abs = abs_min
        if avg_abs is None:
            avg_abs = base + float(z_values.mean())

        print(f"{f:<25} | {base:>8.1f} | {min_abs:>8.1f} | {avg_abs:>8.1f} | {min_rel:>8.1f}")

if __name__ == "__main__":
    analyze_bins()
