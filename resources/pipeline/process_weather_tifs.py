import os
import subprocess

# --- Configuration ---
INPUT_TIF_FILES = ['placeholder1.tif', 'placeholder2.tif']  # Example placeholder files
OUTPUT_DIR_25M = './tiles_25m'
OUTPUT_DIR_100M = './tiles_100m'
TARGET_EPSG = 'EPSG:3857'  # Web Mercator

# --- Helper Functions ---
def run_command(command):
    """Runs a shell command and checks for errors."""
    print(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: The command '{command[0]}' was not found. Please ensure GDAL is installed and in your PATH.")
        return False

def create_directory_if_not_exists(directory_path):
    """Creates a directory if it doesn't already exist."""
    if not os.path.exists(directory_path):
        print(f"Creating directory: {directory_path}")
        os.makedirs(directory_path)
    else:
        print(f"Directory already exists: {directory_path}")

# --- Main Processing Logic ---
def main():
    """Main function to process TIF files."""
    print("Starting TIF processing pipeline...")

    # Create output directories
    create_directory_if_not_exists(OUTPUT_DIR_25M)
    create_directory_if_not_exists(OUTPUT_DIR_100M)

    if not INPUT_TIF_FILES:
        print("No input TIF files configured. Exiting.")
        return

    for input_tif in INPUT_TIF_FILES:
        print(f"\nProcessing file: {input_tif}")

        # Check if the input file exists (basic check, as they are placeholders for now)
        # In a real scenario, you'd want a robust check here.
        # if not os.path.exists(input_tif):
        #     print(f"Warning: Input file {input_tif} not found. Skipping.")
        #     continue

        base_filename = os.path.splitext(os.path.basename(input_tif))[0]

        # Define temporary filenames
        temp_25m_tif = f"{base_filename}_25m_temp.tif"
        temp_100m_tif = f"{base_filename}_100m_temp.tif"

        # --- Downscaling to 25m ---
        print(f"\nDownscaling {input_tif} to 25m...")
        gdalwarp_25m_command = [
            'gdalwarp',
            '-t_srs', TARGET_EPSG,
            '-tr', '25', '25',
            '-r', 'bilinear',  # Resampling method
            '-co', 'COMPRESS=LZW',
            input_tif,
            temp_25m_tif
        ]
        if not run_command(gdalwarp_25m_command):
            print(f"Failed to downscale {input_tif} to 25m. Skipping further processing for this file.")
            # Attempt to clean up partial temp file if it was created
            if os.path.exists(temp_25m_tif):
                os.remove(temp_25m_tif)
            continue

        # --- Tiling for 25m ---
        print(f"\nGenerating 25m tiles for {temp_25m_tif}...")
        output_tile_dir_25m_for_file = os.path.join(OUTPUT_DIR_25M, base_filename)
        create_directory_if_not_exists(output_tile_dir_25m_for_file) # gdal2tiles creates subfolder for tiles
        
        gdal2tiles_25m_command = [
            'gdal2tiles.py',
            '-p', 'mercator',
            '-z', "8-15",  # Example zoom levels
            '-w', 'none',  # Do not generate web viewer files
            temp_25m_tif,
            output_tile_dir_25m_for_file # gdal2tiles will create tiles inside this directory
        ]
        if not run_command(gdal2tiles_25m_command):
            print(f"Failed to generate 25m tiles for {temp_25m_tif}.")
        else:
            print(f"Successfully generated 25m tiles in {output_tile_dir_25m_for_file}")


        # --- Downscaling to 100m ---
        print(f"\nDownscaling {input_tif} to 100m...")
        gdalwarp_100m_command = [
            'gdalwarp',
            '-t_srs', TARGET_EPSG,
            '-tr', '100', '100',
            '-r', 'bilinear',
            '-co', 'COMPRESS=LZW',
            input_tif,
            temp_100m_tif
        ]
        if not run_command(gdalwarp_100m_command):
            print(f"Failed to downscale {input_tif} to 100m. Skipping further processing for this file.")
            if os.path.exists(temp_100m_tif):
                os.remove(temp_100m_tif)
            # Clean up 25m temp file as well if 100m processing fails mid-way for a file
            if os.path.exists(temp_25m_tif):
                 print(f"Cleaning up temporary file: {temp_25m_tif}")
                 os.remove(temp_25m_tif)
            continue

        # --- Tiling for 100m ---
        print(f"\nGenerating 100m tiles for {temp_100m_tif}...")
        output_tile_dir_100m_for_file = os.path.join(OUTPUT_DIR_100M, base_filename)
        create_directory_if_not_exists(output_tile_dir_100m_for_file)

        gdal2tiles_100m_command = [
            'gdal2tiles.py',
            '-p', 'mercator',
            '-z', "6-12",  # Example zoom levels
            '-w', 'none',
            temp_100m_tif,
            output_tile_dir_100m_for_file
        ]
        if not run_command(gdal2tiles_100m_command):
            print(f"Failed to generate 100m tiles for {temp_100m_tif}.")
        else:
            print(f"Successfully generated 100m tiles in {output_tile_dir_100m_for_file}")

        # --- Clean up temporary files ---
        print("\nCleaning up temporary files...")
        for temp_file in [temp_25m_tif, temp_100m_tif]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"Removed temporary file: {temp_file}")
                except OSError as e:
                    print(f"Error removing temporary file {temp_file}: {e}")
            else:
                print(f"Temporary file not found for cleanup (may have failed earlier): {temp_file}")
        
        print(f"\nFinished processing for file: {input_tif}")

    print("\nAll TIF files processed.")

if __name__ == '__main__':
    main()
    print("Running process_weather_tifs.py...")