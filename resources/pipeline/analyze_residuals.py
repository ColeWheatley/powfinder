import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import sys
from datetime import datetime

# --- Configuration ---
WEATHER_VARIABLES = [ # Should match process_task_queue.py for consistency
    "temperature_2m", "relative_humidity_2m", "shortwave_radiation",
    "cloud_cover", "snow_depth", "snowfall", "wind_speed_10m",
    "weather_code", "freezing_level_height", "surface_pressure"
]

# --- Paths ---
THIS_DIR = Path(__file__).parent.resolve()
RESIDUALS_CSV_PATH = THIS_DIR / "residuals.csv"
ALL_POINTS_JSON_PATH = THIS_DIR.parent / "meteo_api" / "all_points.json" # ../meteo_api/all_points.json

# Create analysis directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ANALYSIS_DIR = THIS_DIR / "analysis" / f"run_{TIMESTAMP}"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON_PATH = ANALYSIS_DIR / "residual_summary.json"
HISTOGRAM_PLOT_PATH = ANALYSIS_DIR / "histogram_errors.png"
SCATTER_ELEV_PLOT_PATH = ANALYSIS_DIR / "scatter_error_vs_elev.png"

def load_data(residuals_file_path, all_points_file_path):
    """Loads residuals and merges elevation data."""
    try:
        residuals_df = pd.read_csv(residuals_file_path)
        if residuals_df.empty:
            print(f"Warning: Residuals file {residuals_file_path} is empty. No analysis will be performed.", file=sys.stderr)
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"Error: Residuals file not found at {residuals_file_path}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Residuals file {residuals_file_path} is empty or malformed.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading residuals CSV {residuals_file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(all_points_file_path, 'r') as f:
            all_points_data = json.load(f)

        validation_points = [
            {"lat": p["latitude"], "lon": p["longitude"], "elevation": p["elevation"]}
            for p in all_points_data.get("coordinates", []) if p.get("is_validation", False)
        ]
        if not validation_points:
            print(f"Warning: No validation points with elevation found in {all_points_file_path}. Elevation scatter plots will be affected.", file=sys.stderr)
            # Add an empty elevation column if no validation points are found to avoid merge errors later
            residuals_df['elevation'] = np.nan
            return residuals_df

        elevation_df = pd.DataFrame(validation_points)
        # Round coordinates for merging to avoid float precision issues
        residuals_df['lat_round'] = residuals_df['lat'].round(5)
        residuals_df['lon_round'] = residuals_df['lon'].round(5)
        elevation_df['lat_round'] = elevation_df['lat'].round(5)
        elevation_df['lon_round'] = elevation_df['lon'].round(5)

        # Merge data
        merged_df = pd.merge(residuals_df, elevation_df[['lat_round', 'lon_round', 'elevation']],
                             on=['lat_round', 'lon_round'], how='left')

        # Clean up temporary columns
        merged_df.drop(columns=['lat_round', 'lon_round'], inplace=True)
        if 'lat_y' in merged_df.columns: # if merge creates _x, _y due to original lat/lon cols
            merged_df.drop(columns=['lat_y', 'lon_y'], inplace=True, errors='ignore')
            merged_df.rename(columns={'lat_x': 'lat', 'lon_x': 'lon'}, inplace=True, errors='ignore')

        if merged_df['elevation'].isnull().any():
            print(f"Warning: Some rows in residuals data did not have matching elevation data.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Warning: All points file not found at {all_points_file_path}. Elevation data will be missing.", file=sys.stderr)
        residuals_df['elevation'] = np.nan # Add elevation column with NaNs
        return residuals_df
    except Exception as e:
        print(f"Warning: Error processing elevation data from {all_points_file_path}: {e}. Elevation data might be incomplete.", file=sys.stderr)
        if 'elevation' not in residuals_df.columns:
             residuals_df['elevation'] = np.nan
        return residuals_df

    return merged_df

def calculate_error_metrics(df):
    """Calculates MAE and RMSE for each weather variable."""
    metrics = {}
    for var in WEATHER_VARIABLES:
        error_col = f"error_{var}"
        actual_col = f"actual_{var}"
        pred_col = f"pred_{var}"

        if error_col not in df.columns:
            print(f"Warning: Error column {error_col} not found in residuals data. Skipping metrics for {var}.", file=sys.stderr)
            metrics[var] = {"mae": None, "rmse": None}
            continue

        # Drop rows where error is NaN for metric calculation for this specific variable
        var_errors = df[error_col].dropna()
        if var_errors.empty:
            print(f"Warning: No valid error data for {var} after dropping NaNs. Skipping metrics.", file=sys.stderr)
            metrics[var] = {"mae": None, "rmse": None}
            continue

        # Alternative using actual and predicted if error column is problematic or for direct sklearn use
        # if actual_col in df.columns and pred_col in df.columns:
        #     valid_data = df[[actual_col, pred_col]].dropna()
        #     if not valid_data.empty:
        #         mae = mean_absolute_error(valid_data[actual_col], valid_data[pred_col])
        #         rmse = np.sqrt(mean_squared_error(valid_data[actual_col], valid_data[pred_col]))
        #     else: ...
        # else: ...

        mae = var_errors.abs().mean()
        rmse = np.sqrt((var_errors**2).mean())
        metrics[var] = {"mae": mae, "rmse": rmse}
    return metrics

def plot_histograms(df, output_path):
    """Generates and saves histograms of error terms."""
    if df.empty:
        print("Dataframe for histograms is empty. Skipping plot.", file=sys.stderr)
        return

    num_vars = len(WEATHER_VARIABLES)
    # Adjust layout: try to make it squarish or fit typical screen ratios
    if num_vars <= 0: return
    cols = int(np.ceil(np.sqrt(num_vars)))
    rows = int(np.ceil(num_vars / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    for i, var in enumerate(WEATHER_VARIABLES):
        error_col = f"error_{var}"
        if error_col in df.columns and not df[error_col].dropna().empty:
            ax = axes[i]
            df[error_col].plot(kind='hist', ax=ax, bins=20, alpha=0.7)
            ax.set_title(f"Error: {var}")
            ax.set_xlabel("Error Value")
            ax.set_ylabel("Frequency")
        elif i < len(axes): # Still turn off axis if column missing
             axes[i].set_title(f"{var} (No data)")
             axes[i].axis('off')


    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"Saved error histograms to {output_path}")
    except Exception as e:
        print(f"Error saving histogram plot to {output_path}: {e}", file=sys.stderr)
    plt.close(fig)


def plot_scatter_elevation(df, output_path):
    """Generates and saves scatter plots of error vs. elevation."""
    if df.empty or 'elevation' not in df.columns or df['elevation'].dropna().empty:
        print("Dataframe for scatter plots is empty or missing elevation data. Skipping plot.", file=sys.stderr)
        return

    num_vars = len(WEATHER_VARIABLES)
    if num_vars <= 0: return
    cols = int(np.ceil(np.sqrt(num_vars)))
    rows = int(np.ceil(num_vars / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4)) # Slightly wider for scatter
    axes = axes.flatten()

    for i, var in enumerate(WEATHER_VARIABLES):
        error_col = f"error_{var}"
        if error_col in df.columns and not df[[error_col, 'elevation']].dropna().empty:
            ax = axes[i]
            # Filter out NaNs for this specific plot
            plot_df = df[[error_col, 'elevation']].dropna()
            if not plot_df.empty:
                ax.scatter(plot_df['elevation'], plot_df[error_col], alpha=0.5)
                ax.set_title(f"Error ({var}) vs. Elevation")
                ax.set_xlabel("Elevation (m)")
                ax.set_ylabel(f"Error: {var}")
                ax.grid(True)
            else: # Should not happen if initial check passed, but as safeguard
                axes[i].set_title(f"{var} vs. Elev (No valid data)")
                axes[i].axis('off')
        elif i < len(axes):
            axes[i].set_title(f"{var} vs. Elev (No error data)")
            axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"Saved error vs. elevation scatter plots to {output_path}")
    except Exception as e:
        print(f"Error saving scatter plot to {output_path}: {e}", file=sys.stderr)
    plt.close(fig)


def main():
    print("Starting residual analysis...")
    print(f"Analysis results will be saved to: {ANALYSIS_DIR}")

    # Load data
    data_df = load_data(RESIDUALS_CSV_PATH, ALL_POINTS_JSON_PATH)

    if data_df.empty:
        print("No data loaded. Exiting analysis.", file=sys.stderr)
        # Create empty summary if no data
        try:
            with open(SUMMARY_JSON_PATH, 'w') as f:
                json.dump({var: {"mae": None, "rmse": None} for var in WEATHER_VARIABLES}, f, indent=4)
            print(f"Wrote empty/null summary to {SUMMARY_JSON_PATH}")
        except Exception as e:
            print(f"Error writing empty summary: {e}", file=sys.stderr)
        return

    # Calculate error metrics
    error_metrics = calculate_error_metrics(data_df)
    
    # Add metadata to the summary
    summary_data = {
        "timestamp": TIMESTAMP,
        "analysis_date": datetime.now().isoformat(),
        "total_validation_points": len(data_df),
        "metrics": error_metrics
    }
    
    try:
        with open(SUMMARY_JSON_PATH, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Saved error metrics to {SUMMARY_JSON_PATH}")
    except Exception as e:
        print(f"Error saving summary JSON to {SUMMARY_JSON_PATH}: {e}", file=sys.stderr)

    # Generate plots
    plot_histograms(data_df, HISTOGRAM_PLOT_PATH)
    plot_scatter_elevation(data_df, SCATTER_ELEV_PLOT_PATH)

    # Create a latest symlink for easy access to most recent analysis
    latest_link = THIS_DIR / "analysis" / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(ANALYSIS_DIR.name)
        print(f"Created symlink to latest analysis: {latest_link}")
    except Exception as e:
        print(f"Warning: Could not create latest symlink: {e}")

    print("Residual analysis finished.")
    print(f"Results saved in: {ANALYSIS_DIR}")
    
    # Print summary metrics for quick reference
    print("\n=== Quick Summary ===")
    for var, metrics in error_metrics.items():
        if metrics["mae"] is not None and metrics["rmse"] is not None:
            print(f"{var:20s}: MAE={metrics['mae']:8.4f}, RMSE={metrics['rmse']:8.4f}")
        else:
            print(f"{var:20s}: No valid data")

if __name__ == "__main__":
    # For direct execution, we might need to create dummy input files
    # if they don't exist. This helps in isolated testing of this script.

    # Create dummy residuals.csv
    if not RESIDUALS_CSV_PATH.exists():
        print(f"Creating dummy {RESIDUALS_CSV_PATH.name} for local testing.")
        header = ["lat", "lon", "time", "task_type"]
        for var_name in WEATHER_VARIABLES:
            header.extend([f"pred_{var_name}", f"actual_{var_name}", f"error_{var_name}"])

        dummy_residual_data = [
            # lat, lon, time, task_type, pred_temp, actual_temp, error_temp, pred_hum, actual_hum, error_hum ...
            [47.2, 11.3, "2025-05-23T10:30:00", "validate"] + [10+i, 9+i, 1.0, 60+i, 62+i, -2.0, 100+i, 110+i, -10.0, 1+i*0.1, 1.1+i*0.1, -0.1, 0,0,0, 0,0,0, 10,10,0, 0,0,0, 2000,2100,-100, 900,901,-1] for i in range(10) # 10 rows
        ]
        # Flatten data to match header
        flat_data = []
        for i in range(5): # Create 5 rows of dummy data
            row = [47.2 + i*0.1, 11.3 + i*0.1, "2025-05-23T10:30:00", "validate"]
            for j, var in enumerate(WEATHER_VARIABLES):
                pred_val = 10 + j + i*0.5
                actual_val = 9.5 + j + i*0.6
                error_val = pred_val - actual_val
                row.extend([pred_val, actual_val, error_val])
            flat_data.append(row)

        dummy_residuals_df = pd.DataFrame(flat_data, columns=header)
        dummy_residuals_df.to_csv(RESIDUALS_CSV_PATH, index=False)

    # Create dummy all_points.json
    if not ALL_POINTS_JSON_PATH.exists():
        print(f"Creating dummy {ALL_POINTS_JSON_PATH.name} for local testing.")
        dummy_points_content = {
            "coordinates": [
                {"latitude": 47.2, "longitude": 11.3, "elevation": 500, "is_validation": True},
                {"latitude": 47.3, "longitude": 11.4, "elevation": 600, "is_validation": True},
                {"latitude": 47.4, "longitude": 11.5, "elevation": 700, "is_validation": True},
                {"latitude": 47.5, "longitude": 11.6, "elevation": 800, "is_validation": True},
                {"latitude": 47.6, "longitude": 11.7, "elevation": 900, "is_validation": True},
                {"latitude": 48.0, "longitude": 12.0, "elevation": 550, "is_validation": False} # Non-validation point
            ]
        }
        ALL_POINTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ALL_POINTS_JSON_PATH, 'w') as f:
            json.dump(dummy_points_content, f)

    main()
