import os
import json
import numpy as np
import pandas as pd

def run_def_calculation(file_name, ex_RER_df):
    # Sets up the saving output directory 
    # Ensure the "out" directory exists
    out_dir = os.path.join(os.getcwd(), "out")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Create a subfolder inside "out" named after the file
    plot_dir = os.path.join(out_dir, file_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # 9 second smoothing of data
    # Convert 'ExTime' to Timedelta and set as index
    # Fill missing RER values before smoothing
    ex_RER_df['RER'] = ex_RER_df['RER'].ffill().bfill()
    
    # Then do rolling mean
    ex_RER_df['RER_smooth'] = (
        ex_RER_df
        .set_index(pd.to_timedelta(ex_RER_df['ExTime'], unit='s'))['RER']
        .rolling('9s', center=True)
        .mean()
        .ffill()
        .bfill()
        .reset_index(drop=True)
    )

    # Explicitly fill NaNs with nearest available values
    ex_RER_df['RER_smooth'] = ex_RER_df['RER_smooth'].ffill().bfill()

    # Optional: reset index if you want 'ExTime' back as a column
    # Use drop=True so we don’t duplicate ExTime
    ex_RER_df = ex_RER_df.reset_index(drop=True)

    # Average of all breaths in baseline 
    last_60s_rest_df = ex_RER_df[(ex_RER_df['ExTime'] >= -60) & (ex_RER_df['ExTime'] <= 0)]
    RER_rest_mean = last_60s_rest_df['RER_smooth'].mean()
    if pd.isna(RER_rest_mean):
        # fallback to nearest non-NaN value
        RER_rest_mean = ex_RER_df['RER_smooth'].ffill().bfill().iloc[0]

    # RER_area = mean RER * time at end of exercise (last time index)
    time_at_end = ex_RER_df['ExTime'].max()
    rer_area = RER_rest_mean * time_at_end
    print(f"{file_name} - RER_area: {rer_area:.4f}")
           
    # Exercise SS RER (mean last 60 seconds of exercise)
    SS_df = ex_RER_df[(ex_RER_df['ExTime'] >= (time_at_end - 60)) & (ex_RER_df['ExTime'] <= time_at_end)]
    RER_ex_mean = SS_df['RER_smooth'].mean()
    if pd.isna(RER_ex_mean):
        RER_ex_mean = ex_RER_df['RER_smooth'].ffill().bfill().iloc[-1]

    # Delta RER
    delta_SS_RER = RER_rest_mean - RER_ex_mean

    # Minimum RER during exercise
    mask = ex_RER_df['ExTime'] >= 0
    if ex_RER_df.loc[mask, 'RER_smooth'].empty:
        min_RER = RER_rest_mean  # fallback if no positive times
    else:
        min_RER = ex_RER_df.loc[mask, 'RER_smooth'].min()
    delta_RER = RER_rest_mean - min_RER

    # RER_deficit = RER_area - integral under curve
    if ex_RER_df.loc[mask, 'RER_smooth'].empty:
        rer_deficit = 0  # fallback
    else:
        integral_rer = np.trapz(ex_RER_df.loc[mask, 'RER_smooth'], ex_RER_df.loc[mask, 'ExTime'])
        rer_deficit = integral_rer - rer_area

    # Prepare results dictionary
    results_dict = {
        "file": file_name,
        "RER_deficit_trapz_9p": rer_deficit,
        "RER_SS_Delta_9p":delta_SS_RER,
        "RER_Delta_9p": delta_RER
    }


    # Save the results dictionary to participant .json file
    json_file_name = f"{file_name}_results.json"
    json_path = os.path.join(out_dir, json_file_name)
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {json_path}")

    return results_dict
