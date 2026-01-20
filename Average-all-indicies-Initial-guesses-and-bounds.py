import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# === SETTINGS ===
# Define folder containing input data files (Excel format)
data_folder = os.path.join(os.getcwd(), "data")
# Define output directory for saving averaged results
out_dir = os.path.join(os.getcwd(), "out", "Sample Averages")
os.makedirs(out_dir, exist_ok=True) # Create directory if it doesn't exist
# Define directory inside output folder for plots
plots_dir = os.path.join(out_dir, "plots")
os.makedirs(plots_dir, exist_ok=True) # Create directory if it doesn't exist
# List of variables to process, analyze, and plot
variables_to_plot = [
    'RER',
    'Ti/Ttot',
    'VO2',
    'VCO2',
    'VE(STPD)',
    'Vt',
    'RR',
    'PetCO2',
    'O2_pulse',
    'EE',
    'FAox',
    'Matched HR',
    'FAox'
     #'RER_CF', 'VCO2_CF', 'FAox_CF', 'RER_norm_VE',
]
group_1_ids = ['BS_04', 'BS_05', 'BS_14','BS_20', 'BS_21', 'BS_25',
               'BS_29', 'BS_30', 'BS_31', 'BS_33', 'BS_35', 'BS_36', 'BS_37', 'BS_39',
               'BS_40', 'BS_41', 'BS_43']
group_2_ids = ['MRS_03', 'MRS_07', 'MRS_08', 'MRS_09', 'MRS_11', 'MRS_13',
               'MRS_14', 'MRS_17','MRS_18', 'MRS_19', 'MRS_22', 'MRS_25',
               'MRS_26', 'MRS_27', 'MRS_29', 'MRS_30', 'MRS_31', 'MRS_31',
               'MRS_32', 'MRS_33']
# === Initialize dictionaries to collect deviation start times and time to reach 63% response,
# separated by session type (Baseline or Followup) and variable
all_file_dev_times = {
    'Baseline': {var: [] for var in variables_to_plot},
    'Followup': {var: [] for var in variables_to_plot}
}
# Also store on-kinetics means per file
all_file_on_stats = {
    'Baseline': {var: [] for var in variables_to_plot},
    'Followup': {var: [] for var in variables_to_plot}
}
all_file_time_to_63 = {
    'Baseline': {var: [] for var in variables_to_plot},
    'Followup': {var: [] for var in variables_to_plot}
}
# Initialize dictionary for off kinetics storage ===
all_file_off_stats = {
    'Baseline': {var: [] for var in variables_to_plot},
    'Followup': {var: [] for var in variables_to_plot}
}
# === Add separate storage for Followup Group 1 and Group 2 ===
all_file_dev_times_g1 = {var: [] for var in variables_to_plot}
all_file_dev_times_g2 = {var: [] for var in variables_to_plot}
all_file_time_to_63_g1 = {var: [] for var in variables_to_plot}
all_file_time_to_63_g2 = {var: [] for var in variables_to_plot}
all_file_on_stats_g1 = {var: [] for var in variables_to_plot}
all_file_on_stats_g2 = {var: [] for var in variables_to_plot}
all_file_off_stats_g1 = {var: [] for var in variables_to_plot}
all_file_off_stats_g2 = {var: [] for var in variables_to_plot}
# Load Excel files
excel_files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx') and ('_V1' in f or '_V4' in f)]
print(f"Found {len(excel_files)} .xlsx files: {excel_files}")
# Lists to store dataframes corresponding to Baseline (V1) and Followup (V4) sessions
dfs_V1 = []
dfs_V4_group1 = []
dfs_V4_group2 = []
# === Process each Excel file individually ===
for file in excel_files:
    # Read file into pandas DataFrame
    df = pd.read_excel(os.path.join(data_folder, file))
   
    # Check if 'ExTime' exists, skip and report if not
    if 'ExTime' not in df.columns:
        print(f"Skipping file '{file}' because it does NOT contain the 'ExTime' column.")
        continue # Skip this file and move to the next
   
    # Calculate O2 pulse (VO2 / HR) if columns exist
    if 'VO2' in df.columns and 'Matched HR' in df.columns:
        df['O2_pulse'] = (df['VO2']*1000) / df['Matched HR']
    else:
        print(f"Skipping {file}: missing VO2 or HR for O2_pulse calculation")
        df['O2_pulse'] = np.nan
    # Calculate Respiratory Exchange Ratio (RER) if columns available (for MRS_study redudndant, needed for BEESWEET files)
    if 'VCO2' in df.columns and 'VO2' in df.columns:
        df['RER'] = df['VCO2'] / df['VO2']
    else:
        print(f"Skipping {file}: missing VO2 or VCO2 column")
        continue # Skip this file if required columns missing
    # Calculate tidal volume (Vt = VE(STPD) / RR) if columns exist
    for col in df.columns: # Convert all columns to numeric (except 'Work' column which might contain markers)
        if col != 'Work':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'VE(STPD)' in df.columns and 'RR' in df.columns:
        df['Vt'] = df['VE(STPD)'] / df['RR']
    else:
        print(f"Skipping {file}: missing VE(STPD) or RR column")
        continue
    # Calculate RER normalized by ventilation (RER_norm_VE = RER / VE(STPD))
    if 'RER' in df.columns and 'VE(STPD)' in df.columns:
        df['RER_norm_VE'] = df['RER'] / df['VE(STPD)']
    else:
        print(f"Skipping {file}: missing RER or VE(STPD) for RER_norm_VE calculation")
        continue
    # Calculate ventilation-corrected RER and VCO2 (RER_CF and VCO2_CF) using pre-exercise ventilation as baseline
    if {'VCO2', 'VO2', 'VE(STPD)', 'ExTime'}.issubset(df.columns):
        ve_pre = df[df['ExTime'] < 0]['VE(STPD)'] # Extract ventilation before exercise
        if not ve_pre.empty:
            mean_ve_pre = ve_pre.mean() # Mean pre-exercise ventilation
            # Adjust RER and VCO2 values by scaling ventilation relative to pre-exercise mean
            df['RER_CF'] = df['VCO2'] / ((df['VE(STPD)'] / mean_ve_pre) * df['VO2'])
            df['VCO2_CF'] = df['VCO2'] / (df['VE(STPD)'] / mean_ve_pre)
        else:
            print(f"Skipping {file}: no pre-exercise VE(STPD) for CF calculations")
            df['RER_CF'] = np.nan
            df['VCO2_CF'] = np.nan
    else:
        print(f"Skipping {file}: missing required columns for RER_CF or VCO2_CF")
        df['RER_CF'] = np.nan
        df['VCO2_CF'] = np.nan
    # Calculate fatty acid oxidation estimate (FAox) = 1.67 * VO2 - 1.67 * VCO2
    if 'VO2' in df.columns and 'VCO2' in df.columns:
        df['FAox'] = (1.67 * df['VO2']) - (1.67 * df['VCO2'])
    else:
        print(f"Skipping {file}: missing VO2 or VCO2 for FAox calculation")
        df['FAox'] = np.nan
    # Calculate corrected fatty acid oxidation (FAox_CF) using ventilation-corrected VCO2
    if 'VO2' in df.columns and 'VCO2_CF' in df.columns:
        df['FAox_CF'] = (1.67 * df['VO2']) - (1.67 * df['VCO2_CF'])
    else:
        print(f"Skipping {file}: missing VO2 or VCO2_CF for FAox_CF calculation")
        df['FAox_CF'] = np.nan
    # Calculate Energy Expenditure (EE) if VO2 and RER exist
    if 'VO2' in df.columns and 'RER' in df.columns and 'ExTime' in df.columns:
        # Calculate mean RER during rest (ExTime < 0)
        RER_rest = df[df['ExTime'] < 0]['RER']
        if not RER_rest.empty:
            RER_rest_mean = RER_rest.mean()
            df['EE'] = df['VO2'] * 60 * (3.851 + (1.081 * RER_rest_mean))
        else:
            print(f"Skipping {file}: no resting RER to calculate EE")
            df['EE'] = np.nan
    else:
        print(f"Skipping {file}: missing VO2 or RER for EE calculation")
        df['EE'] = np.nan
    # Convert all columns to numeric (except 'Work' column which might contain markers)
    for col in df.columns:
        if col != 'Work':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Identify session type by filename: 'V1' → Baseline, 'V4' → Followup
    if 'V1' in file:
        session = 'Baseline'
        dfs_V1.append(df)
    elif 'V4' in file:
        session = 'Followup'
        # Extract the participant ID from filename (e.g., MRS_01)
        base_name = os.path.splitext(file)[0]
        parts = base_name.split('_')
        participant_id = '_'.join(parts[:2])
   
        # Assign to correct group and link to correct stats storage
        if participant_id in group_1_ids:
            dfs_V4_group1.append(df)
            this_dev_times = all_file_dev_times_g1
            this_time_to_63 = all_file_time_to_63_g1
            this_on_stats = all_file_on_stats_g1
            this_off_stats = all_file_off_stats_g1
        elif participant_id in group_2_ids:
            dfs_V4_group2.append(df)
            this_dev_times = all_file_dev_times_g2
            this_time_to_63 = all_file_time_to_63_g2
            this_on_stats = all_file_on_stats_g2
            this_off_stats = all_file_off_stats_g2
        else:
            print(f"⚠️ WARNING: {file} (participant {participant_id}) did not match any group!")
            continue
    else:
        print(f"Skipping: {file} (no V1/V4 tag)")
        continue
    # Get maximum exercise time
    max_ex_time = df['ExTime'].max()
    # === For each variable, calculate deviation start time and time to reach 63% of response ===
    for var in variables_to_plot:
        if var not in df.columns:
            continue # Skip if variable not found in this file
        # Split data into pre-exercise (baseline) and post-exercise
        pre_rows = df[df['ExTime'] < 0]
        post_rows = df[df['ExTime'] >= 0]
        # Skip if baseline or post-exercise data missing
        if pre_rows.empty or post_rows.empty:
            continue
        # Calculate baseline mean over last 5 minutes before exercise start
        baseline_mean = df[(df['ExTime'] >= -300) & (df['ExTime'] <= 0)][var].mean()
        # Calculate mean at end of exercise (last minute, 310–370s)
        end_exercise_mean = df[(df['ExTime'] >= 310) & (df['ExTime'] <= 370)][var].mean()
        # Store baseline & end-exercise mean for ON kinetics
        all_file_on_stats[session][var].append({
            'Baseline_Mean': baseline_mean,
            'End_Exercise_Mean': end_exercise_mean
        })
        # Target value = baseline + 63% of difference to end-exercise mean
        target = baseline_mean + (0.63 * (end_exercise_mean - baseline_mean))
        # Calculate absolute difference from baseline post-exercise
        abs_diff = np.abs(post_rows[var] - baseline_mean)
        # Identify when difference exceeds 5% of baseline magnitude → considered "deviation"
        deviation = abs_diff > 0.05 * abs(baseline_mean)
        ex_times_post = post_rows['ExTime'].values
        is_deviating = False
        dev_start = None
        # Find first sustained deviation lasting at least 15 seconds
        for i in range(len(deviation)):
            if deviation.iloc[i]:
                if not is_deviating:
                    dev_start = ex_times_post[i] # Mark deviation start time
                    is_deviating = True
                elif (ex_times_post[i] - dev_start) >= 15:
                    # Once sustained deviation found, record start time and stop loop
                    all_file_dev_times[session][var].append(dev_start)
                    break
            else:
                # Reset if deviation not sustained
                is_deviating = False
                dev_start = None
        if dev_start is None:
            # No sustained deviation found for this variable in this file
            continue
        # Extract post-exercise data after deviation start time
        post_after_dev = post_rows[post_rows['ExTime'] >= dev_start]
        if post_after_dev.empty:
            continue
        # Find first time point when variable reaches or exceeds 63% target value
        reached_idx = post_after_dev[post_after_dev[var] >= target].index
        if len(reached_idx) == 0:
            continue # Target never reached
        time_reach_63 = post_after_dev.loc[reached_idx[0], 'ExTime']
        # Calculate duration from deviation start to reaching 63%
        time_to_63 = time_reach_63 - dev_start
        # Store this time duration for later stats
        all_file_time_to_63[session][var].append(time_to_63)
        # === OFF kinetics calculations ===
        # Mean of last 60s of file
        max_time = df['ExTime'].max()
        end_recovery_mean = df[df['ExTime'] >= (max_time - 60)][var].mean()
        diff = end_exercise_mean - end_recovery_mean
        if np.isnan(diff) or diff == 0:
            continue
        if diff > 0:
            target_off_tau = end_exercise_mean - 0.27 * diff
        else:
            target_off_tau = end_exercise_mean + 0.27 * abs(diff)
        post_ex_rows_off = df[df['ExTime'] >= 370]
        if post_ex_rows_off.empty:
            continue
        if diff > 0:
            changed = post_ex_rows_off[var] <= end_exercise_mean - 0.05 * abs(diff)
        else:
            changed = post_ex_rows_off[var] >= end_exercise_mean + 0.05 * abs(diff)
        ex_times_post_off = post_ex_rows_off['ExTime'].values
        is_changing = False
        change_start = None
        off_time_delay = None
        for i in range(len(changed)):
            if changed.iloc[i]:
                if not is_changing:
                    change_start = ex_times_post_off[i]
                    is_changing = True
                elif (ex_times_post_off[i] - change_start) >= 15:
                    off_time_delay = change_start - 370
                    break
            else:
                is_changing = False
                change_start = None
        off_tau = None
        if off_time_delay is not None:
            sustained = post_ex_rows_off[post_ex_rows_off['ExTime'] >= change_start]
            if diff > 0:
                reached = sustained[sustained[var] <= target_off_tau]
            else:
                reached = sustained[sustained[var] >= target_off_tau]
            if not reached.empty:
                off_tau = reached.iloc[0]['ExTime'] - 370
        all_file_off_stats[session][var].append({
            'Mean_End_Exercise': end_exercise_mean,
            'Mean_End_Recovery': end_recovery_mean,
            'Off_Time_Delay': off_time_delay,
            'Off_Tau': off_tau
        })
# === Function to average multiple DataFrames by interpolating missing data points ===
def average_dfs(dfs):
    """
    Takes a list of DataFrames and returns a single DataFrame with averaged values
    for each time point, interpolating to fill missing data and aligning time indices.
    """
    if not dfs:
        return pd.DataFrame() # Return empty DataFrame if input list is empty
    # Set 'ExTime' as index for all dataframes and remove duplicate indices (keep first)
    for i in range(len(dfs)):
        if 'ExTime' in dfs[i].columns:
            dfs[i].set_index('ExTime', inplace=True, drop=False)
        dfs[i] = dfs[i][~dfs[i].index.duplicated(keep='first')]
    # Get union of all time points from all DataFrames, then sort ascending
    all_times = dfs[0].index
    for df in dfs[1:]:
        all_times = all_times.union(df.index)
    all_times = all_times.sort_values()
    reindexed_dfs = []
    for df in dfs:
        df_numeric = df.infer_objects() # Convert columns to best possible dtypes
        # Reindex to include all unioned times, interpolate missing data linearly, and fill edges
        reindexed = df_numeric.reindex(all_times).interpolate(method='index').ffill().bfill()
        reindexed_dfs.append(reindexed)
    # Concatenate along columns, creating a multi-level column index to separate dataframes
    combined = pd.concat(reindexed_dfs, axis=1, keys=range(len(reindexed_dfs)))
    # Columns to average (exclude 'Work' as it holds markers, not data)
    columns = [col for col in dfs[0].columns if col != 'Work']
    averaged = pd.DataFrame(index=all_times)
    # Calculate average across all dataframes for each column
    for col in columns:
        averaged[col] = combined.xs(col, axis=1, level=1).mean(axis=1)
    # Restore 'ExTime' as a regular column
    averaged['ExTime'] = averaged.index
    averaged.reset_index(drop=True, inplace=True)
    # Initialize 'Work' column with NaNs (will be set later)
    averaged['Work'] = np.nan
    return averaged
# Compute average DataFrames for Baseline and Followup sessions
baseline_avg = average_dfs(dfs_V1)
followup_avg_group1 = average_dfs(dfs_V4_group1)
followup_avg_group2 = average_dfs(dfs_V4_group2)
# === Function to add manual marker values into averaged DataFrames ===
def set_manual_markers(avg_df):
    """
    Insert markers in 'Work' column at key time points:
    1 at exercise start (0s), 2 near exercise end (370), 3 at last data point.
    """
    if avg_df.empty:
        return avg_df
    avg_df['Work'] = np.nan # Reset 'Work' column to NaN
    # Find indices closest to specified times
    idx_1 = (np.abs(avg_df['ExTime'] - 0)).idxmin()
    idx_2 = (np.abs(avg_df['ExTime'] - 370)).idxmin()
    idx_3 = avg_df.index[-1]
    # Assign marker values at these indices
    avg_df.loc[idx_1, 'Work'] = 1
    avg_df.loc[idx_2, 'Work'] = 2
    avg_df.loc[idx_3, 'Work'] = 3
    return avg_df
# Add manual markers to averaged dataframes
baseline_avg = set_manual_markers(baseline_avg)
followup_avg_group1 = set_manual_markers(followup_avg_group1)
followup_avg_group2 = set_manual_markers(followup_avg_group2)
# === Calculate simplified summary statistics (mean, count) for deviation and time to 63% ===
on_kinetics_stats = {'Baseline': {}, 'Followup': {}}
for session in ['Baseline', 'Followup']:
    for var in variables_to_plot:
        valid_td = [t for t in all_file_dev_times[session][var] if t is not None]
        valid_t63 = [t for t in all_file_time_to_63[session][var] if t is not None]
        valid_means = all_file_on_stats[session][var]
        if valid_td and valid_t63 and valid_means:
            df_means = pd.DataFrame(valid_means)
            on_kinetics_stats[session][var] = {
                'Baseline_Mean': df_means['Baseline_Mean'].mean(),
                'End_Exercise_Mean': df_means['End_Exercise_Mean'].mean(),
                'Mean_Time_Delay': np.mean(valid_td),
                'Mean_Time_to_63': np.mean(valid_t63),
                'N_Files': len(valid_td)
            }
        else:
            on_kinetics_stats[session][var] = {
                'Baseline_Mean': None,
                'End_Exercise_Mean': None,
                'Mean_Time_Delay': None,
                'Mean_Time_to_63': None,
                'N_Files': 0
            }
           
## === Calculate summary statistics for off kinetics (Baseline only, unchanged) ===
off_kinetics_stats = {'Baseline': {}}
for var, vals in all_file_off_stats['Baseline'].items():
    if vals:
        df_off = pd.DataFrame(vals)
        stats = {
            'Mean_End_Exercise': df_off['Mean_End_Exercise'].mean(),
            'Mean_End_Recovery': df_off['Mean_End_Recovery'].mean(),
            'Mean_Off_Time_Delay': df_off['Off_Time_Delay'].mean(),
            'Mean_Off_Tau': df_off['Off_Tau'].mean(),
            'N_Files': len(vals)
        }
    else:
        stats = {
            'Mean_End_Exercise': None,
            'Mean_End_Recovery': None,
            'Mean_Off_Time_Delay': None,
            'Mean_Off_Tau': None,
            'N_Files': 0
        }
    off_kinetics_stats['Baseline'][var] = stats
# === Calculate ON kinetics stats for Followup Group 1 and Group 2 ===
on_kinetics_stats_group1 = {}
on_kinetics_stats_group2 = {}
for var in variables_to_plot:
    dev_times_g1 = [t for t in all_file_dev_times_g1[var] if t is not None]
    dev_times_g2 = [t for t in all_file_dev_times_g2[var] if t is not None]
    t63_g1 = [t for t in all_file_time_to_63_g1[var] if t is not None]
    t63_g2 = [t for t in all_file_time_to_63_g2[var] if t is not None]
    means_g1 = pd.DataFrame(all_file_on_stats_g1[var])
    means_g2 = pd.DataFrame(all_file_on_stats_g2[var])
    if dev_times_g1:
        on_kinetics_stats_group1[var] = {
            'Baseline_Mean': means_g1['Baseline_Mean'].mean() if not means_g1.empty else None,
            'End_Exercise_Mean': means_g1['End_Exercise_Mean'].mean() if not means_g1.empty else None,
            'Mean_Time_Delay': np.mean(dev_times_g1),
            'Mean_Time_to_63': np.mean(t63_g1) if t63_g1 else None,
            'N_Files': len(dev_times_g1)
        }
    else:
        on_kinetics_stats_group1[var] = {
            'Baseline_Mean': None,
            'End_Exercise_Mean': None,
            'Mean_Time_Delay': None,
            'Mean_Time_to_63': None,
            'N_Files': 0
        }
    if dev_times_g2:
        on_kinetics_stats_group2[var] = {
            'Baseline_Mean': means_g2['Baseline_Mean'].mean() if not means_g2.empty else None,
            'End_Exercise_Mean': means_g2['End_Exercise_Mean'].mean() if not means_g2.empty else None,
            'Mean_Time_Delay': np.mean(dev_times_g2),
            'Mean_Time_to_63': np.mean(t63_g2) if t63_g2 else None,
            'N_Files': len(dev_times_g2)
        }
    else:
        on_kinetics_stats_group2[var] = {
            'Baseline_Mean': None,
            'End_Exercise_Mean': None,
            'Mean_Time_Delay': None,
            'Mean_Time_to_63': None,
            'N_Files': 0
        }
# === Calculate OFF kinetics stats for Followup Group 1 and Group 2 ===
off_kinetics_stats_group1 = {}
off_kinetics_stats_group2 = {}
for var in variables_to_plot:
    vals_g1 = []
    vals_g2 = []
    off_vals = all_file_off_stats.get('Followup', {}).get(var, [])
    for i in range(len(dfs_V4_group1)):
        if i < len(off_vals):
            val = off_vals[i]
            if val:
                vals_g1.append(val)
    for i in range(len(dfs_V4_group2)):
        idx = i + len(dfs_V4_group1)
        if idx < len(off_vals):
            val = off_vals[idx]
            if val:
                vals_g2.append(val)
    if vals_g1:
        df_off_g1 = pd.DataFrame(vals_g1)
        off_kinetics_stats_group1[var] = {
            'Mean_End_Exercise': df_off_g1['Mean_End_Exercise'].mean(),
            'Mean_End_Recovery': df_off_g1['Mean_End_Recovery'].mean(),
            'Mean_Off_Time_Delay': df_off_g1['Off_Time_Delay'].mean(),
            'Mean_Off_Tau': df_off_g1['Off_Tau'].mean(),
            'N_Files': len(vals_g1)
        }
    else:
        off_kinetics_stats_group1[var] = {
            'Mean_End_Exercise': None,
            'Mean_End_Recovery': None,
            'Mean_Off_Time_Delay': None,
            'Mean_Off_Tau': None,
            'N_Files': 0
        }
    if vals_g2:
        df_off_g2 = pd.DataFrame(vals_g2)
        off_kinetics_stats_group2[var] = {
            'Mean_End_Exercise': df_off_g2['Mean_End_Exercise'].mean(),
            'Mean_End_Recovery': df_off_g2['Mean_End_Recovery'].mean(),
            'Mean_Off_Time_Delay': df_off_g2['Off_Time_Delay'].mean(),
            'Mean_Off_Tau': df_off_g2['Off_Tau'].mean(),
            'N_Files': len(vals_g2)
        }
    else:
        off_kinetics_stats_group2[var] = {
            'Mean_End_Exercise': None,
            'Mean_End_Recovery': None,
            'Mean_Off_Time_Delay': None,
            'Mean_Off_Tau': None,
            'N_Files': 0
        }
# === Calculate ON kinetics stats for Followup Group 1 and Group 2 ===
on_kinetics_stats_group1 = {}
on_kinetics_stats_group2 = {}
for var in variables_to_plot:
    vals_g1 = []
    vals_g2 = []
    # Collect ON kinetics deviation times from files in group1
    for i, df in enumerate(dfs_V4_group1):
        file_dev_times = all_file_dev_times['Followup'].get(var, [])
        if i < len(file_dev_times):
            val = file_dev_times[i]
            if val is not None:
                vals_g1.append(val)
    # Collect ON kinetics deviation times from files in group2
    for i, df in enumerate(dfs_V4_group2):
        file_dev_times = all_file_dev_times['Followup'].get(var, [])
        if i < len(file_dev_times):
            val = file_dev_times[i]
            if val is not None:
                vals_g2.append(val)
    # Calculate means & stds for Group 1
    if vals_g1:
        on_kinetics_stats_group1[var] = {
            'Mean_Time_Delay': np.mean(vals_g1),
            'Std': np.std(vals_g1),
            'N_Files': len(vals_g1)
        }
    else:
        on_kinetics_stats_group1[var] = {'Mean_Time_Delay': None, 'Std': None, 'N_Files': 0}
    # Calculate means & stds for Group 2
    if vals_g2:
        on_kinetics_stats_group2[var] = {
            'Mean_Time_Delay': np.mean(vals_g2),
            'Std': np.std(vals_g2),
            'N_Files': len(vals_g2)
        }
    else:
        on_kinetics_stats_group2[var] = {'Mean_Time_Delay': None, 'Std': None, 'N_Files': 0}
# === ✅ Calculate OFF kinetics stats for Followup Group 1 and Group 2 ===
off_kinetics_stats_group1 = {}
off_kinetics_stats_group2 = {}
for var in variables_to_plot:
    off_stats_g1 = pd.DataFrame(all_file_off_stats_g1[var])
    off_stats_g2 = pd.DataFrame(all_file_off_stats_g2[var])
    if not off_stats_g1.empty:
        off_kinetics_stats_group1[var] = {
            'Mean_End_Exercise': off_stats_g1['Mean_End_Exercise'].mean(),
            'Mean_End_Recovery': off_stats_g1['Mean_End_Recovery'].mean(),
            'Mean_Off_Time_Delay': off_stats_g1['Off_Time_Delay'].mean(),
            'Mean_Off_Tau': off_stats_g1['Off_Tau'].mean(),
            'N_Files': len(off_stats_g1)
        }
    else:
        off_kinetics_stats_group1[var] = {
            'Mean_End_Exercise': None,
            'Mean_End_Recovery': None,
            'Mean_Off_Time_Delay': None,
            'Mean_Off_Tau': None,
            'N_Files': 0
        }
    if not off_stats_g2.empty:
        off_kinetics_stats_group2[var] = {
            'Mean_End_Exercise': off_stats_g2['Mean_End_Exercise'].mean(),
            'Mean_End_Recovery': off_stats_g2['Mean_End_Recovery'].mean(),
            'Mean_Off_Time_Delay': off_stats_g2['Off_Time_Delay'].mean(),
            'Mean_Off_Tau': off_stats_g2['Off_Tau'].mean(),
            'N_Files': len(off_stats_g2)
        }
    else:
        off_kinetics_stats_group2[var] = {
            'Mean_End_Exercise': None,
            'Mean_End_Recovery': None,
            'Mean_Off_Time_Delay': None,
            'Mean_Off_Tau': None,
            'N_Files': 0
        }
# === Plot averaged variables and overlay mean deviation timings ===
for var in variables_to_plot:
    plt.figure(figsize=(12, 7))
    # Plot Baseline
    if not baseline_avg.empty and var in baseline_avg.columns:
        plt.plot(baseline_avg['ExTime'], baseline_avg[var], label='Baseline (V1)', color='black')
    # Plot Followup Group 1
    if not followup_avg_group1.empty and var in followup_avg_group1.columns:
        plt.plot(followup_avg_group1['ExTime'], followup_avg_group1[var], label='Followup Group 1 (V4)', color='red')
    # Plot Followup Group 2
    if not followup_avg_group2.empty and var in followup_avg_group2.columns:
        plt.plot(followup_avg_group2['ExTime'], followup_avg_group2[var], label='Followup Group 2 (V4)', color='blue')
    plt.title(f"Averaged {var} by Group")
    plt.xlabel("ExTime (s)")
    plt.ylabel(var)
    plt.legend()
    plt.grid(True)
    # Save
    out_file = os.path.join(plots_dir, f"{var.replace('/', '_')}_baseline_followup_groups.png")
    plt.savefig(out_file)
    plt.close()
# Save averaged data and final On/Off Kinetics to Excel workbook ===
out_path = os.path.join(out_dir, "averaged_T1_V1_V4_with_markers.xlsx")
with pd.ExcelWriter(out_path) as writer:
    # Baseline averaged data and kinetics stats (unchanged)
    if not baseline_avg.empty:
        baseline_avg.to_excel(writer, sheet_name="Baseline_V1")
   
  
   
    # Follow-up averaged data by group (if any)
    if not followup_avg_group1.empty:
        followup_avg_group1.to_excel(writer, sheet_name="Followup_Group1_V4")
    if not followup_avg_group2.empty:
        followup_avg_group2.to_excel(writer, sheet_name="Followup_Group2_V4")
    # Baseline ON kinetics summary
    pd.DataFrame(on_kinetics_stats['Baseline']).T.to_excel(writer, sheet_name="OnKinetics_Baseline")
    # Baseline OFF kinetics summary
    pd.DataFrame(off_kinetics_stats['Baseline']).T.to_excel(writer, sheet_name="OffKinetics_Baseline")
print(f"\n✅ Saved all results to {out_path}")