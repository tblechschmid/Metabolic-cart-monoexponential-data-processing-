def run_smoothed_timpoints(file_name, ex_RER_df):
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt

    # Output directories
    out_dir = os.path.join(os.getcwd(), "out")
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, file_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calculate FAox (g/min)
    if 'VO2' in ex_RER_df and 'VCO2' in ex_RER_df:
        ex_RER_df['FAox'] = 1.67 * ex_RER_df['VO2'] - 1.67 * ex_RER_df['VCO2']
    else:
        ex_RER_df['FAox'] = np.nan
    
    # Calculat CHOox (g/min)
    if 'VCO2' in ex_RER_df and 'VO2' in ex_RER_df:
        ex_RER_df['CHOox'] = 4.55 * ex_RER_df['VCO2'] - 3.21 * ex_RER_df['VO2']
    else:
        ex_RER_df['CHOox'] = np.nan
    
    #Calculate FAox_NC (g/min) - nitrogen corrected fatty acid oxidation
    if 'VO2' in ex_RER_df and 'VCO2' in ex_RER_df: #Experimental and unvalidated correction using combination of Weir and Frayn equations 
        ex_RER_df['FAox_NC'] = 3.57 * ex_RER_df['VO2'] - 4.63 * ex_RER_df['VCO2']
    else:
        ex_RER_df['FAox_NC'] = np.nan
        
    #Calculate CHOox_NC (g/min) - nitrogen corrected fatty acid oxidation
    if 'VO2' in ex_RER_df and 'VCO2' in ex_RER_df: #Experimental and unvalidated correction factor using combination of Weir and Frayn equations 
    
        ex_RER_df['CHOox_NC'] = 0.13 * ex_RER_df['VCO2'] - 0.37 * ex_RER_df['VO2']
    else:
        ex_RER_df['CHOox_NC'] = np.nan
        
    # Protein oxidation (g/min)
    if 'VO2' in ex_RER_df and 'VCO2' in ex_RER_df: #Experimental and unvalidated Correction factor based on RER normalized to VE. 
    
        ex_RER_df['PROox'] = 9.63 * ex_RER_df['VCO2'] - 6.18 * ex_RER_df['VO2']
    else:
        ex_RER_df['PROox'] = np.nan

    # Calculate O2_Pulse 
    if 'VO2' in ex_RER_df.columns and 'Matched HR' in ex_RER_df.columns:
        ex_RER_df['O2_Pulse'] = (ex_RER_df['VO2'] *1000) / ex_RER_df['Matched HR']
    else:
        ex_RER_df['O2_Pulse'] = np.nan

    # Baseline for A0 init (last 60s rest)
    last_60s_rest_df = ex_RER_df[(ex_RER_df['ExTime'] >= -60) & (ex_RER_df['ExTime'] <= 0)]

    variables_to_smooth = [
        
        'RER', 
        'RER_norm_VE',
        'VE', 'VO2',
        'Ti_Ttot',
        'VCO2',
        'FAox',
        'FAox_NC',
        'CHOox',
        'CHOox_NC',
        'PROox',
        'Vt',
        'RR',
        'Matched HR',
        'O2_Pulse'
    ]

    # A0 initialization
    variable_a0_init = {}
    for var in variables_to_smooth:
        if var in last_60s_rest_df.columns:
            variable_a0_init[f"{var}_A0_init"] = last_60s_rest_df[var].mean()
        else:
            variable_a0_init[f"{var}_A0_init"] = np.nan  # Ensure keys always exist

    # Filter positive time
    ex_RER_df_positive = ex_RER_df[ex_RER_df['ExTime'] >= 0].copy()
    if ex_RER_df_positive.empty:
        print(f"No positive time values in {file_name}. Skipping.")
        return variable_a0_init  # Still return A0 values

    # Convert to TimedeltaIndex
    ex_RER_df_positive['ExTime_timedelta'] = pd.to_timedelta(ex_RER_df_positive['ExTime'], unit='s')
    ex_RER_df_positive.set_index('ExTime_timedelta', inplace=True)

    # Apply smoothing
    for var in variables_to_smooth:
        if var in ex_RER_df_positive:
            ex_RER_df_positive[f"{var}_smoothed"] = (
                ex_RER_df_positive[var].rolling('9s', center=True).mean().bfill().ffill()
            )

    x_data = ex_RER_df_positive['ExTime'].values
    results_dict = {"file": file_name}
    results_dict.update(variable_a0_init)

    # Save evaluation points
    eval_times = [0, 30, 60, 120, 180, 240, 300, 360]
    for eval_time in eval_times:
        for var in variables_to_smooth:
            if var in ex_RER_df_positive:
                idx = (np.abs(x_data - eval_time)).argmin()
                results_dict[f"{var}_on_{eval_time}s_9p"] = ex_RER_df_positive.iloc[idx][f"{var}_smoothed"]

    # Plotting (safe in parallel)
    for var in variables_to_smooth:
        if var in ex_RER_df_positive:
            plt.figure(figsize=(10, 5))
            plt.plot(x_data, ex_RER_df_positive[var].values, label=f"Original {var}", color='gray', alpha=0.4)
            plt.plot(x_data, ex_RER_df_positive[f"{var}_smoothed"].values, label=f"Smoothed {var} (9s)", color='blue')
            plt.title(f"{var} Smoothed vs Original - {file_name}")
            plt.xlabel("Time (s)")
            plt.ylabel(var)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f"{file_name}_{var}_smoothed_plot_9p.png"))
            plt.close()

    return results_dict  # Return results dict for JSON aggregation
