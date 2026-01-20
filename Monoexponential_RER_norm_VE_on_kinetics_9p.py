import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import json
from openpyxl import load_workbook, Workbook

# Define the mono-exponential model function
def model(x, A0, A1, TD, tau):
    x = np.asarray(x, dtype=float)
    return np.where(x < TD, A0, A0 - (A0 - A1) * (1 - np.exp(-(x - TD) / tau)))

# 9-point moving average smoothing function
def moving_average(data, window_size=9):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Main function to run kinetics model for RER
def run_kinetics(file_name, ex_RER_df): 

    # Sets up the saving output directory 
    out_dir = os.path.join(os.getcwd(), "out")  # Ensure the "out" directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plot_dir = os.path.join(out_dir, file_name)  # Create a subfolder inside "out" named after the file
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Average of all breaths in baseline 
    last_60s_ex_df = ex_RER_df[(ex_RER_df.iloc[:, 0] >= -60) & (ex_RER_df.iloc[:, 0] <= 0)]
    A0_mean = last_60s_ex_df['RER_norm_VE'].mean()
    print(f"A0_mean = {A0_mean}")

    # Filter only positive RER values
    ex_RER_df_positive = ex_RER_df[ex_RER_df['ExTime'] >= 0]
    if ex_RER_df_positive.empty:
        print(f"No positive RER values in {file_name}. Skipping this file.")
        return None
    else:
        print(f"Removed {ex_RER_df.shape[0] - ex_RER_df_positive.shape[0]} negative exercise time values.")
        print(f"{ex_RER_df_positive.shape[0]} positive exercise time values.")

    # Extract time and RER (all positive exercise values from here)
    x_data = np.array(ex_RER_df_positive.iloc[:, 0])
    y_data = np.array(ex_RER_df_positive.iloc[:, 2])
    
    ex_RER_df_positive = ex_RER_df_positive.copy()
    ex_RER_df_positive['ExTime_timedelta'] = pd.to_timedelta(ex_RER_df_positive['ExTime'], unit='s')
    ex_RER_df_positive.set_index('ExTime_timedelta', inplace=True)
    
    # Smooth column 2 (RER_norm_VE values) over a 9-second window centered on time index
    y_data_smooth_series = ex_RER_df_positive.iloc[:, 2].rolling('9s', center=True).mean()
    
    # Fill NaNs at edges
    y_data_smooth_series = ex_RER_df_positive.iloc[:, 2].rolling('9s', center=True).mean()
    y_data_smooth_series = y_data_smooth_series.bfill().ffill()
    
    # Convert to numpy array for modeling
    y_data_smooth = y_data_smooth_series.values
    
    # Use original ExTime column for x_data (seconds)
    x_data = ex_RER_df_positive['ExTime'].values
    
    # Set initial guesses for the equation
    A0_init = A0_mean  # mean of End ex
    A1_init = np.mean(y_data_smooth[x_data >= (x_data[-1] - 60)])  # mean last 60 sec of exovery
    tau_init = 2.8  

    # Prepare TD_initial guess
    upper_threshold = A0_mean * 0.95
    # Time and RER arrays
    time_vals = x_data
    RER_vals = y_data_smooth
    # Create boolean mask where RER <95% A0 mean and is sustained for more than 15 seconds
    above_threshold = RER_vals < upper_threshold
    # Minimum sustained duration requirement 
    min_duration = 15  # time in seconds that you are using a metrci where RER falls and remains lower for more than 15 seconds 
    # Default time delay (default commonly used in VO2 uptake kinetics)
    td_init = 5.25
    found = False
    # search for the first time RER falls below the threshold and reamains lower for >= 15 seconds 
    for i in range(len(time_vals)):
        if above_threshold[i]:
            td_start_time = time_vals[i]
            for j in range(i + 1, len(time_vals)):
                if not above_threshold[j]:
                    break
                duration = time_vals[j] - td_start_time
                if duration >= min_duration:
                    td_init = td_start_time
                    print(f"td_init{td_init}")
                    found = True
                    break
            if found:
                break
    # If no valid TD found, print fallback message  
    if not found:
        print(f"No sustained -5% RER_norm_VE change found; fallback to 10 s")

    # Set lower bounds for the equation
    A0_lb = A0_init - 1e-3  # constrains the lower bound to very close to A0_mean
    A1_lb = y_data_smooth.min()  # Contrains lowest RER observed in recovery 
    td_lb = 7.2  # 2SD below fastest collegiate athlete onset (Kilding et al. 2006)
    tau_lb = 23.1  # 2SD below typical postmenopausal values (Alexander et al. 2003)

    # Set upper bounds
    A0_ub = A0_mean + 1e-3  # Again constrains the upper bound very close to A0_mean
    A1_ub = y_data_smooth.max()  # Highest y-value
    td_ub = 281.2  # 4*tau = 98% Steady State (Martinis and Paterson 2015)
    tau_ub = x_data.max()  # No steady state recovery if tau exceeds this

    # Consolidate values
    initial_guess = [A0_init, A1_init, td_init, tau_init]
    lower_bounds = [A0_lb, A1_lb, td_lb, tau_lb]
    upper_bounds = [A0_ub, A1_ub, td_ub, tau_ub]

    try:
        params, covariance = curve_fit(model, x_data, y_data_smooth, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
    except RuntimeError:
        print(f"Model fit failed")
        return None

    A0, A1, TD, tau = params
    errors = np.sqrt(np.diag(covariance))
    from scipy.stats import t # Uses t-score methods to calculate confidence intervals. This does not assume normal distribution like z-score and is a but more conservative. 
    dof = max(0, len(x_data) - len(params))
    t_crit = t.ppf(0.975, dof)
    ci = t_crit * errors

    from scipy.integrate import quad
    def fitted_RER(t):
        return model(t, A0, A1, TD, tau)
    
    # Calculate RER deficit 
    x_start = 0 # Integration bounds lower
    x_end = x_data.max() # Integration bounds upper 
    area_under_curve = (quad(fitted_RER, x_start, x_end)[0]) # Perform definite integregral for the final model results and convert it to liters
    area_under_curve_trapz = np.trapz(y_data, x=x_data)
    rest_RER_area = A0 * x_end
    RER_def =  rest_RER_area - area_under_curve 
    RER_def_RAW = rest_RER_area - area_under_curve_trapz

    # RER primary component  
    RER_on_pc = (A0 - A1)
    RER_on_MRT = TD + tau

    # Calculate time to steady state Martinis and Paterson 2015.
    TT_ss = tau * 4

    # Calculate the net energetic cost of walking using the Weir equation 
    
    # Compute R²
    final_y_fit = model(x_data, *params)
    final_residuals = y_data_smooth - final_y_fit
    ss_res = np.sum((y_data_smooth - final_y_fit) ** 2)
    ss_tot = np.sum((y_data_smooth - np.mean(y_data_smooth)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compute RMSE
    rmse = np.sqrt(np.mean(final_residuals**2))
    
    # Report values each minute or closest value to it if it doesn't exist
    def get_rer_norm_ve_at_time(t):
        idx = np.argmin(np.abs(x_data - t))  # Index of closest value
        return y_data[idx]

    # Print to console (troubleshooting step and so you don't have to open the summary file to see results, comment out if needed)
    print(f"\nResults for: {file_name}")
    print(f"A0_on_9p = {A0:.4f} ± {ci[0]:.4f}")
    print(f"A1_on_9p = {A1:.4f} ± {ci[1]:.4f}")
    print(f"RER_on_PC_9p = {RER_on_pc:.4f} ± {ci[1]:.4f}")
    print(f"TD_on_9p = {TD:.4f} ± {ci[2]:.4f}")
    print(f"tau_on_9p = {tau:.4f} ± {ci[3]:.4f}")
    print(f"r^2_on_9p = {r_squared}")
    print(f"RER_def_9p = {RER_def:.4f}")
    print(f"Time to steady state ex RER_9p = {TT_ss:.4f}")

    
    # Plot smoothed data and final result 
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, label="Original RER", color='gray', alpha=0.4)
    plt.plot(x_data, y_data_smooth, label="Smoothed RER (9-point MA)", color='blue')
    plt.plot(x_data, model(x_data, *params), label="Fitted Model", color='red', linestyle='--')
    plt.title(f"RER_norm_VE Kinetics 9p - {file_name}")
    equation_text = f"RER(t) = {A0:.2f} - ({A1:.2f} - {A0:.2f}) × [1 - exp(-(t - {TD:.2f}) / {tau:.2f})]"
    plt.text(
        0.05, 0.95, equation_text,
        transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', color='black',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7)
    )
    plt.xlabel("Time (s)")
    plt.ylabel("RER")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plot_dir, f"{file_name}_RER_norm_VE_on_kinetics_plot_9p.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()


    results_dict = {
        "file": file_name,
        "RER_norm_VE_on_A0_init_9p": A0_init,
        "RER_norm_VE_on_A1_init_9p": A1_init,
        "RER_norm_VE_on_A0_9p": A0,
        "RER_norm_VE_on_A1_9p": A1,
        "RER_norm_VE_on_PCA_9p": RER_on_pc,
        "RER_norm_VE_on_TD_9p": TD,
        "RER_norm_VE_on_tau_9p": tau,
        "RER_norm_VE_on_MRT_9p":RER_on_MRT, 
        "RER_norm_VE_on_Final_R^2_9p": r_squared,
        "RER_norm_VE_on_RMSE_9p":rmse,
        "RER_norm_VE_on_CI_A0_9p": ci[0],
        "RER_norm_VE_on_CI_A1_9p": ci[1],
        "RER_norm_VE_on_PC_9p"
        "RER_norm_VE_on_CI_TD_9p": ci[2],
        "RER_norm_VE_on_CI_tau_9p": ci[3],
        "RER_norm_VE_on_def_9p": RER_def,
        "RER_norm_VE_on_def_RAW":RER_def_RAW,
        "RER_norm_VE_on_TT_ss_9p": TT_ss, 
        # "RER_norm_VE_on_0s_9p":  get_rer_norm_ve_at_time(0),
        # "RER_norm_VE_on_30s_9p":  get_rer_norm_ve_at_time(30),
        # "RER_norm_VE_on_60s_9p":  get_rer_norm_ve_at_time(60),
        # "RER_norm_VE_on_120s_9p": get_rer_norm_ve_at_time(120),
        # "RER_norm_VE_on_180s_9p": get_rer_norm_ve_at_time(180),
        # "RER_norm_VE_on_240s_9p": get_rer_norm_ve_at_time(240),
        # "RER_norm_VE_on_300s_9p": get_rer_norm_ve_at_time(300),
        # "RER_norm_VE_on_360s_9p": get_rer_norm_ve_at_time(360),
    }
  
    # Save the results dictionary to participant .json file
    json_file_name = f"{file_name}_results.json"

    out_dir = os.path.join(os.getcwd(), "out")
    if not os.path.exists(out_dir):
         os.makedirs(out_dir)

    json_path = os.path.join(out_dir, json_file_name)
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4) 
    print(f"Results saved to {json_path}")

    return results_dict
