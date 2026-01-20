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

# Main function to run kinetics model for o2_pulse
def run_kinetics(file_name, rec_o2_pulse_df):
    
    # Sets up the saving output directory 
    out_dir = os.path.join(os.getcwd(), "out")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plot_dir = os.path.join(out_dir, file_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Average of all breaths in baseline 
    last_60s_ex_df = rec_o2_pulse_df[(rec_o2_pulse_df.iloc[:, 0] >= -60) & (rec_o2_pulse_df.iloc[:, 0] <= 0)]
    A0_mean = last_60s_ex_df.iloc[:, 1].mean()
    print(f"A0_mean = {A0_mean}")
    
    # Filter only positive o2_pulse values
    rec_o2_pulse_df_positive = rec_o2_pulse_df[rec_o2_pulse_df['ExTime'] >= 0]
    if rec_o2_pulse_df_positive.empty:
        print(f"No positive o2_pulse values in {file_name}. Skipping this file.")
        return None
    else:
        print(f"Removed {rec_o2_pulse_df.shape[0] - rec_o2_pulse_df_positive.shape[0]} negative exercise time values.")  # removed negative ExTime rows
        print(f"{rec_o2_pulse_df_positive.shape[0]} positive exercise time values.")  # number of positive ExTime rows

    # Extract time and o2_pulse (all positive exercise values from here)
    x_data = np.array(rec_o2_pulse_df_positive.iloc[:, 0])
    y_data = np.array(rec_o2_pulse_df_positive.iloc[:, 1])

    # Initialize placeholder for all removed outliers
    all_outlier_x = np.array([])  # To store x-values of all removed outliers
    all_outlier_y = np.array([])  # To store y-values of all removed outliers


    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="Observed o2_pulse", color='blue', s=20)
    plt.title(f"o2_pulse Kinetics - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("o2_pulse (mL/beat)")
    plt.legend()
    plot_path = os.path.join(plot_dir, f"{file_name}_o2_pulse_off_kinetics_plot.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    # Initialize last successful td_init with default fallback (10s) # fallback td_init
    last_successful_td_init = 18.1
    
    # Iterative outlier removal and model fitting
    for iteration in range(5):
        print(f"\nIteration {iteration + 1}")
        
        # Set initial guesses for the equation
        A0_init = A0_mean # mean of End ex
        A1_init = np.mean(y_data[x_data >= (x_data[-1] - 60)]) # mean last 60 sec of recovery
        tau_init = 21.5
        
        # Prepare TD_initial guess (First breath where o2_pulse experiences a -5% change and is sustained for 15 seconds)
        upper_threshold =  A0_mean * 0.95  # 5% decrease threshold
        time_vals = x_data
        o2_pulse_vals = y_data

        # Clean NaNs and infinite values
        o2_pulse_vals = pd.to_numeric(o2_pulse_vals, errors='coerce')
        valid_mask = np.isfinite(o2_pulse_vals)
        time_vals = time_vals[valid_mask]
        o2_pulse_vals = o2_pulse_vals[valid_mask]

        # Create a boolean mask where HR < 95% of A0_mean
        above_threshold = o2_pulse_vals < upper_threshold
        # Minimum sustained duration requirement
        min_duration = 15  # seconds
        # Default time delay value
        td_init = 18.7 # Value from the baseline average of all participants 
        found = False  # Flag to check if a valid TD was found
        
        # TD calculation start - search for the first time HR stays above threshold for >= 15 seconds
        for i in range(len(time_vals)):
            if above_threshold[i]:
                td_start_time = time_vals[i]
                # Look ahead to see how long the variable stays above threshold
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
            print(f"No sustained -5% O2 pulse change found; fallback")

        # Set lower bounds
        A0_lb = A0_init - 1e-3  # slight tolerance
        A1_lb = y_data.min()
        td_lb = 0
        tau_lb = 0
        
        # Set upper bounds
        A0_ub = A0_mean + 1e-3  # slight tolerance
        A1_ub = y_data.max() 
        td_ub = x_data.max()
        tau_ub = x_data.max()
        
        initial_guess = [A0_init, A1_init, td_init, tau_init]
        lower_bounds = [A0_lb, A1_lb, td_lb, tau_lb]
        upper_bounds = [A0_ub, A1_ub, td_ub, tau_ub]

        try:
            params, covariance = curve_fit(model, x_data, y_data, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
            # If successful, update last successful td_init
            last_successful_td_init = params[2]  # TD parameter
        except RuntimeError:
            print(f"Model fit failed in iteration {iteration + 1}, using last successful td_init = {last_successful_td_init}")  # fallback on last known td_init
            # Use last successful td_init for next iteration without changing params or x_data, y_data
            continue  # continue with next iteration

        y_fit = model(x_data, *params)
        residuals = y_data - y_fit
        num_SD = 2.5  # outlier removal cutoff
        std_resid = np.std(residuals)
        
        mask = np.abs(residuals) <= (std_resid * num_SD) 

        outlier_x = x_data[~mask]
        outlier_y = y_data[~mask]
        
        all_outlier_x = np.concatenate((all_outlier_x, outlier_x))
        all_outlier_y = np.concatenate((all_outlier_y, outlier_y))

        x_data = x_data[mask]
        y_data = y_data[mask]
        print(f"Remaining data points: {len(x_data)}")  # number of points after outlier removal

    # Final fit using cleaned data
    A0, A1, TD, tau = params
    errors = np.sqrt(np.diag(covariance))
    from scipy.stats import t 
    dof = max(0, len(x_data) - len(params))
    t_crit = t.ppf(0.975, dof)
    ci = t_crit * errors  # 95% confidence intervals
    
    from scipy.integrate import quad
    def fitted_o2_pulse(t):
        return model(t, A0, A1, TD, tau)

    x_start = 0
    x_end = x_data.max()
    area_under_curve = (quad(fitted_o2_pulse, x_start, x_end)[0]) / 60  # normalize by 60 sec
    rest_o2_pulse_area = (A1 / 60) * x_end
    print(f"Rest_o2_pulse_area = {rest_o2_pulse_area}") 
    print(f"Area under curve = {area_under_curve}")
    EP_O2_pulse = area_under_curve - rest_o2_pulse_area  # excess post-exercise oxygen pulse
    
    o2_pulse_off_pc = (A0 - A1)
    EP_O2_pulse_estimate = (tau / 60) * o2_pulse_off_pc  # estimate from tau and offset difference
    o2_pulse_off_MRT = TD + tau

    TT_ss = tau * 4  # time to steady state, ~4 time constants

    final_y_fit = model(x_data, *params)
    final_residuals = y_data - final_y_fit
    
    ss_res = np.sum((y_data - final_y_fit) ** 2)  # residual sum of squares  
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)  # total sum of squares  
    r_squared = 1 - (ss_res / ss_tot)  # coefficient of determination
    
    rmse = np.sqrt(np.mean(final_residuals**2))

    
    print(f"\nResults for: {file_name}")
    print(f"A0_off = {A0:.4f} ± {ci[0]:.4f}")
    print(f"A1_off = {A1:.4f} ± {ci[1]:.4f}")
    print(f"TD_off = {TD:.4f} ± {ci[2]:.4f}")
    print(f"tau_off = {tau:.4f} ± {ci[3]:.4f}")
    print(f"r^2_off = {r_squared}")
    print(f"EP_O2_pulse = {EP_O2_pulse:.4f}")
    print(f"EP_O2_pulse_estimate = {EP_O2_pulse_estimate:.4f}")
    print(f"Time to steady state rec = {TT_ss:.4f}")

    plt.figure(figsize=(10, 4))
    plt.scatter(x_data, final_residuals, color='purple', s=20)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Residuals - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Residuals (Observed - Fitted)")
    plt.grid(True)
    residuals_plot_path = os.path.join(plot_dir, f"{file_name}_o2_pulse_off_residuals.png")
    plt.savefig(residuals_plot_path)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="Observed o2_pulse", color='blue', s=20)
    plt.scatter(all_outlier_x, all_outlier_y, label="Outliers (≥ 2.5 SD)", color='red', s=20, marker='x')
    plt.plot(x_data, model(x_data, *params), label="Fitted Model", color='red', linestyle='--')

    equation_text = f"O₂ pulse(t) = {A0:.2f} - ({A1:.2f} - {A0:.2f}) × [1 - exp(-(t - {TD:.2f}) / {tau:.2f})]"
    plt.text(
        0.05, 0.95, equation_text,
        transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', color='black',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7)
    )

    plt.title(f"o2_pulse Kinetics - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("o2_pulse (mL/beat)")
    plt.legend()
    plot_path = os.path.join(plot_dir, f"{file_name}_o2_pulse_off_kinetics_plot.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    results_dict = {
        "file": file_name,
        "O2_pulse_off_A0": A0, # mL O2/beat 
        "O2_pulse_off_A1": A1, # mL O2/beat 
        "O2_pulse_off_PCA" :o2_pulse_off_pc, # mL O2/beat 
        "O2_pulse_off_TD": TD, # sec
        "O2_pulse_off_tau": tau, # sec
        "O2_pulse_off_MRT": o2_pulse_off_MRT, 
        "O2_pulse_off_Final_R^2": r_squared,
        "O2_pulse_off_rmse":rmse, # mL O2/beat 
        "O2_pulse_off_CI_A0": ci[0], # mL O2/beat 
        "O2_pulse_off_CI_A1": ci[1], # mL O2/beat 
        "O2_pulse_off_CI_TD": ci[2], # sec
        "O2_pulse_off_CI_tau": ci[3], # sec
        "O2_pulse_off_EP_O2_pulse": EP_O2_pulse, # mL O2 * min / beat
        "O2_pulse_off_EP_O2_pulse_estimate": EP_O2_pulse_estimate,  # mL O2 * min / beat
        "O2_pulse_off_TT_ss": TT_ss # sec
    }
    json_file_name = f"{file_name}_results.json"

    out_dir = os.path.join(os.getcwd(), "out")
    if not os.path.exists(out_dir):
         os.makedirs(out_dir)

    json_path = os.path.join(out_dir, json_file_name)
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4) 
    print(f"Results saved to {json_path}")

    return results_dict
