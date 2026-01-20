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

# Main function to run kinetics model for VCO2
def run_kinetics(file_name, rec_VCO2_df):
    
    # Sets up the saving output directory 
    # Ensure the "out" directory exists
    out_dir = os.path.join(os.getcwd(), "out")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Create a subfolder inside "out" named after the file
    plot_dir = os.path.join(out_dir, file_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Average of all breaths in baseline 
    last_60s_ex_df = rec_VCO2_df[(rec_VCO2_df.iloc[:, 0] >= -60) & (rec_VCO2_df.iloc[:, 0] <= 0)]
    A0_mean = last_60s_ex_df.iloc[:, 1].mean()
    print(f"A0_mean = {A0_mean}")
    
    # Filter only positive VCO2 values
    rec_VCO2_df_positive = rec_VCO2_df[rec_VCO2_df['ExTime'] >= 0]
    if rec_VCO2_df_positive.empty:
        print(f"No positive VCO2 values in {file_name}. Skipping this file.")
        return None
    else:
        print(f"Removed {rec_VCO2_df.shape[0] - rec_VCO2_df_positive.shape[0]} negative exercise time values.")
        print(f"{rec_VCO2_df_positive.shape[0]} positive exercise time values.")

    # Extract time and VCO2 (all positive exercise values from here)
    x_data = np.array(rec_VCO2_df_positive.iloc[:, 0])
    y_data = np.array(rec_VCO2_df_positive.iloc[:, 1])

    # Initialize placeholder for all removed outliers
    all_outlier_x = np.array([])  # To store x-values of all removed outliers
    all_outlier_y = np.array([])  # To store y-values of all removed outliers

    # Iterative outlier removal and model fitting
    for iteration in range(5):
        print(f"\nIteration {iteration + 1}")
        
        # Set initial guesses for the equation
        A0_init = A0_mean # mean of End ex
        A1_init = np.mean(y_data[x_data >= (x_data[-1] - 60)]) # mean last 60 sec of recovery
        tau_init = 23.2
        
        # Prepare TD_initial guess (First breath where VCO2 experiences a -5% change and is sustained for 15 seconds)
        upper_threshold =  A0_mean* 0.95
        # Time and VCO2 arrays
        time_vals = x_data
        VCO2_vals = y_data
        # Create a boolean mask where VCO2 < 95% of A0_mean
        above_threshold = VCO2_vals < upper_threshold
        # Minimum sustained duration requirement
        min_duration = 15  # seconds
        # Default time delay value
        td_init = 19.4
        found = False  # Flag to check if a valid TD was found
        # Search for the first time VCO2 stays above threshold for >= 30 seconds
        for i in range(len(time_vals)):
            if above_threshold[i]:
                td_start_time = time_vals[i]
                # Look ahead to see how long it stays above threshold
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
            print(f"No sustained -5% VCO2 change found; fallback to 20.1 s")
        
        #Set lower bounds for the equation
        A0_lb = A0_init - 1e-3  # constrains the lower bound to very close to A0_mean
        A1_lb = y_data.min()  #Contrains lowest VCO2 observed in recovery 
        td_lb = 0  # competetive collegiate atheltes demonstrate VCO2 uptake kinetics that occur at 8.6 seconds (2SD less than this is 7.2s),   Kilding AE, Winter EM, Fysh M. A comparison of pulmonary oxygen uptake kinetics in middle- and long-distance runners. Int J Sports Med. 2006 May;27(5):419-26. doi: 10.1055/s-2005-865778. PMID: 16729386.
        tau_lb = 0 # competetive collegiate atheltes demonstrate VCO2 uptake kinetics that occur at 24.3 seconds (2SD less than this is 23.1),   Kilding AE, Winter EM, Fysh M. A comparison of pulmonary oxygen uptake kinetics in middle- and long-distance runners. Int J Sports Med. 2006 May;27(5):419-26. doi: 10.1055/s-2005-865778. PMID: 16729386.
        
        #Set upper bounds
        A0_ub = A0_mean + 1e-3 # Again constrains the upper bound very close to A0_mean
        A1_ub = y_data.max() # Highest y-value
        td_ub = x_data.max() # No change in VCO2 from end exercise 
        tau_ub = x_data.max() #If tau reaches this value, no steady state recovery was observed
        
        # Consolidate values
        initial_guess = [A0_init, A1_init, td_init, tau_init]
        lower_bounds = [A0_lb, A1_lb, td_lb, tau_lb]
        upper_bounds = [A0_ub, A1_ub, td_ub, tau_ub]

        try:
            params, covariance = curve_fit(model, x_data, y_data, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
        except RuntimeError:
            print(f"Model fit failed in iteration {iteration + 1}")
            return None

        y_fit = model(x_data, *params)
        residuals = y_data - y_fit
        num_SD = 2.5 # Number of standard deviations you want to use for your filter (Wiggins et al. 2019 = 3SD) *change as necessary 
        std_resid = np.std(residuals)
        
        # Creates a mask to filter out points that are greater than num_SD of the residuals in either direction 
        mask = np.abs(residuals) <= (std_resid * num_SD) 

        # Collect the outliers for this iteration
        outlier_x = x_data[~mask]
        outlier_y = y_data[~mask]
        
        # Append the current iteration's outliers to the overall list
        all_outlier_x = np.concatenate((all_outlier_x, outlier_x))
        all_outlier_y = np.concatenate((all_outlier_y, outlier_y))

        x_data = x_data[mask]
        y_data = y_data[mask]
        print(f"Remaining data points: {len(x_data)}")

    # Final fit using cleaned data
    A0, A1, TD, tau = params
    errors = np.sqrt(np.diag(covariance))
    from scipy.stats import t # Uses t-score methods to calculate confidence intervals. This does not assume normal distribution like z-score and is a but more conservative. 
    dof = max(0, len(x_data) - len(params))
    t_crit = t.ppf(0.975, dof)
    ci = t_crit * errors
    
    #Calculate EPOC
    from scipy.integrate import quad
    # Define the fitted VCO2 function for integration
    def fitted_VCO2(t):
        return model(t, A0, A1, TD, tau)
    # Integration bounds
    x_start = 0
    x_end = x_data.max() #final x value
    area_under_curve = (quad(fitted_VCO2, x_start, x_end)[0]) / 60 # Perform definite integregral for the final model results and convert it to liters  
    rest_VCO2_area = (A1 / 60) * x_end # dividing by 60 converts the result to L
    print(f"Rest_VCO2_area = {rest_VCO2_area}") 
    print(f"Area under curve = {area_under_curve}")
    EPOC = area_under_curve - rest_VCO2_area
    
    # Poole & Jones 2012 estimate
    VCO2_off_pc = (A0- A1)
    EPOC_estimate = (tau/60) * VCO2_off_pc
    VCO2_MRT = TD + tau

    # Calculate time to steady state Martinis and Paterson 2015.
    TT_ss = tau * 4


    # Plot residuals of the final fit
    final_y_fit = model(x_data, *params)
    final_residuals = y_data - final_y_fit
    
    # Compute R²
    ss_res = np.sum((y_data - final_y_fit) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean(final_residuals**2))
    
    # Print to console (troubleshooting step and so you don't have to open the summary file to see results, comment out if needed)    
    print(f"\nResults for: {file_name}")
    print(f"A0_off = {A0:.4f} ± {ci[0]:.4f}")
    print(f"A1_off = {A1:.4f} ± {ci[1]:.4f}")
    print(f"TD_off = {TD:.4f} ± {ci[2]:.4f}")
    print(f"tau_off = {tau:.4f} ± {ci[3]:.4f}")
    print(f"r^2_off = {r_squared}")
    print(f"EPOC = {EPOC:.4f}")
    print(f"EPOC_estimate = {EPOC_estimate:.4f}")
    print(f"Time to steady state rec = {TT_ss:.4f}")

    plt.figure(figsize=(10, 4))
    plt.scatter(x_data, final_residuals, color='purple', s=20)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Residuals - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Residuals (Observed - Fitted)")
    plt.grid(True)
    residuals_plot_path = os.path.join(plot_dir, f"{file_name}_VCO2_off_residuals.png")
    plt.savefig(residuals_plot_path)
    plt.show()
    plt.close()

    # Final VCO2 model plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="Observed VCO2", color='blue', s=20)
    plt.scatter(all_outlier_x, all_outlier_y, label="Outliers (≥ 2.5 SD)", color='red', s=20, marker='x')
    plt.plot(x_data, model(x_data, *params), label="Fitted Model", color='red', linestyle='--')

    equation_text = f"VCO₂(t) = {A0:.2f} - ({A1:.2f} - {A0:.2f}) × [1 - exp(-(t - {TD:.2f}) / {tau:.2f})]"
    plt.text(
        0.05, 0.95, equation_text,
        transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', color='black',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7)
    )

    plt.title(f"VCO2 Kinetics - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("VCO2 (L/min)")
    plt.legend()
    plot_path = os.path.join(plot_dir, f"{file_name}_VCO2_off_kinetics_plot.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    results_dict = {
            
        "file": file_name,
        "VCO2_off_A0": A0, # L/min
        "VCO2_off_A1": A1, # L/min
        "VCO2_off_PCA":VCO2_off_pc,# L/min
        "VCO2_off_TD": TD, # sec
        "VCO2_off_tau": tau, # sec
        "VCO2_off_MRT" : VCO2_MRT, # sec
        "VCO2_off_Final_R^2":r_squared,
        "VCO2_off_RMSE":rmse, # L/min
        "VCO2_off_CI_A0": ci[0], # L/min
        "VCO2_off_CI_A1": ci[1], # L/min
        "VCO2_off_CI_TD": ci[2], # sec
        "VCO2_off_CI_tau": ci[3], # sec
        "VCO2_off_EP_VCO2": EPOC, # Liters
        "VCO2_off_EP_VCO2_estimate": EPOC_estimate, # Liters
        "VCO2_off_TT_ss": TT_ss # sec
        
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