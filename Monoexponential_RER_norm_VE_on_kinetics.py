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

# Main function to run kinetics model for RER_norm_VE
def run_kinetics(file_name, ex_RER_df): 
    
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
    last_60s_ex_df = ex_RER_df[(ex_RER_df.iloc[:, 0] >= -60) & (ex_RER_df.iloc[:, 0] <= 0)]
    A0_mean = last_60s_ex_df['RER_norm_VE'].mean()
    print(f"A0_mean = {A0_mean}")
    
    # Filter only positive RER_norm_VE values
    ex_RER_norm_VE_df_positive = ex_RER_df[ex_RER_df['ExTime'] >= 0]
    if ex_RER_norm_VE_df_positive.empty:
        print(f"No positive RER_norm_VE values in {file_name}. Skipping this file.")
        return None
    else:
        print(f"Removed {ex_RER_df.shape[0] - ex_RER_norm_VE_df_positive.shape[0]} negative exercise time values.")
        print(f"{ex_RER_norm_VE_df_positive.shape[0]} positive exercise time values.")

    # Extract time and RER_norm_VE (all positive exercise values from here)
    x_data = np.array(ex_RER_norm_VE_df_positive.iloc[:, 0])
    y_data = np.array(ex_RER_norm_VE_df_positive['RER_norm_VE'])

    # Initialize placeholder for all removed outliers
    all_outlier_x = np.array([])  # To store x-values of all removed outliers
    all_outlier_y = np.array([])  # To store y-values of all removed outliers

    # Iterative outlier removal and model fitting
    for iteration in range(5):
        print(f"\nIteration {iteration + 1}")
        
        # Set initial guesses for the equation
        A0_init = A0_mean # mean of End ex
        A1_init = np.mean(y_data[x_data >= (x_data[-1] - 60)]) # mean last 60 sec of exovery
        tau_init = 2.78 
        
        # Prepare TD_initial guess (First breath where RER_norm_VE experiences a -5% change and is sustained for 15 seconds)
        upper_threshold =  A0_mean* 0.95
        # Time and RER_norm_VE arrays
        time_vals = x_data
        RER_norm_VE_vals = y_data
        # Create a boolean mask where RER_norm_VE < 95 % of A0_mean
        above_threshold = RER_norm_VE_vals < upper_threshold
        # Minimum sustained duration requirement
        min_duration = 15  # seconds
        # Default time delay value
        td_init = 5.25 
        found = False  # Flag to check if a valid TD was found
        # Search for the first time RER_norm_VE stays above threshold for >= 30 seconds
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
            print(f"No sustained -5% RER_norm_VE change found; fallback to 6.5 s")
        
        #Set lower bounds for the equation
        A0_lb = A0_init - 1e-3  # constrains the lower bound to very close to A0_mean
        A1_lb = y_data.min()  #Contrains lowest RER_norm_VE observed in recovery 
        td_lb = 0  # competetive collegiate atheltes demonstrate RER_norm_VE uptake kinetics that occur at 8.6 seconds (2SD less than this is 7.2s),   Kilding AE, Winter EM, Fysh M. A comparison of pulmonary oxygen uptake kinetics in middle- and long-distance runners. Int J Sports Med. 2006 May;27(5):419-26. doi: 10.1055/s-2005-865778. PMID: 16729386.
        tau_lb = 0  # competetive collegiate atheltes demonstrate RER_norm_VE uptake kinetics that occur at 24.3 seconds (2SD less than this is 23.1),   Kilding AE, Winter EM, Fysh M. A comparison of pulmonary oxygen uptake kinetics in middle- and long-distance runners. Int J Sports Med. 2006 May;27(5):419-26. doi: 10.1055/s-2005-865778. PMID: 16729386.
        
        #Set upper bounds
        A0_ub = A0_mean + 1e-3 # Again constrains the upper bound very close to A0_mean
        A1_ub = y_data.max() # Highest y-value
        td_ub = x_data.max() # 286.1 = 3SD greater than TD in combined data.  4* tau = 98% Steady State (Marias and Paterson 2015), Alexander et al. impaired postmenopausal women: (57.1 + (6.6*2)* 4 = 281.2, Wiggins et al. 2019 used "x_data.max() / 2" 
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
    
    #Calculate RER_norm_VE_def
    from scipy.integrate import quad
    # Define the fitted RER_norm_VE function for integration
    def fitted_RER_norm_VE(t):
        return model(t, A0, A1, TD, tau)
    # Integration bounds
    x_start = 0
    x_end = x_data.max() #final x value
    area_under_curve = (quad(fitted_RER_norm_VE, x_start, x_end)[0]) # Perform definite integregral for the final model results and convert it to liters  
    rest_RER_norm_VE_area = A0 * x_end 
    print(f"Rest_RER_norm_VE_area = {rest_RER_norm_VE_area}") 
    print(f"Area under curve = {area_under_curve}")
    RER_norm_VE_def =  rest_RER_norm_VE_area - area_under_curve
    
    # Poole & Jones 2012 estimate
    RER_norm_VE_on_pc = (A0- A1)
    RER_norm_VE_def_estimate = (tau/60) * RER_norm_VE_on_pc
    RER_norm_VE_on_MRT = TD + tau

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
    
    # # Report modeled RER at each minute 
    # evaluation_times = [60, 120, 180, 240, 300, 360]
    # evaluation_times = np.array(evaluation_times)
    # modeled_RER_norm_VE_values = model(evaluation_times, A0, A1, TD, tau)
    
    # Calculate spearmanrho btweem model y-values and treadmill belt speed at matched time indeces
    # Filter to ExTime ≥ 0 and drop any missing Belt_speed values
    valid_rows = ex_RER_df.copy()
    valid_rows = valid_rows[(valid_rows['ExTime'] >= 0) & (valid_rows['TM_Belt_speed'].notna())]
    
    # Extract Belt_speed and corresponding modeled RER_norm_VE values
    belt_speed_values = valid_rows['TM_Belt_speed'].values
    time_values = valid_rows['ExTime'].values
    modeled_RER_norm_VE = model(time_values, A0, A1, TD, tau)
    from scipy.stats import spearmanr
    rho, p_value = spearmanr(belt_speed_values, modeled_RER_norm_VE)
    
    # Print to console (troubleshooting step and so you don't have to open the summary file to see results, comment out if needed)    
    print(f"\nResults for: {file_name} complete")

    # Make plot for RER_norm_VE
    plt.figure(figsize=(10, 4))
    plt.scatter(x_data, final_residuals, color='purple', s=20)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Residuals - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Residuals (Observed - Fitted)")
    plt.grid(True)
    residuals_plot_path = os.path.join(plot_dir, f"{file_name}_RER_norm_VE__on_residuals.png")
    plt.savefig(residuals_plot_path)
    plt.show()
    plt.close()

    # Final RER_norm_VE model plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="Observed RER_norm_VE", color='blue', s=20)
    plt.scatter(all_outlier_x, all_outlier_y, label="Outliers (≥ 2.5 SD)", color='red', s=20, marker='x')
    plt.plot(x_data, model(x_data, *params), label="Fitted Model", color='red', linestyle='--')

    equation_text = f"RER_norm_VE(t) = {A0:.2f} - ({A1:.2f} - {A0:.2f}) × [1 - exp(-(t - {TD:.2f}) / {tau:.2f})]"
    plt.text(
        0.05, 0.95, equation_text,
        transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', color='black',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7)
    )

    plt.title(f"RER_norm_VE Kinetics - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("RER_norm_VE (RER/L/min)")
    plt.legend()
    plot_path = os.path.join(plot_dir, f"{file_name}_RER_norm_VE_on_kinetics_plot.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    results_dict = {
            
        "file": file_name,
        "RER_norm_VE_on_A0": A0,
        "RER_norm_VE_on_A1": A1,
        "RER_norm_VE_on_PCA": RER_norm_VE_on_pc,
        "RER_norm_VE_on_TD": TD,
        "RER_norm_VE_on_tau": tau,
        "RER_norm_VE_on_MRT":RER_norm_VE_on_MRT,
        "RER_norm_VE_on_Final_R^2":r_squared,
        "RER_norm_VE_on_RMSE":rmse,
        "RER_norm_VE_on_CI_A0": ci[0],
        "RER_norm_VE_on_CI_A1": ci[1],
        "RER_norm_VE_on_CI_TD": ci[2],
        "RER_norm_VE_on_CI_tau": ci[3],
        "RER_norm_VE_on_def": RER_norm_VE_def,
        "RER_norm_VE_on_def_estimate": RER_norm_VE_def_estimate,
        "RER_norm_VE_on_TT_ss": TT_ss,
        # "RER_norm_VE_model_at_60s":  modeled_RER_norm_VE_values[0],
        # "RER_norm_VE_model_at_120s": modeled_RER_norm_VE_values[1],
        # "RER_norm_VE_model_at_180s": modeled_RER_norm_VE_values[2],
        # "RER_norm_VE_model_at_240s": modeled_RER_norm_VE_values[3],
        # "RER_norm_VE_model_at_300s": modeled_RER_norm_VE_values[4],
        # "RER_norm_VE_model_at_360s": modeled_RER_norm_VE_values[5],
        "RER_norm_VE_Belt_Speed_Spearman_r": rho,
        "RER_norm_VE_Belt_Speed_Spearman_p_value": p_value
        
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