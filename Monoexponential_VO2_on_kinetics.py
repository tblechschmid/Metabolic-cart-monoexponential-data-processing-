import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import pandas as pd
import os
import json
#from openpyxl import load_workbook, Workbook

# Define the mono-exponential model function
def model(x, A0, A1, TD, tau):
    x = np.asarray(x, dtype=float)
    return np.where(x < TD, A0, A0 + (A1 - A0) * (1 - np.exp(-(x - TD) / tau)))

# Main function to run kinetics model for VO2
def run_kinetics(file_name, ex_VO2_df):
    
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
    last_60s_rest_df = ex_VO2_df[(ex_VO2_df['ExTime'] >= -60) & (ex_VO2_df['ExTime'] <= 0)]
    A0_mean = last_60s_rest_df['VO2'].mean()
    print(f"A0_mean{A0_mean}")
    
    # Filter only positive VO2 values
    ex_VO2_df_positive = ex_VO2_df[ex_VO2_df['ExTime'] >= 0]
    if ex_VO2_df_positive.empty:
        print(f"No positive VO2 values in {file_name}. Skipping this file.")
        return None
    else:
        print(f"Removed {ex_VO2_df.shape[0] - ex_VO2_df_positive.shape[0]} negative exercise time values.")
        print(f"{ex_VO2_df_positive.shape[0]} positive exercise time values.")

    # Extract time and VO2 (all positive exercise values from here)
    x_data = np.array(ex_VO2_df_positive['ExTime'])
    y_data = np.array(ex_VO2_df_positive['VO2'])

    # Initialize placeholder for all removed outliers
    all_outlier_x = np.array([])  # To store x-values of all removed outliers
    all_outlier_y = np.array([])  # To store y-values of all removed outliers

    # Iterative outlier removal and model fitting
    for iteration in range(5):
        print(f"\nIteration {iteration + 1}")
        
        # Set initial guesses for the equation
        A0_init = A0_mean # mean of rest
        A1_init = np.mean(y_data[x_data >= (x_data[-1] - 60)]) # mean last 60 sec of exercise
        tau_init = 31.5 # Alexander et al. 2003 unimpaired postmenopausal women
        
        # Prepare TD_initial guess (First breath where VO2 experiences a +5% change and is sustained for 15 seconds)
        upper_threshold =  A0_mean* 1.10
        # Time and VO2 arrays
        time_vals = x_data
        vo2_vals = y_data
        # Create a boolean mask where VO2 > 105% of A0_mean
        above_threshold = vo2_vals > upper_threshold
        # Minimum sustained duration requirement
        min_duration = 15  # seconds
        # Default time delay value
        td_init = 5.8
        found = False  # Flag to check if a valid TD was found
        # Search for the first time VO2 stays above threshold for >= 30 seconds
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
            print(f"No sustained +5% VO2 change found; fallback")
        
        #Set lower bounds for the equation
        A0_lb = A0_init - 1e-3  # constrains the lower bound to very close to A0_mean
        A1_lb = A0_mean  #Contrains VO2 ot resting VO2
        td_lb = 0  # competetive collegiate atheltes demonstrate VO2 uptake kinetics that occur at 13.1 seconds (2SD less than this is 11.9),   Kilding AE, Winter EM, Fysh M. A comparison of pulmonary oxygen uptake kinetics in middle- and long-distance runners. Int J Sports Med. 2006 May;27(5):419-26. doi: 10.1055/s-2005-865778. PMID: 16729386.
        tau_lb = 0  # competetive collegiate atheltes demonstrate VO2 uptake kinetics that occur at 12.3 seconds (2SD less than this is 11.3),   Kilding AE, Winter EM, Fysh M. A comparison of pulmonary oxygen uptake kinetics in middle- and long-distance runners. Int J Sports Med. 2006 May;27(5):419-26. doi: 10.1055/s-2005-865778. PMID: 16729386.
        
        #Set upper bounds
        A0_ub = A0_mean + 1e-3 # Again constrains the upper bound very close to A0_mean
        A1_ub = y_data.max() # Highest y-value
        td_ub = x_data.max() # 4* tau = 98% SS, Alexander et al. impaired postmenopausal women: (58.0 + (9.3*2)* 4 = 306.4, Wiggins et al. 2019 used "x_data.max() / 2" 
        tau_ub = x_data.max() #If tau reaches this value, no steady state was observed
        
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
    
    #Calculate O2 deficit
    from scipy.integrate import quad
    # Define the fitted VO2 function for integration
    def fitted_vo2(t):
        return A0 + (A1 - A0) * (1 - np.exp(-(t - TD) / tau)) if t >= TD else A0
    # Integration bounds
    x_start = 0
    x_end = x_data.max()  # final x value converted to minutes
    area_under_curve = (quad(fitted_vo2, x_start, x_end)[0]) / 60 # Perform definite integral for the final model results and convert it to liters  
    total_area = (A1 / 60) * x_end # End result to L
    print(f"Total exercise area = {total_area}") 
    print(f"Area under curve = {area_under_curve}")
    O2_def = total_area - area_under_curve
    
    # Poole & Jones 2012 estimate
    VO2_on_PC = (A1-A0)
    O2_def_estimate = (tau/60) * VO2_on_PC
    VO2_MRT = TD + tau

    # Calculate time to steady state Martinis and Paterson 2015.
    TT_ss = tau * 4

    # Plot residuals of the final fit
    final_y_fit = model(x_data, *params)
    final_residuals = y_data - final_y_fit
    
    # Compute R²
    ss_res = np.sum((y_data - final_y_fit) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    # Compute RSME
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Calculate energy expenditure Weir nitrogen adjusted equation
    RER_rest_mean = last_60s_rest_df['RER'].mean()# Compute with the average of rest 
    last_60s_ex_df = ex_VO2_df[(ex_VO2_df['ExTime'] >= 310) & (ex_VO2_df['ExTime'] <= 370)] # These are because there is a 10 second ramp and a 6 minute walk test (i.e 360 seconds), this could be rewritten to just look at last 60 seconds
    RER_ex_mean =  last_60s_ex_df['RER'].mean()# Compute with average of last minute of exercise
    Resting_EE = (A0*60) * (3.851+(1.081*RER_rest_mean)) # kcal/hour
    Walking_EE =  (A1*60) * (3.851+(1.081*RER_ex_mean)) # kcal/hour
    Walking_economy = (Walking_EE) / 3216.688 # the demoninator is a conversion factor to get m/s and EE in kcal/hour into units kcal/m at the speed of the treadmill test used (0.89m/s). This will need to be changed depending on the walking speed!   
    Net_energetic_cst_walking = (Walking_EE - Resting_EE) / 3216.688 # same denominator as above units kcal/m
        
    # Print to console (troubleshooting step and so you don't have to open the summary file to see results, comment out if needed)    
    print(f"\nResults for: {file_name}")
    print(f"A0 = {A0:.4f} ± {ci[0]:.4f}")
    print(f"A1 = {A1:.4f} ± {ci[1]:.4f}")
    print(f"TD = {TD:.4f} ± {ci[2]:.4f}")
    print(f"tau = {tau:.4f} ± {ci[3]:.4f}")
    print(f"r^2 = {r_squared}")
    print(f"O2 deficit = {O2_def:.4f}")
    print(f"O2_def_estimate = {O2_def_estimate:.4f}")
    print(f"Time to steady state = {TT_ss:.4f}")

    plt.figure(figsize=(10, 4))
    plt.scatter(x_data, final_residuals, color='purple', s=20)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Residuals - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Residuals (Observed - Fitted)")
    plt.grid(True)
    residuals_plot_path = os.path.join(plot_dir, f"{file_name}_VO2_on_residuals.png")
    plt.savefig(residuals_plot_path)
    plt.show()
    plt.close()

    # Final VO2 model plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="Observed VO2", color='blue', s=20)
    plt.scatter(all_outlier_x, all_outlier_y, label="Outliers (≥ 2.5 SD)", color='red', s=20, marker='x')
    plt.plot(x_data, model(x_data, *params), label="Fitted Model", color='red', linestyle='--')

    equation_text = f"VO₂(t) = {A0:.2f} + ({A1:.2f} - {A0:.2f}) × [1 - exp(-(t - {TD:.2f}) / {tau:.2f})]"
    plt.text(
        0.05, 0.95, equation_text,
        transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', color='black',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7)
    )

    plt.title(f"VO2 Kinetics - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("VO2 (L/min)")
    plt.legend()
    plot_path = os.path.join(plot_dir, f"{file_name}_VO2_on_kinetics_plot.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    results_dict = {
            
        "file": file_name,
        "Resting_EE":Resting_EE, # kcal/hr
        "Walking_EE":Walking_EE, # kcal/hr
        "Walking_economy":Walking_economy, # kcal/meter
        "Net_energetic_cst_walking":Net_energetic_cst_walking, # kcal/m
        "VO2_on_A0": A0, # L/min
        "VO2_on_A1": A1, # L/min
        "VO2_on_PCA": VO2_on_PC, # L/min
        "VO2_on_TD": TD, #  sec
        "VO2_on_tau": tau, # sec
        "VO2_on_MRT": VO2_MRT, # sec
        "VO2_on_RMSE":rmse, # L/min
        "VO2_Final_R^2":r_squared,
        "VO2_on_CI_A0": ci[0], # L/min
        "VO2_on_CI_A1": ci[1], # L/min
        "VO2_on_CI_TD": ci[2], # sec 
        "VO2_on_CI_tau": ci[3], # sec 
        "VO2_on_O2_def": O2_def, # Liters
        "VO2_on_O2_def_estimate": O2_def_estimate, # Liters
        "Total_VO2_Liters":total_area,
        "VO2_on_TT_ss": TT_ss # sec

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