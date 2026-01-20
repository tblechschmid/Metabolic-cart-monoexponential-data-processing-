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
    return np.where(x < TD, A0, A0 + (A1 - A0) * (1 - np.exp(-(x - TD) / tau)))

# Main function to run kinetics model for Vt
def run_kinetics(file_name, ex_Vt_df):
    
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
    last_60s_rest_df = ex_Vt_df[(ex_Vt_df['ExTime'] >= -60) & (ex_Vt_df['ExTime'] <= 0)]
    A0_mean = last_60s_rest_df['Vt'].mean()
    print(f"A0_mean{A0_mean}")
    
    # Filter only positive Vt values
    ex_Vt_df_positive = ex_Vt_df[ex_Vt_df['ExTime'] >= 0]
    if ex_Vt_df_positive.empty:
        print(f"No positive Vt values in {file_name}. Skipping this file.")
        return None
    else:
        print(f"Removed {ex_Vt_df.shape[0] - ex_Vt_df_positive.shape[0]} negative exercise time values.")
        print(f"{ex_Vt_df_positive.shape[0]} positive exercise time values.")

    # Extract time and Vt (all positive exercise values from here)
    x_data = np.array(ex_Vt_df_positive['ExTime'])
    y_data = np.array(ex_Vt_df_positive['Vt'])

    # Initialize placeholder for all removed outliers
    all_outlier_x = np.array([])  # To store x-values of all removed outliers
    all_outlier_y = np.array([])  # To store y-values of all removed outliers

    # Iterative outlier removal and model fitting
    for iteration in range(5):
        print(f"\nIteration {iteration + 1}")
        
        # Set initial guesses for the equation
        A0_init = A0_mean # mean of rest
        A1_init = np.mean(y_data[x_data >= (x_data[-1] - 60)]) # mean last 60 sec of exercise
        tau_init = 10.9 
        
        # Prepare TD_initial guess (First breath where Vt experiences a +5% change and is sustained for 15 seconds)
        upper_threshold =  A0_mean* 1.10
        # Time and Vt arrays
        time_vals = x_data
        Vt_vals = y_data
        # Create a boolean mask where Vt > 105% of A0_mean
        above_threshold = Vt_vals > upper_threshold
        # Minimum sustained duration requirement
        min_duration = 15  # seconds
        # Default time delay value
        td_init = 13.8
        found = False  # Flag to check if a valid TD was found
        # Search for the first time Vt stays above threshold for >= 30 seconds
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
            print(f"No sustained +5% Vt change found; fallback")
        
        #Set lower bounds for the equation
        A0_lb = A0_init - 1e-3  # constrains the lower bound to very close to A0_mean
        A1_lb = A0_mean  #Contrains Vt ot resting Vt
        td_lb = 0  
        tau_lb = 0 
        
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
    
    #Calculate Vt deficit
    from scipy.integrate import quad
    # Define the fitted Vt function for integration
    def fitted_Vt(t):
        return A0 + (A1 - A0) * (1 - np.exp(-(t - TD) / tau)) if t >= TD else A0
    # Integration bounds
    x_start = 0
    x_end = x_data.max()  # final x value converted to minutes
    area_under_curve = (quad(fitted_Vt, x_start, x_end)[0]) / 60 # Perform definite integral for the final model results and convert it to liters  
    total_area = (A1 / 60) * x_end # End result to L
    print(f"Total exercise area = {total_area}") 
    print(f"Area under curve = {area_under_curve}")
    Vt_def = total_area - area_under_curve
    
    # Poole & Jones 2012 estimate
    Vt_on_PC = (A1-A0)
    Vt_def_estimate = (tau/60) * Vt_on_PC
    VT_MRT = TD + tau

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
    print(f"A0 = {A0:.4f} ± {ci[0]:.4f}")
    print(f"A1 = {A1:.4f} ± {ci[1]:.4f}")
    print(f"TD = {TD:.4f} ± {ci[2]:.4f}")
    print(f"tau = {tau:.4f} ± {ci[3]:.4f}")
    print(f"r^2 = {r_squared}")
    print(f"Vt deficit = {Vt_def:.4f}")
    print(f"Vt_def_estimate = {Vt_def_estimate:.4f}")
    print(f"Time to steady state = {TT_ss:.4f}")

    plt.figure(figsize=(10, 4))
    plt.scatter(x_data, final_residuals, color='purple', s=20)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Residuals - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Residuals (Observed - Fitted)")
    plt.grid(True)
    residuals_plot_path = os.path.join(plot_dir, f"{file_name}_Vt_on_residuals.png")
    plt.savefig(residuals_plot_path)
    plt.show()
    plt.close()

    # Final Vt model plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="Observed Vt", color='blue', s=20)
    plt.scatter(all_outlier_x, all_outlier_y, label="Outliers (≥ 2.5 SD)", color='red', s=20, marker='x')
    plt.plot(x_data, model(x_data, *params), label="Fitted Model", color='red', linestyle='--')

    equation_text = f"Vt(t) = {A0:.2f} + ({A1:.2f} - {A0:.2f}) × [1 - exp(-(t - {TD:.2f}) / {tau:.2f})]"
    plt.text(
        0.05, 0.95, equation_text,
        transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', color='black',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7)
    )

    plt.title(f"Vt Kinetics - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Vt (L/min)")
    plt.legend()
    plot_path = os.path.join(plot_dir, f"{file_name}_Vt_on_kinetics_plot.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    results_dict = {
            
        "file": file_name,
        "Vt_on_A0": A0, # L/breath
        "Vt_on_A1": A1, # L/breath
        "Vt_on_PCA": Vt_on_PC, # L/breath
        "Vt_on_TD": TD, # sec
        "Vt_on_tau": tau, # sec
        "Vt_on_MRT": VT_MRT, # sec 
        "Vt_Final_R^2":r_squared,
        "Vt_on_RMSE":rmse, # L/breath
        "Vt_on_CI_A0": ci[0], # L/breath
        "Vt_on_CI_A1": ci[1], # L/breath
        "Vt_on_CI_TD": ci[2], # sec
        "Vt_on_CI_tau": ci[3], # sec
        "Vt_on_deficit": Vt_def, # L * min / breath
        "Vt_on_def_estimate": Vt_def_estimate, # L * min / breath
        "Vt_on_TT_ss": TT_ss # sec
        
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