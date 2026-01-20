def run_kinetics(file_name, ex_RER_df): 
    import numpy as np
    import pandas as pd
    import os
    import json
    from scipy.optimize import curve_fit
    from scipy.stats import t
    from scipy.integrate import quad
    import matplotlib.pyplot as plt
    from openpyxl import load_workbook, Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    def model(x, A0, A1, TD, tau):
        x = np.asarray(x, dtype=float)
        return np.where(x < TD, A0, A0 - (A0 - A1) * (1 - np.exp(-(x - TD) / tau)))

    # Output directories
    out_dir = os.path.join(os.getcwd(), "out")
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, file_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Baseline
    last_60s_rest_df = ex_RER_df[(ex_RER_df['ExTime'] >= -60) & (ex_RER_df['ExTime'] <= 0)]
    A0_mean = last_60s_rest_df['RER'].mean()
    print(f"A0_mean = {A0_mean}")

    variables_to_smooth = ['RER', 'RER_norm_VE', 'VE', 'VO2', 'Ti_Ttot', 'VCO2', 'Vt', 'RR']
    variable_a0_init = {f"{var}_A0_init": last_60s_rest_df[var].mean() for var in variables_to_smooth}

    # Filter for post-onset
    ex_RER_df_positive = ex_RER_df[ex_RER_df['ExTime'] >= 0].copy()
    if ex_RER_df_positive.empty:
        print(f"No positive RER values in {file_name}. Skipping this file.")
        return None

    print(f"Removed {ex_RER_df.shape[0] - ex_RER_df_positive.shape[0]} negative values. Keeping {ex_RER_df_positive.shape[0]} rows.")

    # Convert to TimedeltaIndex
    ex_RER_df_positive['ExTime_timedelta'] = pd.to_timedelta(ex_RER_df_positive['ExTime'], unit='s')
    ex_RER_df_positive.set_index('ExTime_timedelta', inplace=True)

    # Smooth all variables
    for var in variables_to_smooth:
        ex_RER_df_positive[f"{var}_smoothed"] = (
            ex_RER_df_positive[var]
            .rolling('9s', center=True)
            .mean()
            .bfill()
            .ffill()
        )

    x_data = ex_RER_df_positive['ExTime'].values
    y_data = ex_RER_df_positive['RER'].values
    y_data_smooth = ex_RER_df_positive['RER_smoothed'].values

    # Initial guess
    A0_init = A0_mean
    A1_init = np.mean(y_data_smooth[x_data >= (x_data[-1] - 60)])
    tau_init = 50.8

    # Time delay guess
    upper_threshold = A0_mean * 0.95
    above_threshold = y_data_smooth < upper_threshold
    min_duration = 15
    td_init = 32.7
    found = False
    for i in range(len(x_data)):
        if above_threshold[i]:
            td_start = x_data[i]
            for j in range(i + 1, len(x_data)):
                if not above_threshold[j]:
                    break
                if (x_data[j] - td_start) >= min_duration:
                    td_init = td_start
                    found = True
                    break
            if found: break

    if not found:
        print("No sustained -5% drop found, using td_init = 10s")

    # Bounds
    initial_guess = [A0_init, A1_init, td_init, tau_init]
    lower_bounds = [A0_init - 1e-3, y_data_smooth.min(), 7.2, 23.1] 
    upper_bounds = [A0_init + 1e-3, y_data_smooth.max(), 281.2, x_data.max()]

    def get_var_at_time(varname, t):
        idx = np.argmin(np.abs(x_data - t))
        return ex_RER_df_positive.iloc[idx][f"{varname}_smoothed"]

    results_dict = {"file": file_name}
    results_dict.update(variable_a0_init)

    eval_times = [0, 30, 60, 120, 180, 240, 300, 360]
    for eval_time in eval_times:
        for var in ['RER', 'RER_norm_VE', 'VE', 'VO2', 'VCO2', 'Ti_Ttot','Vt', 'RR']:
            results_dict[f"{var}_on_{eval_time}s_9p"] = get_var_at_time(var, eval_time)

    fit_failed = False

    try:
        params, covariance = curve_fit(model, x_data, y_data_smooth, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
    except RuntimeError:
        print("Fit failed.")
        fit_failed = True

    # Always plot smoothed variables
    for var in variables_to_smooth:
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

    if not fit_failed:
        A0, A1, TD, tau = params
        errors = np.sqrt(np.diag(covariance))
        dof = max(0, len(x_data) - len(params))
        t_crit = t.ppf(0.975, dof)
        ci = t_crit * errors

        fitted = model(x_data, *params)
        ss_res = np.sum((y_data_smooth - fitted) ** 2)
        ss_tot = np.sum((y_data_smooth - np.mean(y_data_smooth)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((y_data_smooth - fitted) ** 2))
        TT_ss = tau * 4
        rest_RER_area = A0 * x_data.max()
        auc = quad(lambda t: model(t, *params), 0, x_data.max())[0]
        auc_trapz = np.trapz(y_data, x=x_data)
        RER_def = rest_RER_area - auc
        RER_def_RAW = rest_RER_area - auc_trapz
        RER_PCA = A0 - A1 # primary component amplitude
        RER_MRT = TD + tau

        results_dict.update({
            # "RER_on_A0_init_9p": A0_init,
            # "RER_on_A1_init_9p": A1_init,
            "RER_on_A0_9p": A0,
            "RER_on_A1_9p": A1,
            "RER_on_PCA_9p": RER_PCA,
            "RER_on_TD_9p": TD,
            "RER_on_tau_9p": tau,
            "RER_on_MRT_9p" : RER_MRT,
            "RER_on_Final_R^2_9p": r_squared,
            "RER_on_RMSE_9p": rmse,
            "RER_on_CI_A0_9p": ci[0],
            "RER_on_CI_A1_9p": ci[1],
            "RER_on_CI_TD_9p": ci[2],
            "RER_on_CI_tau_9p": ci[3],
            "RER_on_def_9p": RER_def,
            "RER_on_def_RAW_9p": RER_def_RAW,
            "RER_on_TT_ss_9p": TT_ss,
        })

        # Fitted RER plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, label="Original RER", color='gray', alpha=0.4)
        plt.plot(x_data, y_data_smooth, label="Smoothed RER (9s)", color='blue')
        plt.plot(x_data, fitted, label="Fitted Model", color='red', linestyle='--')
        plt.title(f"RER Kinetics - {file_name}")
        eq = f"RER(t) = {A0:.2f} - ({A1:.2f} - {A0:.2f}) × [1 - exp(-(t - {TD:.2f}) / {tau:.2f})]"
        plt.text(0.05, 0.95, eq, transform=plt.gca().transAxes,
                 fontsize=10, va='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        plt.xlabel("Time (s)")
        plt.ylabel("RER")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"{file_name}_RER_on_kinetics_plot_9p.png"))
        plt.close()
   
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
