# Import needed packages / libraries
from pathlib import Path # Object-oriented filesystem paths (used to handle folders/files like 'data' and 'out')
import json # Read/write JSON files (store results from each run into .json files)
import traceback # Capture and format detailed error information for logging
from datetime import datetime # Get current date/time for timestamping errors in logs
from concurrent.futures import ProcessPoolExecutor, as_completed # Run tasks in parallel using multiple CPU cores (ProcessPoolExecutor runs each file in its own process;
# as_completed lets us check results as soon as each finishes)
import pandas as pd  # Data manipulation and analysis (used here to aggregate JSON results into Excel)
import File_processing # Custom preprocessing script that prepares dataframes from each Excel file
import subprocess # Needed to run the Excel_Summary_Averages.py script after all the .json files are complete. 

# Import the scripts that you need to run in the working directory (i.e. the folder that Batch_process.py is located in)
## On Kinetics scripts that will be run
import Monoexponential_VO2_on_kinetics
import Monoexponential_VCO2_on_kinetics
import Monoexponential_PetCO2_on_kinetics
import Monoexponential_RER_norm_VE_on_kinetics
import Monoexponential_VE_on_kinetics
import Monoexponential_Vt_on_kinetics
import Monoexponential_RR_on_kinetics
import Monoexponential_HR_on_kinetics
import Monoexponential_RER_on_kinetics
import RER_deficit_9p
import Monoexponential_Ti_Ttot_on_kinetics
import Monoexponential_O2_pulse_on_kinetics

## Off Kinetics scripts that will be run
import Monoexponential_VO2_off_kinetics
import Monoexponential_VCO2_off_kinetics
import Monoexponential_PetCO2_off_kinetics
import Monoexponential_VE_off_kinetics
import Monoexponential_Vt_off_kinetics
import Monoexponential_RR_off_kinetics
import Monoexponential_HR_off_kinetics
import Monoexponential_Ti_Ttot_off_kinetics
import Monoexponential_O2_pulse_off_kinetics

## Additional Graphing and smoothed timepoint reporting
import Smoothed_timepoints_9s
import RER_graph_9p

# Define folders and paths
data_folder = Path("data")  # Folder containing your Excel files
results_folder = Path("out")
results_folder.mkdir(exist_ok=True)
error_log_path = Path("error_log.txt")  # Path for error logging

# Check for Excel files in the data folder
excel_files = list(data_folder.glob("*.xlsx"))

# Function to process a single file
def process_file(file_path):
    file_name = file_path.stem  # without .xlsx
    try:
        # Preprocess the Excel file and rutrun the dataframes from the File_processing.py script
        (
            file_name,
            ex_VO2_df,
            rec_VO2_df,
            ex_VCO2_df,
            rec_VCO2_df,
            ex_VE_df,
            rec_VE_df,
            ex_Vt_df,
            rec_Vt_df,
            ex_RER_df,
            RER_graph_df,
            ex_Ti_Ttot_df,
            rec_Ti_Ttot_df,
            ex_o2_pulse_df,
            rec_o2_pulse_df,
        ) = File_processing.preprocess_excel_file(file_path)
        
    # print errors in the File_processing.py script
    except Exception as e:
        with open(error_log_path, "a") as log_file:
            log_file.write(f"\n[{datetime.now()}] Error preprocessing file: {file_path.name}\n")
            log_file.write(f"Error message: {str(e)}\n")
            log_file.write(f"{traceback.format_exc()}\n")
        print(f"Preprocessing error in {file_path.name}. Logged to error_log.txt")
        return None

    # Define all script runs in the working direectory in a list of tuples abd tell python which function in that script that you want to run
    runs = [
        ## On kinetics scripts that you want to run
        ("Monoexponential_VO2_on_kinetics", Monoexponential_VO2_on_kinetics.run_kinetics, ex_VO2_df),
        ("Monoexponential_VCO2_on_kinetics", Monoexponential_VCO2_on_kinetics.run_kinetics, ex_VCO2_df),
        ("Monoexponential_PetCO2_on_kinetics", Monoexponential_PetCO2_on_kinetics.run_kinetics, ex_VCO2_df),
        ("Monoexponential_VE_on_kinetics", Monoexponential_VE_on_kinetics.run_kinetics, ex_VE_df),
        ("Monoexponential_Vt_on_kinetics", Monoexponential_Vt_on_kinetics.run_kinetics, ex_Vt_df),
        ("Monoexponential_RR_on_kinetics", Monoexponential_RR_on_kinetics.run_kinetics, ex_VO2_df),
        ("Monoexponential_HR_on_kinetics", Monoexponential_HR_on_kinetics.run_kinetics, ex_VO2_df),
        ("Monoexponential_RER_on_kinetics", Monoexponential_RER_on_kinetics.run_kinetics, ex_RER_df),# note the R^2 for this fitting is usally very low. A function with more terms will be a greater fit and may provide a better predictor for CVD and related events. 
        ("RER_deficit_9p", RER_deficit_9p.run_def_calculation, ex_RER_df),
        ("Monoexponential_RER_norm_VE_on_kinetics", Monoexponential_RER_norm_VE_on_kinetics.run_kinetics, ex_RER_df),
        ("Monoexponential_Ti_Ttot_on_kinetics", Monoexponential_Ti_Ttot_on_kinetics.run_kinetics, ex_Ti_Ttot_df),
        ("Monoexponential_O2_pulse_on_kinetics", Monoexponential_O2_pulse_on_kinetics.run_kinetics, ex_o2_pulse_df),

        ## Off kinetics scripts that you want to run
        ("Monoexponential_VO2_off_kinetics", Monoexponential_VO2_off_kinetics.run_kinetics, rec_VO2_df),
        ("Monoexponential_VCO2_off_kinetics", Monoexponential_VCO2_off_kinetics.run_kinetics, rec_VCO2_df),
        ("Monoexponential_PetCO2_off_kinetics", Monoexponential_PetCO2_off_kinetics.run_kinetics, rec_VCO2_df),
        ("Monoexponential_VE_off_kinetics", Monoexponential_VE_off_kinetics.run_kinetics, rec_VE_df),
        ("Monoexponential_Vt_off_kinetics", Monoexponential_Vt_off_kinetics.run_kinetics, rec_Vt_df),
        ("Monoexponential_RR_off_kinetics", Monoexponential_RR_off_kinetics.run_kinetics, rec_VO2_df),
        ("Monoexponential_HR_off_kinetics", Monoexponential_HR_off_kinetics.run_kinetics, rec_VO2_df),
        ("Monoexponential_Ti_Ttot_off_kinetics", Monoexponential_Ti_Ttot_off_kinetics.run_kinetics, rec_Ti_Ttot_df),
        ("Monoexponential_O2_pulse_off_kinetics", Monoexponential_O2_pulse_off_kinetics.run_kinetics, rec_o2_pulse_df),

        ## Graphing and smoothed timepoint reporting 
        ("Smoothed_timepoints_9s", Smoothed_timepoints_9s.run_smoothed_timpoints, ex_RER_df), # Reports values at each minute from 9-second smoothed data for all metabolic cart variables
        ("RER_graph_9p", RER_graph_9p.run_graph, RER_graph_df),
        
        ## For future use
        # ("Monoexponential_RER_norm_VE_on_kinetics_9p", Monoexponential_RER_norm_VE_on_kinetics_9p.run_kinetics, ex_RER_df),
        # ("Monoexponential_RER_VE_CF_on_kinetics", Monoexponential_RER_VE_CF_on_kinetics.run_kinetics, ex_RER_df),
    ]

    # Run each module (i.e. the script and function) individually for each excel file with its own error handling
    results_per_file = {}
    for name, func, df in runs:
        try:
            res = func(file_name, df)
            results_per_file[name] = res  # store result, even if None
        except Exception as e:
            results_per_file[name] = None  # ensure key exists
            with open(error_log_path, "a") as log_file:
                log_file.write(f"\n[{datetime.now()}] Error in {name} for file: {file_path.name}\n")
                log_file.write(f"Error message: {str(e)}\n")
                log_file.write(f"{traceback.format_exc()}\n")
            print(f"Error in {name} for {file_path.name}. Logged to error_log.txt")

    # Save results to JSON per input Excel file
    json_path = results_folder / f"{file_name}_results.json"
    with open(json_path, "w") as jf:
        json.dump(results_per_file, jf, indent=4)
    return file_name

# Main parallel execution: allows access to all available CPU cores for paralell processing instead of serial processing 
def main():
    excel_files = list(data_folder.glob("*.xlsx")) # Collect all Excel files with .xlsx extension in the data_folder directory
    if not excel_files:  # If no Excel files are found, print a message and stop execution
        print("No Excel files found in data folder.") 
        return
    # Create a pool of worker processes using all available CPU cores
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, f): f for f in excel_files} # Submit each file to be processed in parallel by process_file() and Store mapping between the future object (task) and the file it belongs to
        for fut in as_completed(futures):  # Iterate over completed tasks as they finish (not in the order submitted)
            file_path = futures[fut] # Retrieve the file path associated with this future
            try:
                result_file = fut.result() # Get the result returned by process_file()
                if result_file: # If a result was returned (not None), print confirmation
                    print(f"Finished processing {result_file}")
            except Exception as e: # If any error occurred while processing this file, print the file name and error message
                print(f"Unexpected error in {file_path.name}: {e}")

# Aggregation script to put all results into a single .json file (run separately after all JSON files are created)
def aggregate_json_to_excel():
    all_results = []
    for json_file in results_folder.glob("*.json"):
        with open(json_file, "r") as jf:
            data = json.load(jf)
            # Flatten JSON: combine per-run results into a single dictionary
            flat_res = {"file": json_file.stem.replace("_results", "")}
            for run_name, run_res in data.items():
                if run_res:
                    for k, v in run_res.items():
                        flat_res[f"{run_name}_{k}"] = v
            all_results.append(flat_res)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel("Kinetics_summary.xlsx", index=False)
        print("Aggregated results saved to Kinetics_summary.xlsx")
    else:
        print("No JSON results to aggregate.")

if __name__ == "__main__":
    main()

    # Run Excel_Summary_Averages.py after all files are processed
    try:
        print("Running Excel_Summary_Averages.py to compile results...")
        subprocess.run(["python", "Excel_Summary_Averages.py"], check=True)
    except Exception as e:
        print(f"❌ Error running Excel_Summary_Averages.py: {e}")
        
print("Kinetics Analysis complete")


