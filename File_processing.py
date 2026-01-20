# The purpose of this script is to take an individual excel file and make a bunch of data frames. These data frames are then taken by Batch_bropcess.py and passed to the appropriate scripts to run kinetic modeling, graphing, etc. 
# import the packacges that you need 
import os
import pandas as pd
import numpy as np  # Needed for np.nan

# Class to import an Excel file into a Pandas DataFrame
class ExcelImporter:
    def __init__(self, file_path):
        self.file_name = os.path.basename(file_path).replace(".xlsx", "")
        self.df = None
        self.load_excel(file_path)

    def load_excel(self, file_path):
        self.df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")
        # Rename 5th column to 'Work'
        cols = list(self.df.columns)
        if len(cols) > 4:
            cols[4] = 'Work'
            self.df.columns = cols
        print(f"Imported Excel file: {self.file_name}")

# Function to preprocess the Excel file and segment it into rest, exercise, and recovery
def preprocess_excel_file(file_path):
    excel_importer = ExcelImporter(file_path)
    raw_df = excel_importer.df

    # Rename 5th column to 'Work' again just to be safe (if needed)
    if 'Work' not in raw_df.columns:
        raw_df.rename(columns={raw_df.columns[4]: 'Work'}, inplace=True)
 
    # Rename 'Belt_speed' to 'TM_Belt_speed' if present
    if 'Belt_speed' in raw_df.columns:
        raw_df.rename(columns={'Belt_speed': 'TM_Belt_speed'}, inplace=True)

    # Identify phase markers by 'Work' column values
    start_ex_index = raw_df[raw_df['Work'] == 1].index[0]
    end_ex_index = raw_df[raw_df['Work'] == 2].index[0]
    end_file_index = raw_df[raw_df['Work'] == 3].index[0]

    # Create rest phase dataframe to compute A0_VE
    df_rest = raw_df.iloc[0:start_ex_index + 1].copy()

    max_rest_time = df_rest.iloc[:, 2].max()
    last_minute_rest = df_rest[df_rest.iloc[:, 2] >= max_rest_time - 60]

    # Add calculated columns to raw_df
    raw_df['RER'] = raw_df['VCO2'] / raw_df['VO2']
    # raw_df['VCO2_CF'] = raw_df['VCO2'] / (raw_df['VE(STPD)'] / A0_VE) #unvalidated correction factor that normalizes VCO2 to VE
    # raw_df['RER_CF'] = (raw_df['VCO2'] / (raw_df['VE(STPD)'] / A0_VE)) / raw_df['VO2'] #unvalidated correction factor that normalizes RER to VE
    raw_df['RER_norm_VE'] = raw_df['RER'] / raw_df['VE(STPD)']
    raw_df['Vt'] = raw_df['VE(STPD)'] / raw_df['RR']

    # Note: Skip adding O2_pulse to raw_df

    # Now re-slice rest, exercise, recovery with updated raw_df (new columns included)
    df_rest = raw_df.iloc[0:start_ex_index + 1].copy()
    df_exercise = raw_df.iloc[start_ex_index:end_ex_index + 1].copy()
    df_recovery = raw_df.iloc[end_ex_index:end_file_index + 1].copy()

    # Remove the last 20 seconds of rest phase
    max_rest_time = df_rest.iloc[:, 2].max()
    
    df_rest = df_rest[df_rest.iloc[:, 2] <= max_rest_time - 20].copy()

    # Remove the first 20 seconds of exercise phase
    min_ex_time = df_exercise.iloc[:, 2].min()
    df_exercise = df_exercise[df_exercise.iloc[:, 2] >= min_ex_time + 20].copy()

    # Combine rest and exercise into a new dataframe
    df_ex = pd.concat([df_rest, df_exercise], ignore_index=True)
    

    # Create new dataframe using last minute of exercise and all of recovery
    max_ex_time = df_exercise.iloc[:, 2].max()
    last_min_exercise = df_exercise[df_exercise.iloc[:, 2] >= max_ex_time - 60].copy()
    df_recov = pd.concat([last_min_exercise, df_recovery], ignore_index=True)

    # Reset time in recovery dataframe so that transition from exercise to recovery is the new zero
    offset_time = last_min_exercise.iloc[-1, 2]
    df_recov.iloc[:, 2] = df_recov.iloc[:, 2] - offset_time

    # Remove 20 seconds before and after the new zero point for recovery (phase I uptake kinetic response removal)
    df_rec = df_recov[(df_recov.iloc[:, 2] < -20) | (df_recov.iloc[:, 2] > 20)].copy()
   
    # Add MAtched HR column if Not present
    for df in [df_ex, df_rec]:
        if 'Matched HR' not in df.columns:
            df['Matched HR'] = np.nan
    
    ##### Create final dataframes to return #####

    ex_VO2_df = df_ex[['ExTime', 'VO2', 'RER', 'RR', 'Matched HR']].copy()
    rec_VO2_df = df_rec[['ExTime', 'VO2', 'RER','RR', 'Matched HR']].copy()

    ex_VCO2_df = df_ex[['ExTime', 'VCO2', 'PetCO2']].copy()
    rec_VCO2_df = df_rec[['ExTime', 'VCO2', 'PetCO2']].copy()

    ex_VE_df = df_ex[['ExTime', 'VE(STPD)']].copy()
    ex_VE_df.rename(columns={'VE(STPD)': 'VE'}, inplace=True) # Cleaner name used in the other scripts
    rec_VE_df = df_rec[['ExTime', 'VE(STPD)']].copy()
    rec_VE_df.rename(columns={'VE(STPD)': 'VE'}, inplace=True) # Cleaner name used in the other scripts

    ex_Vt_df = df_ex[['ExTime', 'Vt']].copy()
    rec_Vt_df = df_rec[['ExTime', 'Vt']].copy()

    ex_RER_df = df_ex[['ExTime', 'RER','TM_Belt_speed', 'VE(STPD)', 'Ti/Ttot', 'VO2', 'VCO2', 'Vt', 'RR','Matched HR']].copy()
    ex_RER_df.rename(columns={'VE(STPD)': 'VE', 'Ti/Ttot': 'Ti_Ttot'}, inplace=True)  # Cleaner name used in the other scripts
    # Add RER_norm_VE column (RER divided by VE) with zero replaced by NaN for safety
    ex_RER_df['RER_norm_VE'] = ex_RER_df['RER'] / ex_RER_df['VE'].replace(0, np.nan)

    ex_Ti_Ttot_df = df_ex[['ExTime', 'Ti/Ttot']].copy()
    ex_Ti_Ttot_df.rename(columns={'Ti/Ttot': 'Ti_Ttot'}, inplace=True)  # Cleaner name used in the other scripts
    rec_Ti_Ttot_df = df_rec[['ExTime', 'Ti/Ttot']].copy()
    rec_Ti_Ttot_df.rename(columns={'Ti/Ttot': 'Ti_Ttot'}, inplace=True)  # Cleaner name used in the other scripts

    RER_graph_df = df_ex[['ExTime', 'RER']].copy()
    RER_graph_df['Marker'] = raw_df['Work']
    RER_graph_df.dropna(subset=['ExTime', 'RER'], inplace=True)
    
    # Add RER_norm_VE column (RER divided by VE) with zero replaced by NaN for safety
    RER_graph_df['RER_norm_VE'] = ex_RER_df['RER'] / ex_RER_df['VE'].replace(0, np.nan)

    # Create O2_pulse dataframes conditionally, without adding to raw_df. Only handles this if HR data column is present. Spits out empty dataframe and skips the calcualtion otherwise.
    ex_o2_pulse_df = pd.DataFrame({
        'ExTime': df_ex['ExTime'],
        'O2_pulse': (df_ex['VO2'] * 1000) / df_ex['Matched HR']  # NaNs if HR missing
    })
    rec_o2_pulse_df = pd.DataFrame({
        'ExTime': df_rec['ExTime'],
        'O2_pulse': (df_rec['VO2'] * 1000) / df_rec['Matched HR']  # NaNs if HR missing
    })

# These are the data frames that Batch_process.py will pass to the different scripts 
    return (
        excel_importer.file_name,
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
    )
