import pandas as pd
from pathlib import Path
import json

# Folder containing JSON files
results_folder = Path("out")
json_files = list(results_folder.glob("*.json"))

if not json_files:
    print("No JSON files found in 'out' folder.")
    exit()

# Determine variable order from all runs in batch process
batch_order = []
for json_file in json_files:
    with open(json_file, "r") as jf:
        data = json.load(jf)
        for run_name, run_res in data.items():
            if run_res:
                for k in run_res.keys():
                    if k not in batch_order:
                        batch_order.append(k)  # Add unique keys in order

# Collect all results
all_results = []
for json_file in json_files:
    with open(json_file, "r") as jf:
        data = json.load(jf)
        flat_res = {"file": json_file.stem.replace("_results", "")}
        for run_name, run_res in data.items():
            if run_res:
                for k, v in run_res.items():
                    flat_res[f"{k}"] = v  # Remove script name prefix
        all_results.append(flat_res)

# Create initial dataframe
summary_df = pd.DataFrame(all_results)

# DEBUG: check what files and extracted info look like
print(summary_df[['file']])  # check file names are loaded

# Extract Trial_ID (e.g., 'MRS_01_V1_T2' from 'MRS_01_V1_T2_IC')
summary_df['Trial_ID'] = summary_df['file'].str.extract(r'(.*?_T\d)')[0]

# Extract Participant and Visit
summary_df['Participant'] = summary_df['Trial_ID'].str.extract(r'(^.*?)(_V\d)')[0]
summary_df['Visit'] = summary_df['Trial_ID'].str.extract(r'(^.*?)(_V\d)')[1]  # _V1, _V4

# Extract Trial Number (e.g., _T2, _T3)
summary_df['Trial_Num'] = summary_df['Trial_ID'].str.extract(r'(_T\d)')[0]

# DEBUG: check that extracted columns are correct
print(summary_df[['file', 'Trial_ID', 'Participant', 'Visit', 'Trial_Num']])


# Function to pivot and order variables by visit (_V1 then _V4)
def pivot_and_group_by_visit(df, trial_filter=None):
    if trial_filter:
        df = df[df['Trial_Num'].isin(trial_filter)]
    grouped = df.drop(columns=['file', 'Trial_ID', 'Trial_Num']).groupby(['Participant', 'Visit']).mean(numeric_only=True).reset_index()
    pivot_df = grouped.pivot(index='Participant', columns='Visit')
    pivot_df.columns = [f"{col}{visit}" for col, visit in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Order: all _V1 first, then all _V4
    ordered_cols = ['Participant']
    for visit in ['_V1', '_V4']:
        for var in batch_order:
            col_name = f"{var}{visit}"
            if col_name in pivot_df.columns:
                ordered_cols.append(col_name)
    return pivot_df[ordered_cols]

# 3-trial by visit averages (all trials)
pivot_df = pivot_and_group_by_visit(summary_df)

# T2_T3 averages 
t2t3_pivot = pivot_and_group_by_visit(summary_df, trial_filter=['_T2', '_T3'])

# T1 by visit
t1_pivot = pivot_and_group_by_visit(summary_df, trial_filter=['_T1'])

# T2 by visit
t2_pivot = pivot_and_group_by_visit(summary_df, trial_filter=['_T2'])

# T3 by visit
t3_pivot = pivot_and_group_by_visit(summary_df, trial_filter=['_T3'])

# Write all sheets to Excel 
output_file = "Kinetics_summary.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
    pivot_df.to_excel(writer, sheet_name='3-trial by visit', index=False)
    t2t3_pivot.to_excel(writer, sheet_name='T2_T3 averages by visit', index=False)
    t1_pivot.to_excel(writer, sheet_name='T1 by visit', index=False)
    t2_pivot.to_excel(writer, sheet_name='T2 by visit', index=False)
    t3_pivot.to_excel(writer, sheet_name='T3 by visit', index=False)

print(f"Aggregated Excel file saved as '{output_file}'")

