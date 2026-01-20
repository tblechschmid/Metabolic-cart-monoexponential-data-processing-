# Metabolic-cart-monoexponential-data-processing-
Collection of scripts that will batch process monoexponential analysis of excel files exported from Vyaire Vmax metabolic carts. This script will run in parallel on all available CPU cores. 

1]	Save your data from the Vyaire Vmax cart in .xlxs format.
  a)	Use additional scripts to add HR and treadmill belt speed if necessary.
  b) Files should be labeled as V1 for baseline and V4 for follow-up.
  c) For participants with multiple tests on a single day, files should be saved with the extensions "_V1_T1" for treadmill test #1 at baseline, "_V1_T2" for treadmill test #2 at baseline, etc.
    c)i}	If you have called the visit variables differently than the _V1_T1 format for example, you will need to change the script (or your saving convention) to match your naming convention or the variable to match the script.  
2]	Place your .xlxs data files in the “data” subfolder within the working directory. 
3]	(ONLY IF NEEDED) Edit “File_processing” to pull data from the columns that you want to run. 
  a)	If you edit variables, make sure that you have python throw nan values (see the code for ‘Matched HR’ for reference). 
  b)	Edit the results dictionary to include the variables that you want in your analysis.
4]	(If no previsously published data for your protocol is avaiable) Run the “Average-all-indices-initial-guesses-and-bounds.py” script. to get the estimates for initial guesses of each measured parameter.
  a)	Open the output excel file and navigate to the baseline summary tabs for on- and off- kinetics. 
  b)	Then edit each model initial guesses in each of the scripts accordingly. 
5]	(ONLY IF NEEDED) Edit “Batch_process” by commenting in/out the scripts that you want to run.
  a)	This will then only run the scripts in the folder that you need and shorten analysis time. 
6]	Make sure all the files that python will be running the script on are closed.
  a)	Example 1: You cannot have the “Kinetics_summary.xlsx” file open when you run batch process.
  b)	Example #2: You also cannot have a participant treadmill data excel file open, it will throw an error and you will have to start the script over to include that participant. 
7]	Run batch process by clicking the play button or pressing f5 (Varies by IDE).
8]	When batch_process is done, open the “Kinetics_summary.xlsx” file and view the results.  
  a)	The file will have tabs that take the averages of the participant files from all files for a participant in _V1 and _V4, separate tab for the average of the second two treadmill tests if available, and data from individual tests.

