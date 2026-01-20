import os
import pandas as pd
import matplotlib.pyplot as plt

def run_graph(file_name, RER_graph_df):
    # Ensure the "out/<file_name>" directory exists
    out_dir = os.path.join(os.getcwd(), "out", file_name)
    os.makedirs(out_dir, exist_ok=True)

    # Convert 'ExTime' to timedelta and set as index for rolling
    RER_graph_df['ExTime_timedelta'] = pd.to_timedelta(RER_graph_df['ExTime'], unit='s')
    RER_graph_df.set_index('ExTime_timedelta', inplace=True)
    
    # Apply 9-second rolling average (centered)
    RER_graph_df['RER_smoothed'] = RER_graph_df['RER'].rolling('9s', center=True).mean()
    
    # Reset index to restore original
    RER_graph_df.reset_index(drop=True, inplace=True)

    # Calculate mean RER before exercise (ExTime < 0)
    pre_exercise_mask = RER_graph_df['ExTime'] < 0
    neg_rer_mean = RER_graph_df.loc[pre_exercise_mask, 'RER_smoothed'].mean()

    # Debug prints (optional)
    print(f"[{file_name}] Mean RER smoothed pre-exercise: {neg_rer_mean}")

    # Identify key times
    start_time = 0  # Always ExTime = 0
    end_times = RER_graph_df.loc[RER_graph_df['Marker'] == 2, 'ExTime']
    end_time = end_times.iloc[0] if not end_times.empty else None

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(RER_graph_df['ExTime'], RER_graph_df['RER_smoothed'], label="Smoothed RER", color='blue')

    # Vertical line at ExTime = 0
    plt.axvline(x=start_time, color='green', linestyle='--', label='Start Exercise')

    # Vertical line at first Work == 2
    if end_time is not None:
        plt.axvline(x=end_time, color='orange', linestyle='--', label='Work = 2')

    # Horizontal line for mean of pre-exercise smoothed values
    if pd.notna(neg_rer_mean):
        plt.axhline(y=neg_rer_mean, color='blue', linestyle=':', linewidth=1.5, label='Mean RER pre-exercise')
        plt.text(RER_graph_df['ExTime'].min(), neg_rer_mean, f'{neg_rer_mean:.2f}', color='blue', va='bottom')

    # Adjust y-limits
    y_min = min(RER_graph_df['RER_smoothed'].min(),
                neg_rer_mean if pd.notna(neg_rer_mean) else float('inf'))
    y_max = max(RER_graph_df['RER_smoothed'].max(),
                neg_rer_mean if pd.notna(neg_rer_mean) else float('-inf'))
    plt.ylim(y_min - 0.1, y_max + 0.1)

    # Final plot formatting
    plt.title(f"Smoothed RER Curves - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("RER")
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(out_dir, f"{file_name}_RER_smoothed_plot.png")
    plt.savefig(plot_path, dpi=300)
    # plt.show()
    plt.close()
