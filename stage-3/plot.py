import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
cpu_df = pd.read_csv("cpu_times.csv")
gpu_df = pd.read_csv("gpu_times.csv")

# Phases to compare (exclude TOTAL and RESULT and percentage columns)
phases = ['GStime', 'GBtime', 'STtime', 'NMS', 'DTtime', 'Hystime']

# Compute average times for each phase
cpu_avg = cpu_df[phases].mean()
gpu_avg = gpu_df[phases].mean()

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    'CPU': cpu_avg,
    'GPU': gpu_avg
})

# Plotting
plt.figure(figsize=(10, 6))
comparison_df.plot(kind='bar', log=True)
plt.title('Average Phase Execution Time: CPU vs GPU (Log Scale)')
plt.ylabel('Time (microseconds, log scale)')
plt.xticks(rotation=45)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

# Save the plot as a PDF
plt.savefig("cpu_gpu_comparison_log.pdf")

