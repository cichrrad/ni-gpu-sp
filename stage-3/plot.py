#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Plot SIZE vs TOTAL execution time from CSV logs and save the figure."
    )
    parser.add_argument("csvfile", help="Path to the CSV file containing log data")
    parser.add_argument("--output", default="plot.png",
                        help="Output image file for the saved figure (default: plot.png)")
    args = parser.parse_args()

    # Load CSV data and convert columns to numeric
    df = pd.read_csv(args.csvfile)
    df['SIZE'] = pd.to_numeric(df['SIZE'], errors='coerce')
    df['TOTAL'] = pd.to_numeric(df['TOTAL'], errors='coerce')

    # Sort the data by SIZE to create a coherent line plot
    df.sort_values('SIZE', inplace=True)

    # Create the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(df['SIZE'], df['TOTAL'], marker='o', linestyle='-', color='b')
    plt.xlabel("File Size (bytes)")
    plt.ylabel("Total Execution Time (microseconds)")
    plt.title("Correlation between File Size and Total Execution Time")
    plt.grid(True)
    plt.tight_layout()

    # Save the figure to file and do not show it interactively
    plt.savefig(args.output+"_size_x_total")
    print(f"Figure saved as {args.output}_[charttype].png")

if __name__ == "__main__":
    main()

