import csv
import os
from typing import Dict, Any
from pathlib import Path

def save_to_csv(
    results: Dict[str, Any],
    topology_nodes_count: int,
    topology_file: str,
    app_count: int,
    allocation_method: str,
    results_folder: str = "Results"
) -> None:
    """
    Save simulation parameters and results to a CSV file in a Results folder.
    Prompts user to append to existing file or create a new one.
    """
    # Create Results folder if it doesn't exist
    results_path = Path(results_folder)
    results_path.mkdir(exist_ok=True)

    # Define CSV headers
    headers = [
        "Run ID", "Topology Nodes Count", "Topology File", "App Count", "Allocation Method",
        "Allocated Count", "Fog Node Allocations", "Cloud Allocations", "Utilized Fog Nodes",
        "Total CPU (MIPS)", "Total RAM (GB)", "Total Storage (GB)", "Total Bandwidth (Gbps)",
        "Total Latency (s)", "Total Makespan (s)", "Total Workload (WU)", "Total Energy (W)", "Total Cost (€)"
    ]

    # Find existing CSV files and determine next file number
    csv_files = list(results_path.glob("simulation_results_*.csv"))
    if csv_files:
        latest_num = max(int(f.stem.split('_')[-1]) for f in csv_files if f.stem.startswith("simulation_results_"))
        default_file = results_path / f"simulation_results_{latest_num:03d}.csv"
        next_file_num = latest_num + 1
    else:
        default_file = results_path / "simulation_results_001.csv"
        next_file_num = 1  # Start at 001 if no files exist

    # Prompt user for append or new file
    print(f"Existing results file: {default_file.name if csv_files else 'None'}")
    while True:  # Loop until a valid input is provided
        choice = input(f"Do you want to append to the existing '{default_file.name}' (a), create a new file (n), or exit (e)? [a/n/e]: ").strip().lower()
        
        if choice == 'e':
            print("Exiting without saving.")
            return
        elif choice == 'a' and csv_files and os.path.exists(default_file):
            csv_filename = default_file
            # Get next Run ID from existing file
            with open(csv_filename, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # Skip header
                run_id = sum(1 for row in reader if row) + 1  # Count existing rows
            break
        elif choice == 'n':
            csv_filename = results_path / f"simulation_results_{next_file_num if csv_files else 1:03d}.csv"
            run_id = 1
            break
        else:
            print("Invalid input. Please enter 'a', 'n', or 'e'.")

    # Prepare data row
    data = [
        f"Simulation Run {run_id:03d}",
        topology_nodes_count,
        topology_file,
        app_count,
        allocation_method,
        results['allocated_count'],
        results['fog_node_allocations'],
        results['cloud_allocations'],
        results['utilized_fog_nodes_count'],
        int(results['total_CPU']),  # Convert MIPS to integer
        int(results['total_RAM'] / 1024),  # Convert MB to GB and ensure integer
        int(results['total_storage'] / 1024),  # Convert MB to GB and ensure integer
        int(results['total_Bandwidth'] / 1000),  # Convert Mbps to Gbps and ensure integer
        int(results['total_Latency']),
        int(results['total_Makespan']),
        int(results['total_Workload']),
        int(results['total_Energy']),
        int(results['total_Cost'])
    ]

    # Write or append to CSV
    write_header = not os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(headers)
        writer.writerow(data)

    print(f"Results saved to: {csv_filename}")