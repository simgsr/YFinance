import pandas as pd
import os

# Step 1: Ask for input CSV file path
print("Step 1: Input File Selection")
file_path = input("Please enter the full path to your CSV file containing tickers: ")

# Read the CSV file
try:
    df = pd.read_csv(file_path)
    tickers = df.iloc[:, 0].tolist()  # Get tickers from first column
    total_tickers = len(tickers)
except FileNotFoundError:
    print("Error: File not found. Please check the path and try again.")
    exit()
except Exception as e:
    print(f"Error: Could not read the file: {e}")
    exit()

if total_tickers == 0:
    print("Error: No tickers found in the CSV file.")
    exit()

# Step 2: Ask for number of tickers per output file
print(f"\nStep 2: Configuration")
print(f"Found {total_tickers} tickers in the file.")
while True:
    try:
        tickers_per_file = int(input("How many tickers do you want in each output CSV file? "))
        if tickers_per_file <= 0:
            print("Please enter a positive number.")
            continue
        if tickers_per_file > total_tickers:
            print(f"Cannot exceed total number of tickers ({total_tickers}).")
            continue
        break
    except ValueError:
        print("Please enter a valid integer.")

# Step 3: Generate output CSV files
print(f"\nStep 3: Generating Output Files")
num_files = (total_tickers + tickers_per_file - 1) // tickers_per_file
output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else "."
base_name = os.path.splitext(os.path.basename(file_path))[0]

for i in range(num_files):
    start_idx = i * tickers_per_file
    end_idx = min((i + 1) * tickers_per_file, total_tickers)
    chunk = tickers[start_idx:end_idx]

    new_df = pd.DataFrame(chunk, columns=[df.columns[0]])
    output_file = f"{output_dir}/{base_name}_part{i+1}.csv"

    new_df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(chunk)} tickers")

print(f"\nCompleted! Generated {num_files} CSV files.")
