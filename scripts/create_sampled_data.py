# Script to sample 50 rows per label from multiple large CSVs
import pandas as pd
import os
import csv

# List of CSV files to process
csv_files = [
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
]

# Labels to sample
labels = ['BENIGN', 'DDos', 'Dos', 'Port Scan', 'Brute Force']

# How many samples per label
n_samples = 50

# Store samples here
samples = {label: [] for label in labels}
header = None

for file in csv_files:
    path = file if os.path.exists(file) else os.path.join('d:/test', file)
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            file_header = next(reader)
            if header is None:
                header = file_header
            for row in reader:
                if len(row) < 2:
                    continue
                label = row[-1].strip()
                if label in samples and len(samples[label]) < n_samples:
                    samples[label].append(row)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Write to new CSV
output_file = 'sampled_data.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for label in labels:
        for row in samples[label]:
            writer.writerow(row)

print(f"Sampled data written to {output_file}")
