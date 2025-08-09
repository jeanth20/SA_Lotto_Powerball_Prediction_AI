import csv
import numpy as np
import pandas as pd

# Open the CSV file for reading
with open("cleaned_lotto_powerball.csv", "r", newline="") as open_file:
    csv_reader = csv.reader(open_file)

    # Read all rows into a list for frequency calculation
    rows = list(csv_reader)

# Extract all ball numbers (b1 to b6 and BonusBall) into a flat list
all_numbers = []
for row in rows[1:]:  # Skip the header row
    all_numbers.extend([int(row[i].strip()) for i in range(2, 7)])  # b1 to BonusBall

# Calculate frequency using pandas
frequency = pd.Series(all_numbers).value_counts().sort_index()

# Save frequency data to a new CSV file
with open("frequency_lotto_powerball.csv", "w", newline="") as freq_file:
    csv_writer = csv.writer(freq_file)
    csv_writer.writerow(["Number", "Frequency"])  # Write header
    for number, freq in frequency.items():
        csv_writer.writerow([number, freq])

with open("frequency_lotto_powerball_h_l.csv", "w", newline="") as freq_file:
    csv_writer = csv.writer(freq_file)
    frequency = frequency.sort_values(ascending=False)
    csv_writer.writerow(["Number", "Frequency"])  # Write header
    for number, freq in frequency.items():
        csv_writer.writerow([number, freq])


print("Frequency data saved to 'frequency_lotto.csv'")
