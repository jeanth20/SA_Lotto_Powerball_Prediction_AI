import csv

# Open the CSV file for reading
with open("lotto2015_2025 - Sheet1.csv", "r", newline="") as open_file:
    csv_reader = csv.reader(open_file)

    # Create a new file to save the cleaned data
    with open("cleaned_lotto.csv", "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)

        # Write the header row for the cleaned data
        csv_writer.writerow(["Year", "Date", "b1", "b2", "b3", "b4", "b5", "b6", "BonusBall"])

        # Iterate through each row in the original file
        for row in csv_reader:
            # Ensure the row has the expected number of columns (4 in this case)
            if len(row) == 4:
                year = row[0].strip()  # Extract the year
                date = row[1].strip()  # Extract the date
                series = row[2].strip()  # Extract the series of numbers
                bonus_ball = row[3].strip()  # Extract the bonus ball

                # Remove newline characters and join the series numbers with commas
                series_cleaned = ",".join(series.splitlines())

                # Split the cleaned series into individual numbers
                series_numbers = series_cleaned.split(",")

                # Write the cleaned data to the new file
                csv_writer.writerow([year, date] + series_numbers + [bonus_ball])

# Close the files after processing
open_file.close()
output_file.close()