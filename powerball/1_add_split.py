import csv
# https://za.national-lottery.com/powerball/results/history
# Open the CSV file for reading
with open("powerball2009_2025 - Sheet1.csv", "r", newline="", encoding="utf-8") as open_file:
    csv_reader = csv.reader(open_file)

    # Create a new file to save the cleaned data
    with open("cleaned_lotto_powerball.csv", "w", newline="", encoding="utf-8") as output_file:
        csv_writer = csv.writer(output_file)

        # Write the header row for the cleaned data
        csv_writer.writerow(["Draw Date", "b1", "b2", "b3", "b4", "b5", "b6", "Jackpot", "Outcome"])

        # Skip the header row in the input file
        next(csv_reader)

        # Iterate through each row in the original file
        for row in csv_reader:
            # Combine multiline "Draw Date" and "Results" into single lines
            draw_date = " ".join(row[0].splitlines()).strip()
            results = " ".join(row[1].splitlines()).strip()
            jackpot = row[2].strip()
            outcome = row[3].strip()

            # Split the results into individual numbers
            numbers = results.split()

            # Ensure there are exactly 6 numbers in the results
            if len(numbers) == 6:
                # Write the cleaned data to the new file
                csv_writer.writerow([draw_date] + numbers + [jackpot, outcome])
