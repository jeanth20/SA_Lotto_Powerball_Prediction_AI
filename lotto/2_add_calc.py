import csv
import numpy as np
import pandas as pd

# check for consecutive numbers
def has_consecutive(nums):
    nums_sorted = sorted(nums)
    return any(nums_sorted[i + 1] - nums_sorted[i] == 1 for i in range(len(nums_sorted) - 1))

# Open the CSV file for reading
with open("cleaned_lotto.csv", "r", newline="") as open_file:
    csv_reader = csv.reader(open_file)

    # Create a new file to save the cleaned data
    with open("calculated_lotto.csv", "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)

        # Write the header row for the calculated data
        csv_writer.writerow(["Year", "Date", "Day", "b1", "b2", "b3", "b4", "b5", "b6", "BonusBall","mean", "median", "std", "min", "max", "range", "sum_without_bonus", "sum_with_bonus", "even_count", "odd_count", "low_range_count", "high_range_count", "prev_common_count", "has_consecutive"])

        # Skip the header row in the input file
        next(csv_reader)  # Add this line to skip the first row (header)

        # Initialize a variable to store the previous row's balls
        prev_balls = None

        # Iterate through each row in the original file
        for row in csv_reader:

            # Convert the date and time from the row
            draw_date = pd.to_datetime(row[1] + " " + row[0])
            day_of_week = draw_date.day_name()  # Get the name of the day

            b1 = int(row[2].strip())  # Extract the value b1
            b2 = int(row[3].strip())  # Extract the value b2
            b3 = int(row[4].strip())  # Extract the value b3
            b4 = int(row[5].strip())  # Extract the value b4
            b5 = int(row[6].strip())  # Extract the value b5
            b6 = int(row[7].strip())  # Extract the value b6
            bb = int(row[8].strip())  # Extract the value bb
            
           # Create a list of the main ball numbers
            balls = [b1, b2, b3, b4, b5, b6]

            # Calculate statistics using numpy
            mean = np.mean(balls)
            median = np.median(balls)
            std = np.std(balls)
            min_val = np.min(balls)
            max_val = np.max(balls)
            range_val = max_val - min_val

            # Sum of balls
            sum_without_bonus = sum(balls)

            # Sum including bonus ball
            sum_with_bonus = sum_without_bonus + bb

            # Count of even and odd numbers
            even_count = sum(1 for n in balls if n % 2 == 0)
            odd_count = 6 - even_count

            # Count of numbers in certain ranges
            low_range_count = sum(1 for n in balls if n <= 25)
            high_range_count = 6 - low_range_count

            # Calculate the count of common numbers with the previous row
            if prev_balls is None:
                prev_common_count = 0
            else:
                prev_common_count = len(set(balls) & set(prev_balls))

            # Update the previous balls for the next iteration
            prev_balls = balls
            
            # Check for consecutive numbers
            has_consecutive_flag = has_consecutive(balls)

            # Write the row to the output file
            csv_writer.writerow([row[0], row[1], day_of_week, b1, b2, b3, b4, b5, b6, bb,
                                 mean, median, std, min_val, max_val, range_val, 
                                 sum_without_bonus, sum_with_bonus,
                                 even_count, odd_count,
                                 low_range_count, high_range_count,
                                 prev_common_count, has_consecutive_flag])


# Close the files after processing
open_file.close()
output_file.close()
