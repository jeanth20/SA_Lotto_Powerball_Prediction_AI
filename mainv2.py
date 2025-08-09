import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import datetime
import subprocess
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# Function to process data and create calculated_lotto_powerball.csv
def process_data():
    try:
        # First, try to load existing calculated data
        if os.path.exists("calculated_lotto_powerball.csv"):
            return pd.read_csv("calculated_lotto_powerball.csv")
    except:
        pass

    # If calculated data doesn't exist or failed to load, create it
    # Check for consecutive numbers
    def has_consecutive(nums):
        nums_sorted = sorted(nums)
        return any(nums_sorted[i + 1] - nums_sorted[i] == 1 for i in range(len(nums_sorted) - 1))

    # First, ensure we have a clean CSV file
    if not os.path.exists("cleaned_lotto_powerball.csv"):
        # Run the cleanup process first
        try:
            # Change to powerball directory and run cleanup
            original_dir = os.getcwd()
            os.chdir("powerball")

            # Run the split script to create cleaned data
            result = subprocess.run(["python", "1_add_split.py"],
                                  capture_output=True, text=True, timeout=30)

            # Copy the cleaned file to main directory
            if os.path.exists("cleaned_lotto_powerball.csv"):
                import shutil
                shutil.copy("cleaned_lotto_powerball.csv", "../cleaned_lotto_powerball.csv")

            os.chdir(original_dir)
        except Exception as e:
            # If cleanup fails, return error
            raise Exception(f"Failed to create clean data: {str(e)}")

    # Open the cleaned CSV file for reading
    with open("cleaned_lotto_powerball.csv", "r", newline="") as open_file:
        csv_reader = csv.reader(open_file)

        # Create a new file to save the calculated data
        with open("calculated_lotto_powerball.csv", "w", newline="") as output_file:
            csv_writer = csv.writer(output_file)

            # Write the header row for the calculated data
            csv_writer.writerow(["Draw Date", "b1", "b2", "b3", "b4", "b5", "b6", "Jackpot", "Outcome",
                                "mean", "median", "std", "min", "max", "range", "sum_without_bonus",
                                "sum_with_bonus", "even_count", "odd_count", "low_range_count",
                                "high_range_count", "prev_common_count", "has_consecutive"])

            # Skip the header row in the input file
            header = next(csv_reader)

            # Initialize a variable to store the previous row's balls
            prev_balls = None

            # Iterate through each row in the original file
            for row in csv_reader:
                try:
                    # Clean the data by removing quotes and extra whitespace
                    b1 = int(row[1].strip().strip('"'))  # Extract the value b1
                    b2 = int(row[2].strip().strip('"'))  # Extract the value b2
                    b3 = int(row[3].strip().strip('"'))  # Extract the value b3
                    b4 = int(row[4].strip().strip('"'))  # Extract the value b4
                    b5 = int(row[5].strip().strip('"'))  # Extract the value b5
                    b6 = int(row[6].strip().strip('"'))  # Extract the value b6
                except (ValueError, IndexError) as e:
                    # Skip rows that can't be parsed
                    print(f"Skipping row due to parsing error: {row}, Error: {e}")
                    continue

                # Create a list of the main ball numbers
                balls = [b1, b2, b3, b4, b5]  # b6 is the bonus ball

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
                sum_with_bonus = sum_without_bonus + b6

                # Count of even and odd numbers
                even_count = sum(1 for n in balls if n % 2 == 0)
                odd_count = 5 - even_count

                # Count of numbers in certain ranges
                low_range_count = sum(1 for n in balls if n <= 25)
                high_range_count = 5 - low_range_count

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
                csv_writer.writerow([row[0], b1, b2, b3, b4, b5, b6, row[7], row[8],
                                    mean, median, std, min_val, max_val, range_val,
                                    sum_without_bonus, sum_with_bonus,
                                    even_count, odd_count,
                                    low_range_count, high_range_count,
                                    prev_common_count, has_consecutive_flag])

    return pd.read_csv("calculated_lotto_powerball.csv")

# Function to calculate engineered features
def calculate_engineered_features(history):
    features = []
    for i in range(len(history)):
        row = history.iloc[max(i-5, 0):i]
        current = history.iloc[i][['b1', 'b2', 'b3', 'b4', 'b5']].values

        hit_rate_last_5 = 0
        if not row.empty:
            past_numbers = row[['b1', 'b2', 'b3', 'b4', 'b5']].values.flatten()
            hit_rate_last_5 = len(set(current) & set(past_numbers)) / 5

        avg_gap = np.mean(np.diff(sorted(current)))

        features.append({
            'hit_rate_last_5': hit_rate_last_5,
            'avg_gap_between_numbers': avg_gap,
        })

    return pd.DataFrame(features)

# Ensure model prediction is unique (not a past draw)
def get_valid_prediction(prediction_main, prediction_powerball, df):
    expected_cols = ['b1', 'b2', 'b3', 'b4', 'b5']
    if not all(col in df.columns for col in expected_cols):
        return prediction_main, prediction_powerball  # Cannot validate if columns missing

    # Create set for easy comparison
    drawn_sets = df[expected_cols].apply(lambda row: set(row), axis=1)
    predicted_set = set(prediction_main)

    # If predicted main numbers match a past draw, generate alternative
    if predicted_set in drawn_sets.values:
        # Alternative strategy: shuffle slightly
        alternative = list(predicted_set)
        np.random.shuffle(alternative)
        alternative = sorted(alternative[:5])

        # Check again
        if set(alternative) not in drawn_sets.values:
            return alternative, prediction_powerball
        else:
            # Last resort: random new numbers
            new_numbers = np.random.choice(range(1, 51), 5, replace=False)
            return sorted(new_numbers), prediction_powerball

    return prediction_main, prediction_powerball

# Function to calculate number likelihoods and train models
def calculate_likelihoods(df):
    # Add engineered features
    engineered = calculate_engineered_features(df)
    df = pd.concat([df.reset_index(drop=True), engineered.reset_index(drop=True)], axis=1)

    # Update feature list
    features = ['mean', 'median', 'std', 'min', 'max', 'range', 'sum_without_bonus',
                'even_count', 'odd_count', 'low_range_count', 'high_range_count',
                'prev_common_count', 'hit_rate_last_5', 'avg_gap_between_numbers']

    targets_main = ['b1', 'b2', 'b3', 'b4', 'b5']
    target_powerball = ['b6']

    X = df[features].copy()
    y_main = df[targets_main].copy()
    y_powerball = df[target_powerball].copy()

    # Train/test split
    X_train, _, y_train_main, _ = train_test_split(X, y_main, test_size=0.2, random_state=42)
    _, _, y_train_powerball, _ = train_test_split(X, y_powerball, test_size=0.2, random_state=42)

    # Train models separately
    model_main = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model_main.fit(X_train, y_train_main)

    model_powerball = RandomForestClassifier(n_estimators=100, random_state=42)
    model_powerball.fit(X_train, y_train_powerball.values.ravel())

    # Calculate likelihoods for main numbers (1-50)
    main_probabilities = []
    for estimator in model_main.estimators_:
        main_probabilities.append(estimator.predict_proba(X.iloc[-1:]))

    # Aggregate probabilities for main numbers
    main_number_likelihood = {i: 0 for i in range(1, 51)}
    for i, probs in enumerate(main_probabilities):
        for idx, prob in enumerate(probs[0]):
            if idx + 1 <= 50:  # Ensure we only consider numbers 1-50
                main_number_likelihood[idx + 1] += prob

    # Average the probability across the 5 targets for main numbers
    for num in main_number_likelihood:
        main_number_likelihood[num] = round((main_number_likelihood[num] / 5) * 100, 2)

    # Calculate likelihoods for powerball (1-20)
    powerball_probs = model_powerball.predict_proba(X.iloc[-1:])

    # Create powerball likelihood dictionary
    powerball_likelihood = {i: 0 for i in range(1, 21)}
    for idx, prob in enumerate(powerball_probs[0]):
        if idx + 1 <= 20:  # Ensure we only consider numbers 1-20
            powerball_likelihood[idx + 1] = round(prob * 100, 2)

    # Create dataframes for display
    main_likelihood_df = pd.DataFrame({
        'Number': list(main_number_likelihood.keys()),
        'Likelihood (%)': list(main_number_likelihood.values())
    }).sort_values(by='Likelihood (%)', ascending=False)

    powerball_likelihood_df = pd.DataFrame({
        'Number': list(powerball_likelihood.keys()),
        'Likelihood (%)': list(powerball_likelihood.values())
    }).sort_values(by='Likelihood (%)', ascending=False)

    # Make predictions for next draw
    next_input = X.iloc[-1:].copy()
    prediction_main = model_main.predict(next_input)[0]
    prediction_powerball = model_powerball.predict(next_input)[0]

    # Ensure 5 unique main numbers
    main_numbers = list(dict.fromkeys(prediction_main))
    while len(main_numbers) < 5:
        new_num = np.random.randint(1, 51)  # 1-50
        if new_num not in main_numbers:
            main_numbers.append(new_num)

    main_numbers.sort()

    # Ensure valid powerball number
    powerball_number = int(prediction_powerball)
    if powerball_number < 1 or powerball_number > 20:
        powerball_number = np.random.randint(1, 21)

    return main_likelihood_df, powerball_likelihood_df, model_main, model_powerball, X, features

# Function to create heatmap for main numbers
def plot_main_heatmap(df):
    all_numbers = np.concatenate([df['b1'], df['b2'], df['b3'], df['b4'], df['b5']])
    counts_main = pd.Series(all_numbers).value_counts().sort_index()

    # Create a full range 1-50, even if some numbers have 0 counts
    all_main_numbers = pd.Series(range(1, 51))
    counts_main = counts_main.reindex(all_main_numbers).fillna(0)

    # Create a 5x10 grid for the heatmap
    heatmap_data = counts_main.values.reshape(5, 10)

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title("Main Numbers (1-50) Frequency Heatmap")
    ax.set_xlabel("Number Group")
    ax.set_ylabel("Number Range")

    return fig

# Function to create heatmap for powerball
def plot_powerball_heatmap(df):
    powerball_numbers = df['b6'].values
    counts_pb = pd.Series(powerball_numbers).value_counts().sort_index()

    all_pb_numbers = pd.Series(range(1, 21))
    counts_pb = counts_pb.reindex(all_pb_numbers).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 2))
    sns.heatmap(counts_pb.values.reshape(2, 10), annot=True, fmt=".0f", cmap="viridis", cbar=True, ax=ax)
    ax.set_title("PowerBall (1-20) Frequency Heatmap")
    ax.set_xlabel("Number Group")
    ax.set_ylabel("Number Range")

    return fig

# Function to calculate likelihood for user-selected numbers
def calculate_user_numbers_likelihood(main_numbers, powerball_number, model_main, model_powerball, X, features):
    # Create a feature vector for the user's numbers
    user_features = X.iloc[-1:].copy()  # Use the latest draw's features

    # Get probabilities for each number
    main_probs = []
    for i, estimator in enumerate(model_main.estimators_):
        probs = estimator.predict_proba(user_features)[0]
        if main_numbers[i] <= len(probs):
            main_probs.append(probs[main_numbers[i] - 1])
        else:
            main_probs.append(0.01)  # Default low probability if out of range

    # Get probability for powerball
    powerball_probs = model_powerball.predict_proba(user_features)[0]
    if powerball_number <= len(powerball_probs):
        powerball_prob = powerball_probs[powerball_number - 1]
    else:
        powerball_prob = 0.01  # Default low probability if out of range

    # Calculate average likelihood
    all_probs = main_probs + [powerball_prob]
    likelihood = round(np.mean(all_probs) * 100, 2)

    return likelihood

# Function to plot number trends over time
def plot_number_trends(df, selected_numbers):
    if not selected_numbers:
        return None

    # Create a dataframe with draw dates and selected numbers
    trends_data = pd.DataFrame()
    trends_data['Draw Date'] = df['Draw Date']

    # Convert to datetime for proper plotting
    trends_data['Draw Date'] = pd.to_datetime(trends_data['Draw Date'], format='%A %d %B %Y', errors='coerce')

    # Track occurrences of each selected number
    for num in selected_numbers:
        trends_data[f'Number {num}'] = ((df['b1'] == num) |
                                        (df['b2'] == num) |
                                        (df['b3'] == num) |
                                        (df['b4'] == num) |
                                        (df['b5'] == num) |
                                        (df['b6'] == num)).astype(int)

    # Calculate rolling frequency (last 10 draws)
    for num in selected_numbers:
        trends_data[f'Frequency {num}'] = trends_data[f'Number {num}'].rolling(10).sum()

    # Plot the trends
    fig, ax = plt.subplots(figsize=(12, 6))
    for num in selected_numbers:
        ax.plot(trends_data['Draw Date'], trends_data[f'Frequency {num}'], label=f'Number {num}')

    ax.set_title('Number Frequency Trends (Rolling 10-draw window)')
    ax.set_xlabel('Draw Date')
    ax.set_ylabel('Frequency in last 10 draws')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

# Function to plot patterns in winning combinations
def plot_winning_patterns(df):
    # Calculate various patterns
    patterns = pd.DataFrame()
    patterns['Even Count'] = df['even_count']
    patterns['Odd Count'] = 5 - df['even_count']
    patterns['Low Range Count'] = df['low_range_count']
    patterns['High Range Count'] = 5 - df['low_range_count']
    patterns['Sum Without Bonus'] = df['sum_without_bonus']
    patterns['Has Consecutive'] = df['has_consecutive'].astype(int)

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Plot even/odd distribution
    even_counts = patterns['Even Count'].value_counts().sort_index()
    axes[0, 0].bar(even_counts.index, even_counts.values, color='skyblue')
    axes[0, 0].set_title('Distribution of Even Numbers')
    axes[0, 0].set_xlabel('Number of Even Numbers')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_xticks(range(6))

    # Plot odd distribution
    odd_counts = patterns['Odd Count'].value_counts().sort_index()
    axes[0, 1].bar(odd_counts.index, odd_counts.values, color='lightcoral')
    axes[0, 1].set_title('Distribution of Odd Numbers')
    axes[0, 1].set_xlabel('Number of Odd Numbers')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xticks(range(6))

    # Plot low range distribution
    low_counts = patterns['Low Range Count'].value_counts().sort_index()
    axes[1, 0].bar(low_counts.index, low_counts.values, color='lightgreen')
    axes[1, 0].set_title('Distribution of Low Range Numbers (1-25)')
    axes[1, 0].set_xlabel('Number of Low Range Numbers')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xticks(range(6))

    # Plot high range distribution
    high_counts = patterns['High Range Count'].value_counts().sort_index()
    axes[1, 1].bar(high_counts.index, high_counts.values, color='gold')
    axes[1, 1].set_title('Distribution of High Range Numbers (26-50)')
    axes[1, 1].set_xlabel('Number of High Range Numbers')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xticks(range(6))

    # Plot sum distribution
    axes[2, 0].hist(patterns['Sum Without Bonus'], bins=20, color='mediumseagreen')
    axes[2, 0].set_title('Distribution of Sum of Main Numbers')
    axes[2, 0].set_xlabel('Sum')
    axes[2, 0].set_ylabel('Frequency')

    # Plot consecutive numbers
    has_consecutive = patterns['Has Consecutive'].value_counts()
    axes[2, 1].pie(
        has_consecutive.values,
        labels=['No Consecutive', 'Has Consecutive'],
        autopct='%1.1f%%',
        colors=['lightblue', 'salmon'],
        startangle=90
    )
    axes[2, 1].set_title('Presence of Consecutive Numbers')

    plt.tight_layout()
    return fig

# Global variables to store processed data and models
df = None
main_likelihood_df = None
powerball_likelihood_df = None
model_main = None
model_powerball = None
X = None
features = None

# Initialize data and models
def initialize_data():
    global df, main_likelihood_df, powerball_likelihood_df, model_main, model_powerball, X, features, current_model_type, training_data_info

    # Process data and create calculated file
    df = process_data()

    # Calculate likelihoods and get models
    main_likelihood_df, powerball_likelihood_df, model_main, model_powerball, X, features = calculate_likelihoods(df)

    # Set model type info
    current_model_type = "Full Dataset"
    training_data_info = f"All {len(df)} available draws"

    return "Data processed and models trained successfully!"

# Global variables to track model type
current_model_type = "Full Dataset"
training_data_info = ""

# Function to get predictions
def get_predictions():
    global df, model_main, model_powerball, X, current_model_type, training_data_info

    if model_main is None or model_powerball is None:
        return "Please initialize data first!", "", "", "", "", ""

    # Get the prediction from the model
    next_input = X.iloc[-1:].copy()
    prediction_main = model_main.predict(next_input)[0]
    prediction_powerball = model_powerball.predict(next_input)[0]

    # Ensure 5 unique main numbers
    main_numbers = list(dict.fromkeys(prediction_main))
    while len(main_numbers) < 5:
        new_num = np.random.randint(1, 51)  # 1-50
        if new_num not in main_numbers:
            main_numbers.append(new_num)

    main_numbers.sort()

    # Ensure valid powerball number
    powerball_number = int(prediction_powerball)
    if powerball_number < 1 or powerball_number > 20:
        powerball_number = np.random.randint(1, 21)

    # Add the validation for unique predictions
    main_numbersc, powerball_numberc = get_valid_prediction(main_numbers, powerball_number, df)

    # Calculate likelihood of this prediction
    pred_likelihood = calculate_user_numbers_likelihood(main_numbersc, powerball_numberc,
                                                       model_main, model_powerball, X, features)

    # Calculate stats for predicted numbers
    even_count = sum(1 for n in main_numbersc if n % 2 == 0)
    odd_count = 5 - even_count
    number_sum = sum(main_numbersc)
    number_sum_with_bonus = number_sum + powerball_numberc
    low_range_count = sum(1 for n in main_numbersc if n <= 25)
    high_range_count = 5 - low_range_count
    number_range = max(main_numbersc) - min(main_numbersc)
    number_mean = np.mean(main_numbersc)
    number_median = np.median(main_numbersc)

    # Check for consecutive numbers
    sorted_nums = sorted(main_numbersc)
    has_consecutive = any(b - a == 1 for a, b in zip(sorted_nums, sorted_nums[1:]))

    # Check if set of numbers was drawn before
    set_already_drawn = False
    if df is not None:
        drawn_sets = df[['b1', 'b2', 'b3', 'b4', 'b5']].apply(lambda row: set(row), axis=1)
        if any(set(main_numbersc) == drawn_set for drawn_set in drawn_sets):
            set_already_drawn = True

    # Create stats dataframe
    stats_df = pd.DataFrame({
        'Stat': [
            'Mean', 'Median', 'Range', 'Sum Without PowerBall', 'Sum With PowerBall',
            'Even Count', 'Odd Count', 'Low Range Count (1-25)', 'High Range Count (26-50)',
            'Has Consecutive Numbers', 'Already Drawn Before'
        ],
        'Value': [
            round(number_mean, 2), number_median, number_range, number_sum, number_sum_with_bonus,
            even_count, odd_count, low_range_count, high_range_count,
            'Yes' if has_consecutive else 'No', 'Yes' if set_already_drawn else 'No'
        ]
    })

    main_numbers_str = ', '.join(map(str, main_numbersc))

    # Enhanced prediction text with model information
    model_info = f"🤖 **Model Used:** {current_model_type}\n"
    if training_data_info:
        model_info += f"📊 **Training Data:** {training_data_info}\n\n"
    else:
        model_info += "\n"

    prediction_text = f"{model_info}🎯 **Prediction:**\nMain Numbers: {main_numbers_str}\nPowerBall: {powerball_numberc}\nLikelihood: {pred_likelihood}%"

    model_status = f"Using {current_model_type} Model"

    return prediction_text, main_numbers_str, str(powerball_numberc), f"{pred_likelihood}%", stats_df, model_status

# Function to check user numbers
def check_user_numbers(num1, num2, num3, num4, num5, powerball):
    global model_main, model_powerball, X, features, current_model_type, training_data_info

    if model_main is None or model_powerball is None:
        return "Please initialize data first!"

    user_main_numbers = [int(num1), int(num2), int(num3), int(num4), int(num5)]
    user_powerball = int(powerball)

    user_likelihood = calculate_user_numbers_likelihood(user_main_numbers, user_powerball,
                                                       model_main, model_powerball, X, features)

    # Get prediction likelihood for comparison
    _, _, _, pred_likelihood_str, _, _ = get_predictions()
    pred_likelihood = float(pred_likelihood_str.replace('%', ''))

    # Enhanced result with model information
    model_info = f"🤖 **Analysis using {current_model_type} Model**\n"
    if training_data_info:
        model_info += f"📊 **Training Data:** {training_data_info}\n\n"

    result = f"{model_info}🎯 **Your Numbers:** {', '.join(map(str, user_main_numbers))} | PowerBall: {user_powerball}\n"
    result += f"📈 **Your Likelihood:** {user_likelihood}%\n\n"

    if user_likelihood > pred_likelihood:
        result += "✅ Your numbers have a higher likelihood than our AI prediction! 🎉"
    elif user_likelihood < pred_likelihood:
        result += "📊 Our AI prediction has a higher likelihood than your numbers."
    else:
        result += "⚖️ Your numbers have the same likelihood as our AI prediction."

    return result

# Function to get recent draws
def get_recent_draws():
    try:
        recent_df = pd.read_csv("cleaned_lotto_powerball.csv").head(10)
        return recent_df
    except:
        return pd.DataFrame({"Error": ["Could not load data"]})

# Function to get calculated draws
def get_calculated_draws():
    try:
        calc_df = pd.read_csv("calculated_lotto_powerball.csv").head(100)
        return calc_df
    except:
        return pd.DataFrame({"Error": ["Could not load data"]})

# Function to get likelihood tables
def get_likelihood_tables():
    global main_likelihood_df, powerball_likelihood_df

    if main_likelihood_df is None or powerball_likelihood_df is None:
        empty_df = pd.DataFrame({"Message": ["Please initialize data first!"]})
        return empty_df, empty_df

    return main_likelihood_df, powerball_likelihood_df

# Function to generate heatmaps
def generate_heatmaps():
    global df

    if df is None:
        return None, None

    main_heatmap = plot_main_heatmap(df)
    powerball_heatmap = plot_powerball_heatmap(df)

    return main_heatmap, powerball_heatmap

# Function to generate trends plot
def generate_trends_plot(selected_numbers):
    global df

    if df is None:
        return None

    if not selected_numbers:
        return None

    # Convert selected numbers to integers
    trend_numbers = [int(num) for num in selected_numbers]

    return plot_number_trends(df, trend_numbers)

# Function to generate patterns plot
def generate_patterns_plot():
    global df

    if df is None:
        return None

    return plot_winning_patterns(df)

# Function to add new draw to the main CSV file
def add_new_draw(draw_date, num1, num2, num3, num4, num5, powerball, jackpot, outcome):
    try:
        # Looking at the existing data, I notice there are two formats:
        # 1. Multiline format (older entries): "Friday\n25 April 2025","5\n 9\n 13\n 46\n 47\n 11"
        # 2. Single line format (newer entries): Tuesday 5 August 2025,10 32 34 40 48 14
        # Let's use the single line format for consistency with recent entries

        formatted_date = draw_date
        formatted_results = f"{num1} {num2} {num3} {num4} {num5} {powerball}"

        # Read the existing file
        powerball_file = "powerball/powerball2009_2025 - Sheet1.csv"

        # Read all existing rows
        with open(powerball_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            rows = list(reader)

        # Insert the new row at position 1 (after header, before existing data)
        new_row = [formatted_date, formatted_results, jackpot, outcome]
        rows.insert(1, new_row)

        # Write back to file with minimal quoting
        with open(powerball_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(rows)

        return f"✅ Successfully added new draw: {draw_date} - Numbers: {num1}, {num2}, {num3}, {num4}, {num5}, {powerball}"

    except Exception as e:
        return f"❌ Error adding new draw: {str(e)}"

# Function to run cleanup scripts
def run_cleanup_scripts():
    try:
        results = []

        # Change to powerball directory
        original_dir = os.getcwd()
        os.chdir("powerball")

        # Run script 1: Clean and split data
        try:
            result1 = subprocess.run(["python", "1_add_split.py"],
                                   capture_output=True, text=True, timeout=30)
            if result1.returncode == 0:
                results.append("✅ Step 1: Data cleaning completed successfully")
            else:
                error_msg = result1.stderr.strip() if result1.stderr else "Unknown error"
                results.append(f"❌ Step 1 failed: {error_msg}")
        except subprocess.TimeoutExpired:
            results.append("❌ Step 1 timed out")
        except Exception as e:
            results.append(f"❌ Step 1 error: {str(e)}")

        # Run script 2: Calculate statistics
        try:
            result2 = subprocess.run(["python", "2_add_calc.py"],
                                   capture_output=True, text=True, timeout=30)
            if result2.returncode == 0:
                results.append("✅ Step 2: Statistics calculation completed successfully")
            else:
                error_msg = result2.stderr.strip() if result2.stderr else "Unknown error"
                # Check for common pandas/numpy issues
                if "numpy.dtype size changed" in error_msg:
                    results.append("❌ Step 2 failed: Pandas/NumPy compatibility issue detected")
                    results.append("💡 Suggestion: Try running 'pip uninstall pandas numpy -y && pip install pandas numpy' to fix compatibility")
                elif "invalid literal for int()" in error_msg:
                    results.append("❌ Step 2 failed: Data format issue - CSV formatting problem detected")
                    results.append("💡 Suggestion: The CSV data may have formatting issues. Check the powerball CSV file.")
                else:
                    results.append(f"❌ Step 2 failed: {error_msg}")
        except subprocess.TimeoutExpired:
            results.append("❌ Step 2 timed out")
        except Exception as e:
            results.append(f"❌ Step 2 error: {str(e)}")

        # Run script 3: Calculate frequencies
        try:
            result3 = subprocess.run(["python", "3_cal_freq.py"],
                                   capture_output=True, text=True, timeout=30)
            if result3.returncode == 0:
                results.append("✅ Step 3: Frequency calculation completed successfully")
            else:
                error_msg = result3.stderr.strip() if result3.stderr else "Unknown error"
                if "numpy.dtype size changed" in error_msg:
                    results.append("❌ Step 3 failed: Pandas/NumPy compatibility issue detected")
                else:
                    results.append(f"❌ Step 3 failed: {error_msg}")
        except subprocess.TimeoutExpired:
            results.append("❌ Step 3 timed out")
        except Exception as e:
            results.append(f"❌ Step 3 error: {str(e)}")

        # Copy files to main directory (even if some steps failed)
        try:
            import shutil
            files_copied = []
            if os.path.exists("cleaned_lotto_powerball.csv"):
                shutil.copy("cleaned_lotto_powerball.csv", "../cleaned_lotto_powerball.csv")
                files_copied.append("cleaned_lotto_powerball.csv")
            if os.path.exists("calculated_lotto_powerball.csv"):
                shutil.copy("calculated_lotto_powerball.csv", "../calculated_lotto_powerball.csv")
                files_copied.append("calculated_lotto_powerball.csv")

            if files_copied:
                results.append(f"✅ Files copied to main directory: {', '.join(files_copied)}")
            else:
                results.append("⚠️ No files were available to copy")
        except Exception as e:
            results.append(f"❌ File copy error: {str(e)}")

        # Change back to original directory
        os.chdir(original_dir)

        return "\n".join(results)

    except Exception as e:
        # Make sure we change back to original directory even if there's an error
        try:
            os.chdir(original_dir)
        except:
            pass
        return f"❌ General error: {str(e)}"

# Function to validate draw input
def validate_draw_input(draw_date, num1, num2, num3, num4, num5, powerball, jackpot, outcome):
    errors = []

    # Check date format
    if not draw_date or len(draw_date.strip()) == 0:
        errors.append("Draw date is required")

    # Check main numbers (1-50)
    main_numbers = [num1, num2, num3, num4, num5]
    for i, num in enumerate(main_numbers, 1):
        if not isinstance(num, (int, float)) or num < 1 or num > 50:
            errors.append(f"Main number {i} must be between 1 and 50")

    # Check for duplicates in main numbers
    if len(set(main_numbers)) != 5:
        errors.append("Main numbers must be unique")

    # Check powerball (1-20)
    if not isinstance(powerball, (int, float)) or powerball < 1 or powerball > 20:
        errors.append("PowerBall must be between 1 and 20")

    # Check jackpot format
    if not jackpot or len(jackpot.strip()) == 0:
        errors.append("Jackpot amount is required")

    # Check outcome
    if not outcome or len(outcome.strip()) == 0:
        errors.append("Outcome is required")

    if errors:
        return False, "❌ Validation errors:\n" + "\n".join(errors)
    else:
        return True, "✅ Input validation passed"

# Combined function to add draw and run cleanup
def add_draw_and_cleanup(draw_date, num1, num2, num3, num4, num5, powerball, jackpot, outcome):
    # First validate input
    is_valid, validation_msg = validate_draw_input(draw_date, num1, num2, num3, num4, num5, powerball, jackpot, outcome)

    if not is_valid:
        return validation_msg

    # Add the new draw
    add_result = add_new_draw(draw_date, int(num1), int(num2), int(num3), int(num4), int(num5), int(powerball), jackpot, outcome)

    if "❌" in add_result:
        return add_result

    # Run cleanup scripts
    cleanup_result = run_cleanup_scripts()

    return f"{add_result}\n\n📊 Cleanup Scripts Results:\n{cleanup_result}\n\n🔄 Please click 'Initialize Data & Train Models' to refresh the AI models with new data."

# Function to filter data by date range
def filter_data_by_date_range(df, start_date, end_date):
    try:
        # Convert Draw Date column to datetime
        df_copy = df.copy()
        df_copy['Draw Date'] = pd.to_datetime(df_copy['Draw Date'], format='%A %d %B %Y', errors='coerce')

        # Convert input dates to datetime
        start_dt = pd.to_datetime(start_date, errors='coerce')
        end_dt = pd.to_datetime(end_date, errors='coerce')

        if pd.isna(start_dt) or pd.isna(end_dt):
            return None, "Invalid date format. Please use YYYY-MM-DD format."

        # Filter data
        filtered_df = df_copy[(df_copy['Draw Date'] >= start_dt) & (df_copy['Draw Date'] <= end_dt)]

        if len(filtered_df) == 0:
            return None, "No data found in the specified date range."

        # Convert back to original format for consistency
        filtered_df['Draw Date'] = df['Draw Date'].iloc[filtered_df.index]

        return filtered_df, f"Filtered to {len(filtered_df)} draws from {start_date} to {end_date}"

    except Exception as e:
        return None, f"Error filtering data: {str(e)}"

# Function to filter data by number of recent draws
def filter_data_by_recent_draws(df, num_draws):
    try:
        if num_draws <= 0:
            return None, "Number of draws must be greater than 0."

        if num_draws > len(df):
            return df, f"Using all available {len(df)} draws (requested {num_draws} but only {len(df)} available)."

        filtered_df = df.head(num_draws)
        return filtered_df, f"Using most recent {len(filtered_df)} draws."

    except Exception as e:
        return None, f"Error filtering data: {str(e)}"

# Function to train model with custom data range
def train_custom_model(filter_type, start_date, end_date, num_draws):
    global df, main_likelihood_df, powerball_likelihood_df, model_main, model_powerball, X, features, current_model_type, training_data_info

    try:
        # Load the full dataset first
        if df is None:
            df = process_data()

        # Filter data based on user selection
        if filter_type == "Date Range":
            filtered_df, filter_msg = filter_data_by_date_range(df, start_date, end_date)
            current_model_type = "Custom Date Range"
            training_data_info = f"{len(filtered_df) if filtered_df is not None else 0} draws from {start_date} to {end_date}"
        else:  # Recent Draws
            filtered_df, filter_msg = filter_data_by_recent_draws(df, int(num_draws))
            current_model_type = "Recent Draws"
            training_data_info = f"Last {len(filtered_df) if filtered_df is not None else 0} draws"

        if filtered_df is None:
            return f"❌ {filter_msg}", None, None, "", ""

        # Check minimum data requirements
        if len(filtered_df) < 10:
            return f"❌ Insufficient data: Need at least 10 draws for training, got {len(filtered_df)}", None, None, "", ""

        # Train models with filtered data
        main_likelihood_df, powerball_likelihood_df, model_main, model_powerball, X, features = calculate_likelihoods(filtered_df)

        # Get training statistics
        training_stats = f"""
📊 **Training Statistics:**
- Total draws used: {len(filtered_df)}
- Date range: {filtered_df['Draw Date'].iloc[-1]} to {filtered_df['Draw Date'].iloc[0]}
- Training features: {len(features)}
- Model type: Random Forest with {len(model_main.estimators_)} estimators

✅ Models trained successfully with custom data range!

🎯 **Model Ready:** Go to the Predictions tab to generate new predictions using this custom-trained model!
        """

        return f"✅ {filter_msg}\n{training_stats}", main_likelihood_df, powerball_likelihood_df, f"Custom Model: {current_model_type}", "Ready for predictions with custom model"

    except Exception as e:
        return f"❌ Error training custom model: {str(e)}", None, None, "", ""

# Function to get data range info
def get_data_range_info():
    try:
        if df is None:
            temp_df = process_data()
        else:
            temp_df = df

        # Convert dates for analysis
        temp_df_copy = temp_df.copy()
        temp_df_copy['Draw Date'] = pd.to_datetime(temp_df_copy['Draw Date'], format='%A %d %B %Y', errors='coerce')

        # Remove any rows with invalid dates
        valid_dates = temp_df_copy.dropna(subset=['Draw Date'])

        if len(valid_dates) == 0:
            return "No valid dates found in dataset"

        earliest_date = valid_dates['Draw Date'].min().strftime('%Y-%m-%d')
        latest_date = valid_dates['Draw Date'].max().strftime('%Y-%m-%d')
        total_draws = len(temp_df)

        info = f"""
📅 **Available Data Range:**
- Earliest draw: {earliest_date}
- Latest draw: {latest_date}
- Total draws: {total_draws}

💡 **Recommendations:**
- For recent patterns: Use last 100-500 draws
- For long-term analysis: Use last 1000+ draws
- For specific periods: Use date range filter
        """

        return info

    except Exception as e:
        return f"Error getting data info: {str(e)}"

# Function to reset to full dataset
def reset_to_full_dataset():
    global df, main_likelihood_df, powerball_likelihood_df, model_main, model_powerball, X, features, current_model_type, training_data_info

    try:
        # Reload full dataset
        df = process_data()

        # Train with full dataset
        main_likelihood_df, powerball_likelihood_df, model_main, model_powerball, X, features = calculate_likelihoods(df)

        # Reset model type info
        current_model_type = "Full Dataset"
        training_data_info = f"All {len(df)} available draws"

        return f"✅ Reset to full dataset: {len(df)} draws used for training", main_likelihood_df, powerball_likelihood_df

    except Exception as e:
        return f"❌ Error resetting to full dataset: {str(e)}", None, None

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="🎰 Lotto Powerball Prediction AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎰 Lotto Powerball Prediction AI")

        # Initialize button
        with gr.Row():
            init_btn = gr.Button("Initialize Data & Train Models", variant="primary", size="lg")
            init_status = gr.Textbox(label="Status", interactive=False)

        init_btn.click(initialize_data, outputs=init_status)

        # Main tabs
        with gr.Tabs():
            # Predictions Tab
            with gr.TabItem("🔮 Predictions"):
                # Model status display
                with gr.Row():
                    current_model_display = gr.Textbox(
                        label="Current Model Status",
                        value="Click 'Initialize Data & Train Models' or train a custom model",
                        interactive=False
                    )

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## AI Prediction for Next Draw")
                        get_pred_btn = gr.Button("Get New Prediction", variant="primary", size="lg")
                        prediction_output = gr.Textbox(label="Prediction Details", lines=6, interactive=False)

                        with gr.Row():
                            main_nums_output = gr.Textbox(label="Main Numbers", interactive=False)
                            powerball_output = gr.Textbox(label="PowerBall", interactive=False)
                            likelihood_output = gr.Textbox(label="Likelihood", interactive=False)

                        stats_output = gr.DataFrame(label="Prediction Statistics")

                    with gr.Column(scale=1):
                        gr.Markdown("## Check Your Numbers")
                        gr.Markdown("*Uses the same model as the AI prediction*")

                        with gr.Row():
                            num1 = gr.Number(label="Number 1", value=1, minimum=1, maximum=50)
                            num2 = gr.Number(label="Number 2", value=2, minimum=1, maximum=50)
                        with gr.Row():
                            num3 = gr.Number(label="Number 3", value=3, minimum=1, maximum=50)
                            num4 = gr.Number(label="Number 4", value=4, minimum=1, maximum=50)
                        with gr.Row():
                            num5 = gr.Number(label="Number 5", value=5, minimum=1, maximum=50)
                            user_powerball = gr.Number(label="PowerBall", value=1, minimum=1, maximum=20)

                        check_btn = gr.Button("Check Likelihood", variant="secondary")
                        user_result = gr.Textbox(label="Your Numbers Analysis", lines=8, interactive=False)

                get_pred_btn.click(
                    get_predictions,
                    outputs=[prediction_output, main_nums_output, powerball_output, likelihood_output, stats_output, current_model_display]
                )

                check_btn.click(
                    check_user_numbers,
                    inputs=[num1, num2, num3, num4, num5, user_powerball],
                    outputs=user_result
                )

            # Training Configuration Tab
            with gr.TabItem("⚙️ Training Config"):
                gr.Markdown("## Custom Model Training")
                gr.Markdown("""
                **Configure the data range for training the AI model:**
                - Use recent draws for capturing latest patterns
                - Use specific date ranges for analyzing particular periods
                - Experiment with different ranges to optimize predictions
                """)

                # Data info section
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Dataset Information")
                        data_info_display = gr.Markdown(value="Click 'Get Data Info' to see available data range")
                        get_info_btn = gr.Button("Get Data Info", variant="secondary")

                        get_info_btn.click(get_data_range_info, outputs=data_info_display)

                # Training configuration section
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Training Configuration")

                        filter_type = gr.Radio(
                            choices=["Recent Draws", "Date Range"],
                            value="Recent Draws",
                            label="Filter Type",
                            info="Choose how to filter the training data"
                        )

                        # Recent draws configuration
                        with gr.Group(visible=True) as recent_draws_group:
                            gr.Markdown("**Recent Draws Configuration:**")
                            num_draws_input = gr.Slider(
                                minimum=50,
                                maximum=2000,
                                value=500,
                                step=50,
                                label="Number of Recent Draws",
                                info="How many of the most recent draws to use for training"
                            )

                        # Date range configuration
                        with gr.Group(visible=False) as date_range_group:
                            gr.Markdown("**Date Range Configuration:**")
                            with gr.Row():
                                start_date_input = gr.Textbox(
                                    label="Start Date",
                                    placeholder="2024-01-01",
                                    info="Format: YYYY-MM-DD"
                                )
                                end_date_input = gr.Textbox(
                                    label="End Date",
                                    placeholder="2025-12-31",
                                    info="Format: YYYY-MM-DD"
                                )

                        # Training controls
                        with gr.Row():
                            train_custom_btn = gr.Button("Train Custom Model", variant="primary", size="lg")
                            reset_full_btn = gr.Button("Reset to Full Dataset", variant="secondary")

                        # Show/hide groups based on filter type
                        def update_visibility(filter_choice):
                            if filter_choice == "Recent Draws":
                                return gr.update(visible=True), gr.update(visible=False)
                            else:
                                return gr.update(visible=False), gr.update(visible=True)

                        filter_type.change(
                            update_visibility,
                            inputs=filter_type,
                            outputs=[recent_draws_group, date_range_group]
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Training Results")
                        training_status = gr.Textbox(
                            label="Training Status",
                            lines=12,
                            interactive=False,
                            placeholder="Configure training parameters and click 'Train Custom Model'"
                        )

                        model_status = gr.Textbox(
                            label="Model Status",
                            interactive=False,
                            placeholder="No custom model trained yet"
                        )

                        prediction_status = gr.Textbox(
                            label="Prediction Status",
                            interactive=False,
                            placeholder="Ready for predictions after training"
                        )

                # Updated likelihood tables for custom training
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Custom Training - Main Numbers Likelihood")
                        custom_main_likelihood = gr.DataFrame(label="Main Numbers Likelihood (Custom)")

                    with gr.Column():
                        gr.Markdown("### Custom Training - PowerBall Likelihood")
                        custom_powerball_likelihood = gr.DataFrame(label="PowerBall Likelihood (Custom)")

                # Event handlers
                train_custom_btn.click(
                    train_custom_model,
                    inputs=[filter_type, start_date_input, end_date_input, num_draws_input],
                    outputs=[training_status, custom_main_likelihood, custom_powerball_likelihood,
                            model_status, prediction_status]
                )

                reset_full_btn.click(
                    reset_to_full_dataset,
                    outputs=[training_status, custom_main_likelihood, custom_powerball_likelihood]
                )

            # Data & Likelihoods Tab
            with gr.TabItem("📊 Data & Likelihoods"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Recent 10 Draws")
                        recent_draws = gr.DataFrame(label="Recent Draws")

                        refresh_recent_btn = gr.Button("Refresh Recent Draws")
                        refresh_recent_btn.click(get_recent_draws, outputs=recent_draws)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Main Numbers Likelihood (%)")
                        main_likelihood_table = gr.DataFrame(label="Main Numbers Likelihood")

                    with gr.Column():
                        gr.Markdown("## Powerball Likelihood (%)")
                        powerball_likelihood_table = gr.DataFrame(label="Powerball Likelihood")

                refresh_likelihood_btn = gr.Button("Refresh Likelihood Tables")
                refresh_likelihood_btn.click(
                    get_likelihood_tables,
                    outputs=[main_likelihood_table, powerball_likelihood_table]
                )

            # Heatmaps Tab
            with gr.TabItem("🔥 Heatmaps"):
                gr.Markdown("## Calculated 100 Recent Draws")
                calculated_draws = gr.DataFrame(label="Calculated Draws")

                refresh_calc_btn = gr.Button("Refresh Calculated Draws")
                refresh_calc_btn.click(get_calculated_draws, outputs=calculated_draws)

                gr.Markdown("""
                **How to read the heatmap:**
                - For main numbers (1-50): The position (row, col) corresponds to number = (row * 10) + col + 1
                  Example: (0,0) → number 1, (0,9) → number 10, (4,9) → number 50
                - For powerball (1-20): The position (row, col) corresponds to number = (row * 10) + col + 1
                  Example: (0,0) → number 1, (1,9) → number 20
                """)

                gr.Markdown("## Frequency Heatmaps")

                with gr.Row():
                    main_heatmap_plot = gr.Plot(label="Main Numbers Heatmap")
                    powerball_heatmap_plot = gr.Plot(label="PowerBall Heatmap")

                generate_heatmaps_btn = gr.Button("Generate Heatmaps")
                generate_heatmaps_btn.click(
                    generate_heatmaps,
                    outputs=[main_heatmap_plot, powerball_heatmap_plot]
                )

            # Trends Tab
            with gr.TabItem("📈 Trends"):
                gr.Markdown("## Number Trend Analysis")

                trend_numbers_input = gr.CheckboxGroup(
                    choices=[str(i) for i in range(1, 51)],
                    value=["1", "15", "30", "45"],
                    label="Select numbers to analyze trends",
                    interactive=True
                )

                trends_plot = gr.Plot(label="Number Frequency Trends")

                generate_trends_btn = gr.Button("Generate Trends Plot")
                generate_trends_btn.click(
                    generate_trends_plot,
                    inputs=trend_numbers_input,
                    outputs=trends_plot
                )

            # Patterns Tab
            with gr.TabItem("🧩 Patterns"):
                gr.Markdown("## Winning Combination Patterns")
                gr.Markdown("""
                **Understanding the Patterns:**
                - **Even/Odd Distribution**: Shows how many even numbers typically appear in winning combinations
                - **Low/High Range**: Shows the distribution of numbers from the lower range (1-25)
                - **Sum Distribution**: Shows the typical sum of the main numbers in winning combinations
                - **Consecutive Numbers**: Shows how often consecutive numbers appear in winning combinations
                """)

                patterns_plot = gr.Plot(label="Winning Patterns Analysis")

                generate_patterns_btn = gr.Button("Generate Patterns Plot")
                generate_patterns_btn.click(
                    generate_patterns_plot,
                    outputs=patterns_plot
                )

            # Data Management Tab
            with gr.TabItem("📝 Data Management"):
                gr.Markdown("## Add New Draw & Update Data")
                gr.Markdown("""
                **Instructions:**
                1. Enter the latest draw information below
                2. Click 'Add Draw & Run Cleanup' to add the draw and process all data
                3. After successful processing, go back to the Predictions tab and click 'Initialize Data & Train Models' to refresh the AI models
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### New Draw Information")

                        draw_date_input = gr.Textbox(
                            label="Draw Date",
                            placeholder="e.g., Friday 26 April 2025",
                            info="Enter the full date as it appears in the lottery results"
                        )

                        gr.Markdown("**Main Numbers (1-50):**")
                        with gr.Row():
                            main_num1 = gr.Number(label="Number 1", minimum=1, maximum=50, value=1)
                            main_num2 = gr.Number(label="Number 2", minimum=1, maximum=50, value=2)
                            main_num3 = gr.Number(label="Number 3", minimum=1, maximum=50, value=3)

                        with gr.Row():
                            main_num4 = gr.Number(label="Number 4", minimum=1, maximum=50, value=4)
                            main_num5 = gr.Number(label="Number 5", minimum=1, maximum=50, value=5)
                            powerball_num = gr.Number(label="PowerBall (1-20)", minimum=1, maximum=20, value=1)

                        with gr.Row():
                            jackpot_input = gr.Textbox(
                                label="Jackpot Amount",
                                placeholder="e.g., R25,000,000.00",
                                info="Enter the jackpot amount as it appears"
                            )
                            outcome_input = gr.Dropdown(
                                choices=["Roll", "Won"],
                                label="Outcome",
                                value="Roll"
                            )

                        add_and_cleanup_btn = gr.Button("Add Draw & Run Cleanup", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### Processing Status")
                        processing_output = gr.Textbox(
                            label="Results",
                            lines=15,
                            interactive=False,
                            placeholder="Click 'Add Draw & Run Cleanup' to see processing results here..."
                        )

                        gr.Markdown("### Manual Cleanup (Advanced)")
                        gr.Markdown("Run cleanup scripts without adding new data:")
                        cleanup_only_btn = gr.Button("Run Cleanup Scripts Only", variant="secondary")

                        cleanup_output = gr.Textbox(
                            label="Cleanup Results",
                            lines=8,
                            interactive=False
                        )

                # Event handlers
                add_and_cleanup_btn.click(
                    add_draw_and_cleanup,
                    inputs=[draw_date_input, main_num1, main_num2, main_num3, main_num4, main_num5,
                           powerball_num, jackpot_input, outcome_input],
                    outputs=processing_output
                )

                cleanup_only_btn.click(
                    run_cleanup_scripts,
                    outputs=cleanup_output
                )

        # Load initial data on startup
        demo.load(get_recent_draws, outputs=recent_draws)

    return demo

# Main execution
if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        debug=True
    )