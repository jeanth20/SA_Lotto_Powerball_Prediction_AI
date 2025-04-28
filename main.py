import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("🎰 Lotto Powerball Prediction AI")

# Function to process data and create calculated_lotto_powerball.csv
def process_data():
    # Check for consecutive numbers
    def has_consecutive(nums):
        nums_sorted = sorted(nums)
        return any(nums_sorted[i + 1] - nums_sorted[i] == 1 for i in range(len(nums_sorted) - 1))

    # Open the CSV file for reading
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
                b1 = int(row[1].strip())  # Extract the value b1
                b2 = int(row[2].strip())  # Extract the value b2
                b3 = int(row[3].strip())  # Extract the value b3
                b4 = int(row[4].strip())  # Extract the value b4
                b5 = int(row[5].strip())  # Extract the value b5
                b6 = int(row[6].strip())  # Extract the value b6

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
def plot_winning_patterns2(df):
    # Calculate various patterns
    patterns = pd.DataFrame()
    patterns['Even Count'] = df['even_count']
    patterns['Low Range Count'] = df['low_range_count']
    patterns['Sum Without Bonus'] = df['sum_without_bonus']
    patterns['Has Consecutive'] = df['has_consecutive'].astype(int)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot even/odd distribution
    even_counts = patterns['Even Count'].value_counts().sort_index()
    axes[0, 0].bar(even_counts.index, even_counts.values)
    axes[0, 0].set_title('Distribution of Even Numbers in Winning Combinations')
    axes[0, 0].set_xlabel('Number of Even Numbers')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_xticks(range(6))

    # Plot low/high range distribution
    low_counts = patterns['Low Range Count'].value_counts().sort_index()
    axes[0, 1].bar(low_counts.index, low_counts.values)
    axes[0, 1].set_title('Distribution of Low Range Numbers (1-25)')
    axes[0, 1].set_xlabel('Number of Low Range Numbers')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xticks(range(6))

    # Plot sum distribution
    axes[1, 0].hist(patterns['Sum Without Bonus'], bins=20, color='green')
    axes[1, 0].set_title('Distribution of Sum of Main Numbers')
    axes[1, 0].set_xlabel('Sum')
    axes[1, 0].set_ylabel('Frequency')

    # Plot consecutive numbers
    has_consecutive = patterns['Has Consecutive'].value_counts()
    axes[1, 1].pie(has_consecutive.values, labels=['No Consecutive', 'Has Consecutive'],
                  autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    axes[1, 1].set_title('Presence of Consecutive Numbers')

    plt.tight_layout()
    return fig

# Function to plot patterns in winning combinations (expanded version)
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

# Main application
with st.spinner("Processing data... This may take a moment."):
    # Process data and create calculated file
    df = process_data()

    # Calculate likelihoods and get models
    main_likelihood_df, powerball_likelihood_df, model_main, model_powerball, X, features = calculate_likelihoods(df)

# Create sidebar for predictions and user input
with st.sidebar:
    st.header("🔮 Prediction & Analysis")

    # Predict next draw
    st.subheader("Predicted Numbers for Powerball Next Draw")

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

    # Display prediction
    st.markdown(f"**Main Numbers:** {', '.join(map(str, main_numbers))}")
    st.markdown(f"**PowerBall:** {powerball_number}")

    # Calculate likelihood of this prediction
    pred_likelihood = calculate_user_numbers_likelihood(main_numbers, powerball_number,
                                                       model_main, model_powerball, X, features)
    st.markdown(f"**Likelihood:** {pred_likelihood}%")

    # Separator
    st.markdown("---")

    # User input for checking likelihood
    st.subheader("Check Your Numbers")

    # Input for main numbers
    user_main_numbers = []
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        user_main_numbers.append(st.number_input("Number 1", min_value=1, max_value=50, value=1))
    with col2:
        user_main_numbers.append(st.number_input("Number 2", min_value=1, max_value=50, value=2))
    with col3:
        user_main_numbers.append(st.number_input("Number 3", min_value=1, max_value=50, value=3))
    with col4:
        user_main_numbers.append(st.number_input("Number 4", min_value=1, max_value=50, value=4))
    with col5:
        user_main_numbers.append(st.number_input("Number 5", min_value=1, max_value=50, value=5))

    # Input for powerball
    user_powerball = st.number_input("PowerBall", min_value=1, max_value=20, value=1)

    # Check button
    if st.button("Check Likelihood"):
        user_likelihood = calculate_user_numbers_likelihood(user_main_numbers, user_powerball,
                                                           model_main, model_powerball, X, features)
        st.markdown(f"**Your Numbers Likelihood:** {user_likelihood}%")

        # Compare with prediction
        if user_likelihood > pred_likelihood:
            st.success("Your numbers have a higher likelihood than our prediction! 🎉")
        elif user_likelihood < pred_likelihood:
            st.warning("Our prediction has a higher likelihood than your numbers.")
        else:
            st.info("Your numbers have the same likelihood as our prediction.")

    # Separator
    st.markdown("---")

    # Number trend analysis
    st.subheader("Number Trend Analysis")
    trend_numbers = st.multiselect("Select numbers to analyze trends",
                                  options=list(range(1, 51)),
                                  default=[1, 15, 30, 45])

# Main content area
# st.header("📊 Lotto Powerball Analysis")

# Display tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Data & Likelihoods", "Heatmaps", "Trends", "Patterns"])

with tab1:
    # Display the data
    st.subheader("📊 Recent 10 Draws")
    st.dataframe(pd.read_csv("cleaned_lotto_powerball.csv").head(10))

    # Display likelihood tables
    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("🔢 Main Numbers Likelihood (%)")
        st.dataframe(main_likelihood_df)

    with col2:
        st.subheader("🎯 Powerball Likelihood (%)")
        st.dataframe(powerball_likelihood_df)

with tab2:
    st.subheader("📊 Calculated 100 Recent Draws")
    st.dataframe(pd.read_csv("calculated_lotto_powerball.csv").head(100))

    # Add explanation for heatmap
    st.info("""
    **How to read the heatmap:**
    - For main numbers (1-50): The position (row, col) corresponds to number = (row * 10) + col + 1
      Example: (0,0) → number 1, (0,9) → number 10, (4,9) → number 50
    - For powerball (1-20): The position (row, col) corresponds to number = (row * 10) + col + 1
      Example: (0,0) → number 1, (1,9) → number 20
    """)

    # Display heatmaps
    st.subheader("🔥 Frequency Heatmaps")
    st.pyplot(plot_main_heatmap(df))
    st.pyplot(plot_powerball_heatmap(df))

with tab3:
    # Display number trends
    st.subheader("� Number Frequency Trends")
    if trend_numbers:
        trend_fig = plot_number_trends(df, trend_numbers)
        if trend_fig:
            st.pyplot(trend_fig)
    else:
        st.info("Select numbers in the sidebar to view their trends over time.")

with tab4:
    # Display winning patterns
    st.subheader("🧩 Winning Combination Patterns")
    st.info("""
    **Understanding the Patterns:**
    - **Even/Odd Distribution**: Shows how many even numbers typically appear in winning combinations
    - **Low/High Range**: Shows the distribution of numbers from the lower range (1-25)
    - **Sum Distribution**: Shows the typical sum of the main numbers in winning combinations
    - **Consecutive Numbers**: Shows how often consecutive numbers appear in winning combinations
    """)
    
    patterns_fig = plot_winning_patterns(df)
    st.pyplot(patterns_fig)


