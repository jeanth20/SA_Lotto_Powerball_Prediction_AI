import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# --- Adjusted code to fix problem (5 numbers from 1-50, PowerBall from 1-20) ---

# New columns to calculate
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

# Load data
df = pd.read_csv('calculated_lotto_powerball.csv')

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
X_train, X_test, y_train_main, y_test_main = train_test_split(X, y_main, test_size=0.2, random_state=42)
_, _, y_train_powerball, y_test_powerball = train_test_split(X, y_powerball, test_size=0.2, random_state=42)

# Train models separately
model_main = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model_main.fit(X_train, y_train_main)

model_powerball = RandomForestClassifier(n_estimators=100, random_state=42)
model_powerball.fit(X_train, y_train_powerball.values.ravel())

# Predict
next_input = X.iloc[-1:].copy()

prediction_main = model_main.predict(next_input)[0]
prediction_powerball = model_powerball.predict(next_input)[0]

main_numbers = list(dict.fromkeys(prediction_main))
while len(main_numbers) < 5:
    new_num = np.random.randint(1, 51)  # 1-50
    if new_num not in main_numbers:
        main_numbers.append(new_num)

main_numbers.sort()

powerball_number = int(prediction_powerball)
if powerball_number < 1 or powerball_number > 20:
    powerball_number = np.random.randint(1, 21)

print("Predicted Main Numbers:", main_numbers)
print("Predicted PowerBall:", powerball_number)

# --- Heatmap Function ---
def plot_clean_heatmap(df):
    all_numbers = np.concatenate([df['b1'], df['b2'], df['b3'], df['b4'], df['b5']])
    counts_main = pd.Series(all_numbers).value_counts().sort_index()

    # Create a full range 1-50, even if some numbers have 0 counts
    all_main_numbers = pd.Series(range(1, 51))
    counts_main = counts_main.reindex(all_main_numbers).fillna(0)

    plt.figure(figsize=(14, 4))
    sns.heatmap(counts_main.values.reshape(5, 10), annot=True, fmt=".0f", cmap="coolwarm", cbar=True)
    plt.title("Main Numbers (1-50) Frequency Heatmap")
    plt.xlabel("Number Group")
    plt.ylabel("Number Range")
    # legend_elements = ['(0,0) → number 1', '(0,1) → number 2', '(0,9) → number 10', '(1,0) → number 11', '(1,9) → number 20', 'The number = (row * 10) + column + 1']
    # plt.legend(handles=legend_elements, loc='upper right')

    plt.show()
    
plot_clean_heatmap(df)

def plot_powerball_heatmap(df):
    powerball_numbers = df['b6'].values
    counts_pb = pd.Series(powerball_numbers).value_counts().sort_index()

    all_pb_numbers = pd.Series(range(1, 21))
    counts_pb = counts_pb.reindex(all_pb_numbers).fillna(0)

    plt.figure(figsize=(8, 2))
    sns.heatmap(counts_pb.values.reshape(2, 10), annot=True, fmt=".0f", cmap="viridis", cbar=True)
    plt.title("PowerBall (1-20) Frequency Heatmap")
    plt.xlabel("Number Group")
    plt.ylabel("Number Range")
    plt.show()

plot_powerball_heatmap(df)
