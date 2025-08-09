import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv('calculated_lotto.csv')

features = ['mean', 'median', 'std', 'min', 'max', 'range', 'sum_without_bonus',
            'even_count', 'odd_count', 'low_range_count', 'high_range_count',
            'prev_common_count', 'has_consecutive']
targets = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'BonusBall']

X = df[features].copy()
y = df[targets].copy()
X['has_consecutive'] = X['has_consecutive'].astype(int)

# Train model
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

# Ask user for their combination
print("Enter your 6 numbers and 1 PowerBall number:")
user_input = input("Please enter the 6 numbers and PowerBall, separated by spaces: ").split()

try:
    user_nums = list(map(int, user_input[:6]))
    user_bonus = int(user_input[6])
    
    if len(user_nums) != 6 or user_bonus < 1 or user_bonus > 50:
        raise ValueError("Invalid input. Please make sure you enter 6 numbers and a valid PowerBall number.")
except ValueError as e:
    print(f"Error: {e}")
    exit()

# Feature values for the given input (the same structure as in the training data)
user_features = pd.DataFrame([{
    'mean': df['mean'].mean(),
    'median': df['median'].mean(),
    'std': df['std'].mean(),
    'min': df['min'].min(),
    'max': df['max'].max(),
    'range': df['range'].max(),
    'sum_without_bonus': sum(user_nums),
    'even_count': sum(1 for x in user_nums if x % 2 == 0),
    'odd_count': sum(1 for x in user_nums if x % 2 != 0),
    'low_range_count': sum(1 for x in user_nums if x <= 25),
    'high_range_count': sum(1 for x in user_nums if x > 25),
    'prev_common_count': 0,  # Placeholder, could be calculated based on previous data
    'has_consecutive': int(any(user_nums[i] + 1 == user_nums[i + 1] for i in range(5)))
}])

# Predict the likelihood score for this input
probabilities = [estimator.predict_proba(user_features)[0] for estimator in model.estimators_]

# Calculate the predicted likelihood score
user_probs = []

for i, num in enumerate(user_nums):
    user_probs.append(probabilities[i][num - 1])

user_bonus_prob = probabilities[6][user_bonus - 1]
user_probs.append(user_bonus_prob)

likelihood_score = round(np.mean(user_probs) * 100, 2)

# Display the result
print(f"\nYour Combination: {user_nums} + PowerBall: {user_bonus}")
print(f"Likelihood: {likelihood_score}%")
