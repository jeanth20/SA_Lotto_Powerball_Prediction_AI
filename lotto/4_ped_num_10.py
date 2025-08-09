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

X.loc[:, 'has_consecutive'] = X['has_consecutive'].astype(int)

# Train/test split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

# Use latest input
next_input = X.iloc[-1:].copy()

# Get probabilities for each number slot
probabilities = [estimator.predict_proba(next_input)[0] for estimator in model.estimators_]

# Generate combinations
combinations = []
for _ in range(10):
    main_nums = set()
    probs = []
    for i in range(6):
        prob_dist = probabilities[i]
        choices = np.argsort(prob_dist)[::-1]  # most likely first
        for num in choices:
            if num not in main_nums:
                main_nums.add(num)
                probs.append(prob_dist[num])
                break
    main_nums = sorted([n + 1 for n in main_nums])  # shift index to actual lotto numbers

    # Bonus number
    bonus_probs = probabilities[6]
    bonus_num = np.argmax(bonus_probs) + 1
    probs.append(bonus_probs[bonus_num - 1])

    # Score as average confidence
    likelihood = round(np.mean(probs) * 100, 2)
    combinations.append((main_nums, bonus_num, likelihood))

# Display results
for i, (nums, bonus, score) in enumerate(combinations, 1):
    print(f"Combo {i}: {nums} + PowerBall: {bonus} — Likelihood: {score}%")
