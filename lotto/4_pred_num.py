import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('calculated_lotto.csv')

# Features and targets
features = ['mean', 'median', 'std', 'min', 'max', 'range', 'sum_without_bonus',
            'even_count', 'odd_count', 'low_range_count', 'high_range_count',
            'prev_common_count']
targets = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']

X = df[features].copy()
y = df[targets].copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Predict using latest row
next_input = X.iloc[-1:].copy()
prediction = model.predict(next_input)[0]

# Ensure 6 unique main numbers
main_numbers = list(prediction[:6])

# Replace duplicates
main_numbers = list(dict.fromkeys(main_numbers))  # Remove duplicates
while len(main_numbers) < 6:
    new_num = np.random.randint(1, 53)  # Assuming range 1–52
    if new_num not in main_numbers:
        main_numbers.append(new_num)

main_numbers.sort()

print("Predicted Numbers:", main_numbers)
