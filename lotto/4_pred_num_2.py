import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import csv
from datetime import date

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
main_numbers = list(dict.fromkeys(main_numbers))  # Remove duplicates
while len(main_numbers) < 6:
    new_num = np.random.randint(1, 53)  # Assuming range 1–52
    if new_num not in main_numbers:
        main_numbers.append(new_num)

main_numbers.sort()

# Calculate likelihoods for all numbers 1-52
probabilities = [estimator.predict_proba(next_input)[0] for estimator in model.estimators_]

# Aggregate probabilities across all 6 targets
number_likelihood = {i: 0 for i in range(1, 53)}
for probs in probabilities:
    for idx, prob in enumerate(probs):
        number_likelihood[idx + 1] += prob

# Average the probability across the 6 targets
for num in number_likelihood:
    number_likelihood[num] = round((number_likelihood[num] / 6) * 100, 2)

# Create full table
full_likelihood_df = pd.DataFrame({
    'Number': list(number_likelihood.keys()),
    'Likelihood (%)': list(number_likelihood.values())
})

# Sort by number
full_likelihood_df = full_likelihood_df.sort_values('Number')

print("\nPredicted Numbers:", main_numbers)
print("\nLikelihood Table:")
print(full_likelihood_df.to_string(index=False))

# Create a new file to save the perdiction data
date = date.today()
with open("prediction" + date.strftime("%Y-%m-%d") + ".csv", "w", newline="") as output_file:
    csv_writer = csv.writer(output_file)

    csv_writer.writerow(["Predicted Numbers"])
    csv_writer.writerow(main_numbers)

    # Write the header row
    csv_writer.writerow(["Number", "Likelihood (%)"])

    # Write the data rows
    for number, likelihood in number_likelihood.items():
        csv_writer.writerow([number, likelihood])
