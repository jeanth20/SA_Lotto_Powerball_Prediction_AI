import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations
import random

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

# Ask user how many combinations to generate
try:
    num_combos = int(input("How many combinations do you want to generate? "))
except ValueError:
    num_combos = 10

# Use the last input as base
next_input = X.iloc[[-1]].copy()

# Predict probabilities
probabilities = [estimator.predict_proba(next_input)[0] for estimator in model.estimators_]

# Get ranges
num_range = list(range(1, len(probabilities[0]) + 1))
bonus_range = list(range(1, len(probabilities[6]) + 1))

# Generate N unique random combinations
combinations_set = set()
results = []

while len(results) < num_combos:
    main_nums = set()
    probs = []

    while len(main_nums) < 6:
        i = len(main_nums)
        prob_dist = probabilities[i]
        choices = np.arange(1, len(prob_dist) + 1)
        chosen = np.random.choice(choices, p=prob_dist / prob_dist.sum())
        if chosen not in main_nums:
            main_nums.add(chosen)
            probs.append(prob_dist[chosen - 1])

    bonus_dist = probabilities[6]
    bonus_choices = np.arange(1, len(bonus_dist) + 1)
    bonus_num = np.random.choice(bonus_choices, p=bonus_dist / bonus_dist.sum())
    probs.append(bonus_dist[bonus_num - 1])

    sorted_main = tuple(sorted(main_nums))
    full_combo = (sorted_main, bonus_num)
    if full_combo not in combinations_set:
        combinations_set.add(full_combo)
        results.append((sorted_main, bonus_num, round(np.mean(probs) * 100, 2)))

# Evaluate best and worst from generated set
results.sort(key=lambda x: x[2], reverse=True)
best_generated = results[0]
worst_generated = results[-1]

# Sample limited number of all combinations to find overall best/worst
sampled_all = set()
sample_results = []

while len(sample_results) < 1000:
    sample_main = tuple(sorted(random.sample(num_range, 6)))
    sample_bonus = random.choice(bonus_range)
    if (sample_main, sample_bonus) not in sampled_all:
        sampled_all.add((sample_main, sample_bonus))
        probs = [probabilities[i][n - 1] for i, n in enumerate(sample_main)]
        probs.append(probabilities[6][sample_bonus - 1])
        likelihood = round(np.mean(probs) * 100, 4)
        sample_results.append((sample_main, sample_bonus, likelihood))

sample_results.sort(key=lambda x: x[2], reverse=True)
overall_best = sample_results[0]
overall_worst = sample_results[-1]

# Print output
print("\nGenerated Combinations:\n")
for i, (nums, bonus, score) in enumerate(results, 1):
    print(f"Combo {i}: {list(nums)} + PowerBall: {bonus} — Likelihood: {score}%")

print("\nMost Likely from Generated:")
print(f"{list(best_generated[0])} + PowerBall: {best_generated[1]} — Likelihood: {best_generated[2]}%")

print("\nLeast Likely from Generated:")
print(f"{list(worst_generated[0])} + PowerBall: {worst_generated[1]} — Likelihood: {worst_generated[2]}%")

print("\nOverall Most Likely Combination (sampled):")
print(f"{list(overall_best[0])} + PowerBall: {overall_best[1]} — Likelihood: {overall_best[2]}%")

print("\nOverall Least Likely Combination (sampled):")
print(f"{list(overall_worst[0])} + PowerBall: {overall_worst[1]} — Likelihood: {overall_worst[2]}%")


# just create 10
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.model_selection import train_test_split

# # Load and prepare data
# df = pd.read_csv('calculated_lotto.csv')

# features = ['mean', 'median', 'std', 'min', 'max', 'range', 'sum_without_bonus',
#             'even_count', 'odd_count', 'low_range_count', 'high_range_count',
#             'prev_common_count', 'has_consecutive']
# targets = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'BonusBall']

# X = df[features].copy()
# y = df[targets].copy()
# X['has_consecutive'] = X['has_consecutive'].astype(int)

# # Train model
# X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# model = MultiOutputClassifier(rf)
# model.fit(X_train, y_train)

# # Use the last input as base
# next_input = X.iloc[[-1]].copy()

# # Predict probabilities
# probabilities = [estimator.predict_proba(next_input)[0] for estimator in model.estimators_]

# # Generate combinations with randomness based on probability
# combinations = set()
# attempts = 0

# while len(combinations) < 10 and attempts < 100:
#     attempts += 1
#     main_nums = set()
#     probs = []
#     for i in range(6):
#         prob_dist = probabilities[i]
#         choices = np.arange(1, len(prob_dist) + 1)
#         chosen = np.random.choice(choices, p=prob_dist / prob_dist.sum())
#         while chosen in main_nums:
#             chosen = np.random.choice(choices, p=prob_dist / prob_dist.sum())
#         main_nums.add(chosen)
#         probs.append(prob_dist[chosen - 1])

#     # Bonus ball
#     bonus_dist = probabilities[6]
#     bonus_choices = np.arange(1, len(bonus_dist) + 1)
#     bonus_num = np.random.choice(bonus_choices, p=bonus_dist / bonus_dist.sum())
#     probs.append(bonus_dist[bonus_num - 1])

#     sorted_main = tuple(sorted(main_nums))
#     full_combo = (sorted_main, bonus_num)
#     if full_combo not in combinations:
#         likelihood = round(np.mean(probs) * 100, 2)
#         combinations.add((sorted_main, bonus_num, likelihood))

# # Display results
# for i, (nums, bonus, score) in enumerate(combinations, 1):
#     print(f"Combo {i}: {list(nums)} + PowerBall: {bonus} — Likelihood: {score}%")
