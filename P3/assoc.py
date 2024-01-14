#!/bin/python3

import pandas as pd

# Please download this as a custom package --> type "apyori"
# To load custom packages, do not refresh the page. Instead, click on the reset button on the Console.
from apyori import apriori

dataset = pd.read_csv("store_data.csv", header=None, index_col=None)

# Transforming the list into a list of lists, so that each transaction can be indexed easier
transactions = []
for i, row in dataset.iterrows():
    # Remove nan values produced by having different number of items in each row
    row = row.dropna().astype(str)
    transactions.append(row.tolist())

rules = apriori(transactions, min_support=0.005, min_confidence=0.25, min_lift=3)
# Support: number of transactions containing set of times / total number of transactions
# .      --> products that are bought at least 3 times a day --> 21 / 7501 = 0.0027
# Confidence: Should not be too high, as then this will lead to obvious rules

# Try many combinations of values to experiment with the model

results = []
for item in rules:
    for rule in item.ordered_statistics:
        results.append(
            {
                "items_base": set(rule.items_base),
                "items_add": set(rule.items_add),
                "support": item.support,
                "confidence": rule.confidence,
                "lift": rule.lift,
            }
        )

# Convert the list of dictionaries into a dataframe
df_results = pd.DataFrame(results)

for i, rule in df_results.iterrows():
    print(f"Rule: {set(rule['items_base'])} -> {set(rule['items_add'])}")
    print(f"Support: {rule['support']}")
    print(f"Confidence: {rule['confidence']}")
    print(f"Lift: {rule['lift']}")
    print("=====================================")

# Save the rules into a csv file
df_results.to_csv("results.csv")
