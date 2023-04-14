import pandas as pd

# Read the actual links from CSV file
df_actual = pd.read_csv('final_dataset.csv')

# Read the predicted links from CSV file
df_pred = pd.read_csv('final_predicted_links.csv')

# Calculate the true positives, false positives, and false negatives
tp = len(pd.merge(df_actual, df_pred, on=['country', 'organization']))
fp = len(df_pred) - tp
fn = len(df_actual) - tp

# Calculate the precision, recall, and F1-score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

# Print the evaluation metrics
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1_score)
