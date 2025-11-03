# From lines 13-19 was made by ChatGPT and 67-82 was made by ChatGPT
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Column names (from adult.names)
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]

# Load datasets
train = pd.read_csv("adult.test", header=None, names=columns, skiprows=1, na_values=" ?")
test = pd.read_csv("adult.data", header=None, names=columns, na_values=" ?")

# Clean missing values
train.dropna(inplace=True)
test.dropna(inplace=True)

def backward_chain(row):
    """Goal-driven inference: tries to prove '>50K' first"""
    goal = '>50K'

    if goal == '>50K':
        #  Education-based rules 
        if row['education'] in ['Masters', 'Doctorate']:
            if row['hours_per_week'] > 40:
                return '>50K'

        #  Occupation-based rules 
        if row['occupation'] in ['Exec-managerial', 'Prof-specialty']:
            if row['hours_per_week'] >= 45 or row['education_num'] >= 13:
                return '>50K'

        #  Marital and work rules 
        if row['marital_status'] == 'Married-civ-spouse' and row['hours_per_week'] >= 45:
            return '>50K'

        #  Capital gain rule 
        if row['capital_gain'] > 5000:
            return '>50K'

        #  There is a gender bias in the dataset
        if row['sex'] == 'Male' and row['hours_per_week'] > 50 and row['education_num'] >= 12:
            return '>50K'

        #  Older age rule 
        if row['age'] > 40 and row['education_num'] >= 12 and row['hours_per_week'] > 45:
            return '>50K'

        # If none of the above rules prove >50K
        return '<=50K'

    else:
        # For completeness, infer <=50K
        if row['age'] < 25 and row['education_num'] < 10:
            return '<=50K'
        elif row['hours_per_week'] < 33:
            return '<=50K'
        else:
            return '>50K'

# Apply backward chaining inference
test['predicted_backward'] = test.apply(backward_chain, axis=1)

# Prepare ground truth and predictions
y_true = test['income'].str.strip()
y_pred_backward = test['predicted_backward']

# Evaluate accuracy and metrics
backward_accuracy = accuracy_score(y_true, y_pred_backward)
backward_precision = precision_score(y_true, y_pred_backward, pos_label='>50K')
backward_recall = recall_score(y_true, y_pred_backward, pos_label='>50K')
backward_f1 = f1_score(y_true, y_pred_backward, pos_label='>50K')

print("=== Backward Chaining Evaluation ===")
print(f"Accuracy:  {backward_accuracy:.4f}")
print(f"Precision: {backward_precision:.4f}")
print(f"Recall:    {backward_recall:.4f}")
print(f"F1 Score:  {backward_f1:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_backward))

# Optional sample output
print("\nSample predictions:")
print(test[['age', 'education', 'occupation', 'hours_per_week', 'marital_status', 'sex', 
            'capital_gain', 'income', 'predicted_backward']].head(10))
