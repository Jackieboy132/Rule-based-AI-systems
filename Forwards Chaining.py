# 130-159 was made by chatgpt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Column names (from adult.names)
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]

# Load datasets copied from Backwards Chaining
train = pd.read_csv("adult.test", header=None, names=columns, skiprows=1, na_values=" ?")
test = pd.read_csv("adult.data", header=None, names=columns, na_values=" ?")

#clearn missing values
train.dropna(inplace=True)
test.dropna(inplace=True)

# Normalize income labels
train["income"] = train["income"].str.strip().str.replace(".", "", regex=False)
test["income"] = test["income"].str.strip().str.replace(".", "", regex=False)


def make_facts(person):
    """Turn each record into a set of symbolic facts."""
    facts = set()

    # Age categories
    if person['age'] < 25:
        facts.add("young")
    elif person['age'] < 45:
        facts.add("middle_aged")
    else:
        facts.add("older")

# symbolic categories for attributes: Education, Hours worked, Marital status, Occupation, Capital gain/loss, Gender
    # Education
    if person['education'] in ['Masters', 'Doctorate']:
        facts.add("advanced_degree")
    elif person['education'] in ['Bachelors', 'Prof-school']:
        facts.add("bachelor_level")
    elif person['education'] in ['HS-grad', 'Some-college']:
        facts.add("mid_education")
    else:
        facts.add("low_education")

    # Hours worked
    if person['hours_per_week'] > 50:
        facts.add("very_long_hours")
    elif person['hours_per_week'] >= 40:
        facts.add("full_time")
    else:
        facts.add("part_time")

    # Marital status
    if person['marital_status'] == 'Married-civ-spouse':
        facts.add("married")

    # Occupation
    if person['occupation'] in ['Exec-managerial', 'Prof-specialty']:
        facts.add("professional")
    elif person['occupation'] in ['Craft-repair', 'Sales']:
        facts.add("skilled")
    else:
        facts.add("non_professional")

    # Capital gain/loss
    if person['capital_gain'] > 0:
        facts.add("invests")
    if person['capital_loss'] > 0:
        facts.add("loss_reported")

    # Gender
    if person['sex'].strip().lower() == 'male':
        facts.add("is_male")
    else:
        facts.add("is_female")

    return facts

rules = [
    # High earners
    {"if": {"advanced_degree", "full_time"}, "then": "income>50K"},
    {"if": {"bachelor_level", "professional", "full_time"}, "then": "income>50K"},
    {"if": {"professional", "very_long_hours"}, "then": "income>50K"},
    {"if": {"married", "professional"}, "then": "income>50K"},
    {"if": {"older", "bachelor_level", "invests"}, "then": "income>50K"},
    {"if": {"married", "very_long_hours"}, "then": "income>50K"},
    {"if": {"invests"}, "then": "income>50K"},

    # Lower earners
    {"if": {"young", "low_education"}, "then": "income<=50K"},
    {"if": {"low_education", "non_professional"}, "then": "income<=50K"},
    {"if": {"is_female", "part_time"}, "then": "income<=50K"},
    {"if": {"is_male", "low_education", "part_time"}, "then": "income<=50K"},
    {"if": {"non_professional", "mid_education"}, "then": "income<=50K"},
    {"if": {"loss_reported", "non_professional"}, "then": "income<=50K"},
]

# Forward chaining implementation
def forward_chain(facts, rules):
    inferred = set()
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule["if"].issubset(facts) and rule["then"] not in facts:
                facts.add(rule["then"])
                inferred.add(rule["then"])
                changed = True
    return inferred

# Predict function using forward chaining
def predict_forward_chain(row):
    facts = make_facts(row)
    inferred = forward_chain(facts.copy(), rules)
    if "income>50K" in inferred:
        return ">50K"
    elif "income<=50K" in inferred:
        return "<=50K"
    else:
        # Default fallback rule
        if row['education_num'] >= 13 and row['hours_per_week'] > 40:
            return ">50K"
        else:
            return "<=50K"


# Apply to test dataset
test['predicted_forward'] = test.apply(predict_forward_chain, axis=1)

# Evaluate Forward Chaining System
y_true = test['income'].str.strip()
y_pred_forward = test['predicted_forward']

correct = (y_true == y_pred_forward).sum()
total = len(test)
accuracy_forward = correct / total

print(f"\nImproved Forward Chaining System Accuracy: {accuracy_forward:.2%}")
print(f"Correct predictions: {correct} / {total}\n")


print("=== Forward Chaining Evaluation ===")
forward_accuracy = accuracy_score(y_true, y_pred_forward)
forward_precision = precision_score(y_true, y_pred_forward, pos_label='>50K')
forward_recall = recall_score(y_true, y_pred_forward, pos_label='>50K')
forward_f1 = f1_score(y_true, y_pred_forward, pos_label='>50K')

print(f"Accuracy:  {forward_accuracy:.4f}")
print(f"Precision: {forward_precision:.4f}")
print(f"Recall:    {forward_recall:.4f}")
print(f"F1 Score:  {forward_f1:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_forward))


print("\nSample Predictions:")
print(test[['age', 'education', 'hours_per_week', 'marital_status', 'occupation', 'income', 'predicted_forward']].head(10))
