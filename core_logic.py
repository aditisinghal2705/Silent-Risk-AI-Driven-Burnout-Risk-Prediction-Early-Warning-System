import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# STEP 1: Understand and Load the Data
# ----------------------------------------------------------------------------
# We load the dataset using Pandas. Pandas is like Excel for Python.
# It allows us to view and manipulate our data easily.
print("Loading data...")
df = pd.read_csv('processed_burnout_data.csv')

# We select the columns that have numbers and impact "Burn Rate"
# These are our "Features" (X)
feature_cols = ['Designation', 'Resource Allocation', 'Mental Fatigue Score', 'Tenure_Days', 'Workload_Intensity']

# We need to make sure there are no missing values in these columns
df = df.dropna(subset=feature_cols + ['Burn Rate'])

X = df[feature_cols]
y = df['Burn Rate'] # This is what we want to predict (Target)

# ----------------------------------------------------------------------------
# STEP 2: Train the Model
# ----------------------------------------------------------------------------
# We split our data: 80% to train the model, 20% to test how good it is.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We use Random Forest. Think of it as a council of many "Decision Trees".
# Each tree makes a guess, and they average it out for the final accurate prediction.
print("Training Random Forest Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# STEP 3: The "Sieve" Function (Classification)
# ----------------------------------------------------------------------------
# This takes the raw decimal from the model and turns it into an actionable Risk Level.
def sieve_risk_category(score):
    """
    Converts raw prediction score into Corporate Wellness risk categories.

    These thresholds match the Risk_Zone labels in processed_burnout_data.csv:
    Low Risk: score < 0.30
    Medium Risk: 0.30 <= score < 0.70
    High Risk: score >= 0.70
    """
    if score < 0.3:
        return "Low Risk (Green)", "No action needed. Maintain current workload."
    elif score < 0.7:
        return "Medium Risk (Yellow)", "Flag for a 1-on-1 meeting; suggest a wellness day."
    else:
        return "High Risk (Red)", "Immediate workload reduction; mandatory HR check-in."

# ----------------------------------------------------------------------------
# STEP 4: The "Why" Output (Feature Importance Explanation)
# ----------------------------------------------------------------------------
# To tell the manager "Why", we calculate the Global Feature Importance.
# This tells us which factors generally drive burnout the most in your company.
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

def explain_employee_burnout(employee_data, predicted_score):
    """
    Provides a simple explanation of why the employee got this score based on their stats
    and the global model importances.
    """
    # Simply using the global importances to explain which features are the biggest drivers 
    # and showing the user's specific stats for those drivers.
    top_feature_1 = feature_importance_df.iloc[0]['Feature']
    top_feature_2 = feature_importance_df.iloc[1]['Feature']
    
    val1 = employee_data[top_feature_1].values[0]
    val2 = employee_data[top_feature_2].values[0]
    
    pct1 = round(feature_importance_df.iloc[0]['Importance'] * 100)
    pct2 = round(feature_importance_df.iloc[1]['Importance'] * 100)

    category, action = sieve_risk_category(predicted_score)
    
    explanation = (
        f"Employee assigned category: {category} ({predicted_score:.2f}).\n"
        f"This is primarily driven by their {top_feature_1} (Value: {val1:.1f}, contributes ~{pct1}% to risk) "
        f"and {top_feature_2} (Value: {val2:.1f}, contributes ~{pct2}% to risk).\n"
        f"Recommended Action: {action}"
    )
    return explanation

def print_terminal_bar_chart(title, rows, label_key, value_key, value_formatter, bar_width=40):
    """
    Prints a simple horizontal bar chart in the terminal.
    """
    print(f"\n{title}")
    print("-" * len(title))

    max_value = max(row[value_key] for row in rows)
    label_width = max(len(str(row[label_key])) for row in rows)

    for row in rows:
        label = str(row[label_key])
        value = row[value_key]
        filled_width = round((value / max_value) * bar_width) if max_value else 0
        bar = "#" * filled_width
        print(f"{label:<{label_width}} | {bar:<{bar_width}} | {value_formatter(value)}")

# ----------------------------------------------------------------------------
# STEP 5: Testing it out!
# ----------------------------------------------------------------------------
print("\n--- CORPORATE WELLNESS SYSTEM TEST ---")
# Let's pick a random employee from our test set
sample_employee = X_test.sample(1, random_state=99)
predicted_burn_rate = model.predict(sample_employee)[0]

print(explain_employee_burnout(sample_employee, predicted_burn_rate))


# ----------------------------------------------------------------------------
# STEP 6: Representations for the Teacher
# ----------------------------------------------------------------------------
print("\nGenerating terminal graphs for your teacher...")

# 1. Feature Importance Chart (Shows WHAT causes burnout)
feature_rows = feature_importance_df.to_dict('records')
print_terminal_bar_chart(
    title="What Drives Burnout? (Feature Importance)",
    rows=feature_rows,
    label_key='Feature',
    value_key='Importance',
    value_formatter=lambda value: f"{value * 100:.1f}%"
)


# 2. Risk Distribution Chart (Shows how many people fall into each Sieve category)
# We apply our sieve to the entire dataset to show the general health of the company
df['Predicted_Burn_Rate'] = model.predict(X)
df['Risk_Category'] = df['Predicted_Burn_Rate'].apply(lambda x: sieve_risk_category(x)[0])

risk_order = ["Low Risk (Green)", "Medium Risk (Yellow)", "High Risk (Red)"]
risk_counts = df['Risk_Category'].value_counts().reindex(risk_order, fill_value=0)
risk_rows = [
    {'Risk_Category': category, 'Count': int(count)}
    for category, count in risk_counts.items()
]
print_terminal_bar_chart(
    title="Company Overall Health (Risk Category Distribution)",
    rows=risk_rows,
    label_key='Risk_Category',
    value_key='Count',
    value_formatter=lambda value: f"{value} employees"
)

print("\nDone! Both graphs are shown in the terminal.")
