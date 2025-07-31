# ========================================================================
#   Imports
# ======================================================================== 

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


# ========================================================================
#   Generate Dummy Data
# ======================================================================== 

# Set random seed for reproducibility
np.random.seed(42)

# Create 10000 rows of realistic ConEd data (90% Found Okay, 10% Needs Repair)
n_samples = 10000

# Generate base features
data = {
    # Generate unique IDs
    'case_id': range(1, n_samples + 1),
    
    # Generate random dates between current date and current date - 365d
    'date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
    
    # Add probability values
    'complaint_type': np.random.choice(['No Heat', 'No Hot Water', 'Gas Smell', 'Power Out', 'Meter Issue'], n_samples, p=[0.3, 0.25, 0.1, 0.2, 0.15]),
    
    # Set building age range using a bell curve centered on 50 yrs with a 150-yr spread; keep between 0 and 120 yrs
    'building_age': np.random.normal(50, 15, n_samples).clip(0, 120).astype(int),
    
    # Randomly choose a borough
    'borough': np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'], n_samples),
    
    # Most values are 0-180 days; some rare cases may be 500+ days
    'days_since_last_service': np.random.exponential(180, n_samples).astype(int),
    
    # Randomly select
    'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples),
    'season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_samples)
}

df = pd.DataFrame(data)

# Create realistic patterns for outcomes
def determine_outcome(row):
    # Gas smell almost always needs repair
    if row['complaint_type'] == 'Gas Smell':
        return 'Needs Repair' if np.random.random() > 0.1 else 'Found Okay'
    
    # Old building + winter + no heat = usually needs repair
    if row['building_age'] > 40 and row['season'] == 'Winter' and row['complaint_type'] == 'No Heat':
        return 'Needs Repair' if np.random.random() > 0.2 else 'Found Okay'
    
    # New building + summer + no hot water = usually okay
    if row['building_age'] < 10 and row['season'] == 'Summer' and row['complaint_type'] == 'No Hot Water':
        return 'Found Okay' if np.random.random() > 0.05 else 'Needs Repair'
    
    # Default: 90% Found Okay
    return 'Found Okay' if np.random.random() > 0.1 else 'Needs Repair'

df['status'] = df.apply(determine_outcome, axis=1)

# Save to CSV
df.to_csv('coned_sample_data.csv', index=False)

# Show what we created
print(f"Created {len(df)} records")
print("\nStatus distribution:")
print(df['status'].value_counts())
print(f"\nClass balance: {df['status'].value_counts(normalize=True) * 100}")



# ========================================================================
#   Weight Data
# ======================================================================== 

# Prepare data
X = pd.get_dummies(df.drop(['case_id', 'date', 'status'], axis=1))

# Create a Boolean to assign 1 to 'Found Okay' and 0 to 'Needs Repair'
y = (df['status'] == 'Found Okay').astype(int)

# Do an 80/20 split and stratify to keep proportions of test data the same as the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Count each of the unique values ('Found Okay', 'Needs Repair')
classes = np.unique(y_train)

# Calculate weights
weights = compute_class_weight('balanced', classes=classes, y=y_train)

# Generate dictionary with weights for each class
class_weight_dict = dict(zip(classes, weights))

print(f"Class weights: {class_weight_dict}")
# Output: {0: 5.26, 1: 0.58} - gives more importance to rare class (0 = 'Needs Repair' if that's the less frequent class)

# Train with class weights
model_weighted = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
model_weighted.fit(X_train, y_train)

# Compare to model without weights
model_normal = RandomForestClassifier(random_state=42)
model_normal.fit(X_train, y_train)

# Print to the terminal
print("\nWITHOUT class weights:")
print(classification_report(y_test, model_normal.predict(X_test)))

print("\nWITH class weights:")
print(classification_report(y_test, model_weighted.predict(X_test)))