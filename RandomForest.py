import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
df = pd.read_csv('/Users/karlbrzoska/Downloads/1/aggregatedOutput/004-3YK3J151X6.csv')
df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'])
df['time_since_start'] = (df['timestamp_iso'] - df['timestamp_iso'].iloc[0]).dt.total_seconds() / 60

# Drop unnecessary columns
df = df.drop(['participant_full_id', 'timestamp_iso'], axis=1)

# Filter out instances with more than 80% missing values
threshold = len(df.columns) * 0.8
df = df.dropna(thresh=threshold)
# Identify outcome events lasting at least 15 minutes
positive_outcome_duration = 15  # Duration in minutes for a sustained low glucose event
df['outcome_event'] = 0  # Initialize column with no positive outcome
for i in range(len(df) - positive_outcome_duration):
    if (df['glucose'].iloc[i:i+positive_outcome_duration] < 70).all():
        df['outcome_event'].iloc[i+positive_outcome_duration-1] = 1

# Sequence generation for training data
sequence_length = 60  # 55 minutes observation + 5 minutes prediction
sequences_X = []
sequences_y = []
# Generate sequences ensuring no positive outcome overlaps with the observation window
for i in range(len(df) - sequence_length):
    if df['outcome_event'].iloc[i] == 0:  # Ensure no positive outcome at the start of sequence
        sequences_X.append(df.drop('outcome_event', axis=1).iloc[i:i+sequence_length].values.flatten())
        sequences_y.append(df['outcome_event'].iloc[i+sequence_length-1])

# Convert to numpy arrays for machine learning processing
sequences_X = np.array(sequences_X)
sequences_y = np.array(sequences_y)
# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(sequences_X, sequences_y, test_size=0.2, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the StandardScaler and Imputer
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

# Transform the validation and test data
X_val = imputer.transform(X_val)
X_val = scaler.transform(X_val)
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Train the Random Forest classifier with class weights to handle imbalance
random_forest = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)
random_forest.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = random_forest.predict(X_val)

# Evaluate the model
classification_report_result = classification_report(y_val, y_val_pred)
roc_auc_score_result = roc_auc_score(y_val, y_val_pred)

print(classification_report_result)
print("ROC-AUC score:", roc_auc_score_result)
