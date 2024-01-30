import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from keras.regularizers import l2
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from keras.metrics import Precision, Recall
# Load the dataset
df = pd.read_csv('/Users/karlbrzoska/Downloads/1/aggregatedOutput/004-3YK3J151X6.csv')

# Convert timestamp column to datetime format and calculate time since start in minutes
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
        sequences_X.append(df.drop('outcome_event', axis=1).iloc[i:i+sequence_length].values)
        sequences_y.append(df['outcome_event'].iloc[i+sequence_length-1])

# Convert to numpy arrays for machine learning processing
sequences_X = np.array(sequences_X)
sequences_y = np.array(sequences_y)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

for train, test in kfold.split(sequences_X, sequences_y):
    # Split data
    X_train, y_train = sequences_X[train], sequences_y[train]
    X_test, y_test = sequences_X[test], sequences_y[test]

    # Create a pipeline with SMOTE and scaling
    pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
        ('smote', SMOTE(random_state=42))
    ])

    # Reshape and preprocess data using pipeline
    num_features = sequences_X.shape[2]
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_train_preprocessed, y_train_preprocessed = pipeline.fit_resample(X_train_reshaped, y_train)
    X_train_preprocessed = X_train_preprocessed.reshape(-1, sequences_X.shape[1], num_features)

    # Preprocess test data
    X_test_preprocessed = SimpleImputer(strategy='mean').fit_transform(X_test.reshape(X_test.shape[0], -1))
    X_test_preprocessed = StandardScaler().fit_transform(X_test_preprocessed).reshape(-1, sequences_X.shape[1], num_features)

    # Define the CNN model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequences_X.shape[1], num_features), kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

    # Fit the model
    history = model.fit(X_train_preprocessed, y_train_preprocessed, epochs=50, batch_size=64, verbose=1)

    # Prediction and evaluation
    y_pred_probs = model.predict(X_test_preprocessed)
    y_pred = (y_pred_probs > 0.5).astype("int32")  # Adjust threshold if necessary
    
    print(f'Classification Report for fold {fold_no}:\n{classification_report(y_test, y_pred)}\n')
    
    fold_no += 1

print('Cross-validation completed.')
