import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
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
X_train, X_temp, y_train, y_temp = train_test_split(sequences_X, sequences_y, test_size=0.2, random_state=42, shuffle=True, stratify=sequences_y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

# Initialize the StandardScaler and Imputer
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
X_train = imputer.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_train = scaler.fit_transform(X_train).reshape(X_train.shape[0], sequence_length, -1)
# Transform the validation and test data
X_val = imputer.transform(X_val.reshape(X_val.shape[0], -1))
X_val = scaler.transform(X_val).reshape(X_val.shape[0], sequence_length, -1)
X_test = imputer.transform(X_test.reshape(X_test.shape[0], -1))
X_test = scaler.transform(X_test).reshape(X_test.shape[0], sequence_length, -1)

# Apply RandomUnderSampler without reshaping
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
X_train_resampled = X_train_resampled.reshape(-1, sequence_length, X_train.shape[2])

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights = dict(enumerate(class_weights))
query_input = Input(shape=(X_train_resampled.shape[1], X_train_resampled.shape[2]))
value_input = Input(shape=(X_train_resampled.shape[1], X_train_resampled.shape[2]))

# LSTM layers with return_sequences=True to return the sequence of hidden states
lstm_query = LSTM(100, return_sequences=True)(query_input)
lstm_value = LSTM(100, return_sequences=True)(value_input)

# Attention mechanism using Keras layers
# Note: This code assumes that the LSTM layers return sequences (return_sequences=True)
# You may need to adjust the attention mechanism depending on your specific requirements.
attention = Attention()([lstm_query, lstm_value])

# LSTM layer for further processing
lstm_final = LSTM(50)(attention)

# Dense layers
dense = Dense(100, activation='relu')(lstm_final)
dropout = Dropout(0.2)(dense)
output = Dense(1, activation='sigmoid')(dropout)  # Binary classification

# Create a model using query and value inputs and the attention mechanism
model = Model(inputs=[query_input, value_input], outputs=output)

# Compile model with F1 score as a metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1Score()])

# Define early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# Train model with the resampled data
history = model.fit(
    [X_train_resampled, X_train_resampled],
    y_train_resampled, 
    validation_data=([X_val, X_val], y_val),
    epochs=50, batch_size=64, 
    class_weight=class_weights,
    callbacks=[early_stopping_callback]
)

# Evaluate the model on the test data containing different participants
test_results = model.evaluate([X_test, X_test], y_test)
test_loss, test_accuracy, test_f1_score = test_results

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1_score}")
