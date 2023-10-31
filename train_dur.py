import pandas as pd
import numpy as np
import re
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

df = pd.read_csv('finish_df.csv')

df['duration_seconds'] = df['duration'].apply(lambda x: pd.to_timedelta(x).total_seconds())
df['duration_seconds'] = df['duration_seconds'].astype(int)

selected_features = ['duration_seconds','statement']  #
df = df[selected_features]
df

# Select the 'duration_seconds' column
duration_seconds = df['duration_seconds'].values.reshape(-1, 1)

# Define the number of clusters (you can adjust this based on your preference)
num_clusters = 3

# Create a K-Means clustering model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(duration_seconds)

# Replace the 'duration_seconds' column with cluster labels
df['duration_cluster'] = kmeans.labels_

# Explore the cluster statistics
cluster_stats = df.groupby('duration_cluster')['duration_seconds'].describe()
print(cluster_stats)

# Text preprocessing and feature engineering
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_statement = tfidf_vectorizer.fit_transform(df['statement'])

# Select features and target variable
X_statement = X_statement.toarray()  

# Ensure all feature matrices have the same number of rows
assert X_statement.shape[0] #== X_categorical.shape[0] == X_duration.shape[0]

# Concatenate the feature matrices
X = np.hstack((X_statement,))

y = df['duration_cluster']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start the timer
start_time = time.time()

# Train the XGBoost regression model
model_duration = XGBRegressor()
model_duration.fit(X_train, y_train)

joblib.dump(model_duration, 'duration_cluster_model.bin')
joblib.dump(tfidf_vectorizer, 'duration_tfidf_vectorizer.bin')

# Make predictions
y_pred = model_duration.predict(X_test)
y_pred.round().astype(int)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared (R^2) score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R^2) Score: {r2}")

# End the timer
end_time = time.time()

# Calculate the duration
training_duration = end_time - start_time

# Print the duration
print(f"Model training took {training_duration:.2f} seconds")








