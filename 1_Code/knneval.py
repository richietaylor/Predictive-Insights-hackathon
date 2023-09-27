import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the training data
train_data = pd.read_csv('processed_train.csv')

# Removing unnecessary columns
train_data.drop(columns=['Person_id'], inplace=True)

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Target'])
y_train = train_data['Target']

# Determine the best number of neighbors
neighbors_range = list(range(1, 51))  # check from 1 to 50 neighbors
cross_val_scores = []

for n in neighbors_range:
    knn_model = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn_model, X_train, y_train, cv=5)
    cross_val_scores.append(scores.mean())

best_neighbors = neighbors_range[np.argmax(cross_val_scores)]
print(f"Best number of neighbors: {best_neighbors}")

# Creating the KNN model with the best number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=best_neighbors)

# Fitting the KNN model to the training data
knn_model.fit(X_train, y_train)

# Load the test data
test_data = pd.read_csv('processed_test.csv')

# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=['Person_id'])

# Making predictions on the test data (predicting probabilities for the positive class)
y_test_prob_pred_knn = knn_model.predict_proba(test_data_processed)[:, 1]

# Creating a DataFrame with "person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_knn
})

# Path to save the CSV file
predictions_csv_path = 'knn.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)
