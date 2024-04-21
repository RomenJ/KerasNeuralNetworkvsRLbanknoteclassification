# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression

# Load the dataset
banknotes = pd.read_csv("banknotes.csv")

# Display basic information about the dataset
print("Shape:", banknotes.shape)
print("Head:\n", banknotes.head(10))
print("Info:\n", banknotes.info())

# Describe the data
print('Dataset stats:\n', banknotes.describe())

# Count the number of observations per class
print('Observations per class:\n', banknotes["class"].value_counts())

# Define features and labels
X = banknotes.drop('class', axis=1).values
y = banknotes['class'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a sequential model
model = Sequential()

# Add dense layers
model.add(Dense(32, input_shape=(4,), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

# Display a summary of the model architecture
model.summary()

# Train the model for 20 epochs
history = model.fit(X_train, y_train, epochs=20, verbose=0)

# Plot accuracy over the iterations
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy Over Iterations')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("Model Accuracy Over Iterations.jpg")
plt.show()

# Evaluate the neural network model accuracy on the test set
y_pred_prob_nn = model.predict(X_test)
y_pred_nn = np.round(y_pred_prob_nn).astype(int)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

# Print the neural network model accuracy
print('Neural Network Accuracy:', accuracy_nn)
# Print additional evaluation metrics for neural network model
print('Neural Network Confusion Matrix:\n', confusion_matrix(y_test, y_pred_nn))
print('Neural Network Classification Report:\n', classification_report(y_test, y_pred_nn))

# Train a logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluate the logistic regression model accuracy on the test set
y_pred_lr = log_reg.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)



# Print the logistic regression model accuracy
print('Logistic Regression Accuracy:', accuracy_lr)
# Print additional evaluation metrics for logistic regression model
print('Logistic Regression Confusion Matrix:\n', confusion_matrix(y_test, y_pred_lr))
print('Logistic Regression Classification Report:\n', classification_report(y_test, y_pred_lr))

# Plotting comparison of accuracy between models
models = ['Neural Network', 'Logistic Regression']
accuracies = [accuracy_nn, accuracy_lr]
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)  # Set y-axis limit for better visualization
plt.savefig("Model Accuracy Comparison.jpg")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

# Calcula las probabilidades de predicción para ambos modelos
y_pred_prob_lr = log_reg.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_prob_lr)

fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_pred_prob_nn)

# Calcula el AUC para ambos modelos
auc_lr = roc_auc_score(y_test, y_pred_prob_lr)
auc_nn = roc_auc_score(y_test, y_pred_prob_nn)

# Grafica la curva ROC
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.4f})', color='green')
plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {auc_nn:.4f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
