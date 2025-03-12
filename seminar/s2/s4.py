'''Aplică abilitățile de clasificare pe un set de date real de tip spam vs. non-spam.
Utilizează setul de date Spambase '''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the Spambase dataset
#    Make sure spambase.csv is in your current working directory or provide the full path.
df = pd.read_csv("C:/Users/stud/PycharmProjects/pythonProject/seminar/sampbase.data", header=None)

# The Spambase dataset typically has:
# - 57 numeric columns for features
# - 1 numeric column (the 58th) for the label (1 = spam, 0 = not spam)

# 2. Separate the features (X) and the target label (y)
X = df.iloc[:, :-1]   # All rows, all columns except the last
y = df.iloc[:, -1]    # All rows, only the last column

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Train a Naïve Bayes classifier (MultinomialNB is common for spam detection)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Predict on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = (y_test == y_pred).mean()
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report (Precision, Recall, F1-score)
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)