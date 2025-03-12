import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

'''Încarcă un set de date mic pentru clasificare din scikit-learn (de exemplu, setul de date Iris sau Wine).
Implementează un clasificator Naïve Bayes.'''

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split the dataset into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Naïve Bayes classifier->gaussian, nu ce am facut la seminar
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# 4. Predict on the test set
y_pred = nb_classifier.predict(X_test)

# 5. Print out the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# (Optional) Display the confusion matrix for deeper insight
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)