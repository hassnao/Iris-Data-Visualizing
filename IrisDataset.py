#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
    iris['species'],
    test_size=0.2,
    random_state=42
)

# Create a support vector machine classifier
svm = SVC()

# Fit the classifier to the training data
svm.fit(X_train, y_train)

# Predict the species of the test data
y_pred = svm.predict(X_test)

# Print the accuracy and classification report of the classifier on the test data
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Visualize the data and the results
sns.pairplot(iris, hue='species')
plt.show()


# In[ ]:




