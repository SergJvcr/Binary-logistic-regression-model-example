# Standard operational package imports
import numpy as np
import pandas as pd
# Important imports for preprocessing, modeling, and evaluation
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
# Visualization package imports
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df_original = pd.read_csv('google_data_analitics\\Invistico_Airline.csv')
print(df_original.head(10))

# Data exploration, data cleaning, and model preparation
print(df_original.dtypes)
print(df_original.info())
print(df_original.describe(include='all'))
print(df_original.shape)

# Check the number of satisfied customers in the dataset
print(df_original['satisfaction'].value_counts(dropna=False))
# 54.7 percent (71,087/129,880) of customers were satisfied. 
# While this is a simple calculation, this value can be compared to a logistic regression model's accuracy.

# Check for missing values
print(df_original.isna().sum())

# Drop the rows with missing values
df_subset = df_original.dropna(axis=0).reset_index(drop=True)
print(df_subset.shape)

# Prepare the data
# If you want to create a plot (sns.regplot) of your model to visualize results later, 
# the independent variable Inflight entertainment cannot be "of type int" and 
# the dependent variable satisfaction cannot be "of type object"
df_subset = df_subset.astype({'Inflight entertainment' : float}) # to change datatype
print(df_subset.dtypes)

# Convert the categorical column 'satisfaction' into numeric through one-hot encoding
df_subset['satisfaction'] = OneHotEncoder(drop='first').fit_transform(df_subset[['satisfaction']]).toarray()
print(df_subset.head(10))

# Create the training and testing data
X = df_subset[['Inflight entertainment']]
y = df_subset['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
print(f'The size of the X_train is {X_train.shape[0]} rows, {round(X_train.shape[0] / df_subset.shape[0] * 100, 2)}%')
print(f'The size of the X_test is {X_test.shape[0]} rows, {round(X_test.shape[0] / df_subset.shape[0] * 100, 2)}%')
print(f'The size of the y_train is {y_train.shape[0]} rows, {round(y_train.shape[0] / df_subset.shape[0] * 100, 2)}%')
print(f'The size of the y_train is {y_test.shape[0]} rows, {round(y_test.shape[0] / df_subset.shape[0] * 100, 2)}%')

# Model building
# Fit a LogisticRegression model to the data
classifier = LogisticRegression().fit(X_train, y_train)

# Obtain parameter estimates
print(f'Coefficient (beta 1) is {round(classifier.coef_[0][0], 5)}')
print(f'Intercept (beta 0) is {round(classifier.intercept_[0] ,5)}')

# Create a plot of a model
plt.title('The model: Inflight entertainment VS Satisfaction')
sns.regplot(x=df_subset['Inflight entertainment'], y=df_subset['satisfaction'], 
            color='gray', logistic=True, ci=None, line_kws=dict(color="r"))
plt.ylabel('Satisfaction')
plt.show()

# Results and evaluation
# Save predictions
y_pred = classifier.predict(X_test)  # testing our model with the testing subset of the data
print(y_pred)
# Use the predict_proba and predict functions on X_test
# Use predict_proba to output a probability
print(classifier.predict_proba(X_test)) # to show testing
# Use predict to output 0's and 1's
print(classifier.predict(X_test)) # to show testing

# Analyze the results
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred))

# Produce a confusion matrix
# Calculate the values for each quadrant in the confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred, labels = classifier.classes_)
# Create the confusion matrix as a visualization
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classifier.classes_)
disp.plot()
plt.title('Confusion matrix')
plt.show()
# Two of the quadrants are under 4,000, which are relatively low numbers. 
# Based on what we know from the data and interpreting the matrix, 
# it's clear that these numbers relate to false positives and false negatives.
# Additionally, the other two quadrants—the true positives and true negatives—are both high numbers above 13,000.
# There isn't a large difference in the number of false positives and false negatives.
# Using more than a single independent variable in the model training process could improve model performance. 
# This is because other variables, like Departure Delay in Minutes, 
# seem like they could potentially influence customer satisfaction.

# Logistic regression accurately predicted satisfaction 80.2 percent of the time.
# The confusion matrix is useful, as it displays a similar amount of true positives and true negatives.
# Customers who rated in-flight entertainment highly were more likely to be satisfied. 
# Improving in-flight entertainment should lead to better customer satisfaction.
# The model is 80.2 percent accurate. This is an improvement over the dataset's customer satisfaction rate of 54.7 percent.
# The success of the model suggests that the airline should invest more in model developement 
# to examine if adding more independent variables leads to better results. 
# Building this model could not only be useful in predicting whether or not a customer 
# would be satisfied but also lead to a better understanding of what independent variables lead to happier customers.

# ROC curve
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_pred, plot_chance_level=True, color='orange')
plt.title('ROC curve: \n Inflight entertainment VS Satisfaction')
plt.show()

print(f'The AUC value is {metrics.roc_auc_score(y_test, y_pred)}')