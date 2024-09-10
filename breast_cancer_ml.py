#Importing the dependencies
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection and Processing"""

#loading data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

print(breast_cancer_dataset)

#loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

#printing first 5 rows
data_frame.head()

#adding target column to dataframe
data_frame['label'] = breast_cancer_dataset.target

#printing last 5 rows
data_frame.tail()

#number of rows and columns in data set
data_frame.shape

#getting some info about the data
data_frame.info()

#checking for missing values
data_frame.isnull().sum()

#statistical measure about the data
data_frame.describe()

#checking the distribution of target variable
data_frame['label'].value_counts()

"""1 represents Benign
0 represents Malignant
"""

data_frame.groupby('label').mean()

"""Seperating the features and target"""

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

print(X)

print(Y)

"""Splitting the data into training data and testing data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training"""

model = LogisticRegression()

#training logistic regression model using the Training data we obtained
model.fit(X_train, Y_train)

"""Model Evaluation

"""

# accuracy on our training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy on training data is ', training_data_accuracy)

"""Testing data

"""

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy on test data is ', test_data_accuracy)

"""Building a predictive system"""

input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259
)
#changing input data into a numpy array for processing
input_data_as_numpy_array = np.asarray(input_data)

# reshaping the numpy array as we predict for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 1):
  print('Breast Cancer is benign')

else:
  print('Breast Cancer is malignant')
