import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# load the dataset
data = np.genfromtxt('CBC_data.csv', delimiter=',', skip_header=True)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2)

# impute missing values in X_train and X_test
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# drop samples with missing values in y_train
mask = ~np.isnan(y_train)
X_train = X_train[mask]
y_train = y_train[mask]

# set up a pipeline with an imputer transformer and a logistic regression model
pipe = Pipeline([
    ('model', LogisticRegression(random_state=42))
])

# train the model on the training data
pipe.fit(X_train, y_train)

# make predictions on the test data
y_pred = pipe.predict(X_test)

# calculate the F1 score
f1 = f1_score(y_test, y_pred)

# print the F1 score
print('F1 score:', f1)
