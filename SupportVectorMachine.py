import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_filename = "C:/Users/aNsHuL's pC/CSIRE Stony Brook 2019 Summer/oasis_longitudinal.csv"
df = pd.read_csv(input_filename)

print(df.head(373))

df = df.loc[df['Visit'] == 1]
# replace M with 0; and F with 1 [converting Male & Female indicators to numeric values]
df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])
# Converted actually refers to a person who was Nondemented on an earlier visit and was later diagnosed as Demented
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
# replace Demented and Nondemented with numeric values
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])
# Drop the columns MRI ID, Visit and Hand from the dataframe because they are not relevant predictor variables
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)  # Drop unnecessary columns

# The SES column had a few NaN values, then I found the median and substituted the NaN values in the dataset
number_SES_null = pd.isnull(df['SES']).sum()
df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)
df.groupby(['EDUC'])['SES'].median()
number_SES_null = pd.isnull(df['SES']).sum()

df.groupby(['EDUC'])['SES'].median()
df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)
values = pd.isnull(df['SES']).value_counts()
print(values)

# The dependent variable is Y
Y = df['Group'].values
# List of independent variables or features
X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]

# Create training and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# Feature scaling
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

acc = []
best_score = 0
kfolds = 5
for c in [0.001, 0.1, 1, 10, 100]:
    LogRegModel = LogisticRegression(C=c)
    scores = cross_val_score(LogRegModel, X_train, Y_train, cv=kfolds, scoring='accuracy')
    score = np.mean(scores)
    if score > best_score:
        best_score = score
        best_parameters = c
SelectedLogRegModel = LogisticRegression(C=best_parameters).fit(X_train_scaled, Y_train)
test_score = SelectedLogRegModel.score(X_test_scaled, Y_test)
PredictedOutput = SelectedLogRegModel.predict(X_test_scaled)
test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)

best_score = 0
for c_parameter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    for gamma_parameter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for k_parameter in ['rbf', 'linear', 'poly', 'sigmoid']:
            svmModel = SVC(kernel=k_parameter, C=c_parameter, gamma=gamma_parameter)
            scores = cross_val_score(svmModel, X_train_scaled, Y_train, cv=kfolds, scoring='accuracy')
            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_parameter_c = c_parameter
                best_parameter_gamma = gamma_parameter
                best_parameter_k = k_parameter
SelectedSVModel = SVC(C=best_parameter_c, gamma=best_parameter_gamma, kernel=best_parameter_k).fit(X_train_scaled,
                                                                                                   Y_train)
test_score = SelectedSVModel.score(X_test_scaled, Y_test)
PredictedOutput = SelectedSVModel.predict(X_test_scaled)
test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)
print("Best accuracy on cross validation set is:", best_score)
print("Best parameter for c is:", best_parameter_c)
print("Best parameter for gamma is:", best_parameter_gamma)
print("Best parameter for kernel is:", best_parameter_k)
print("Test accuracy with the best parameters is:", test_score)
print("Test recall with the best parameters is:", test_recall)
print("Test recall with the best parameter is:", test_auc)
m = 'SVM'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])
