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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_filename = "C:/Users/aNsHuL's pC/CSIRE Stony Brook 2019 Summer/oasis_longitudinal.csv"
df = pd.read_csv(input_filename)

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

# The dependent variable is Y
Y = df['Group'].values
# List of independent variables or features
X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]

# Create training and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

SelectedBoostModel = AdaBoostClassifier().fit(X_train, Y_train)

PredictedOutput = SelectedBoostModel.predict(X_test)
print("--------------------------------------------------------------------------------------------------")

gender_int = 1
input_age = 76
input_EDUC = 12
input_SES = 2
input_MMSE = 29
input_CDR = 0.5
input_eTIV = 1200
input_nWBV = 0.736
input_ASF = 1.146

x_predict = [[gender_int, input_age, input_EDUC, input_SES, input_MMSE, input_CDR, input_eTIV, input_nWBV, input_ASF]]

y_predicted = SelectedBoostModel.predict(x_predict)
if y_predicted[0] == 0:
    print("It is UNLIKELY that you will be diagnosed with Alzheimer's")
elif y_predicted[0] == 1:
    print("It is LIKELY that you will be diagnosed with Alzheimer's")
