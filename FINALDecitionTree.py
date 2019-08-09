import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_filename = "C:/Users/aNsHuL's pC/CSIRE Stony Brook 2019 Summer/oasis_longitudinal.csv"
data_frame = pd.read_csv(input_filename)

data_frame = data_frame.loc[data_frame['Visit'] == 1]
# replace M with 0; and F with 1 [converting Male & Female indicators to numeric values]
data_frame['M/F'] = data_frame['M/F'].replace(['F', 'M'], [0, 1])
# Converted actually refers to a person who was Nondemented on an earlier visit and was later diagnosed as Demented
data_frame['Group'] = data_frame['Group'].replace(['Converted'], ['Demented'])
# replace Demented and Nondemented with numeric values
# Demented = 1; Nondemented = 0
data_frame['Group'] = data_frame['Group'].replace(['Demented', 'Nondemented'], [1, 0])
# Drop the columns MRI ID, Visit and Hand from the dataframe because they are not relevant predictor variables
data_frame = data_frame.drop(['MRI ID', 'Visit', 'Hand'], axis=1)  # Drop unnecessary columns

# The dependent variable is Y
Y = data_frame['Group'].values
# List of independent variables or features
X = data_frame[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]

# Create training and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# Create a DecisionTreeModel and fit it with training dataset
SelectedDecisionTreeModel = DecisionTreeClassifier().fit(X_train, Y_train)

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

y_predicted = SelectedDecisionTreeModel.predict(x_predict)
if y_predicted[0] == 0:
    print("It is UNLIKELY that you will be diagnosed with Alzheimer's")
elif y_predicted[0] == 1:
    print("It is LIKELY that you will be diagnosed with Alzheimer's")
