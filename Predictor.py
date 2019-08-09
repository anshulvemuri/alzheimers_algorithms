from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split


class Predictor:
    def __init__(self, df):
        self.logistic_regression_model = LogisticRegression()
        self.decision_tree_classifier = DecisionTreeClassifier()
        self.random_forest_classifier = RandomForestClassifier()
        self.svm_classifier = SVC(C=1.0, gamma='auto', kernel='rbf', degree=3)
        self.adaboost_classifier = AdaBoostClassifier()
        self.data_frame = df

    def prepare_dataframe(self):
        self.data_frame = self.data_frame.loc[self.data_frame['Visit'] == 1]
        self.data_frame = self.data_frame.loc[self.data_frame['Visit'] == 1].copy()
        # replace M with 0; and F with 1 [converting Male & Female indicators to numeric values]
        self.data_frame['M/F'] = self.data_frame['M/F'].replace(['F', 'M'], [0, 1])
        # Converted actually refers to a person who was Nondemented on an earlier
        # visit and was later diagnosed as Demented
        self.data_frame['Group'] = self.data_frame['Group'].replace(['Converted'], ['Demented'])
        # replace Demented and Nondemented with numeric values
        # Demented = 1; Nondemented = 0
        self.data_frame['Group'] = self.data_frame['Group'].replace(['Demented', 'Nondemented'], [1, 0])
        # Drop the columns MRI ID, Visit and Hand from the dataframe because they are not
        # relevant predictor variables
        self.data_frame = self.data_frame.drop(['MRI ID', 'Visit', 'Hand'], axis=1)

    def fit_models(self):
        # The dependent variable is Y
        Y = self.data_frame['Group'].values
        # List of independent variables or features
        X = self.data_frame[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]
        # Create training and test datasets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
        self.logistic_regression_model.fit(X_train, Y_train)
        self.random_forest_classifier.fit(X_train, Y_train)
        self.adaboost_classifier.fit(X_train, Y_train)
        self.decision_tree_classifier.fit(X_train, Y_train)
        self.svm_classifier.fit(X_train, Y_train)

        print("These are the accuracies of the predictions for each algorithm:")
        print('Logistic accuray score = ', self.logistic_regression_model.score(X_test, Y_test))
        print('SVM accuracy score = ', self.svm_classifier.score(X_test, Y_test))
        print('Random Forest accuracy  = ', self.random_forest_classifier.score(X_test, Y_test))
        print('ADA Boost accuray score = ', self.adaboost_classifier.score(X_test, Y_test))
        print('Decision Tree accuray score = ', self.decision_tree_classifier.score(X_test, Y_test))

    def predict(self, gender_int, input_age, input_EDU, input_SES, input_MMSE, input_CDR, input_TIV, input_WBV,
                input_ATF):
        x_predict = [
            [gender_int, input_age, input_EDU, input_SES, input_MMSE, input_CDR, input_TIV, input_WBV, input_ATF]]
        y_predicted_logistic = self.logistic_regression_model.predict(x_predict)
        y_predicted_dt = self.decision_tree_classifier.predict(x_predict)
        y_predicted_rf = self.random_forest_classifier.predict(x_predict)
        y_predicted_svm = self.svm_classifier.predict(x_predict)
        y_predicted_ada = self.adaboost_classifier.predict(x_predict)
        return y_predicted_logistic, y_predicted_dt, y_predicted_rf, y_predicted_svm, y_predicted_ada
