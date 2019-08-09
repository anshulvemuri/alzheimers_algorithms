import pandas as pd
from Predictor import Predictor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def start_program():
    input_filename = "C:/Users/aNsHuL's pC/CSIRE Stony Brook 2019 Summer/oasis_longitudinal.csv"
    df = pd.read_csv(input_filename)
    predictor_object = Predictor(df)
    predictor_object.prepare_dataframe()
    predictor_object.fit_models()

    print("Please enter your information to start the program")
    gender = input("What is your gender(M or F): ")
    if gender == "M":
        gender_num = 1
        gender = gender_num
    elif gender == "F":
        gender_num = 0
        gender = gender_num

    while True:
        try:
            age = int(input("Please enter age (whole numbers until 126): "))
        except ValueError:
            print('Please enter only whole numbers for age')
        else:
            if 0 < age < 126:
                break

    while True:
        try:
            education = int(input("Enter number of years of education (whole numbers until 24): "))
        except ValueError:
            print('Please enter only whole numbers for education')
        else:
            if 0 < education < 25:
                break

    while True:
        try:
            ses = int(input("Enter socio economic status (1 - 5): "))
        except ValueError:
            print('Please enter only whole numbers between 1 and 5')
        else:
            if 0 < ses < 6:
                break

    while True:
        try:
            mmse = int(input("Enter mini mental status examination score(0 - 30): "))
        except ValueError:
            print('Please enter only whole numbers between 1 and 30')
        else:
            if 0 <= mmse <= 30:
                break

    while True:
        try:
            cdr = int(input("Enter clinical dementia rating(0 - 3): "))
        except ValueError:
            print('Please enter only whole numbers between 0 and 3')
        else:
            if 0 <= cdr <= 3:
                break

    while True:
        try:
            etiv = int(input("Enter total intracranial volume(1100 - 2010): "))
        except ValueError:
            print('Please enter only whole numbers between 1100 and 2010')
        else:
            if 1100 <= etiv <= 2010:
                break

    while True:
        try:
            wbv = float(input("Enter normalized whole brain volume(0.6 - 0.9): "))
        except ValueError:
            print('Please enter values between 0.6 and 0.9')
        else:
            if 0.6 <= wbv <= 0.9:
                break

    while True:
        try:
            asf = float(input("Enter normalized atlas scaling factor: (0.8 - 1.7): "))
        except ValueError:
            print('Please enter values between 0.8 and 1.6')
        else:
            if 0.8 <= asf <= 1.7:
                break

    y_predicted_logistic, y_predicted_dt, y_predicted_rf, y_predicted_svm, y_predicted_ada = predictor_object.predict(gender, age, education, ses, mmse, cdr, etiv, wbv, asf)
    if y_predicted_logistic[0] == 0:
        print("Based on the LOGISTIC REGRESSION, it is UNLIKELY that you will be diagnosed with Alzheimer's")
    elif y_predicted_logistic[0] == 1:
        print("Based on the LOGISTIC REGRESSION, it is LIKELY that you will be diagnosed with Alzheimer's")
    if y_predicted_svm[0] == 0:
        print("Based on the SUPPORT VECTOR MACHINE, it is UNLIKELY that you will be diagnosed with Alzheimer's")
    elif y_predicted_svm[0] == 1:
        print("Based on the SUPPORT VECTOR MACHINE, it is LIKELY that you will be diagnosed with Alzheimer's")
    if y_predicted_dt[0] == 0:
        print("Based on the DECISION TREE CLASSIFIER, it is UNLIKELY that you will be diagnosed with Alzheimer's")
    elif y_predicted_dt[0] == 1:
        print("Based on the DECISION TREE CLASSIFIER, it is LIKELY that you will be diagnosed with Alzheimer's")
    if y_predicted_rf[0] == 0:
        print("Based on the RANDOM FOREST CLASSIFIER, it is UNLIKELY that you will be diagnosed with Alzheimer's")
    elif y_predicted_rf[0] == 1:
        print("Based on the RANDOM FOREST CLASSIFIER, it is LIKELY that you will be diagnosed with Alzheimer's")
    if y_predicted_ada[0] == 0:
        print("Based on the ADAPTIVE BOOSTING, it is UNLIKELY that you will be diagnosed with Alzheimer's")
    elif y_predicted_ada[0] == 1:
        print("Based on the ADAPTIVE BOOSTING, it is LIKELY that you will be diagnosed with Alzheimer's")


start_program()
