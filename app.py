from flask import Flask , render_template , request
app = Flask(__name__)

@app.route("/")
def index():
    symptom_items=[]
    for i in range(1,6):
        symptom_item =str(request.args.get("symptom"+str(i)))
        symptom_items.append(symptom_item)
        symptoms = ",".join(s for s in symptom_items)
    if request.args:
        disease = predictDisease(symptoms)
    else:
        disease = " "
    # return ("""<form action="" method="get">
    #             Symptoms : <input type="text" name="symptoms">
    #             <input type="submit" value="Submit">
    #           </form>"""
    #           + "Predicted Disease: "
    #     + str(disease))
    return render_template('index.html', len = len(symptom_values), symptom_values = sorted(symptom_values), predicted_disease=disease)

# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("datasets/Training.csv").dropna(axis = 1)


encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:,:-1] #getting all the rows and columns except last column that contains the disease
y = data.iloc[:, -1] #getting all the rows of the last column of diseases
X_train, X_test, y_train, y_test =train_test_split( X, y, test_size = 0.2, random_state = 24)

# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Reading the test data
test_data = pd.read_csv("datasets/Testing.csv").dropna(axis=1)
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
symptomslist = X.columns.values
# Creating a symptom index dictionary to encode the input symptoms into numerical form
symptom_index = {}
symptom_values = []
for index, value in enumerate(symptomslist):
    symptom = value.replace("_"," ").title()  
    symptom_values.append(symptom)

    # symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}
# Defining the Function
# Input: string containing symptoms separated by commmas
# Output: Generated predictions by models

# @app.route("/<symptoms>")
def predictDisease(symptoms):

        symptoms = symptoms.split(",")
        
        # creating input data for the models
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            if symptom in data_dict["symptom_index"]: 
                index = data_dict["symptom_index"][symptom]
                input_data[index] = 1
            
        # reshaping the input data and converting it into suitable format for model predictions
        input_data = np.array(input_data).reshape(1,-1)
        # generating individual outputs
        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
        # making final prediction by taking mode of all predictions
        final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
        
        # predictions = {
        #     "rf_model_prediction": rf_prediction,
        #     "naive_bayes_prediction": nb_prediction,
        #     "svm_model_prediction": nb_prediction,
        #     "final_prediction":final_prediction
        # }
        return final_prediction
   

# Testing the function
# print(predictDisease("Bloody Stool,Swelled Lymph Nodes,History Of Alcohol Consumption,Yellow Crust Ooze"))

if __name__ == "__main__":
    app.run(use_reloader = True,debug=True)