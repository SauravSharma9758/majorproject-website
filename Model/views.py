from django.shortcuts import render
from .forms import DiseasePredictionForm
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create your views here.
def home(request):
    return render(request,"home.html")

def about(request):
    return render(request,'about.html')

def predict_disease(request):
    predicted_disease = None
    symptoms_fetched = []
    if request.method == 'POST':
        form = DiseasePredictionForm(request.POST)
        if form.is_valid():
            data = pd.read_csv('C:/Users/z004revf/OneDrive - Siemens Healthineers/Desktop/B.E. Project/Disease Prediction/PredictionModel/PredictionModel/Training.csv')
            symptoms_input = form.cleaned_data['symptoms'].strip().split(',')
            
            symptoms_fetched = data.iloc[0, :].tolist()
            # print(symptoms_fetched)  # Print the fetched symptoms
            
            # Extract the features and target variable
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

            # Define the classifiers
            gnb = GaussianNB()
            dtc = DecisionTreeClassifier()
            rfc = RandomForestClassifier(n_estimators=100)
            svm = SVC(probability=True)
            knn = KNeighborsClassifier()
            lr = LogisticRegression()
            ann = MLPClassifier()
            bnb = BernoulliNB()
            etc = ExtraTreeClassifier()
            
            # Train the classifiers
            gnb.fit(X_train, y_train)
            dtc.fit(X_train, y_train)
            rfc.fit(X_train, y_train)
            svm.fit(X_train, y_train)
            knn.fit(X_train, y_train)
            lr.fit(X_train, y_train)
            ann.fit(X_train, y_train)
            bnb.fit(X_train, y_train)
            etc.fit(X_train, y_train)

            # Transform the user input into feature vector
            user_input = []
            for i in range(len(data.columns)-1):
                if data.columns[i] in symptoms_input:
                    user_input.append(1)
                else:
                    user_input.append(0)

            # Make predictions using the trained classifiers
            gnb_pred = gnb.predict([user_input])
            dtc_pred = dtc.predict([user_input])
            rfc_pred = rfc.predict([user_input])
            svm_pred = svm.predict([user_input])
            knn_pred = knn.predict([user_input])
            lr_pred = lr.predict([user_input])
            ann_pred = ann.predict([user_input])
            bnb_pred = bnb.predict([user_input])
            etc_pred = etc.predict([user_input])

            # Append predictions to a list
            predictions = [gnb_pred[0], dtc_pred[0], rfc_pred[0], svm_pred[0], knn_pred[0], lr_pred[0], ann_pred[0], bnb_pred[0], etc_pred[0]]

            # Find the disease with the highest number of occurrences
            disease_counts = {}
            for prediction in predictions:
                if prediction in disease_counts:
                    disease_counts[prediction] += 1
                else:
                    disease_counts[prediction] = 1

            max_occurrence = max(disease_counts.values())
            predicted_disease = [disease for disease, count in disease_counts.items() if count == max_occurrence]
            print(symptoms_fetched,'sdsd')

            # Print the predicted disease
            if predicted_disease:
                return render(request, "predict.html", {
                    "predicted_disease": predicted_disease[0],
                    "symptoms": symptoms_input
                })
            else:
                return render(request, "predict.html", {
                    "error_message": "No disease prediction found.",
                    "symptoms": symptoms_input
                })

    else:
        form = DiseasePredictionForm()

    return render(request, "predict.html", {"form": form,'predicted_disease':predicted_disease,'symptoms_fetched':symptoms_fetched})
