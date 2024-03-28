from tkinter import *
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

Symptoms = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
            'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
            'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
            'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
            'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
            'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
            'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
            'swollen_extremities', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
            'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
            'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
            'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
            'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
            'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
            'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
            'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
            'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
            'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
            'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
            'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
            'red_sore_around_nose', 'yellow_crust_ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           ' Migraine', 'Cervical spondylosis',
           'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
           'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
           'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']

Symptoms2 = []
for i in range(0, len(Symptoms)):
    Symptoms2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
tr = pd.read_csv("Training.csv")

tr.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37,
                          'Urinary tract infection': 38, 'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)

a = tr[Symptoms]

b = tr[["prognosis"]]
np.ravel(b)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
te = pd.read_csv("Testing.csv")
te.replace({'prognosis': {'Fungal infection': 0,'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3,
                          'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37,
                          'Urinary tract infection': 38, 'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)

a_test = te[Symptoms]
b_test = te[["prognosis"]]
# np.ravel(b_test)


class B_Naive_bayes:
    def __init__(self, symptoms):
        self.symptoms = symptoms
        self.class_probs = None
        self.feature_probs = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes)

        self.class_probs = class_counts / n_samples
        self.feature_probs = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes):
            class_mask = (y == c)
            class_samples = X[class_mask]

            # Estimate feature probabilities using Laplace smoothing
            self.feature_probs[i, :] = (np.sum(class_samples, axis=0) + 1) / (np.sum(class_mask) + 2)

    def predict_proba(self, X):
        class_probs_log = np.log(self.class_probs)
        feature_probs_log = np.log(self.feature_probs)

        # Convert the input list X to a NumPy array
        X_array = np.array(X)

        # Ensure X_array has the same number of features as feature_probs_log
        if X_array.shape[1] < feature_probs_log.shape[1]:
            X_array = np.pad(X_array, ((0, 0), (0, feature_probs_log.shape[1] - X_array.shape[1])))

        # Use broadcasting to compute log likelihoods for each sample and class
        log_likelihoods = X_array @ feature_probs_log.T + (1 - X_array) @ (1 - feature_probs_log.T)

        # posterior probabilities
        log_probs = log_likelihoods + class_probs_log

        # Convert log probabilities back to probabilities
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

def Naive_Bayes():
    t3.delete("0.0", END)
    if ((S1.get() == "-----" and S2.get() == "-----") or (S3.get() == "-----" and S4.get() == "-----")):
        r = Tk()
        aa = Label(r, text="PLEASE ENTER AS PER THE INSTRUCTIONS", fg="red", bg="black", font=("comicsansms", 13),
                   borderwidth=3, relief=SUNKEN)
        aa.grid(row=1, column=1)
        r.mainloop()
    if (Name.get() == ""):
        qw = Tk()
        aqwe = Label(qw, text="PLEASE ENTER PATIENT'S NAME ", fg="red", bg="black", font=("comicsansms", 13),
                     borderwidth=3, relief=SUNKEN)
        aqwe.grid(row=1, column=1)
        qw.mainloop()
    else:
        # Create a new instance of the classifier for each prediction
        nb = B_Naive_bayes(symptoms=Symptoms)
        nb.fit(tr[Symptoms], np.ravel(b))

        # Reset Symptoms2 for each prediction
        Symptoms2 = [0] * len(Symptoms)

        presentsymp = [S1.get(), S2.get(), S3.get(), S4.get()]
        for i in range(0, len(Symptoms)):
            for j in presentsymp:
                if (j == Symptoms[i]):
                    Symptoms2[i] = 1

        iptest = [Symptoms2]
        predict = nb.predict(iptest)
        predicted = predict[0]

        probability = nb.predict_proba(iptest)
        max_prob_index = np.argmax(probability)
        max_prob = probability[0][max_prob_index]

        # Accuracy calculation
        y_true = np.ravel(b_test)
        y_pred = nb.predict(te[Symptoms])
        accuracy = accuracy_score(y_true, y_pred)

        # Display accuracy
        accuracy_text = f"Accuracy of our model : {accuracy:.2%}"
        t_accuracy = Text(root, height=1, width=40, bg="orange", fg="black")
        t_accuracy.grid(row=21, column=1, padx=10)
        t_accuracy.insert(END, accuracy_text)

        S = 0
        for a in range(0, len(disease)):
            if (predicted == a):
                S = 1
                break

        if (S == 1):
            disease_name = disease[a]
            probability_text = f"Probability of Disease : {max_prob:.2%}"

            # Display disease name in one Text widget
            t_disease = Text(root, height=1, width=40, bg="orange", fg="black")
            t_disease.grid(row=19, column=1, padx=10)
            t_disease.insert(END, f"Disease: {disease_name}")

            # Display probability in another Text widget
            t_prob = Text(root, height=1, width=40, bg="orange", fg="black")
            t_prob.grid(row=20, column=1, padx=10)
            t_prob.insert(END, probability_text)
        else:
            t3.insert(END, "Not Found")



root = Tk()
root.configure(background='lightgrey')
root.geometry("800x600")
root.minsize(200, 100)
root.maxsize(2000, 2000)

# entry variables
S1 = StringVar()
S1.set("-----")
S2 = StringVar()
S2.set("-----")
S3 = StringVar()
S3.set("-----")
S4 = StringVar()
S4.set("-----")
Name = StringVar()

# Heading
x = Label(root, justify=LEFT, text="------------------------DISEASE DIAGNOSIS APP--------------------------", fg="white",bg="darkblue", font=("comicsansms", 20))
x.grid(row=1, column=0, columnspan=4, pady=20)  # Adjusted columnspan

#----------------------------------------
NameP = Label(root, text="Name of the Patient", fg="blue", bg="lightgrey", borderwidth=3, relief=SUNKEN)
NameP.grid(row=6, column=0, pady=15, sticky=W)
#------------------------------------------
S1L = Label(root, text="Symptom 1", fg="orange",borderwidth=3,relief=SUNKEN)    #l=label
S1L.grid(row=7, column=0, pady=20, padx=20,sticky=W)

S2L = Label(root, text="Symptom 2", fg="orange",borderwidth=3,relief=SUNKEN)
S2L.grid(row=8, column=0, pady=20,padx=20, sticky=W)

S3L = Label(root, text="Symptom 3", fg="orange",borderwidth=3,relief=SUNKEN)
S3L.grid(row=9, column=0, pady=20,padx=20, sticky=W)

S4L = Label(root, text="Symptom 4", fg="orange",borderwidth=3,relief=SUNKEN)
S4L.grid(row=10, column=0, pady=20, padx=20,sticky=W)

Naive = Label(root, text="RESULT", fg="white", bg="red",borderwidth=3,relief=RAISED)
Naive.grid(row=19, column=0, pady=20, padx=20 ,sticky=W)

SymptomsList = sorted(Symptoms)

inst=Label(root,text="PLEASE   ENTER   ATLEAST   3   SYMPTOMS",fg="red",borderwidth=3,relief=RIDGE)
inst.grid(row=5, column=1,pady=10)

Namet = Entry(root, textvariable=Name)             #t=textfield
Namet.grid(row=6, column=1)

S1t = OptionMenu(root, S1,*SymptomsList)
S1t.grid(row=7, column=1)

S2t = OptionMenu(root, S2,*SymptomsList)
S2t.grid(row=8, column=1)

S3t = OptionMenu(root, S3,*SymptomsList)
S3t.grid(row=9, column=1)

S4t = OptionMenu(root, S4,*SymptomsList)
S4t.grid(row=10, column=1)
#-----------------------------
N = Button(root, text="DISEASE PREDICTION", command=Naive_Bayes, bg="green", fg="white", activeforeground="red")
N.grid(row=10, column=3, padx=10)
#----------------------------------------
#textfileds------------------
t3 = Text(root, height=1, width=40, bg="orange", fg="black")
t3.grid(row=19, column=1, padx=10, columnspan=2)  # Adjusted columnspan

root.mainloop()


