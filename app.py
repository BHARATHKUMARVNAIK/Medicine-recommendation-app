
import streamlit as st
import pandas as pd
from sklearn.svm import SVC
import pickle

import pickle

import pickle

# Correct loading of model and dictionaries
svc = pickle.load(open("/Users/bharathkumarvnaik/Downloads/programing/python/Recommendation-system-project/medicen_recommendation_system/dataset-folder/dataset-f/svc.pkl","rb"))
symptoms_dict = pickle.load(open("/Users/bharathkumarvnaik/Downloads/programing/python/Recommendation-system-project/medicen_recommendation_system/main_folder/symptoms_dict.pkl","rb"))
diseases_list = pickle.load(open("/Users/bharathkumarvnaik/Downloads/programing/python/Recommendation-system-project/medicen_recommendation_system/main_folder/diseases_list.pkl","rb"))



print("Type of svc:", type(svc))
print("Type of symptoms_dict:", type(symptoms_dict))



# Load dataframes
description_df = pd.read_csv("/Users/bharathkumarvnaik/Downloads/programing/python/Recommendation-system-project/medicen_recommendation_system/dataset-folder/dataset-f/description.csv")
precautions_df = pd.read_csv('/Users/bharathkumarvnaik/Downloads/programing/python/Recommendation-system-project/medicen_recommendation_system/dataset-folder/dataset-f/precautions_df.csv')
medication_df = pd.read_csv('/Users/bharathkumarvnaik/Downloads/programing/python/Recommendation-system-project/medicen_recommendation_system/dataset-folder/dataset-f/medications.csv')
diets_df = pd.read_csv('/Users/bharathkumarvnaik/Downloads/programing/python/Recommendation-system-project/medicen_recommendation_system/dataset-folder/dataset-f/diets.csv')
workout_df = pd.read_csv('/Users/bharathkumarvnaik/Downloads/programing/python/Recommendation-system-project/medicen_recommendation_system/dataset-folder/dataset-f/workout_df.csv')

# cleaning columns names
for df in [description_df,precautions_df,medication_df,diets_df,workout_df]:
    df.columns = df.columns.str.strip()



def get_predicted_value(symptoms):
    input_vector = [0] * len(symptoms_dict)
    for s in symptoms:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1
    result = svc.predict([input_vector])[0]
    return result if isinstance(result, str) else diseases_list[result]

    '''

def get_predicted_value(patient_systems):
    input_vector = [0] * len(symptoms_dict)
    for item in patient_systems:
        input_vector[symptoms_dict[item]] = 1
    return svc.predict([input_vector])[0]

    '''

def helper(dis):
    desc = description_df[description_df['Disease'] == dis]['Description']
    desc = " ".join(desc.values)

    pre = precautions_df[precautions_df['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.tolist()[0] if not pre.empty else []

    med = medication_df[medication_df['Disease'] == dis]['Medication']
    med = med.tolist()

    die = diets_df[diets_df['Disease'] == dis]['Diet']
    die = die.tolist()

    wrkout = workout_df[workout_df['disease'] == dis]['workout']
    wrkout = wrkout.tolist()

    return desc, pre, med, die, wrkout



st.set_page_config(page_title="Medicine Recommendation App", layout="centered")

st.title("💊 Medicine Recommendation App")
st.markdown("Enter your symptoms (comma-separated) to get a diagnosis and helpful health tips.")


symptoms_input = st.text_input("Enter your symptoms (e.g. headache, fatigue, vomiting):")


if st.button("Predict Disease"):
    if symptoms_input.strip() == "":
        st.warning("Please enter at least one symptom.")
    else:
        user_symptoms = [s.strip() for s in symptoms_input.split(',')]
        predicted_disease = get_predicted_value(user_symptoms)
        
        st.success(f"### 🦠 Predicted Disease: `{predicted_disease}`")

        desc, pre, med, die, wrkout = helper(predicted_disease)

        st.subheader("📝 Description")
        st.write(desc)

        st.subheader("💊 Recommended Medications")
        st.write(", ".join(med) if med else "No data available.")

        st.subheader("⚠️ Precautions")
        for i, p in enumerate(pre, 1):
            st.markdown(f"{i}. {p}")

        st.subheader("🥗 Recommended Diet")
        for i, d in enumerate(die, 1):
            st.markdown(f"{i}. {d}")

        st.subheader("🏋️‍♂️ Suggested Workout")
        for i, w in enumerate(wrkout, 1):
            st.markdown(f"{i}. {w}")






