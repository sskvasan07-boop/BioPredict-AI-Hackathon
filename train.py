import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Expanded and refined symptoms list
SYMPTOMS_LIST = [
    "skin rash", "itching", "joint pain", "peeling skin", "silver dusting", 
    "nail dents", "prominent veins", "leg swelling", "bruising", "obesity", 
    "chills", "fever", "belly pain", "diarrhea", "vomiting", "bloating", 
    "headache", "weight loss", "nausea", "breathing difficulty", "fatigue", 
    "muscle pain", "cough", "phlegm", "chest pain", "sore throat", 
    "runny nose", "sneezing", "high fever", "sweating", "fast heartbeat", 
    "blurred vision", "excessive hunger", "excessive thirst", "frequent urination", 
    "neck stiffness", "dizziness", "loss of appetite", "red spots", 
    "blistering sores", "yellow ooze", "constipation", "anal pain", 
    "bloody stool", "eye pain", "sinus pressure", "heartburn", "acid reflux", 
    "stomach ulcer", "back pain", "halo around lights", "loss of side vision", 
    "eye redness", "severe headache", "nausea and vomiting", "pain behind the eyes",
    "scaly patches", "swollen blood vessels", "cramps in calves", "fatigued soon",
    "bruise marks", "overweight", "visible blood vessels", "abdominal pain",
    "constipation", "headache", "watery stools", "bloating", "weakness",
    "soreness around anus", "bloody stools", "pain during bowel movements",
    "stiff neck", "swollen joints", "muscle weakness", "pus-filled pimples",
    "blackheads", "scurring", "persistant cough", "shallow breathing",
    "itchy eyes", "runny nose", "sneezing", "sore throat", "abdominal cramps",
    "runny or stuffy nose", "watery eyes", "persistent cough", "body aches",
    "loss of taste", "loss of smell", "chest tightness", "wheezing"
]

def preprocess_text(text):
    text = text.lower()
    symptom_vector = []
    # Weighted Symptom Logic: Assign higher weight to critical symptoms
    CRITICAL_SYMPTOMS = ["breathing difficulty", "high fever", "chest pain", "severe headache"]
    
    for symptom in SYMPTOMS_LIST:
        pattern = r'\b' + re.escape(symptom).replace(' ', r'\s+') + r's?\b'
        val = 1 if re.search(pattern, text) else 0
        
        # Apply weighting if critical
        if val == 1 and symptom in CRITICAL_SYMPTOMS:
            val = 2 # Double weight for classification importance
        symptom_vector.append(val)
    return symptom_vector

def train_model():
    print("Loading dataset...")
    df = pd.read_csv('Symptom2Disease.csv')
    
    # Dataset Expansion: Synthesize common illness data
    common_illnesses = [
        {"label": "Common Cold", "text": "I have been sneezing and have a very sore throat. My nose is runny and I feel a bit tired. I have a stuffy nose and body aches."},
        {"label": "Common Cold", "text": "Runny nose and sneezing for two days. My throat is scratchy and sore. I am coughing a lot."},
        {"label": "Influenza", "text": "I have a very high fever and chills. My muscles ache all over and I am extremely fatigued. Severe body aches and dry cough."},
        {"label": "Influenza", "text": "Suddenly came down with a high fever and muscle aches. Feeling very weak and have chills. My chest feels tight."},
        {"label": "Gastroenteritis", "text": "I have severe diarrhea and I have been vomiting. My stomach has painful abdominal cramps and I am dehydrated."},
        {"label": "Gastroenteritis", "text": "Vomiting and watery diarrhea. I have bloating and sharp abdominal cramps. Nausea is constant."},
        {"label": "Allergy", "text": "My eyes are very itchy and I can't stop sneezing. I also noticed a slight skin rash and watery eyes."},
        {"label": "Allergy", "text": "Itchy eyes and runny nose after going outside. I also have some itching on my skin and hives."},
        {"label": "Glaucoma", "text": "I have been experiencing a severe headache and blurred vision. I can see halos around lights and my eye is red and painful."},
        {"label": "Bronchial Asthma", "text": "I am wheezing and have breathing difficulty. My chest feels tight and I have a persistent cough."},
        {"label": "Bronchial Asthma", "text": "Shortness of breath and wheezing sounds. I feel like I can't get enough air."}
    ] * 30 # Generate 330+ synthetic rows for common cases
    
    df_synthetic = pd.DataFrame(common_illnesses)
    df = pd.concat([df, df_synthetic], ignore_index=True)
    
    print("Preprocessing descriptions...")
    X = np.array([preprocess_text(text) for text in df['text']])
    y = df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    print("Saving files to /models...")
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    with open('models/symptoms_list.pkl', 'wb') as f:
        pickle.dump(SYMPTOMS_LIST, f)
        
    with open('models/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train_model()
