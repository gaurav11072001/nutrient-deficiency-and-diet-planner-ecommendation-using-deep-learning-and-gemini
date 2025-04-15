import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from train_vitamin_model import VitaminNet

def get_user_input():
    print("\nPlease provide your information:")
    
    # Get personal information
    while True:
        try:
            age = int(input("Age: "))
            if 0 <= age <= 120:
                break
            print("Please enter a valid age between 0 and 120")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        gender = input("Gender (Male/Female): ").strip().capitalize()
        if gender in ['Male', 'Female']:
            break
        print("Please enter either 'Male' or 'Female'")
    
    while True:
        diet_type = input("Diet Type (Vegetarian/Non-Vegetarian): ").strip().capitalize()
        if diet_type in ['Vegetarian', 'Non-Vegetarian']:
            break
        print("Please enter either 'Vegetarian' or 'Non-Vegetarian'")
    
    while True:
        living_env = input("Living Environment (Urban/Rural): ").strip().capitalize()
        if living_env in ['Urban', 'Rural']:
            break
        print("Please enter either 'Urban' or 'Rural'")
    
    valid_skin_conditions = ['Normal', 'Dry Skin', 'Rough Skin', 'Pale/Yellow Skin']
    while True:
        print("\nValid skin conditions:")
        for i, condition in enumerate(valid_skin_conditions, 1):
            print(f"{i}. {condition}")
        try:
            choice = int(input("\nEnter the number of your skin condition (1-4): "))
            if 1 <= choice <= len(valid_skin_conditions):
                skin_condition = valid_skin_conditions[choice - 1]
                break
            print("Please enter a number between 1 and 4")
        except ValueError:
            print("Please enter a valid number")
    
    # List all symptoms
    symptoms = [
        'Night Blindness',
        'Dry Eyes',
        'Bleeding Gums',
        'Fatigue',
        'Tingling Sensation',
        'Low Sun Exposure',
        'Reduced Memory Capacity',
        'Shortness of Breath',
        'Loss of Appetite',
        'Fast Heart Rate',
        'Brittle Nails',
        'Weight Loss',
        'Reduced Wound Healing Capacity',
        'Bone Pain'
    ]
    
    print("\nFor each symptom, enter 1 if you have it, 0 if you don't:")
    symptom_values = {}
    for symptom in symptoms:
        while True:
            try:
                value = int(input(f"{symptom}: "))
                if value in [0, 1]:
                    symptom_values[symptom] = value
                    break
                else:
                    print("Please enter 0 or 1")
            except ValueError:
                print("Please enter 0 or 1")
    
    # Create the data dictionary
    sample_data = {
        'Age': age,
        'Gender': gender,
        'Diet Type': diet_type,
        'Living Environment': living_env,
        'Skin Condition': skin_condition,
        **symptom_values
    }
    
    return sample_data

def load_and_preprocess_data(sample_data):
    # Create LabelEncoder for categorical variables
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Diet Type', 'Living Environment', 'Skin Condition']
    
    # Create symptom count feature
    symptom_columns = ['Night Blindness', 'Dry Eyes', 'Bleeding Gums', 'Fatigue', 'Tingling Sensation', 
                      'Low Sun Exposure', 'Reduced Memory Capacity', 'Shortness of Breath', 'Loss of Appetite', 
                      'Fast Heart Rate', 'Brittle Nails', 'Weight Loss', 'Reduced Wound Healing Capacity', 'Bone Pain']
    sample_data['Symptom_Count'] = sum(sample_data[col] for col in symptom_columns)
    
    # Create interaction features
    sample_data['Vision_Issues'] = sample_data['Night Blindness'] + sample_data['Dry Eyes']
    sample_data['Physical_Weakness'] = sample_data['Fatigue'] + sample_data['Shortness of Breath'] + sample_data['Fast Heart Rate']
    sample_data['Neurological_Signs'] = sample_data['Tingling Sensation'] + sample_data['Reduced Memory Capacity']
    
    skin_mapping = {'Normal': 0, 'Dry Skin': 1, 'Rough Skin': 1, 'Pale/Yellow Skin': 2}
    sample_data['Skin_Health'] = sample_data['Reduced Wound Healing Capacity'] + skin_mapping[sample_data['Skin Condition']]
    
    # Create environmental risk score
    sample_data['Environmental_Risk'] = (
        (sample_data['Living Environment'] == 'Urban') + 
        sample_data['Low Sun Exposure'] + 
        (sample_data['Diet Type'] == 'Vegetarian')
    )
    
    # Encode categorical variables
    for col in categorical_columns:
        # Fit on a list containing all possible values to ensure proper encoding
        possible_values = {
            'Gender': ['Male', 'Female'],
            'Diet Type': ['Vegetarian', 'Non-Vegetarian'],
            'Living Environment': ['Urban', 'Rural'],
            'Skin Condition': ['Normal', 'Dry Skin', 'Rough Skin', 'Pale/Yellow Skin']
        }
        le.fit(possible_values[col])
        sample_data[col] = le.transform([sample_data[col]])[0]
    
    # Create feature vector in the correct order
    features = []
    features.append(sample_data['Age'])
    features.append(sample_data['Gender'])
    features.append(sample_data['Diet Type'])
    features.append(sample_data['Living Environment'])
    features.append(sample_data['Skin Condition'])
    
    # Add all symptom columns in order
    for col in symptom_columns:
        features.append(sample_data[col])
    
    # Add engineered features
    features.append(sample_data['Symptom_Count'])
    features.append(sample_data['Vision_Issues'])
    features.append(sample_data['Physical_Weakness'])
    features.append(sample_data['Neurological_Signs'])
    features.append(sample_data['Skin_Health'])
    features.append(sample_data['Environmental_Risk'])
    
    return pd.Series(features)

def predict_deficiency(model, features):
    # Convert to tensor
    X = torch.FloatTensor(features.values.reshape(1, -1))
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
    
    # Map prediction to deficiency type
    deficiency_mapping = {
        0: 'Iron',
        1: 'No Deficiency',
        2: 'Vitamin A',
        3: 'Vitamin B12',
        4: 'Vitamin C',
        5: 'Vitamin D',
        6: 'Zinc'
    }
    return deficiency_mapping[predicted.item()]

def main():
    print("Welcome to the Vitamin Deficiency Predictor!")
    print("This tool will help predict potential vitamin deficiencies based on your symptoms.")
    print("Please note: This is not a medical diagnosis. Always consult with a healthcare professional.")
    
    # Load the trained model
    input_size = 25
    hidden_sizes = [256, 128, 64, 32]
    num_classes = 7
    
    model = VitaminNet(input_size, hidden_sizes, num_classes)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Get user input
    sample_data = get_user_input()
    
    # Preprocess the data
    features = load_and_preprocess_data(sample_data)
    
    # Make prediction
    prediction = predict_deficiency(model, features)
    
    # Display results
    print("\n=== Results ===")
    print(f"Predicted Deficiency: {prediction}")
    
    print("\nSymptoms Reported:")
    symptom_cols = [col for col in sample_data.keys() if sample_data[col] == 1 and isinstance(sample_data[col], int)]
    for symptom in symptom_cols:
        print(f"- {symptom}")
    
    print("\nReminder: This prediction is based on machine learning and should not be used as a substitute")
    print("for professional medical advice. Please consult with a healthcare provider for proper diagnosis.")

if __name__ == "__main__":
    main() 