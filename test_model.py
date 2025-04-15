import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from train_vitamin_model import VitaminNet

def load_and_preprocess_data(sample_data):
    # Create LabelEncoder for categorical variables
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Diet Type', 'Living Environment', 'Skin Condition']
    
    # Create symptom count feature
    symptom_columns = [
        'Night Blindness', 'Dry Eyes', 'Bleeding Gums', 'Fatigue', 
        'Tingling Sensation', 'Low Sun Exposure', 'Memory Loss',
        'Shortness of Breath', 'Loss of Appetite', 'Fast Heart Rate',
        'Muscle Weakness', 'Weight Loss', 'Reduced Wound Healing Capacity',
        'Bone Pain', 'Depression', 'Weakened Immune System', 'Numbness',
        'Sore Throat', 'Cracked Lips', 'Light Sensitivity', 'Itchy Eyes',
        'Headache', 'Diarrhea', 'Confusion', 'Vision Problems',
        'Poor Balance', 'Easy Bruising', 'Heavy Menstrual Bleeding',
        'Blood in Urine', 'Dark Stool'
    ]
    
    sample_data['Symptom_Count'] = sum(sample_data[col] for col in symptom_columns)
    
    # Create interaction features
    sample_data['Vision_Issues'] = sum(sample_data[col] for col in ['Night Blindness', 'Dry Eyes', 'Vision Problems', 'Light Sensitivity', 'Itchy Eyes'])
    sample_data['Physical_Weakness'] = sum(sample_data[col] for col in ['Fatigue', 'Shortness of Breath', 'Fast Heart Rate', 'Muscle Weakness'])
    sample_data['Neurological_Signs'] = sum(sample_data[col] for col in ['Tingling Sensation', 'Memory Loss', 'Confusion', 'Poor Balance', 'Numbness'])
    sample_data['Bleeding_Issues'] = sum(sample_data[col] for col in ['Bleeding Gums', 'Easy Bruising', 'Heavy Menstrual Bleeding', 'Blood in Urine'])
    sample_data['Digestive_Issues'] = sum(sample_data[col] for col in ['Loss of Appetite', 'Diarrhea', 'Dark Stool'])
    
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
    features.append(sample_data['Bleeding_Issues'])
    features.append(sample_data['Digestive_Issues'])
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
        3: 'Vitamin B1 (Thiamine)',
        4: 'Vitamin B12',
        5: 'Vitamin B2 (Riboflavin)',
        6: 'Vitamin B3 (Niacin)',
        7: 'Vitamin B6',
        8: 'Vitamin C',
        9: 'Vitamin D',
        10: 'Vitamin E',
        11: 'Vitamin K',
        12: 'Folate',
        13: 'Zinc'
    }
    return deficiency_mapping[predicted.item()]

def main():
    # Load the trained model with correct architecture
    input_size = 43  # Updated to match the actual number of features from training
    hidden_sizes = [512, 256, 128, 64]  # Updated to match training architecture
    num_classes = 14  # Updated number of deficiency types
    
    model = VitaminNet(input_size, hidden_sizes, num_classes)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Sample test case
    sample_data = {
        'Age': 35,
        'Gender': 'Female',
        'Diet Type': 'Vegetarian',
        'Living Environment': 'Urban',
        'Skin Condition': 'Dry Skin',
        'Night Blindness': 0,
        'Dry Eyes': 0,
        'Bleeding Gums': 0,
        'Fatigue': 1,
        'Tingling Sensation': 0,
        'Low Sun Exposure': 1,
        'Memory Loss': 0,
        'Shortness of Breath': 1,
        'Loss of Appetite': 1,
        'Fast Heart Rate': 0,
        'Muscle Weakness': 1,
        'Weight Loss': 0,
        'Reduced Wound Healing Capacity': 0,
        'Bone Pain': 0,
        'Depression': 0,
        'Weakened Immune System': 0,
        'Numbness': 0,
        'Sore Throat': 0,
        'Cracked Lips': 0,
        'Light Sensitivity': 0,
        'Itchy Eyes': 0,
        'Headache': 0,
        'Diarrhea': 0,
        'Confusion': 0,
        'Vision Problems': 0,
        'Poor Balance': 0,
        'Easy Bruising': 0,
        'Heavy Menstrual Bleeding': 0,
        'Blood in Urine': 0,
        'Dark Stool': 0
    }
    
    # Preprocess the sample data
    features = load_and_preprocess_data(sample_data)
    
    # Make prediction
    prediction = predict_deficiency(model, features)
    print(f"\nPredicted Deficiency: {prediction}")
    
    # Print input symptoms for reference
    print("\nInput Symptoms:")
    symptom_cols = [col for col in sample_data.keys() if sample_data[col] == 1 and isinstance(sample_data[col], int)]
    for symptom in symptom_cols:
        print(f"- {symptom}")

if __name__ == "__main__":
    main() 