import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_vitamin_deficiency_data(n_samples=1000):
    # Define vitamin deficiencies and their primary symptoms
    vitamin_patterns = {
        'Vitamin A': {
            'primary': ['Night Blindness', 'Dry Eyes'],
            'secondary': ['Reduced Wound Healing Capacity', 'Skin Condition'],
            'skin_conditions': ['Dry Skin', 'Rough Skin'],
            'probability': 0.85
        },
        'Vitamin B1 (Thiamine)': {
            'primary': ['Loss of Appetite', 'Fatigue', 'Muscle Weakness'],
            'secondary': ['Memory Loss', 'Numbness'],
            'skin_conditions': ['Normal'],
            'probability': 0.85
        },
        'Vitamin B2 (Riboflavin)': {
            'primary': ['Sore Throat', 'Cracked Lips', 'Light Sensitivity'],
            'secondary': ['Fatigue', 'Itchy Eyes'],
            'skin_conditions': ['Dry Skin'],
            'probability': 0.85
        },
        'Vitamin B3 (Niacin)': {
            'primary': ['Headache', 'Fatigue', 'Depression'],
            'secondary': ['Memory Loss', 'Diarrhea'],
            'skin_conditions': ['Rough Skin'],
            'probability': 0.85
        },
        'Vitamin B6': {
            'primary': ['Depression', 'Confusion', 'Weakened Immune System'],
            'secondary': ['Fatigue', 'Numbness'],
            'skin_conditions': ['Dry Skin', 'Rough Skin'],
            'probability': 0.85
        },
        'Vitamin B12': {
            'primary': ['Tingling Sensation', 'Fatigue', 'Memory Loss'],
            'secondary': ['Loss of Appetite', 'Shortness of Breath'],
            'skin_conditions': ['Pale/Yellow Skin'],
            'probability': 0.85
        },
        'Vitamin C': {
            'primary': ['Bleeding Gums', 'Reduced Wound Healing Capacity'],
            'secondary': ['Fatigue', 'Weight Loss'],
            'skin_conditions': ['Rough Skin'],
            'probability': 0.85
        },
        'Vitamin D': {
            'primary': ['Low Sun Exposure', 'Fatigue', 'Bone Pain'],
            'secondary': ['Weight Loss', 'Depression'],
            'skin_conditions': ['Normal'],
            'probability': 0.85
        },
        'Vitamin E': {
            'primary': ['Muscle Weakness', 'Vision Problems', 'Weakened Immune System'],
            'secondary': ['Numbness', 'Poor Balance'],
            'skin_conditions': ['Dry Skin'],
            'probability': 0.85
        },
        'Vitamin K': {
            'primary': ['Easy Bruising', 'Bleeding Gums', 'Heavy Menstrual Bleeding'],
            'secondary': ['Blood in Urine', 'Dark Stool'],
            'skin_conditions': ['Normal', 'Pale/Yellow Skin'],
            'probability': 0.85
        },
        'Folate': {
            'primary': ['Fatigue', 'Shortness of Breath', 'Memory Loss'],
            'secondary': ['Depression', 'Muscle Weakness'],
            'skin_conditions': ['Pale/Yellow Skin'],
            'probability': 0.85
        },
        'Iron': {
            'primary': ['Fatigue', 'Shortness of Breath', 'Fast Heart Rate'],
            'secondary': ['Loss of Appetite', 'Memory Loss'],
            'skin_conditions': ['Pale/Yellow Skin'],
            'probability': 0.85
        },
        'Zinc': {
            'primary': ['Reduced Wound Healing Capacity', 'Loss of Appetite'],
            'secondary': ['Weight Loss', 'Weakened Immune System'],
            'skin_conditions': ['Rough Skin', 'Dry Skin'],
            'probability': 0.85
        },
        'No Deficiency': {
            'primary': [],
            'secondary': [],
            'skin_conditions': ['Normal'],
            'probability': 0.85
        }
    }

    # Define all possible symptoms
    all_symptoms = [
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

    data = []
    for _ in range(n_samples):
        # Randomly select deficiency
        deficiency = random.choice(list(vitamin_patterns.keys()))
        pattern = vitamin_patterns[deficiency]
        
        # Initialize symptoms
        symptoms = {symptom: 0 for symptom in all_symptoms}
        
        # Add primary symptoms based on probability
        for symptom in pattern['primary']:
            if symptom in symptoms:
                symptoms[symptom] = 1 if random.random() < pattern['probability'] else 0
        
        # Add secondary symptoms with lower probability
        for symptom in pattern['secondary']:
            if symptom in symptoms:
                symptoms[symptom] = 1 if random.random() < (pattern['probability'] * 0.7) else 0
        
        # Add some random noise
        for symptom in all_symptoms:
            if symptom not in pattern['primary'] + pattern['secondary']:
                symptoms[symptom] = 1 if random.random() < 0.1 else 0

        # Generate demographic and environmental data
        age = random.randint(10, 70)
        gender = random.choice(['Male', 'Female'])
        diet_type = random.choice(['Vegetarian', 'Non-Vegetarian'])
        living_env = random.choice(['Urban', 'Rural'])
        
        # Set skin condition based on deficiency
        skin_condition = random.choice(pattern['skin_conditions'])
        
        # Create the record
        record = {
            'Age': age,
            'Gender': gender,
            'Diet Type': diet_type,
            'Living Environment': living_env,
            'Skin Condition': skin_condition,
            'Predicted Deficiency': deficiency
        }
        record.update(symptoms)
        
        data.append(record)
    
    return pd.DataFrame(data)

# Generate the dataset with more samples to account for more classes
df = generate_vitamin_deficiency_data(2000)

# Save the dataset
df.to_csv('data/symptom_based_vitamin_deficiency_dataset_final.csv', index=False)

print("Dataset generated successfully!")
print("\nSample distribution:")
print(df['Predicted Deficiency'].value_counts())

# Display first few rows
print("\nFirst few rows of the dataset:")
print(df.head()) 