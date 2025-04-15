# Vitamin Deficiency Predictor

A machine learning-powered web application that predicts potential vitamin deficiencies based on user symptoms and provides personalized diet recommendations.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Model Training Process](#model-training-process)
- [Installation](#installation)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Directory Structure](#directory-structure)

## Project Overview

This project combines machine learning and web technologies to create an intelligent system that:
1. Predicts vitamin deficiencies based on user symptoms
2. Provides personalized diet recommendations
3. Visualizes health analytics and trends
4. Offers a user-friendly interface for symptom reporting

## Features

### 1. User Authentication
- Secure login and registration system
- Session management
- Password hashing for security

### 2. Interactive Symptom Collection
- Multi-step form interface
- 30+ different symptoms categorized into:
  - Vision & Physical
  - Neurological
  - Blood & Digestion
  - Other Symptoms
- Real-time form validation

### 3. ML-Powered Prediction
- Neural network model for deficiency prediction
- Support for 14 different deficiency types
- Probability scores for multiple potential deficiencies

### 4. Personalized Diet Recommendations
- Integration with Google's Gemini AI
- Custom diet plans based on:
  - Predicted deficiency
  - User's diet type
  - Reported symptoms

### 5. Analytics Dashboard
- Interactive charts and visualizations
- Deficiency distribution analysis
- Age group correlations
- Symptom correlation patterns
- Monthly trend analysis

## Technical Architecture

### Frontend
- HTML5, CSS3, JavaScript
- Bootstrap 5 for responsive design
- Chart.js for data visualization
- Marked.js for markdown rendering

### Backend
- Flask web framework
- SQLite database for user management
- PyTorch for ML model deployment
- Google Gemini API for diet recommendations

### Machine Learning Model
- PyTorch neural network (VitaminNet)
- Multi-layer architecture with:
  - Batch normalization
  - Dropout for regularization
  - ReLU activation

## Model Training Process

### 1. Data Preprocessing

#### Dataset Features
```python
# Basic Features
- Age
- Gender (Male/Female)
- Diet Type (Vegetarian/Non-Vegetarian)
- Living Environment (Urban/Rural)
- Skin Condition (Normal, Dry Skin, Rough Skin, Pale/Yellow Skin)

# Symptom Features (30 binary indicators)
- Vision Related: Night Blindness, Dry Eyes, Vision Problems, Light Sensitivity, Itchy Eyes
- Physical: Fatigue, Shortness of Breath, Fast Heart Rate, Muscle Weakness
- Neurological: Tingling Sensation, Memory Loss, Confusion, Poor Balance, Numbness
- Blood-related: Bleeding Gums, Easy Bruising, Heavy Menstrual Bleeding, Blood in Urine
- Digestive: Loss of Appetite, Diarrhea, Dark Stool
- Others: Low Sun Exposure, Weight Loss, Bone Pain, Depression, etc.
```

#### Feature Engineering Process
```python
# 1. Symptom Count Feature
symptom_columns = [
    'Night Blindness', 'Dry Eyes', 'Bleeding Gums', 'Fatigue', 
    'Tingling Sensation', 'Low Sun Exposure', 'Memory Loss',
    # ... and other symptoms
]
df['Symptom_Count'] = df[symptom_columns].sum(axis=1)

# 2. Grouped Symptom Scores
df['Vision_Issues'] = df[['Night Blindness', 'Dry Eyes', 'Vision Problems', 
                         'Light Sensitivity', 'Itchy Eyes']].sum(axis=1)
df['Physical_Weakness'] = df[['Fatigue', 'Shortness of Breath', 
                             'Fast Heart Rate', 'Muscle Weakness']].sum(axis=1)
df['Neurological_Signs'] = df[['Tingling Sensation', 'Memory Loss', 
                              'Confusion', 'Poor Balance', 'Numbness']].sum(axis=1)
df['Bleeding_Issues'] = df[['Bleeding Gums', 'Easy Bruising', 
                           'Heavy Menstrual Bleeding', 'Blood in Urine']].sum(axis=1)
df['Digestive_Issues'] = df[['Loss of Appetite', 'Diarrhea', 'Dark Stool']].sum(axis=1)

# 3. Skin Health Score
skin_mapping = {
    'Normal': 0, 
    'Dry Skin': 1, 
    'Rough Skin': 1, 
    'Pale/Yellow Skin': 2
}
df['Skin_Health'] = df['Reduced Wound Healing Capacity'] + df['Skin Condition'].map(skin_mapping)

# 4. Environmental Risk Score
df['Environmental_Risk'] = (
    (df['Living Environment'] == 'Urban').astype(int) + 
    df['Low Sun Exposure'].astype(int) + 
    (df['Diet Type'] == 'Vegetarian').astype(int)
)
```

#### Data Standardization
```python
# 1. Categorical Encoding
categorical_columns = ['Gender', 'Diet Type', 'Living Environment', 'Skin Condition']
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 2. Feature Scaling
X = df.drop(['Predicted Deficiency'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### 2. Model Architecture Details

```python
class VitaminNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(VitaminNet, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer with batch normalization and dropout
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
```

#### Architecture Parameters
- Input Size: 43 features (after feature engineering)
- Hidden Layer Sizes: [512, 256, 128, 64]
- Output Size: 14 (deficiency classes)
- Dropout Rate: 0.3
- Activation: ReLU
- Normalization: Batch Normalization after each hidden layer

### 3. Training Configuration Details

```python
# Model Initialization
model = VitaminNet(input_size=43, 
                   hidden_sizes=[512, 256, 128, 64],
                   num_classes=14)

# Training Parameters
learning_rate = 0.001
num_epochs = 1000
batch_size = 64

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=10,
    verbose=True
)
```

### 4. Training Process Details

#### Training Loop
```python
best_accuracy = 0
for epoch in range(num_epochs):
    # Training Phase
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Validation Phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        _, predicted = torch.max(val_outputs.data, 1)
        accuracy = accuracy_score(y_test, predicted)
        
        # Save Best Model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Adjust Learning Rate
        scheduler.step(accuracy)
```

#### Feature Importance Analysis
```python
def compute_feature_importance(model, X):
    X.requires_grad = True
    outputs = model(X)
    importance = torch.zeros(X.shape[1])
    
    for i in range(outputs.shape[1]):
        if i == outputs.shape[1] - 1:
            continue
        
        outputs[:, i].sum().backward(retain_graph=True)
        importance += torch.abs(X.grad).mean(0).detach()
        X.grad.zero_()
    
    return importance
```

### 5. Model Evaluation

#### Performance Metrics
- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Feature Importance Analysis

#### Model Output
The model predicts probabilities for 14 different vitamin deficiency types:
1. Iron
2. No Deficiency
3. Vitamin A
4. Vitamin B1 (Thiamine)
5. Vitamin B12
6. Vitamin B2 (Riboflavin)
7. Vitamin B3 (Niacin)
8. Vitamin B6
9. Vitamin C
10. Vitamin D
11. Vitamin E
12. Vitamin K
13. Folate
14. Zinc


```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export GEMINI_API_KEY="your-api-key"  # Linux/Mac
set GEMINI_API_KEY="your-api-key"     # Windows
```

## Usage

1. Train the model:
```bash
python train_vitamin_model.py
```

2. Run the web application:
```bash
python app.py
```

3. Access the application:
```
http://localhost:5000
```

## API Integration

### Gemini AI Integration
```python
- API Key configuration
- Safety settings implementation
- Generation config setup
- Error handling and retries
```

### Prediction API Endpoint
```python
POST /predict
- Input: User symptoms and information
- Output: 
  - Deficiency prediction
  - Confidence scores
  - Diet recommendations
```

## Directory Structure

```
vitamin-deficiency-predictor/
├── app.py                 # Flask application
├── train_vitamin_model.py # Model training script
├── requirements.txt       # Project dependencies
├── best_model.pth        # Trained model weights
├── static/
│   ├── style.css         # CSS styles
│   └── script.js         # Frontend JavaScript
├── templates/
│   ├── index.html        # Main application template
│   ├── login.html        # Login page
│   └── register.html     # Registration page
└── data/
    └── symptom_based_vitamin_deficiency_dataset_final.csv
```
