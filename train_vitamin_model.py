import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class VitaminNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(VitaminNet, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
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
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def main():
    # Load the dataset
    df = pd.read_csv('data/symptom_based_vitamin_deficiency_dataset_final.csv')

    # Feature Engineering
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

    df['Symptom_Count'] = df[symptom_columns].sum(axis=1)

    # Create interaction features
    df['Vision_Issues'] = df[['Night Blindness', 'Dry Eyes', 'Vision Problems', 'Light Sensitivity', 'Itchy Eyes']].sum(axis=1)
    df['Physical_Weakness'] = df[['Fatigue', 'Shortness of Breath', 'Fast Heart Rate', 'Muscle Weakness']].sum(axis=1)
    df['Neurological_Signs'] = df[['Tingling Sensation', 'Memory Loss', 'Confusion', 'Poor Balance', 'Numbness']].sum(axis=1)
    df['Bleeding_Issues'] = df[['Bleeding Gums', 'Easy Bruising', 'Heavy Menstrual Bleeding', 'Blood in Urine']].sum(axis=1)
    df['Digestive_Issues'] = df[['Loss of Appetite', 'Diarrhea', 'Dark Stool']].sum(axis=1)

    # Create skin health score
    skin_mapping = {'Normal': 0, 'Dry Skin': 1, 'Rough Skin': 1, 'Pale/Yellow Skin': 2}
    df['Skin_Health'] = df['Reduced Wound Healing Capacity'] + df['Skin Condition'].map(skin_mapping)

    # Create environmental risk score
    df['Environmental_Risk'] = ((df['Living Environment'] == 'Urban').astype(int) + 
                              df['Low Sun Exposure'].astype(int) + 
                              (df['Diet Type'] == 'Vegetarian').astype(int))

    # Preprocess the data
    # Convert categorical variables to numerical
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Diet Type', 'Living Environment', 'Skin Condition']
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    # Separate features and target
    X = df.drop(['Predicted Deficiency'], axis=1)
    y = df['Predicted Deficiency']

    # Encode the target variable
    y = le.fit_transform(y)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model parameters
    input_size = X.shape[1]
    hidden_sizes = [512, 256, 128, 64]  # Increased network capacity
    num_classes = len(np.unique(y))
    learning_rate = 0.001
    num_epochs = 1000
    batch_size = 64

    # Initialize the model
    model = VitaminNet(input_size, hidden_sizes, num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # Training loop
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            _, predicted = torch.max(val_outputs.data, 1)
            accuracy = accuracy_score(y_test, predicted)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_model.pth')
            
            scheduler.step(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%')

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Final evaluation
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = accuracy_score(y_test, predicted)
        
        print(f"\nFinal Model Accuracy on Test Set: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, predicted, target_names=le.classes_))

    # Feature importance using gradient-based method
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

    feature_importance = compute_feature_importance(model, X_train)
    feature_importance = pd.DataFrame({
        'feature': list(df.drop(['Predicted Deficiency'], axis=1).columns),
        'importance': feature_importance.numpy()
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

if __name__ == '__main__':
    main() 