```mermaid
%% Level 0 DFD (Context Diagram)
graph TD
    User((User))
    System[Vitamin Deficiency Prediction System]
    Database[(Database)]
    
    User -->|Input symptoms & info| System
    System -->|Predictions & Diet Plans| User
    System -->|Store/Retrieve Data| Database
    Database -->|User Data & Results| System

%% Level 1 DFD
graph TD
    User((User))
    Auth[Authentication System]
    Predict[Prediction Engine]
    Diet[Diet Recommendation System]
    DB[(Database)]
    Model[ML Model]
    
    User -->|Register/Login| Auth
    Auth -->|Verify| DB
    User -->|Input Symptoms| Predict
    Predict -->|Process Data| Model
    Model -->|Deficiency Prediction| Diet
    Diet -->|Generate Plan| User
    DB -->|Store Results| DB

%% ER Diagram
erDiagram
    USERS ||--o{ PREDICTIONS : makes
    USERS {
        int id PK
        string first_name
        string last_name
        string email
        string password
        datetime created_at
    }
    PREDICTIONS ||--o{ SYMPTOMS : has
    PREDICTIONS {
        int id PK
        int user_id FK
        string deficiency_type
        float probability
        datetime prediction_date
        text diet_recommendation
    }
    SYMPTOMS {
        int id PK
        int prediction_id FK
        string symptom_name
        boolean is_present
    }
    DEFICIENCY_TYPES {
        int id PK
        string name
        text description
        text common_symptoms
    }
    PREDICTIONS }|--|| DEFICIENCY_TYPES : predicts

%% System Architecture
graph TD
    subgraph Frontend
        UI[User Interface]
        Forms[Input Forms]
        Charts[Visualization]
    end
    
    subgraph Backend
        Auth[Authentication]
        API[Flask API]
        ML[Machine Learning]
        Diet[Diet Generator]
    end
    
    subgraph Database
        UserDB[(User Data)]
        PredDB[(Predictions)]
    end
    
    subgraph ML_Components
        Model[Neural Network]
        Preprocess[Data Preprocessing]
        Train[Model Training]
    end
    
    UI -->|User Input| Forms
    Forms -->|Submit| API
    API -->|Authenticate| Auth
    Auth -->|Verify| UserDB
    API -->|Process| ML
    ML -->|Get Prediction| Model
    Model -->|Results| Diet
    Diet -->|Plan| API
    API -->|Store| PredDB
    API -->|Display| Charts

%% Data Processing Flow
graph LR
    Input[User Input] -->|Validation| Clean[Data Cleaning]
    Clean -->|Preprocessing| Features[Feature Engineering]
    Features -->|Scaling| Scale[Data Scaling]
    Scale -->|Prediction| Model[Neural Network]
    Model -->|Analysis| Output[Prediction Results]
    Output -->|Generation| Diet[Diet Plan]
    
%% Neural Network Architecture
graph TD
    Input[Input Layer] -->|512 neurons| H1[Hidden Layer 1]
    H1 -->|256 neurons| H2[Hidden Layer 2]
    H2 -->|128 neurons| H3[Hidden Layer 3]
    H3 -->|64 neurons| H4[Hidden Layer 4]
    H4 -->|14 classes| Output[Output Layer]
    
    subgraph Layer_Components
        ReLU[ReLU Activation]
        BN[Batch Normalization]
        Drop[Dropout 0.3]
    end
``` 