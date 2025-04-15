from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import torch
from train_vitamin_model import VitaminNet
from test_model import load_and_preprocess_data, predict_deficiency
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import google.generativeai as genai
import os
from google.api_core import retry
from tenacity import retry, stop_after_attempt, wait_exponential
import requests.exceptions
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import asyncio
import sqlite3
from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash
import re  # Add at the top with other imports

# Configure Gemini API with proper API key
GEMINI_API_KEY = "AIzaSyDpcpNypqfOPVVP7cDUUKXcNx0zsT9xWYM"  # Direct API key assignment

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully")
except Exception as e:
    print(f"Error configuring Gemini API: {str(e)}")

# Initialize Gemini model with safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

generation_config = {
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 40,
    "max_output_tokens": 2048,
}

# Create the model with better error handling
print("Initializing Gemini model...")
model_name = "gemini-1.5-flash"  # Using the stable pro model
try:
    gemini_model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print(f"Model initialized with name: {model_name}")
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
    gemini_model = None

# Test the API with better error handling
def test_gemini_api():
    if not gemini_model:
        return False
    try:
        test_response = gemini_model.generate_content("Hello, are you working?")
        if test_response and hasattr(test_response, 'text'):
            print("Gemini API test successful")
            return True
        else:
            print("Warning: Gemini API test returned unexpected response format")
            return False
    except Exception as e:
        print(f"Warning: Gemini API test failed: {str(e)}")
        return False

# Run API test
api_working = test_gemini_api()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Replace with a secure secret key in production

# Database configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'users.db')

# Database initialization
def init_db():
    try:
        # Ensure the database directory exists
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create predictions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                deficiency_type TEXT NOT NULL,
                probability REAL NOT NULL,
                reported_symptoms TEXT NOT NULL,
                diet_recommendation TEXT,
                diet_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        print(f"Database initialized successfully at {DATABASE_PATH}")
    except sqlite3.Error as e:
        print(f"Database initialization error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Function to migrate the database schema
def migrate_db():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # Check if diet_data column exists in predictions table
        c.execute("PRAGMA table_info(predictions)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'diet_data' not in columns:
            print("Migrating database: Adding diet_data column to predictions table")
            # Add diet_data column if it doesn't exist
            c.execute("ALTER TABLE predictions ADD COLUMN diet_data TEXT")
            conn.commit()
            print("Database migration completed successfully")
        else:
            print("Database schema is up to date")
            
    except sqlite3.Error as e:
        print(f"Database migration error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Initialize the database
init_db()

# Migrate database if needed
migrate_db()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize preprocessing tools
le = LabelEncoder()
scaler = StandardScaler()
categorical_columns = ['Gender', 'Diet Type', 'Living Environment', 'Skin Condition']
possible_values = {
    'Gender': ['Male', 'Female'],
    'Diet Type': ['Vegetarian', 'Non-Vegetarian'],
    'Living Environment': ['Urban', 'Rural'],
    'Skin Condition': ['Normal', 'Dry Skin', 'Rough Skin', 'Pale/Yellow Skin']
}

# Pre-fit label encoders
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder().fit(possible_values[col])

# Pre-define feature groups for faster computation
vision_symptoms = ['Night Blindness', 'Dry Eyes', 'Vision Problems', 'Light Sensitivity', 'Itchy Eyes']
physical_symptoms = ['Fatigue', 'Shortness of Breath', 'Fast Heart Rate', 'Muscle Weakness']
neurological_symptoms = ['Tingling Sensation', 'Memory Loss', 'Confusion', 'Poor Balance', 'Numbness']
bleeding_symptoms = ['Bleeding Gums', 'Easy Bruising', 'Heavy Menstrual Bleeding', 'Blood in Urine']
digestive_symptoms = ['Loss of Appetite', 'Diarrhea', 'Dark Stool']

# Skin condition mapping
skin_mapping = {'Normal': 0, 'Dry Skin': 1, 'Rough Skin': 1, 'Pale/Yellow Skin': 2}

# Load the model once when the app starts
input_size = 43  # Updated to match the actual number of features from training
hidden_sizes = [512, 256, 128, 64]  # Updated to match training architecture
num_classes = 14  # Updated number of deficiency types

model = VitaminNet(input_size, hidden_sizes, num_classes)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set to evaluation mode

# Move model to CPU and optimize for inference
model = model.cpu()
torch.set_grad_enabled(False)

def fast_preprocess_data(data):
    """Optimized preprocessing function"""
    # Create feature vector in the correct order
    features = []
    
    # Basic information
    features.append(float(data['Age']) / 120.0)  # Normalize age
    
    # Encode and append categorical variables
    for col in categorical_columns:
        encoded_val = label_encoders[col].transform([data[col]])[0]
        features.append(float(encoded_val))
    
    # Add all symptom columns in order
    symptom_fields = [
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
    
    for symptom in symptom_fields:
        features.append(float(data[symptom]))
    
    # Calculate Symptom_Count (total number of symptoms present)
    symptom_count = sum(float(data[s]) for s in symptom_fields)
    features.append(symptom_count)
    
    # Calculate and add derived features
    # Vision Issues
    vision_issues = float(sum(data[s] for s in vision_symptoms))
    features.append(vision_issues)
    
    # Physical Weakness
    physical_weakness = float(sum(data[s] for s in physical_symptoms))
    features.append(physical_weakness)
    
    # Neurological Signs
    neurological_signs = float(sum(data[s] for s in neurological_symptoms))
    features.append(neurological_signs)
    
    # Bleeding Issues
    bleeding_issues = float(sum(data[s] for s in bleeding_symptoms))
    features.append(bleeding_issues)
    
    # Digestive Issues
    digestive_issues = float(sum(data[s] for s in digestive_symptoms))
    features.append(digestive_issues)
    
    # Skin Health
    skin_health = float(data['Reduced Wound Healing Capacity'] + skin_mapping[data['Skin Condition']])
    features.append(skin_health)
    
    # Environmental Risk
    environmental_risk = float(
        (data['Living Environment'] == 'Urban') + 
        data['Low Sun Exposure'] + 
        (data['Diet Type'] == 'Vegetarian')
    )
    features.append(environmental_risk)
    
    return np.array(features, dtype=np.float32)

def validate_input(data):
    required_fields = {
        'Age': int,
        'Gender': str,
        'Diet Type': str,
        'Living Environment': str,
        'Skin Condition': str
    }
    
    symptom_fields = [
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
    
    # Check required fields
    for field, field_type in required_fields.items():
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(data[field], field_type) and data[field] is not None:
            try:
                if field_type == int:
                    data[field] = int(data[field])
                else:
                    data[field] = str(data[field])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {field}")
    
    # Validate age range
    if not (1 <= int(data['Age']) <= 120):
        raise ValueError("Age must be between 1 and 120")
    
    # Validate categorical fields
    if data['Gender'] not in possible_values['Gender']:
        raise ValueError("Invalid gender value")
    if data['Diet Type'] not in possible_values['Diet Type']:
        raise ValueError("Invalid diet type")
    if data['Living Environment'] not in possible_values['Living Environment']:
        raise ValueError("Invalid living environment")
    if data['Skin Condition'] not in possible_values['Skin Condition']:
        raise ValueError("Invalid skin condition")
    
    # Check symptom fields
    for field in symptom_fields:
        if field not in data:
            raise ValueError(f"Missing symptom field: {field}")
        if not isinstance(data[field], bool):
            data[field] = bool(data[field])
    
    return data

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'

        try:
            conn = sqlite3.connect(DATABASE_PATH)
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = c.fetchone()
            
            if user and check_password_hash(user[4], password):  # Index 4 is password
                session['user_id'] = user[0]  # Store user ID in session
                session['user_name'] = f"{user[1]} {user[2]}"  # Store full name
                
                if remember:
                    # Session will last longer if remember is checked
                    session.permanent = True
                
                flash('Successfully logged in!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid email or password.', 'danger')
        except sqlite3.Error as e:
            print(f"Login error: {str(e)}")
            flash('An error occurred. Please try again.', 'danger')
        finally:
            conn.close()
            
    return render_template('login.html')

def is_valid_email(email):
    """Validate email format using regex pattern"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('firstName', '').strip()
        last_name = request.form.get('lastName', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')
        confirm_password = request.form.get('confirmPassword')

        # Validate all required fields
        if not all([first_name, last_name, email, password, confirm_password]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        # Validate email format
        if not is_valid_email(email):
            flash('Please enter a valid email address.', 'danger')
            return render_template('register.html')

        # Validate password length
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html')

        # Validate password match
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')

        # Hash password
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            c = conn.cursor()
            
            # Check if email already exists
            c.execute('SELECT email FROM users WHERE email = ?', (email,))
            if c.fetchone() is not None:
                flash('An account with this email already exists.', 'danger')
                return render_template('register.html')

            # Insert new user
            c.execute('''
                INSERT INTO users (first_name, last_name, email, password) 
                VALUES (?, ?, ?, ?)
            ''', (first_name, last_name, email, hashed_password))
            conn.commit()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except sqlite3.IntegrityError:
            flash('An account with this email already exists.', 'danger')
        except sqlite3.Error as e:
            print(f"Registration error: {str(e)}")
            flash('An error occurred during registration. Please try again.', 'danger')
        finally:
            if conn:
                conn.close()
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Update the home route to require login
@app.route('/')
@login_required
def home():
    user_name = session.get('user_name', 'Guest')
    return render_template('index.html', user_name=user_name)

# Add profile route to show prediction history
@app.route('/profile')
@login_required
def profile():
    user_id = session.get('user_id')
    user_name = session.get('user_name', 'Guest')
    
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        c = conn.cursor()
        
        # Get user details
        c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = dict(c.fetchone())
        
        # Get prediction history, ordered by most recent first
        c.execute('''
            SELECT * FROM predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        ''', (user_id,))
        
        predictions = []
        most_recent = None  # Initialize most_recent
        deficiency_counts = {}  # Track deficiency type counts
        
        for row in c.fetchall():
            prediction_data = dict(row)
            
            # Parse the JSON stored in reported_symptoms
            if prediction_data['reported_symptoms']:
                try:
                    prediction_data['reported_symptoms'] = json.loads(prediction_data['reported_symptoms'])
                except json.JSONDecodeError:
                    # If it's not valid JSON, keep it as is
                    pass
            
            # Parse the JSON stored in diet_data
            if prediction_data.get('diet_data') and prediction_data['diet_data'] != 'null':
                try:
                    prediction_data['diet_data'] = json.loads(prediction_data['diet_data'])
                except json.JSONDecodeError:
                    # If it's not valid JSON, set to None
                    prediction_data['diet_data'] = None
            
            # Set most_recent to the first prediction (since they're ordered by date DESC)
            if most_recent is None:
                most_recent = prediction_data
            
            # Track deficiency type counts
            deficiency_type = prediction_data['deficiency_type']
            deficiency_counts[deficiency_type] = deficiency_counts.get(deficiency_type, 0) + 1
                    
            predictions.append(prediction_data)
        
        # Find the most common deficiency type
        most_common = None
        if deficiency_counts:
            most_common_type = max(deficiency_counts.items(), key=lambda x: x[1])[0]
            most_common = {'type': most_common_type, 'count': deficiency_counts[most_common_type]}
            
        return render_template(
            'profile.html', 
            user=user, 
            predictions=predictions, 
            user_name=user_name,
            most_recent=most_recent,  # Add most_recent to template context
            most_common=most_common  # Add most_common to template context
        )
        
    except sqlite3.Error as e:
        flash(f"Error retrieving data: {str(e)}", 'danger')
        return redirect(url_for('home'))
    finally:
        if conn:
            conn.close()

# Add route to delete a prediction from history
@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    user_id = session.get('user_id')
    
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # First verify that the prediction belongs to the current user
        c.execute('SELECT user_id FROM predictions WHERE id = ?', (prediction_id,))
        result = c.fetchone()
        
        if not result:
            flash('Prediction record not found.', 'warning')
            return redirect(url_for('profile'))
            
        if result[0] != user_id:
            flash('You do not have permission to delete this record.', 'danger')
            return redirect(url_for('profile'))
        
        # Delete the prediction
        c.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
        conn.commit()
        
        flash('Prediction record deleted successfully.', 'success')
        
    except sqlite3.Error as e:
        flash(f"Error deleting record: {str(e)}", 'danger')
    finally:
        if conn:
            conn.close()
    
    return redirect(url_for('profile'))

# Add route to clear all prediction history
@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    user_id = session.get('user_id')
    
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # Delete all predictions for this user
        c.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
        conn.commit()
        
        count = c.rowcount
        flash(f'Successfully cleared {count} prediction records from your history.', 'success')
        
    except sqlite3.Error as e:
        flash(f"Error clearing history: {str(e)}", 'danger')
    finally:
        if conn:
            conn.close()
    
    return redirect(url_for('profile'))

# Add retry decorator for the diet recommendation function
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda _: {
        "diet_plan": "Diet recommendations temporarily unavailable. Please try again later.",
        "success": False
    }
)
def get_diet_recommendation(deficiency, symptoms, user_info):
    """Generate personalized diet recommendations using Gemini"""
    if not api_working or not gemini_model:
        print("API not working or model not initialized")
        return {
            "diet_plan": "Diet recommendation service is currently unavailable. Please ensure API key is properly configured.",
            "success": False
        }

    try:
        # Enhanced structured prompt with markdown
        prompt = f"""Create a detailed diet plan for {deficiency} deficiency.
        Patient: {user_info['Age']} years old, {user_info['Gender']}, {user_info['Diet Type']} diet
        Symptoms: {', '.join(symptoms)}

        Format your response using the following strict JSON structure with markdown content:
        
        ```json
        {{
            "title": "Diet Plan for {deficiency} Deficiency",
            "introduction": "Brief paragraph explaining this deficiency and why diet matters",
            "daily_meals": [
                {{
                    "name": "Breakfast",
                    "description": "Description of ideal breakfast",
                    "food_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
                }},
                {{
                    "name": "Mid-morning Snack",
                    "description": "Description of ideal snack",
                    "food_suggestions": ["suggestion 1", "suggestion 2"]
                }},
                {{
                    "name": "Lunch",
                    "description": "Description of ideal lunch",
                    "food_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
                }},
                {{
                    "name": "Evening Snack",
                    "description": "Description of ideal snack",
                    "food_suggestions": ["suggestion 1", "suggestion 2"]
                }},
                {{
                    "name": "Dinner",
                    "description": "Description of ideal dinner",
                    "food_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
                }}
            ],
            "key_nutrients": [
                {{
                    "name": "Nutrient name",
                    "benefits": "Brief description of benefits",
                    "food_sources": ["source 1", "source 2", "source 3"]
                }},
                // Add 2-4 more nutrients
            ],
            "foods_to_avoid": [
                {{
                    "name": "Food category to avoid",
                    "reason": "Why this should be avoided"
                }},
                // Add 2-3 more foods to avoid
            ],
            "supplements": [
                {{
                    "name": "Supplement name",
                    "dosage": "Recommended dosage",
                    "notes": "Important notes about this supplement"
                }},
                // Add 1-2 more supplements if relevant
            ],
            "absorption_tips": [
                "Tip 1 for better absorption",
                "Tip 2 for better absorption",
                "Tip 3 for better absorption"
            ]
        }}
        ```

        Keep it practical and compatible with {user_info['Diet Type']} diet. Include only scientifically accurate information.
        If a section doesn't apply, provide empty arrays but do not omit any JSON fields."""

        print(f"Sending prompt to Gemini API: {prompt}")

        try:
            response = gemini_model.generate_content(prompt)
            print(f"Raw Gemini response: {response}")
            
            if hasattr(response, 'text'):
                text = response.text.strip()
                print(f"Response text: {text}")
                
                # Extract JSON from the response
                # Find the JSON block within the response (between ``` markers)
                try:
                    json_text = text
                    if "```json" in text:
                        json_text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        json_text = text.split("```")[1].split("```")[0].strip()
                    
                    # Parse the JSON
                    diet_data = json.loads(json_text)
                    
                    # Return structured diet data and raw markdown for backward compatibility
                    return {
                        "diet_plan": text,  # Keep original markdown for backward compatibility
                        "diet_data": diet_data,  # Add structured data for modern UI
                        "success": True
                    }
                except (json.JSONDecodeError, IndexError) as json_error:
                    print(f"Error parsing JSON from response: {str(json_error)}")
                    print(f"Failed JSON text: {text}")
                    # Fallback to returning just the text
                    return {
                        "diet_plan": f"## Diet Recommendations for {deficiency} Deficiency\n\n{text}",
                        "success": True
                    }
            elif hasattr(response, 'parts'):
                text = ''.join([part.text for part in response.parts]).strip()
                print(f"Response from parts: {text}")
                
                # Try to extract JSON using the same logic as above
                try:
                    json_text = text
                    if "```json" in text:
                        json_text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        json_text = text.split("```")[1].split("```")[0].strip()
                    
                    # Parse the JSON
                    diet_data = json.loads(json_text)
                    
                    return {
                        "diet_plan": text,  # Keep original markdown for backward compatibility
                        "diet_data": diet_data,  # Add structured data for modern UI
                        "success": True
                    }
                except (json.JSONDecodeError, IndexError) as json_error:
                    print(f"Error parsing JSON from parts response: {str(json_error)}")
                    # Fallback to returning just the text
                    return {
                        "diet_plan": f"## Diet Recommendations for {deficiency} Deficiency\n\n{text}",
                        "success": True
                    }
            else:
                print("No text or parts attribute in response")
                return {
                    "diet_plan": "## Diet Recommendations\n\nUnable to generate recommendations. Please try again.",
                    "success": False
                }
                
        except Exception as api_error:
            print(f"API Error: {str(api_error)}")
            return {
                "diet_plan": "## Error\n\nService temporarily unavailable. Please try again later.",
                "success": False
            }

    except Exception as e:
        print(f"General Error: {str(e)}")
        return {
            "diet_plan": "## Error\n\nAn error occurred while generating recommendations.",
            "success": False
        }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            raise ValueError("No data provided")
        
        # Validate input data
        data = validate_input(data)
        
        # Convert boolean values to integers
        symptom_fields = [
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
        
        for field in symptom_fields:
            data[field] = 1 if data[field] else 0
        
        # Use optimized preprocessing
        features = fast_preprocess_data(data)
        
        # Convert to tensor and get prediction
        X = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get all predictions with probabilities
            all_probs = probabilities.squeeze().numpy()
            all_indices = np.argsort(all_probs)[::-1]  # Sort in descending order
            
            # Filter out 'No Deficiency' and get top 4
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
            
            # Filter predictions to exclude 'No Deficiency'
            filtered_predictions = []
            for idx in all_indices:
                if idx != 1:  # Skip 'No Deficiency'
                    filtered_predictions.append({
                        'deficiency': deficiency_mapping[idx],
                        'probability': float(all_probs[idx]) * 100  # Convert to percentage
                    })
                if len(filtered_predictions) == 4:
                    break
            
            # Get reported symptoms
            reported_symptoms = [field for field in symptom_fields if data[field] == 1]
            
            # Get diet recommendations with the new format
            diet_response = get_diet_recommendation(
                filtered_predictions[0]['deficiency'],
                reported_symptoms,
                {
                    'Age': data['Age'],
                    'Gender': data['Gender'],
                    'Diet Type': data['Diet Type']
                }
            )
            
            # Save prediction to database if user is logged in
            if 'user_id' in session:
                try:
                    user_id = session['user_id']
                    conn = sqlite3.connect(DATABASE_PATH)
                    c = conn.cursor()
                    
                    # Store reported symptoms as a JSON string
                    symptoms_json = json.dumps(reported_symptoms)
                    
                    c.execute('''
                        INSERT INTO predictions 
                        (user_id, deficiency_type, probability, reported_symptoms, diet_recommendation, diet_data)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        user_id,
                        filtered_predictions[0]['deficiency'],
                        filtered_predictions[0]['probability'],
                        symptoms_json,
                        diet_response['diet_plan'],
                        json.dumps(diet_response.get('diet_data', None))
                    ))
                    
                    conn.commit()
                    print(f"Prediction saved for user {user_id}")
                except sqlite3.Error as e:
                    print(f"Error saving prediction: {str(e)}")
                finally:
                    if conn:
                        conn.close()
        
        return jsonify({
            'prediction': filtered_predictions[0]['deficiency'],
            'top_predictions': filtered_predictions,
            'reported_symptoms': reported_symptoms,
            'diet_recommendation': diet_response['diet_plan'],
            'diet_data': diet_response.get('diet_data', None),  # Include structured diet data if available
            'success': diet_response['success']
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True) 