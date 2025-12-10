import pandas as pd
import numpy as np
import pickle
import os
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Get the absolute path to the app directory
# Assuming this script is at the root of the project, App is a subdirectory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, 'App')

def train_knn_model():
    """Train and return the best KNN model for breast cancer prediction"""
    print("Training KNN model...")
    # Define model path
    models_dir = os.path.join(APP_DIR, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, 'breast_cancer_model.pickle')

    # Step 1: Load and prepare the data
    data_path = os.path.join(APP_DIR, 'data', 'breast-cancer.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    data = pd.read_csv(data_path, sep=",")
    # Select relevant features and target variable
    data = data[["diagnosis", "radius_mean", "texture_mean", "perimeter_mean"]]
    predict = "diagnosis"

    # Step 2: Prepare features (X) and target (y)
    x = np.array(data.drop(columns=[predict]))
    y = np.array(data[predict])
    
    # Step 3: Train multiple models to find the best one
    best = 0
    best_model = None
    for i in range(100):
        # Create new random split for each iteration
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
        # Initialize KNN classifier with 9 neighbors
        model = KNeighborsClassifier(n_neighbors=9)
        # Train the model
        model.fit(x_train, y_train)
        # Calculate accuracy
        accuracy = model.score(x_test, y_test)
        
        if accuracy > best:
            best = accuracy
            best_model = model

    print(f"Best KNN model found with accuracy: {best}")
    
    # Save the best model found
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print("KNN Model saved to disk.")

def train_rf_model():
    print("Training RandomForest model...")
    models_dir = os.path.join(APP_DIR, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    rf_model_path = os.path.join(models_dir, 'rf_model_data.pickle')
    
    data_path = os.path.join(APP_DIR, 'data', 'breast-cancer.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    data = pd.read_csv(data_path)
    
    # Clean column names by stripping whitespace
    data.columns = data.columns.str.strip()
    
    # Clean data values by stripping whitespace
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.strip()
    
    # Convert diagnosis to binary (M=1, B=0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Select features and target
    features = ['radius_mean', 'texture_mean', 'perimeter_mean']
    X = data[features]
    y = data['diagnosis']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    model_data = (model, scaler, features)

    with open(rf_model_path, "wb") as f:
        pickle.dump(model_data, f)
    print("RandomForest model data saved to disk.")

if __name__ == "__main__":
    train_knn_model()
    train_rf_model()
