from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os # To handle file paths

# Initialize the Flask application
app = Flask(__name__,template_folder="../ai")

# --- Model Training and Saving ---
# This part will run once when the Flask app starts.
# It trains the model and saves it to a file so it can be loaded later without retraining.

# Define the path to your CSV file
# Ensure 'BostonHousing (1).csv' is in the same directory as your app.py
csv_file_path = "BostonHousing (1).csv"

# Check if the CSV file exists
if not os.path.exists(csv_file_path):
    print(f"Error: CSV file not found at {csv_file_path}")
    print("Please make sure 'BostonHousing (1).csv' is in the same directory as this script.")
    # You might want to exit or raise an exception here in a production environment
    # For now, we'll continue but the model loading will fail if the file isn't there.

try:
    s1 = pd.read_csv(csv_file_path)

    # Check for missing values (as in your original code)
    # print(s1.isnull().sum())

    # Prepare data for the model
    x = s1.drop('price', axis=1) # Features
    y = s1['price'] # Target variable (house price)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=31)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Save the trained model using pickle
    model_filename = 'boston_house_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model trained and saved to {model_filename}")

except FileNotFoundError:
    print(f"Could not find the CSV file at {csv_file_path}. Please ensure it's in the correct directory.")
    model = None # Set model to None if training failed
except Exception as e:
    print(f"An error occurred during model training: {e}")
    model = None # Set model to None if training failed


# --- Flask Routes ---

# Load the trained model when the Flask app starts
# This assumes the model was successfully saved in the previous step
try:
    with open('boston_house_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    loaded_model = None
    print("Error: 'boston_house_model.pkl' not found. Please ensure the model was trained and saved.")
except Exception as e:
    loaded_model = None
    print(f"An error occurred while loading the model: {e}")


# Route for the home page, which renders the input form
@app.route('/')
def index():
    """
    Renders the 'boston_house_data_form.html' template.
    This route handles GET requests to the root URL, showing the form to the user.
    """
    return render_template('boston_house_data_form.html')


# Route for handling predictions from the form submission
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to the '/predict' endpoint.
    It retrieves the 13 input features from the form,
    makes a prediction using the loaded machine learning model,
    and then renders the 'result.html' template to display the prediction.
    """
    if loaded_model is None:
        return "Error: Machine learning model not loaded. Cannot make predictions.", 500

    try:
        # Get the input values from the form
        crim = float(request.form['crim'])
        zn = float(request.form['zn'])
        indus = float(request.form['indus'])
        chas = float(request.form['chas']) # This is 0 or 1 from the select dropdown
        nox = float(request.form['nox'])
        rm = float(request.form['rm'])
        age = float(request.form['age'])
        dis = float(request.form['dis'])
        rad = float(request.form['rad'])
        tax = float(request.form['tax'])
        ptratio = float(request.form['ptratio'])
        b = float(request.form['b'])
        lstat = float(request.form['lstat'])

        # Create a NumPy array from the input values
        # Ensure the order of features matches the order used during model training (x.columns)
        input_features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(input_features)[0] # Get the first (and only) prediction

        # Render the result template with the prediction
        return render_template('result.html', prediction=f"{prediction:.2f}")

    except ValueError:
        return "Error: Invalid input. Please ensure all fields are filled correctly with numbers.", 400
    except Exception as e:
        return f"An unexpected error occurred: {e}", 500


# This block ensures the Flask development server runs only when the script is executed directly
if __name__ == '__main__':
    """
    Entry point for running the Flask application.
    Enables debug mode for automatic reloads and detailed error messages.
    """
    app.run(debug=True)
