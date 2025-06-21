from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
import os
import logging

predict_blueprint = Blueprint('predict', __name__)

# Load pre-trained models and encoders
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../Ml/Obesity')
optimized_voting_classifier = joblib.load(os.path.join(MODEL_DIR, 'optimized_voting_classifier.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
reverse_target_mapping = joblib.load(os.path.join(MODEL_DIR, 'reverse_target_mapping.pkl'))

# Configure logging
logging.basicConfig(level=logging.INFO)

@predict_blueprint.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.json
        logging.info(f"Received input data: {input_data}")

        # Validate input data
        if not input_data:
            raise ValueError("No input data provided.")

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Show exact DataFrame structure for debugging
        logging.info(f"Input DataFrame columns: {list(input_df.columns)}")
        if hasattr(optimized_voting_classifier, 'feature_names_in_'):
            logging.info(f"Expected DataFrame columns: {list(optimized_voting_classifier.feature_names_in_)}")
        
        # Ensure the DataFrame columns are in the exact same order as the training data
        try:
            # Try to reorder columns if feature_names_in_ is available
            if hasattr(optimized_voting_classifier, 'feature_names_in_'):
                feature_names = optimized_voting_classifier.feature_names_in_
                # Make sure all required columns exist
                for col in feature_names:
                    if col not in input_df.columns:
                        return jsonify({"error": f"Missing required column: {col}"}), 400
                # Reorder columns to match the training data
                input_df = input_df[feature_names]
        except Exception as e:
            logging.error(f"Error reordering columns: {e}")
            return jsonify({"error": f"Column ordering error: {str(e)}"}), 400

        # Skip the label encoding step entirely
        # Make prediction directly with the numerical data
        try:
            prediction = optimized_voting_classifier.predict(input_df)
            predicted_label = reverse_target_mapping[prediction[0]]
            logging.info(f"Successful prediction: {predicted_label}")
            return jsonify({"prediction": predicted_label})
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return jsonify({"error": f"Prediction error: {str(e)}"}), 400

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400