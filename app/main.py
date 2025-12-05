import mlflow
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.components.data_preprocessing import preprocess_comment
from src.utils.main_utils import load_object


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri("http://ec2-51-20-74-217.eu-north-1.compute.amazonaws.com:5000") 
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = load_object(vectorizer_path)
    return model, vectorizer


# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("YoutubeSentimentAnalysis", "1", "artifacts/data_transformation/tfidf_vectorizer.pkl") 


@app.route('/')
def home():
    return "Welcome to our flask api"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    print("i am the comment: ",comments)
    print("i am the comment type: ",type(comments))
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        transformed_comments = vectorizer.transform(preprocessed_comments)

        dense_comments = transformed_comments.toarray()  # Convert to dense array
        
        predictions = model.predict(dense_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        # predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)