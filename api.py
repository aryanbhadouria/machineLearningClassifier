from flask import Flask, request, jsonify
from predict import predict_spam

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    prediction = predict_spam(text)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
