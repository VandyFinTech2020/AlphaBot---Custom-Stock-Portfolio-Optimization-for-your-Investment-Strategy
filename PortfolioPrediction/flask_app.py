from flask import Flask, jsonify, request
from predict import get_portfolio_predictions

# Flask app
app = Flask(__name__)

# Routes
@app.route('/predict', methods=['POST'])

def predict():
    req = request.get_json(force=True)
    
    tickers = req['tickers']
    
    print(tickers)
    
    user_portfolio_metrics = get_portfolio_predictions(tickers)
    
    output = jsonify(results=user_portfolio_metrics)

    return output



if __name__ == '__main__':
    app.run(port = 5000, debug=True)
    