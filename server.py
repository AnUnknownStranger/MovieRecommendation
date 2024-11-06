from flask import Flask, jsonify, make_response
from flask_cors import CORS

from backend.auth_routes import *

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/api/data', methods=['GET'])
def get_data():
    print("Hello")
    data = {
        "message": "Hello from Flask!",
        "status": "success",
        "data": [1, 2, 3, 4, 5]
    }
    return jsonify(data)


## Login and Register
@app.route('/register', methods=['POST'])  
def doNewUser():
    return register()

@app.route('/login', methods=['POST'])  
def doReturningUser():
    return login()

@app.route('/logout', methods=['POST'])
def logout():
    response = make_response(jsonify({"message": "Logged out successfully"}))
    response.set_cookie('auth_token', '', expires=0)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=8080)
