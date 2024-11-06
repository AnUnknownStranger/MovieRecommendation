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
@app.route('/login/new_user', methods=['POST'])  
def doNewUser():
    return newUser()

@app.route('/login/returning_user', methods=['POST'])  
def doReturningUser():
    return returningUser()

@app.route('/login')
def doRegister():
    return register()

@app.route('/logout', methods=['POST'])
def logout():
    response = make_response(jsonify({"message": "Logged out successfully"}))
    response.set_cookie('auth_token', '', expires=0)  # Setting the cookie to expire immediately
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5001)
