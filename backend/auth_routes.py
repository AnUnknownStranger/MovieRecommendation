
from flask import request, make_response, jsonify, redirect
from bson import json_util
import bcrypt
import hashlib
import secrets
import re

from response import sendResponse, page_not_found
from config import app, userCollection, channelCollection


def is_valid_email(email):
    email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_pattern, email)


def newUser():
    try:
        newUserDat = request.get_json()
            
        email = newUserDat.get("email", "")
        username = newUserDat.get("username", "")
        password = newUserDat.get("password", "")
        passwordConfirm = newUserDat.get("confirm_password", "")

        print("***************************************")
        print(password)
        print(passwordConfirm)

        if email == "" or username == "" or password == "" or passwordConfirm == "":
            print("EMPTY FIELDS")
            return jsonify({'message_fields': "Fields can't be left empty."}), 400
        
        if not is_valid_email(email):
            return jsonify({"message_email": "Invalid email format."}), 400

        findDupName = userCollection.find_one({"username": username})
        findDupEmail = userCollection.find_one({"email": email})
        
        if findDupEmail:
            return jsonify({"message_email": "Email already in use. Please choose another one."}), 400
        
        if findDupName: 
            return jsonify({"message_username": "Username already exist. Please try another name."}), 400
        
        if password != passwordConfirm:
            return jsonify({"message_password": "Passwords do not match."}), 400
        
        
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)

        userCollection.insert_one({
            "email": email,
            "username": username, 
            "password": hashed_password, 
            "salt": salt, 
            "following": [],
            'followers': [],
            "profile_image": "public/image/mainProfile.png",
            "direct_messages": [],
            "verified": False,
            })
        # print(userCollection)
        return make_response()
    except Exception as e:  # Catch the exception and print it for debugging
        print(e)
        return jsonify({'message': 'An error occurred'}), 500

def returningUser():
    try:
        requestData = request.get_json()
        username = requestData.get("username")
        userPassword = requestData.get("password")

        if not username or not userPassword:
            return jsonify({'message_required': 'Username and password are required.'}), 400

        userRecord = userCollection.find_one({"username": username})

        if userRecord:
            salt = userRecord["salt"]
            hashedPassword = bcrypt.hashpw(userPassword.encode(), salt)

            if hashedPassword == userRecord["password"]:
                token = secrets.token_hex()
                hashedToken = hashlib.sha256(token.encode("utf-8")).hexdigest()
                response = make_response()
                response.set_cookie("auth_tok", value=token, max_age=3600, httponly=True)
                userCollection.update_one(
                    {"username": username},
                    {"$set": {"token": hashedToken}}
                )
                return response
            else:
                return jsonify({'message_invalid': 'Invalid username or password.'}), 400
        else:
            return jsonify({'message_invalid': 'Invalid username or password.'}), 400

    except Exception as e: 
        return jsonify({'message': 'An error occurred'}), 500

def register():
    try:
        return sendResponse(filenamedir="./build/index.html", path=None, mimetype="text/html", xcontenttypeoptions="nosniff", makeresponse=True)
    except Exception:
        return page_not_found()

def getUser():
    try:
        token = request.cookies.get("auth_tok")
        if token:
            hashed_token = hashlib.sha256(token.encode("utf-8")).hexdigest()
            user = userCollection.find_one({"token": hashed_token})

            if user:
                channels = list(channelCollection.find({}))
                user["channels"] = channels
                return json_util.dumps(user)
            else:
                return redirect("/login"), 302
        else:
            return jsonify({"message": "No token found"}), 401
    except Exception as e:
        error_message = f"An error occurred: {e}"
        print("***********ERROR**:", error_message)
        return jsonify({"message": error_message}), 500
