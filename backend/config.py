from flask import Flask

from pymongo import MongoClient

#! "localhost" for server.py, "mongo" for docker

client = MongoClient("mongodb://localhost:27017/")
db = client["MovieRecommendations"]
moveCollection = db["movies_metadata"]
userCollection = db["user"]

app = Flask(__name__)