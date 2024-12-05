from pymongo import MongoClient

client = MongoClient("mongo")
db = client["MovieRecommendations"]
movieCollection = db["movies_metadata"]
userCollection = db["user"]
