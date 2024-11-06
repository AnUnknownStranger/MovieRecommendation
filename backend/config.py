from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["MovieRecommendations"]
moveCollection = db["movies_metadata"]
userCollection = db["user"]
