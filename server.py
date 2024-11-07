from flask import Flask, jsonify, make_response
from flask_cors import CORS

from backend.auth_routes import *
from backend.config import *

from bson import ObjectId 
import hashlib


app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

def get_user_from_token(auth_token):

    hashed_token = hashlib.sha256(auth_token.encode("utf-8")).hexdigest()
    return userCollection.find_one({"token": hashed_token})

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


# Flask route for title search
from bson import ObjectId

@app.route('/search/title', methods=['GET'])
def search_by_title():
    # Ensure the text index on the title is created only once
    if "title_text" not in movieCollection.index_information():
        movieCollection.create_index([("title", "text")], name="title_text")

    title_query = request.args.get('q', '')

    # Step 1: Look for exact match first
    exact_matches = list(movieCollection.find(
        {"title": {"$regex": f"^{title_query}$", "$options": "i"}},
        {"_id": 1, "title": 1}
    ))

    # Step 2: Perform a text search for broader matches
    text_matches = list(movieCollection.find(
        {"$text": {"$search": title_query}},
        {"_id": 1, "title": 1, "score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(10 - len(exact_matches)))

    # Combine exact matches and text search results, prioritizing exact matches
    combined_results = exact_matches + text_matches
    movie_list = [{"id": str(movie["_id"]), "title": movie["title"]} for movie in combined_results]
    
    return jsonify(movie_list)



@app.route('/api/profile', methods=['GET'])
def get_profile():
    auth_token = request.cookies.get('authToken')
    
    if not auth_token:
        return jsonify({"error": "Unauthorized"}), 401

    # Hash the token to match with the database-stored hashed token
    hashed_token = hashlib.sha256(auth_token.encode("utf-8")).hexdigest()

    # Fetch the user by the hashed token
    user_data = userCollection.find_one({"token": hashed_token})

    if not user_data:
        return jsonify({"error": "User not found"}), 404

    # Fetch movie titles for liked and disliked movies
    liked_movie_ids = [ObjectId(movie_id) for movie_id in user_data.get("likes", [])]
    disliked_movie_ids = [ObjectId(movie_id) for movie_id in user_data.get("dislikes", [])]

    # Retrieve movies by IDs and get their titles
    liked_movies = movieCollection.find({"_id": {"$in": liked_movie_ids}})
    disliked_movies = movieCollection.find({"_id": {"$in": disliked_movie_ids}})

    # Convert cursor results to lists of movie titles
    likes = [movie["title"] for movie in liked_movies]
    dislikes = [movie["title"] for movie in disliked_movies]

    # Create profile response
    profile = {
        "name": user_data.get("username"),
        "likes": likes,
        "dislikes": dislikes
    }
    print(profile)

    return jsonify(profile)

@app.route('/api/like', methods=['POST'])
def like_movie():
    auth_token = request.cookies.get('authToken')
    if not auth_token:
        return jsonify({"error": "Unauthorized"}), 401

    user_data = get_user_from_token(auth_token)
    if not user_data:
        return jsonify({"error": "User not found"}), 404

    movie_id = request.json.get("movieId")
    if movie_id:
        # Ensure movie_id is an ObjectId
        movie_id = ObjectId(movie_id) if not isinstance(movie_id, ObjectId) else movie_id
        
        # Add movie ID to likes if not already liked, and remove from dislikes
        userCollection.update_one(
            {"_id": user_data["_id"]},
            {"$addToSet": {"likes": movie_id}, "$pull": {"dislikes": movie_id}}
        )
        return jsonify({"message": "Movie liked successfully"}), 200
    else:
        return jsonify({"error": "Invalid movie ID"}), 400

    
@app.route('/api/dislike', methods=['POST'])
def dislike_movie():
    auth_token = request.cookies.get('authToken')
    if not auth_token:
        return jsonify({"error": "Unauthorized"}), 401

    user_data = get_user_from_token(auth_token)
    if not user_data:
        return jsonify({"error": "User not found"}), 404

    movie_id = request.json.get("movieId")
    if movie_id:
        # Ensure movie_id is an ObjectId
        movie_id = ObjectId(movie_id) if not isinstance(movie_id, ObjectId) else movie_id
        
        # Add movie ID to dislikes if not already disliked, and remove from likes
        userCollection.update_one(
            {"_id": user_data["_id"]},
            {"$addToSet": {"dislikes": movie_id}, "$pull": {"likes": movie_id}}
        )
        return jsonify({"message": "Movie disliked successfully"}), 200
    else:
        return jsonify({"error": "Invalid movie ID"}), 400


@app.route('/api/movie/details', methods=['GET'])
def get_movie_details():
    movie_id = request.args.get('id')
    print(movie_id)
    if not movie_id:
        return jsonify({"error": "Movie ID is required"}), 400
    print("movie_id")
    try:
        # Convert movie_id to ObjectId and fetch the movie document
        movie = movieCollection.find_one({"_id": ObjectId(movie_id)})

        
        if not movie:
            return jsonify({"error": "Movie not found"}), 404

        # Prepare the response with detailed movie info
        movie_details = {
            "id": str(movie["_id"]),
            "title": movie.get("title", "No title available"),
            "overview": movie.get("overview", "No description available"),
            "releaseDate": movie.get("release_date", "Unknown"),
            "runtime": movie.get("runtime", "Unknown"),
            "genres": [genre["name"] for genre in movie.get("genres", []) if isinstance(genre, dict) and "name" in genre],
            "productionCompanies": [company["name"] for company in movie.get("production_companies", []) if isinstance(company, dict) and "name" in company],
            "popularity": movie.get("popularity", "Unknown"),
            "voteAverage": movie.get("vote_average", "Unknown"),
            "voteCount": movie.get("vote_count", "Unknown"),
        }

        return jsonify(movie_details), 200

    except Exception as e:
        print(f"Error fetching movie details: {e}")
        return jsonify({"error": "An error occurred while fetching movie details"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)



