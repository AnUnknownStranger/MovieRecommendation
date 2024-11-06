from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from util import csvParser
import torch
import torch.nn as nn

# Trainable neural network model (optional)
class MovieRecommender(nn.Module):
    def __init__(self, input_dim):
        super(MovieRecommender, self).__init__()
        self.fc = nn.Linear(input_dim, 128)  # Simple linear layer
    
    def forward(self, x):
        return self.fc(x)

def Recommender(Movie_idx, movie_factors, number, data):
    
    # Compute cosine similarity
    similarities = cosine_similarity(movie_factors[Movie_idx].unsqueeze(0), movie_factors).flatten()

    # Sort by similarity score in descending order
    rec_indices = similarities.argsort()[::-1][1:number+1]  
    #remove the original title
    rec_indices = [idx for idx in rec_indices if data.iloc[idx]['title'] != data.iloc[Movie_idx]['title']]
    return rec_indices

#Process data to Pytorch Format
def processData(data):
    tfidf = TfidfVectorizer(max_features=500)
    #convert content describtion to numeric values
    encoded_content = tfidf.fit_transform(data["content"]).toarray()
    
    #encoding genres and companies for ML purposes
    encoded_genres = MultiLabelBinarizer().fit_transform(data['genres'])
    encoded_companies = MultiLabelBinarizer().fit_transform(data['production_companies'])

    #Change the all factors to pytorch format
    genres = torch.tensor(encoded_genres, dtype=torch.float32)
    companies = torch.tensor(encoded_companies, dtype=torch.float32)
    content = torch.tensor(encoded_content, dtype=torch.float32)
    movie_factors = torch.cat([genres, companies, content], dim=1)
    return movie_factors

#Make top movie recommendations recommendations based on the vote average and popularity
def TopRecommendations(data,recommendations,num):
    #Find the corresponding movies
    recommended_movies = data.iloc[recommendations].copy()
    recommended_movies['Score'] = (0.5*recommended_movies['vote_average']) + (0.5*recommended_movies['popularity'])

    recommendations = recommended_movies.sort_values(by='Score', ascending=False).head(num)['title'].to_numpy()

    return recommendations
    

def MakeRecommendation(movie_title):
    #Obatin data by parsing csv file
    data = csvParser.ParseMovieData()

    movie_idx = title_idx_conversion(data,movie_title)
    if movie_idx == None:
        return None
    #Parse movie data
    movie_factors = processData(data)
    # Save the precomputed movie_factors
    torch.save(movie_factors, "Movie_factor.pt")

    # Later, we load the precomputed data and generate recommendations
    precomputed_factors = torch.load("Movie_factor.pt")
    recommended_indices = Recommender(Movie_idx=movie_idx, movie_factors=precomputed_factors, number=10, data=data)

    recommended_movies = TopRecommendations(data,recommended_indices,5)
    
    return recommended_movies

#Find the movie Title
def title_idx_conversion(data, title):
    indices = data.index[data['title'].str.lower() == title.lower()].tolist()
    return indices[0] if indices else None
    
    
def main():
    print(MakeRecommendation("Finding nemo"))

if __name__ == "__main__":
    main()
