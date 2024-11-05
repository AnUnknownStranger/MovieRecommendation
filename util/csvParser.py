import pandas as pd
import ast


def parse_genres(x):
    try:
        genres = ast.literal_eval(x)
        genres_list = []
        for g in genres:
            genres_list.append(g['name'])
        # concate into a single string instead of a list of genres
        result = ', '.join(genres_list)
        return result
    except (TypeError):
        return ""
def ParseMovieData():
    csv_file_path = 'Data/movies_metadata.csv'
    
    # Load the dataset with low_memory=False to avoid DtypeWarning
    df = pd.read_csv(csv_file_path, low_memory=False)
    
    # Ensure 'vote_average' and 'vote_count' columns are numeric
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    
    content_df = df[['title', 'genres', 'overview', 'tagline', 'production_companies']]
    content_df = content_df.fillna('')
    content_df['genres'] = content_df['genres'].apply(parse_genres)
    content_df['content'] = content_df['overview'] + ' ' + content_df['genres'] + ' ' + content_df['overview'] + ' ' + content_df['tagline'] + ' ' + content_df['production_companies']
    return content_df
ParseMovieData()
