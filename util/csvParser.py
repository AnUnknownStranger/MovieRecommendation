import pandas as pd

def ParseMovieData():
    csv_file_path = 'Data/movies_metadata.csv'
    
    # Load the dataset with low_memory=False to avoid DtypeWarning
    df = pd.read_csv(csv_file_path, low_memory=False)
    
    # Ensure 'vote_average' and 'vote_count' columns are numeric
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    
    content_df = df[['title', 'genres', 'overview', 'tagline', 'production_companies']]

    return content_df
ParseMovieData()
