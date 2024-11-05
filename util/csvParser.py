import pandas as pd

def ParseMovieData():
    csv_file_path = 'Data/movies_metadata.csv'
    df = pd.read_csv(csv_file_path)
    print(df)
    pass

ParseMovieData()