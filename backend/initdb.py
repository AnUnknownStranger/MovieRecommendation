from config import moveCollection
import pandas as pd
import json

#used to populate the databse with initial csv data
csv_file_path = "Data/movies_metadata.csv"
df = pd.read_csv(csv_file_path)

data = json.loads(df.to_json(orient="records"))
moveCollection.insert_many(data)

