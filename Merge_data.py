import pandas as pd
import psycopg2

# Load noteevents.csv data from PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,  # Default PostgreSQL port
    dbname="mimic",
    user="postgres",
    password="7077"
)

# Create a cursor object to interact with the database
cur = conn.cursor()

# Execute an SQL query to fetch the necessary data from noteevents table
cur.execute("SELECT * FROM mimiciii.noteevents;")
noteevents_list = cur.fetchall() 
noteevents_df = pd.DataFrame(noteevents_list)

# Load train.csv data from local file
train_df = pd.read_csv('O:\MIMIC3\Train.csv')
print(train_df.head())

# Merge noteevents.csv and train.csv data
merged_df = pd.merge(noteevents_df, train_df, on='row_id', how='inner')

# Extract relevant information based on start and stop index columns
merged_df['extracted_text'] = merged_df.apply(lambda row: row['text'][row['start']:row['end']], axis=1)
merged_df['label'] = merged_df['label'] # You can specify the appropriate column name from the train.csv file

# Preview the extracted information
print(merged_df.head())
cur.close()
conn.close()
