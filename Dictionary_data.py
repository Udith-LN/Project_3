import spacy
import random

# Load spacy model
nlp = spacy.blank('en')

# Create an empty list to store the training data
train_data = []

# Iterate through the train_df DataFrame
for index, row in train_df.iterrows():
    # Extract the relevant information from the row
    text = row['extracted_text']
    start = row['start']
    end = row['end']
    label = row['sbdh']
    
    # Create a dictionary of named entity annotations
    entities = [(start, end, label)]
    
    # Append the sentence and entities as a tuple to the train_data list
    train_data.append((text, {"entities": entities}))

# Print the first few examples of the train_data
print(train_data[:5])
