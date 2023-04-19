# Project_3
Training a SBDH NER Model

Step 1: Download the train.csv file from the github link provided and the noteevents.csv file from the MIMIC 3 database.

Step 2: Merge the train.csv file and the ‘text’ column of the noteevents data(code provided on github - Merge_data)

Step 3: Associate the extracted text from the noteevents to the start and end label (code provided on github - Associate_data)

Step 4: Create a dictionary of named entity annotations (code on github - Dictionary_data)

Step 5: I converted the dictionary to a json file and stored it for reuse

Step 6: Train the data, I did not make use of any config file, instead I wrote a python code with a function to train the model (code on github - Train_data)

Step 7: Calculate the evaluation matrix using Spacy’s built-in functions (code on github - Evaluate_data)

