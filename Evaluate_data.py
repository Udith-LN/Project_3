import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
import json

# Load the trained NER model
nlp = spacy.load('trained_ner_model')

t_df = pd.read_csv('MIMIC_SBDH_with_text.csv', nrows=100, usecols=['text','sbdh'])

# Load the evaluation data
eval_data = train_data 

# Get reference entities with their annotations from the evaluation dataset
predicted_entities = []
for text, _ in eval_data:  # Extract only the text from the tuple
    doc = nlp(text)
    predicted_entities.append([(ent.text, ent.label_) for ent in doc.ents])

def get_evaluation_data_with_annotations(data_path):
    df = pd.read_csv('MIMIC_SBDH_with_text.csv')

    # Extract evaluation data with annotations
    evaluation_data = []
    for index, row in df.iterrows():
        text = row['text']  # Extract text from the dataset
        annotations = {}  # Initialize an empty dictionary for annotations

        # Extract annotations from the dataset and add them to the dictionary
        annotations['entities'] = []
        for keyword, label in row[['text', 'sbdh']].items():
            if not pd.isna(keyword) and not pd.isna(label):
                start = text.find(keyword)  # Find the start index of the keyword in the text
                end = start + len(keyword)  # Calculate the end index of the keyword
                if start != -1:
                    annotations['entities'].append((start, end, label))  

        # Add the text and annotations to the evaluation data list
        evaluation_data.append((text, annotations))

    return evaluation_data


# Check if predicted_entities is not empty
if not predicted_entities:
    print("No entities were predicted by the trained NER model.")
else:
    # Get reference entities with their annotations from the evaluation dataset
    reference_entities = get_evaluation_data_with_annotations(t_df)  

    # Print contents of the lists for debugging
    print("predicted_entities:", predicted_entities)
    print("reference_entities:", reference_entities)

    # Convert predicted and reference entities to flat lists
    predicted_flat = [item for sublist in predicted_entities for item in sublist]
    reference_flat = [item for sublist in reference_entities for item in sublist]

    # Print contents of the flat lists for debugging
    print("predicted_flat:", predicted_flat)
    print("reference_flat:", reference_flat)

    # Extract entity texts and labels separately
    predicted_texts, predicted_labels = zip(*predicted_flat)
    reference_texts, reference_labels = zip(*reference_flat)

    # Calculate evaluation metrics
    precision = precision_score(reference_labels, predicted_labels, average='weighted')
    recall = recall_score(reference_labels, predicted_labels, average='weighted')
    f1_score = f1_score(reference_labels, predicted_labels, average='weighted')

    # Print the evaluation metrics
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1_score:.2f}')
