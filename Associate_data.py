# Associate the extracted text to the label
import pandas as pd 
 
train_df = pd.read_csv('MIMIC_SBDH_with_text.csv')

train_df['extracted_text'] = train_df.apply(lambda row: row['text'][row['start']:row['end']], axis=1)
train_df['sbdh'] = train_df['sbdh'] 
 
print(train_df.head());
