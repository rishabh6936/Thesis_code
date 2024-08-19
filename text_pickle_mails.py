import spacy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

data = pd.read_csv('/Users/rishabhsingh/Downloads/TUmail.txt')

columns = ['Body','From','To']
dataset = pd.DataFrame(columns=columns)

count = 0
for index, row in data.iterrows():
    email_body = row['Text']
    sender = row['Von: (Name)']
    receiver = row['An: (Name)']
    subject = row['Betreff']

    # Append the data to a temporary DataFrame
    temp_df = pd.DataFrame({'Body': [email_body], 'From': [sender], 'To': [receiver], 'Subject': [subject]})

    # Concatenate the temporary DataFrame with the main dataset
    dataset = pd.concat([dataset, temp_df] ,ignore_index=True)
    if index == 20:
        break



file_path = '/Users/rishabhsingh/Mail_dataset/mail.parquet'
dataset.to_parquet(file_path)
dataset.to_pickle("/Users/rishabhsingh/Downloads/TU_mails.pkl")
print(data.head())
