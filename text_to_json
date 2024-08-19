import spacy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

data = pd.read_csv('/Users/rishabhsingh/Downloads/TUmail.txt')

#columns = ['Body','From','To']
dataset = pd.DataFrame()

count = 0
for index, row in data.iterrows():

    # Append the data to a temporary DataFrame
    temp_df = pd.DataFrame([row])

    # Concatenate the temporary DataFrame with the main dataset
    dataset = pd.concat([dataset, temp_df])
    if index == 20:
        break


dataset_json = dataset.to_json(orient='records')

with open('/Users/rishabhsingh/Downloads/tumail.json', 'w') as file:
    file.write(dataset_json)



print(data.head())
