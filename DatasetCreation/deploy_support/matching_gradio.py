import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
import json
from numpy.linalg import norm
import gradio as gr
from sentence_transformers import SentenceTransformer

# necessary function
def cosinesimilarity(vector1, vector2):
    cosine = np.dot(vector1, vector2)/(norm(vector1)*norm(vector2))
    return cosine

def encode_input_and_return_top_n(input_in, db_dff, top_k, new2oldmatching):
    embed1 = model.encode(input_in)
    scores = []
    db_df_in = deepcopy(db_dff)
    db_in = list(set(db_df_in['Câu lệnh có sẵn'].tolist()))
    for i, func in enumerate(db_in):
        embed2 = db_df_in['Embedding'].loc[i]
        scores.append(round(cosinesimilarity(embed1, embed2), 3))
    db_df_in["Điểm"] = scores
    db_df_in.sort_values(by=['Điểm'], inplace=True, ascending=False)
    ids = db_df_in[:top_k].index.tolist()
    output = {new2oldmatching[db_df_in['Câu lệnh có sẵn'][i].strip()]: round(db_df_in['Điểm'][i].item(), 2) for i in ids}
    return output

def image_classifier(Input):
    inputt = Input.lower()
    result = encode_input_and_return_top_n(inputt, db_df, 3, new2oldmatch)
    return result
  
def encode_database(db_in):
    df = pd.DataFrame(list(zip(db_in, [[]]*len(db_in))), columns=["Câu lệnh có sẵn", "Embedding"])
    for i, func in tqdm(enumerate(db_in)):
        embedding2 = model.encode(func)
        df['Embedding'].loc[i] = embedding2
    else:
        print()
        print("Encode database successfully")
    return df
    
model = SentenceTransformer("Huy1432884/function_retrieval")
model.eval()

with open('new2oldmatch.json', 'r') as openfile:
    new2oldmatch = json.load(openfile)
    new2oldmatch = {u.strip().lower(): v.strip() for u, v in new2oldmatch.items()}

database = [cmd.lower() for cmd in new2oldmatch.keys()]
db_df = encode_database(database)

demo = gr.Interface(fn=image_classifier, inputs="text", outputs="label")
demo.launch()