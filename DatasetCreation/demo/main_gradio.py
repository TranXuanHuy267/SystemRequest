import gradio as gr
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import namedtuple
from something import time2date, output2url
import numpy as np
from numpy.linalg import norm
from copy import deepcopy
import pandas as pd
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')

fields = ['device', 'model_name', 'max_source_length', 'max_target_length', 'beam_size']
params = namedtuple('params', field_names=fields)

args = params(
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_name='facebook/mbart-large-50-many-to-many-mmt',
    max_source_length=256,
    max_target_length=256,
    beam_size=1
)

# model = AutoModelForSeq2SeqLM.from_pretrained("Huy1432884/db_retrieval", use_auth_token="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")
model = AutoModelForSeq2SeqLM.from_pretrained("model")
model.to(args.device)
model.eval()

if "mbart" in args.model_name.lower():
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name, src_lang="vi_VN", tgt_lang="vi_VN")
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


def cosinesimilarity(vector1, vector2):
    cosine = np.dot(vector1, vector2)/(norm(vector1)*norm(vector2))
    return cosine

def encode_input_and_return_top_n(input_in, db_dff, top_k, new2oldmatching):
    embed1 = model_matching.encode(input_in)
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

def encode_database(db_in):
    df = pd.DataFrame(list(zip(db_in, [[]]*len(db_in))), columns=["Câu lệnh có sẵn", "Embedding"])
    for i, func in enumerate(db_in):
        embedding2 = model_matching.encode(func)
        df['Embedding'].loc[i] = embedding2
    return df

# model_matching = SentenceTransformer("Huy1432884/function_retrieval")
model_matching = SentenceTransformer("model_matching")
model_matching.eval()

with open('new2oldmatch.json', 'r') as openfile:
    new2oldmatch = json.load(openfile)
    new2oldmatch = {u.strip().lower(): v.strip() for u, v in new2oldmatch.items()}

database = [cmd.lower() for cmd in new2oldmatch.keys()]
db_df = encode_database(database)

def text_analysis(text):
    text = text.lower()

    inputs = tokenizer(
        [text],
        text_target=None,
        padding="longest",
        max_length=args.max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    for k, v in inputs.items():
        inputs[k] = v.to(args.device)


    if "mbart" in args.model_name:
        inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id["vi_VN"]

    outputs = model.generate(
        **inputs,
        max_length=args.max_target_length,
        num_beams=args.beam_size,
        early_stopping=True,
    )

    output_sentences = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    out = json.loads("{" + output_sentences[0] + "}")
    if out['LOẠI BIỂU ĐỒ']=='dashboard':
        if out['CHU KỲ THỜI GIAN']!='tháng':
            chu_ky_in = 'ngày'
        else:
            chu_ky_in = 'tháng'
        out['CHU KỲ THỜI GIAN']='ngày' if out['CHU KỲ THỜI GIAN'] not in ['ngày', 'tháng'] else out['CHU KỲ THỜI GIAN']
        check_dashboard = out['ĐƠN VỊ']+"_"+chu_ky_in
        out['DB URL'] = output2url[check_dashboard]
        out['DATE'] = str(time2date(out)).replace("-", "").replace("-", "")
        out['FINAL URL'] = "https://vsds.viettel.vn"+ out['DB URL'] + "?toDate=" + out['DATE']
        show = {i: out[i] for i in ['LOẠI BIỂU ĐỒ', 'ĐƠN VỊ', 'CHU KỲ THỜI GIAN', 'DB URL', 'DATE', 'FINAL URL']}
    elif out['LOẠI BIỂU ĐỒ']=='biểu đồ':
        show = {i: out[i] for i in ['LOẠI BIỂU ĐỒ', 'ĐƠN VỊ', 'CHU KỲ THỜI GIAN']}
    else:
        result = encode_input_and_return_top_n(text, db_df, 3, new2oldmatch)
        return result
    return show

demo = gr.Interface(
    text_analysis,
    gr.Textbox(placeholder="Enter sentence here..."),
    ["json"],
    examples=[
        ["Mở dashboard vtc ngày hôm qua"],
        ["Mở biểu đồ cột td ngày này"],
        ["Hãy mở biểu đồ cơ cấu của tập đoàn trong ngày hôm nay"],
        ["Tháng này, vtc cần tôi mở biểu đồ rank để cập nhật danh sách khách hàng"],
        ["Các thông số NAT ngày hôm qua đã được ghi nhận trên đát bọt"],
        ["Hôm nay hãy mở của Viettel tt không gian mạng Viettel vtcc để kiểm tra"],
        ["Mở DB CTM ngày gốc"],
        ["Tôi đã sử dụng Dashboard để truy cập thông tin qti vào ngày hôm nay"],
        ["Trưởng phòng đã ra lệnh mở biểu đồ kết hợp đường và cột cho toàn tập đoàn vào hôm nay"]
    ],
)


demo.launch()