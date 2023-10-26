import gradio as gr
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import namedtuple
from something import time2date
import numpy as np
from numpy.linalg import norm
from copy import deepcopy
import pandas as pd
from sentence_transformers import SentenceTransformer
from api_getDashboard_call import call_getDashboard_api_2
from api_getListChart_call import call_getListChart_api_2

import warnings
warnings.filterwarnings('ignore')

fields = ['device', 'model_name', 'max_source_length', 'max_target_length', 'beam_size']
params = namedtuple('params', field_names=fields)

args = params(
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_name='facebook/mbart-large-50-many-to-many-mmt',
    max_source_length=256,
    max_target_length=512,
    beam_size=2
)

model = AutoModelForSeq2SeqLM.from_pretrained("model_text2table_3")
model.to(args.device)
model.eval()

if "mbart" in args.model_name.lower():
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

def from_donvi_to_gPC(don_vi):
    gPC = areacode2groupprofilecode[don_vi]
    return gPC

def best_dashboardName(output_getDashboard_api, chi_tieu, don_vi):
    if len(output_getDashboard_api) > 0:
        t = 0
        for item in output_getDashboard_api:
            if item['dashboardName'].strip().lower() == chi_tieu:
                t = 1
                best_dashboard_id = item['dashboardId']
                break
        if t == 0:
            h = 0
            for item in output_getDashboard_api:
                if item['dashboardName'].strip().lower() == don_vi:
                    h = 1
                    best_dashboard_id = item['dashboardId']
                    break
            if h == 0:
                best_dashboard_id = output_getDashboard_api[0]['dashboardId']
    else:
        best_dashboard_id = 'NULL'
    return best_dashboard_id

def from_chitieu_to_service_id(chi_tieu):
    service_id = chitieu2serviceid[chi_tieu]
    return service_id

def best_chartName(output_getListChart_api):
    if len(output_getListChart_api) > 0:
        chart_id = output_getListChart_api[0]['chartId']
    else:
        chart_id = 'NULL'
    return chart_id

def best_cmd_to_url(cmd_stat):
    url = cmd2url[list(cmd_stat.keys())[0]]
    return url

model_matching = SentenceTransformer("model_matching")
model_matching.eval()

with open('new2oldmatch.json', 'r') as openfile:
    new2oldmatch = json.load(openfile)
    new2oldmatch = {u.strip().lower(): v.strip() for u, v in new2oldmatch.items()}

with open('areacode2groupprofilecode.json', 'r') as openfile:
    areacode2groupprofilecode = json.load(openfile)

with open('chitieu2serviceid.json', 'r') as openfile:
    chitieu2serviceid = json.load(openfile)

with open('cmd2url.json', 'r') as openfile:
    cmd2url = json.load(openfile)

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
    output_api = {
        'command': 'OPEN_URL',
        'URL': 'NULL',
        'chartId': 'NULL',
        'toDate': 'NULL' 
    }

    if out['LOẠI BIỂU ĐỒ']=='dashboard' or \
        (out['ĐƠN VỊ'] != 'Khác' and out['LOẠI BIỂU ĐỒ']=='khác'):
        if out['CHU KỲ THỜI GIAN']!='tháng':
            chu_ky_thoi_gian = 'NGAY'
        else:
            chu_ky_thoi_gian = 'THANG'
        don_vi = out['ĐƠN VỊ']
        gPC = from_donvi_to_gPC(don_vi)
        chi_tieu = out['CHỈ TIÊU']

        data = {
            'token': 'eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNLY0M7M0sNBRSq0oAAuYGJgaG4MEiksS09Kc81NSgZqMgOKGxkBtKakFJZ4pIBFTEyC3tDi1CMaF8fMSc5G01AIAgG6fSngAAAA.mvTqCOqk-VCO0ua7YUiKIWJfgXzKkQC_4UhzMFrCzdA',
            'groupProfileCode': gPC,
            'cycleTime': chu_ky_thoi_gian,
            'areaCode': don_vi,
            'ip': '10.61.142.22'
        }
        token = "eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNDA1sDQ3M9NRSq0ogApYmhmCBIpLEtPSnPNTUoGajMwsTQyNgdpSUgtKPFNAIqYmQG5pcWoRjAvj5yXmImmpBQDfAGr_eAAAAA.iU0Vka0zQCQMUb1I8U6ncOMh5YtrO3b7G4ZhMGYN5yk"
        output_getDashboard_api = call_getDashboard_api_2(data, token)
        dashboard_id = best_dashboardName(output_getDashboard_api, chi_tieu, don_vi)
        todate = str(time2date(out)).replace("-", "").replace("-", "")
        output_api['URL'] = f'/pages/screen/{dashboard_id}'
        output_api['toDate'] = str(todate)

    elif out['LOẠI BIỂU ĐỒ']!= 'khác':
        loai_bieu_do = out['LOẠI BIỂU ĐỒ']
        if out['CHU KỲ THỜI GIAN']=='ngày':
            chu_ky_thoi_gian = 'NGAY'
        elif out['CHU KỲ THỜI GIAN']=='tháng':
            chu_ky_thoi_gian = 'THANG'
        elif out['CHU KỲ THỜI GIAN']=='quý':
            chu_ky_thoi_gian = 'QUY'
        elif out['CHU KỲ THỜI GIAN']=='năm':
            chu_ky_thoi_gian = 'NAM'
        else:
            chu_ky_thoi_gian = 'NGAY'
        
        don_vi = out['ĐƠN VỊ']
        chi_tieu = out['CHỈ TIÊU']
        tieu_de = out['TIÊU ĐỀ']

        serviceId = from_chitieu_to_service_id(chi_tieu)

        data = {
            "ip": "10.61.142.22",
            "serviceId": serviceId,
            "areaCode": don_vi,
            "cycleTime": chu_ky_thoi_gian,
            "keyNameTitle": tieu_de,
            "typeChart": loai_bieu_do
        }

        token = "eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNDKzMLU0NdJRSq0ogAiYmxsZgwSKSxLT0pzzU1KBmozMLE0MjYHaUlILSjxTQCKmJkBuaXFqEYwL4-cl5iJpqQUAoOBjYXgAAAA.0jEJ8Y7YQ6lin1wG9WbeCn1La3nI1URlMyURbhbFyOI"
        output_getListChart_api = call_getListChart_api_2(data, token)
        chart_id = best_chartName(output_getListChart_api)
        todate = str(time2date(out)).replace("-", "").replace("-", "")
        output_api['command'] = 'OPEN_CHART'
        output_api['chartId'] = chart_id
        output_api['toDate'] = str(todate)
    else:
        cmd_stat = encode_input_and_return_top_n(text, db_df, 1, new2oldmatch)
        url = best_cmd_to_url(cmd_stat)
        output_api['URL'] = url
    for key in out.keys():
        output_api[key] = out[key]
    
    return output_api

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