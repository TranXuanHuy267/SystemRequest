import numpy as np
from numpy.linalg import norm
from copy import deepcopy
import pandas as pd
import re
import json

def save_data_to_file(data):
    with open("/Users/tranxuanhuy/Documents/Viettel Digital Talent/Chatbot/Code/demo/save_all/save_all.json", "a") as file:
        file.write(json.dumps(data) + "\n")

def best_chartName(output_getListChart_api):
    if len(output_getListChart_api) > 0:
        chart_id = output_getListChart_api[0]['chartId']
    else:
        chart_id = 'NULL'
    return chart_id

def best_cmd_to_url(cmd_stat, cmd2url):
    url = cmd2url[list(cmd_stat.keys())[0]]
    return url

def process_dashboardName(name):
    result = re.sub(r'[\d.,]+', '', name)
    result = result.replace("_", " ").replace("  ", " ").replace("  ", " ").strip().lower()
    return result

def best_chitieu2id(chitieu_stat, chitieufull2serviceid, chitieu2serviceid):
    service_ids = []
    for i in range(50):
        chitieu_real = chitieufull2serviceid[list(chitieu_stat.keys())[i]]
        service_id = chitieu2serviceid[chitieu_real]
        service_ids.append(service_id)
    return list(set(service_ids))

def from_chitieu_to_service_id(model_matching, chi_tieu, chitieu2serviceid, db_df_2, chitieufull2serviceid):
    if chi_tieu not in chitieu2serviceid.keys():
        chitieu_stat = encode_input_and_return_top_n_chitieu(model_matching, chi_tieu, db_df_2, 50)
        service_ids = best_chitieu2id(chitieu_stat, chitieufull2serviceid, chitieu2serviceid)
    else:
        service_ids = [chitieu2serviceid[chi_tieu]]
        chitieu_stat = encode_input_and_return_top_n_chitieu(model_matching, chi_tieu, db_df_2, 50)
        service_ids+=best_chitieu2id(chitieu_stat, chitieufull2serviceid, chitieu2serviceid)
    return service_ids

def encode_database(db_in, model_matching):
    df = pd.DataFrame(list(zip(db_in, [[]]*len(db_in))), columns=["Câu lệnh có sẵn", "Embedding"])
    for i, func in enumerate(db_in):
        embedding2 = model_matching.encode(func)
        df["Embedding"].loc[i] = embedding2
    return df

def from_donvi_to_gPC(don_vi, areacode2groupprofilecode):
    if don_vi not in areacode2groupprofilecode.keys():
        return []
    else:
        gPCs = areacode2groupprofilecode[don_vi]
    return gPCs

def best_dashboardName(output_getDashboard_api, chi_tieu, don_vi, model_matching_default, areacode2areacodeText):
    if len(output_getDashboard_api) > 0:
        input_user = areacode2areacodeText[don_vi].lower() if chi_tieu=='khác' else chi_tieu
        embedding1 = model_matching_default.encode(input_user)
        best_i = -1
        best_score = -1
        for i in range(len(output_getDashboard_api)):
            db_name = process_dashboardName(output_getDashboard_api[i]['dashboardName'])
            embedding2 = model_matching_default.encode(db_name)
            output_getDashboard_api[i]['score'] = cosinesimilarity(embedding1, embedding2)
            if output_getDashboard_api[i]['score'] > best_score:
                best_i = i
                best_score = output_getDashboard_api[i]['score']
        best_dashboard_id = output_getDashboard_api[best_i]['dashboardId']
    else:
        best_dashboard_id = 'NULL'
    return best_dashboard_id

def encode_input_and_return_top_n_chitieu(model_matching, input_in, db_dff, top_k):
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
    output = {db_df_in['Câu lệnh có sẵn'][i].strip(): round(db_df_in['Điểm'][i].item(), 2) for i in ids}
    return output

def encode_input_and_return_top_n(model_matching, input_in, db_dff, top_k, new2oldmatching):
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

def cosinesimilarity(vector1, vector2):
    cosine = np.dot(vector1, vector2)/(norm(vector1)*norm(vector2))
    return cosine

def time2date(input):
    chu_ky_thoi_gian = input['CHU KỲ THỜI GIAN']
    thu = input['THỨ']
    ngay = input['NGÀY']
    tuan = input['TUẦN']
    thang = input['THÁNG']
    quy = input['QUÝ']
    nam = input['NĂM']
    import datetime
    import calendar
    from dateutil.relativedelta import relativedelta
    current_date = datetime.date.today()
    output = 'error'
    if chu_ky_thoi_gian=='ngày':
# hôm kia
        if ngay=='hôm kia':
            output = current_date - datetime.timedelta(days=2)
# hôm qua
        elif ngay=='hôm qua':
            output = current_date - datetime.timedelta(days=1)
# hôm nay
        elif ngay=='hôm nay':
            output = current_date
# ngày mai
        elif ngay=='mai':
            output = current_date + datetime.timedelta(days=1)
# ngày kia
        elif ngay=='kia':
            output = current_date + datetime.timedelta(days=2)
# đầu
        elif ngay=='đầu':
            if thang=='trước':
                needed_thang = 12 if current_date.month==1 else current_date.month-1
                needed_nam = current_date.year-1 if current_date.month==1 else current_date.year
                output = datetime.date(
                    needed_nam,
                    needed_thang,
                    1
                )
            elif thang=='này':
                needed_thang = current_date.month
                needed_nam = current_date.year
                output = datetime.date(
                    needed_nam,
                    needed_thang,
                    1
                )
            elif thang=='sau':
                needed_thang = 1 if current_date.month==12 else current_date.month+1
                needed_nam = current_date.year+1 if current_date.month==12 else current_date.year
                output = datetime.date(
                    needed_nam,
                    needed_thang,
                    1
                )
            else:
                output = current_date
# cuối
        elif ngay=='cuối':
            if thang=='trước':
                needed_thang = 12 if current_date.month==1 else current_date.month-1
                needed_nam = current_date.year-1 if current_date.month==1 else current_date.year
                output = datetime.date(
                    needed_nam,
                    needed_thang,
                    calendar.monthrange(needed_nam, needed_thang)[1]
                )
            elif thang=='này':
                needed_thang = current_date.month
                needed_nam = current_date.year
                output = datetime.date(
                    needed_nam,
                    needed_thang,
                    calendar.monthrange(needed_nam, needed_thang)[1]
                )
            elif thang=='sau':
                needed_thang = 1 if current_date.month==12 else current_date.month+1
                needed_nam = current_date.year+1 if current_date.month==12 else current_date.year
                output = datetime.date(
                    needed_nam,
                    needed_thang,
                    calendar.monthrange(needed_nam, needed_thang)[1]
                )
            else:
                output = current_date
# xxx
        elif ngay.isdigit():
            ngay = int(ngay)
            if thang.isdigit():
                thang = int(thang)
                if nam.isdigit():
                    nam = int(nam)
                    try:
                        output = datetime.date(nam, thang, ngay)
                    except ValueError:
                        output = current_date
                else:
                    try:
                        output = datetime.date(current_date.year, thang, ngay)
                    except ValueError:
                        output = current_date
            else:
                try:
                    output = datetime.date(current_date.year, current_date.month, ngay)
                except ValueError:
                    output = current_date
        elif ngay=='khác':
# thứ hai tuần sau
# thứ hai tuần này
# thứ hai tuần trước
# thứ hai tuần gần nhất
# thứ hai
            if thu != "khác":
                if tuan=='khác' or tuan=='gần nhất':
                    current_thu = current_date.weekday()+2
                    input_thu = 8 if thu=='chủ nhật' else int(thu)
                    if input_thu < current_thu:
                        daydelta = current_thu - input_thu
                        output = current_date - datetime.timedelta(days=daydelta)
                    elif input_thu == current_thu:
                        output = current_date - datetime.timedelta(days=7)
                    else:
                        daydelta = input_thu - current_thu
                        output = current_date - datetime.timedelta(days=7-daydelta)
                elif tuan=='này':
                    current_thu = current_date.weekday()+2
                    input_thu = 8 if thu=='chủ nhật' else int(thu)
                    if input_thu < current_thu:
                        daydelta = current_thu - input_thu
                        output = current_date - datetime.timedelta(days=daydelta)
                    elif input_thu == current_thu:
                        output = current_date
                    else:
                        daydelta = input_thu - current_thu
                        output = current_date + datetime.timedelta(days=daydelta)
                elif tuan=='trước':
                    current_thu = current_date.weekday()+2
                    input_thu = 8 if thu=='chủ nhật' else int(thu)
                    if input_thu < current_thu:
                        daydelta = current_thu - input_thu + 7
                        output = current_date - datetime.timedelta(days=daydelta)
                    elif input_thu == current_thu:
                        output = current_date - datetime.timedelta(days=7)
                    else:
                        daydelta = input_thu - current_thu
                        output = current_date - datetime.timedelta(days=7-daydelta)
                elif tuan=='sau':
                    current_thu = current_date.weekday()+2
                    input_thu = 8 if thu=='chủ nhật' else int(thu)
                    if input_thu < current_thu:
                        daydelta = 7 - (current_thu - input_thu)
                        output = current_date + datetime.timedelta(days=daydelta)
                    elif input_thu == current_thu:
                        output = current_date + datetime.timedelta(days=7)
                    else:
                        daydelta = input_thu - current_thu
                        output = current_date + datetime.timedelta(days=7+daydelta)
            elif thu=='khác':
                current_thu = current_date.weekday()
                if tuan=='trước' or tuan=='gần nhất': # ngày cuối tuần
                    daydelta = current_thu+1
                    output = current_date - datetime.timedelta(days=daydelta)
                elif tuan=='này':
                    daydelta = 6 - current_thu
                    if daydelta > 0:
                        output = current_date + datetime.timedelta(days=daydelta)
                    else:
                        output = current_date
                elif tuan=='sau':
                    daydelta = 13 - current_thu
                    output = current_date + datetime.timedelta(days=daydelta)
                else:
                    output = current_date
        else: # return default for ngay
            output = current_date
    elif chu_ky_thoi_gian=='tháng': # ngày cuối tháng
        if thu != 'khác' or ngay != 'khác' or tuan != 'khác':
            needed_thang = 12 if current_date.month==1 else current_date.month-1
            needed_nam = current_date.year-1 if current_date.month==1 else current_date.year
            output = datetime.date(
                needed_nam,
                needed_thang,
                calendar.monthrange(needed_nam, needed_thang)[1]
            )

        elif thang=='trước':
            needed_thang = 12 if current_date.month==1 else current_date.month-1
            needed_nam = current_date.year-1 if current_date.month==1 else current_date.year
            output = datetime.date(
                needed_nam,
                needed_thang,
                calendar.monthrange(needed_nam, needed_thang)[1]
            )
        elif thang=='này':
            needed_thang = current_date.month
            needed_nam = current_date.year
            output = datetime.date(
                needed_nam,
                needed_thang,
                calendar.monthrange(needed_nam, needed_thang)[1]
            )
        elif thang=='sau':
            needed_thang = 1 if current_date.month==12 else current_date.month+1
            needed_nam = current_date.year+1 if current_date.month==12 else current_date.year
            output = datetime.date(
                needed_nam,
                needed_thang,
                calendar.monthrange(needed_nam, needed_thang)[1]
            )
        elif thang.isdigit():
            thang = int(thang)
            if nam.isdigit():
                nam = int(nam)
                output = datetime.date(
                    int(nam),
                    int(thang),
                    calendar.monthrange(int(nam), int(thang))[1]
                )
            else:
                if thang > current_date.month:
                    output = datetime.date(
                        current_date.year-1,
                        thang,
                        calendar.monthrange(current_date.year-1, thang)[1]
                    )
                else:
                    output = datetime.date(
                        current_date.year,
                        thang,
                        calendar.monthrange(current_date.year, thang)[1]
                    )
        elif thang=='khác':
            needed_thang = 12 if current_date.month==1 else current_date.month-1
            needed_nam = current_date.year-1 if current_date.month==1 else current_date.year
            output = datetime.date(
                needed_nam,
                needed_thang,
                calendar.monthrange(needed_nam, needed_thang)[1]
            )
        else: # ngay cuoi thang truoc
            needed_thang = 12 if current_date.month==1 else current_date.month-1
            needed_nam = current_date.year-1 if current_date.month==1 else current_date.year
            output = datetime.date(
                needed_nam,
                needed_thang,
                calendar.monthrange(needed_nam, needed_thang)[1]
            )
    elif chu_ky_thoi_gian=='quý':
        if quy in ['1', '2', '3', '4', 'I', 'II', 'III', 'IV']:
            output = current_date
        else:
            output = current_date
    elif chu_ky_thoi_gian=='năm':
        if nam.isdigit():
            output = current_date
        else:
            output = current_date
    elif chu_ky_thoi_gian=='khác':
        output = current_date
    else:
        output = current_date
    if output > current_date - datetime.timedelta(days=2):
        output = current_date - datetime.timedelta(days=2)
    return output