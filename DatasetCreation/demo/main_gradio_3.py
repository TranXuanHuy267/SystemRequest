import gradio as gr
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import namedtuple
from something import time2date, encode_input_and_return_top_n, best_chartName, best_cmd_to_url, encode_database, \
    from_donvi_to_gPC, best_dashboardName, from_chitieu_to_service_id, save_data_to_file
from sentence_transformers import SentenceTransformer
from api_getDashboard_call import call_getDashboard_api_2
from api_getListChart_call import call_getListChart_api_2
import time

import warnings
warnings.filterwarnings('ignore')

fields = ['device', 'model_name', 'max_source_length', 'max_target_length', 'beam_size']
params = namedtuple('params', field_names=fields)

args = params(
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_name='facebook/mbart-large-50-many-to-many-mmt',
    max_source_length=256,
    max_target_length=512,
    beam_size=5
)

model = AutoModelForSeq2SeqLM.from_pretrained("model_text2table_5")
model.to(args.device)
model.eval()

if "mbart" in args.model_name.lower():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model_matching = SentenceTransformer("model_matching")
model_matching.eval()

model_matching_default = SentenceTransformer("model_matching_default")
model_matching_default.eval()

with open('data/new2oldmatch.json', 'r') as openfile:
    new2oldmatch = json.load(openfile)
    new2oldmatch = {u.strip().lower(): v.strip() for u, v in new2oldmatch.items()}

with open('data/areacode2allgroupprofilecode.json', 'r') as openfile:
    areacode2groupprofilecode = json.load(openfile)

with open('data/chitieu2serviceid.json', 'r') as openfile:
    chitieu2serviceid = json.load(openfile)
    chitieu2serviceid = {u.strip().lower(): v for u, v in chitieu2serviceid.items()}


with open('data/cmd2url.json', 'r') as openfile:
    cmd2url = json.load(openfile)

with open('data/chitieufull2serviceid.json', 'r') as openfile:
    chitieufull2serviceid = json.load(openfile)
    chitieufull2serviceid = {u.strip().lower(): v.strip() for u, v in chitieufull2serviceid.items()}

with open('data/areacode2areacodeText.json', 'r') as openfile:
    areacode2areacodeText = json.load(openfile)

with open('/Users/tranxuanhuy/Documents/Viettel Digital Talent/Chatbot/Code/demo/data/bieudo2bieudoviethoa.json', 'r') as openfile:
    bieudo2bieudoviethoa = json.load(openfile)

with open('/Users/tranxuanhuy/Documents/Viettel Digital Talent/Chatbot/Code/demo/data/cmd2score.json', 'r') as openfile:
    cmd2score = json.load(openfile)

database = [cmd.lower() for cmd in new2oldmatch.keys()]
db_df = encode_database(database, model_matching)

database2 = [cmd.lower() for cmd in chitieufull2serviceid.keys()]
db_df2 = encode_database(database2, model_matching_default)

def text_analysis(text):
    logged = {}
    logged['input'] = text
    print('-'*30)
    print('Input:', text)
    text = text.lower()

    output_api = {
        'command': 'OPEN_URL',
        'URL': 'NULL',
        'chartId': 'NULL',
        'toDate': 'NULL',
        'alert': 'NULL'
    }
    cmd_stat = encode_input_and_return_top_n(model_matching, text, db_df, 1, new2oldmatch)
    print('Cmd:', list(cmd_stat.keys())[0])
    print('Score:', list(cmd_stat.values())[0])
    # if list(cmd_stat.values())[0] >= float(cmd2score[list(cmd_stat.keys())[0]]):

    if list(cmd_stat.values())[0] >= 0.95:
        url = best_cmd_to_url(cmd_stat, cmd2url)
        output_api['URL'] = url
        logged['output'] = output_api
        save_data_to_file(logged)
        return output_api

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

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_length=args.max_target_length,
        num_beams=args.beam_size,
        early_stopping=True,
    )
    end = time.time()
    print('Time:', end - start)

    output_sentences = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    out = json.loads("{" + output_sentences[0] + "}")
    output_api = {
        'command': 'OPEN_URL',
        'URL': 'NULL',
        'chartId': 'NULL',
        'toDate': 'NULL',
        'alert': 'NULL'
    }
    
    if out['LOẠI BIỂU ĐỒ']=='dashboard':
        if out['CHU KỲ THỜI GIAN']!='tháng':
            chu_ky_thoi_gian = 'NGAY'
        else:
            chu_ky_thoi_gian = 'THANG'

        todate = str(time2date(out)).replace("-", "").replace("-", "")
        output_api['toDate'] = str(todate)

        don_vi = out['ĐƠN VỊ']
        if out['ĐƠN VỊ'] == 'Khác':
            output_api['alert'] = 'Muốn mở dashboard, hãy nhập đơn vị.'
            logged['output'] = output_api
            save_data_to_file(logged)
            return output_api
        else:
            gPCs = from_donvi_to_gPC(don_vi, areacode2groupprofilecode)
            if gPCs == []:
                output_api['alert'] = 'Đơn vị này chưa có trong hệ thống này.'
                logged['output'] = output_api
                save_data_to_file(logged)
                return output_api
        
        chi_tieu = out['CHỈ TIÊU']
        output_getDashboard_apis = []
        for gPC in gPCs:
            data = {
                'token': 'eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNLY0M7M0sNBRSq0oAAuYGJgaG4MEiksS09Kc81NSgZqMgOKGxkBtKakFJZ4pIBFTEyC3tDi1CMaF8fMSc5G01AIAgG6fSngAAAA.mvTqCOqk-VCO0ua7YUiKIWJfgXzKkQC_4UhzMFrCzdA',
                'groupProfileCode': gPC,
                'cycleTime': chu_ky_thoi_gian,
                'areaCode': don_vi,
                'ip': '10.61.142.22'
            }
            token = "eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNDA1sDQ3M9NRSq0ogApYmhmCBIpLEtPSnPNTUoGajMwsTQyNgdpSUgtKPFNAIqYmQG5pcWoRjAvj5yXmImmpBQDfAGr_eAAAAA.iU0Vka0zQCQMUb1I8U6ncOMh5YtrO3b7G4ZhMGYN5yk"
            output_getDashboard_api = call_getDashboard_api_2(data, token)
            output_getDashboard_apis += output_getDashboard_api

        dashboard_id = best_dashboardName(output_getDashboard_apis, chi_tieu, don_vi, model_matching_default, areacode2areacodeText)

        if dashboard_id == 'NULL':
            output_api['alert'] = 'Hệ thống chưa có dữ liệu về đơn vị này.'
            logged['output'] = output_api
            save_data_to_file(logged)
            return output_api
        
        output_api['URL'] = f'/pages/screen/{dashboard_id}'
        
        # Additionally
        if output_api['alert'] == 'NULL' and out['CHU KỲ THỜI GIAN'] not in ['ngày', 'tháng', 'khác']:
            output_api['alert'] = 'Hệ thống chỉ có dashboard theo chu kỳ ngày hoặc tháng.'

    elif out['LOẠI BIỂU ĐỒ'] not in ['khác', 'dashboard'] and out['LOẠI BIỂU ĐỒ'] in bieudo2bieudoviethoa.keys():
        output_api['command'] = 'OPEN_CHART'
        todate = str(time2date(out)).replace("-", "").replace("-", "")
        output_api['toDate'] = str(todate)

        loai_bieu_do = out['LOẠI BIỂU ĐỒ']
        loai_bieu_do = bieudo2bieudoviethoa[loai_bieu_do]

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
        if out['ĐƠN VỊ'] == 'Khác':
            output_api['alert'] = 'Muốn mở biểu đồ, hãy nhập đơn vị.'
            logged['output'] = output_api
            save_data_to_file(logged)
            return output_api
        
        chi_tieu = out['CHỈ TIÊU']
        if chi_tieu == 'khác':
            for key, value in out.items():
                print(key+ ':', value)
            output_api['alert'] = 'Hãy nhập chỉ tiêu của biểu đồ.'
            logged['output'] = output_api
            save_data_to_file(logged)
            return output_api

        tieu_de = out['TIÊU ĐỀ']

        serviceIds = from_chitieu_to_service_id(model_matching, chi_tieu, chitieu2serviceid, db_df2, chitieufull2serviceid)
        for serviceId in serviceIds:
            data = {
                "ip": "10.61.142.22",
                "serviceId": serviceId,
                "areaCode": don_vi,
                "cycleTime": chu_ky_thoi_gian,
                "keyNameTitle": tieu_de,
                "typeChart": loai_bieu_do
            }
            
            data_2 = {
                "ip": "10.61.142.22",
                "serviceId": str(serviceId),
                "areaCode": don_vi.upper(),
                "cycleTime": chu_ky_thoi_gian,
                "keyNameTitle": '',
                "typeChart": loai_bieu_do
            }
            token = "eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNDKzMLU0NdJRSq0ogAiYmxsZgwSKSxLT0pzzU1KBmozMLE0MjYHaUlILSjxTQCKmJkBuaXFqEYwL4-cl5iJpqQUAoOBjYXgAAAA.0jEJ8Y7YQ6lin1wG9WbeCn1La3nI1URlMyURbhbFyOI"
            output_getListChart_api = call_getListChart_api_2(data_2, token)
            if len(output_getListChart_api) > 0:
                print('serviceId:', str(serviceId))
                break
        
        chart_id = best_chartName(output_getListChart_api)
        if chart_id == 'NULL':
            output_api['alert'] = 'Hệ thống chưa có dữ liệu về đơn vị và chỉ tiêu này.'
            logged['output'] = output_api
            save_data_to_file(logged)
            return output_api
        
        output_api['chartId'] = str(chart_id)
    else:
        url = best_cmd_to_url(cmd_stat, cmd2url)
        output_api['URL'] = url

    """
    else:
        cmd_stat = encode_input_and_return_top_n(model_matching, text, db_df, 1, new2oldmatch)
        url = best_cmd_to_url(cmd_stat, cmd2url)
        output_api['URL'] = url
    """
    print()
    print('Output:')
    for key, value in output_api.items():
        print(key+ ':', value)
    print()
    print('More:')
    for key, value in out.items():
        print(key+ ':', value)
    
    logged['output'] = output_api
    save_data_to_file(logged)
    return output_api

demo = gr.Interface(
    text_analysis,
    gr.Textbox(placeholder="Enter sentence here..."),
    ["json"],
    examples=[
        ["Mở dashboard doanh thu kd quốc tế của vtc ngày hôm qua"],
        ["Các thông số NAT ngày hôm qua đã được ghi nhận trên đát bọt"],
        ["Ngày 26/07, hãy mở db của CTM để kiểm tra tb thực"],
        ["Mở biểu đồ cột chồng tiêu dùng thực tháng trước của tập đoàn"],
        ["Hãy mở biểu đồ đường tỉnh Tuyên Quang tháng 8 về doanh thu dịch vụ"],
        ["Mở báo cáo chốt năm lên"],
        ["Hãy mở báo cáo chăm sóc kh"],
        ["Mở quản lý SMS và email"]
    ],
)


demo.launch(share=False)