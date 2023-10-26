import json
import requests

# Load the JSON data from the file
def call_getListChart_api(data, token_1, token_2):
    api_url = f'https://10.30.164.32:8192/vsds/api/chatbot/getListChart?token={token_2}'
    headers = {'Authorization': f'Bearer {token_2}'}
    response = requests.post(api_url, json=data, headers=headers, verify=False)

    # Kiểm tra phản hồi từ server
    if response.status_code == 200:
        return response
    else:
        return []

def call_getListChart_api_2(data, token):
    with open('/Users/tranxuanhuy/Documents/Viettel Digital Talent/Chatbot/Code/load_api_for_test/data/chart_api_output.jsonl', 'r') as json_file:
        json_list = list(json_file)
    
    for item in json_list:
        inst = json.loads(item)
        if inst['input'] == data:
            return inst['output']
    return []


if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore')
    data = {
        "ip": "10.61.142.22",
        "serviceId": "19",
        "areaCode": "TD",
        "cycleTime": "NGAY",
        "keyNameTitle": "",
        "typeChart": "Biểu đồ cảnh báo 2"
    }

    token_1 = "eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNDKzMLU0NdJRSq0ogAiYmxsZgwSKSxLT0pzzU1KBmozMLE0MjYHaUlILSjxTQCKmJkBuaXFqEYwL4-cl5iJpqQUAoOBjYXgAAAA.0jEJ8Y7YQ6lin1wG9WbeCn1La3nI1URlMyURbhbFyOI"
    token_2 = "eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNLY0M7M0sNBRSq0oAAuYGJgaG4MEiksS09Kc81NSgZqMgOKGxkBtKakFJZ4pIBFTEyC3tDi1CMaF8fMSc5G01AIAgG6fSngAAAA.mvTqCOqk-VCO0ua7YUiKIWJfgXzKkQC_4UhzMFrCzdA"
    output_getListChart_api = call_getListChart_api(data, token_1, token_2)
    print(output_getListChart_api)