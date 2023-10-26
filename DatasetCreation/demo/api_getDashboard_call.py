import requests
import json

# Load the JSON data from the file
def call_getDashboard_api(data, token):
    api_url = f"https://10.30.164.32:8192/vsds/api/chatbot/getDashboard?token={data['token']}&groupProfileCode={data['groupProfileCode']}&cycleTime={data['cycleTime']}&areaCode={data['areaCode']}&ip={data['ip']}"
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.post(api_url, json=data, headers=headers, verify=False)

    # Kiểm tra phản hồi từ server
    if response.status_code == 200:
        return response.json()['data']
    else:
        return []

def call_getDashboard_api_2(data, token):
    with open('/Users/tranxuanhuy/Documents/Viettel Digital Talent/Chatbot/Code/load_api_for_test/data/db_api_output.jsonl', 'r') as json_file:
        json_list = list(json_file)
    
    for item in json_list:
        inst = json.loads(item)
        if inst['input'] == data:
            return inst['output']
    return []

if __name__=='__main__':
    data = {
        'token': 'eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNLY0M7M0sNBRSq0oAAuYGJgaG4MEiksS09Kc81NSgZqMgOKGxkBtKakFJZ4pIBFTEyC3tDi1CMaF8fMSc5G01AIAgG6fSngAAAA.mvTqCOqk-VCO0ua7YUiKIWJfgXzKkQC_4UhzMFrCzdA',
        'groupProfileCode': 'VPTD',
        'cycleTime': 'NGAY',
        'areaCode': 'TD',
        'ip': '10.61.142.22'
    }

    token = "eyJhbGciOiJIUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWyiwuVrJSSk8qSVHSUcpMLFGyMjSzNDA1sDQ3M9NRSq0ogApYmhmCBIpLEtPSnPNTUoGajMwsTQyNgdpSUgtKPFNAIqYmQG5pcWoRjAvj5yXmImmpBQDfAGr_eAAAAA.iU0Vka0zQCQMUb1I8U6ncOMh5YtrO3b7G4ZhMGYN5yk"
    
    output_getDashboard_api = call_getDashboard_api(data, token)
    print(output_getDashboard_api)
