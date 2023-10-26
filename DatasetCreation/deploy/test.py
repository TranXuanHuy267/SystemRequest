import json
import os

import requests

PORT = os.environ.get("APP_PORT", 8080)
HOST = os.environ.get("APP_HOST", "0.0.0.0")

new_port = input(f"Enter new port (default from env: {PORT}): ")
if new_port:
    PORT = int(new_port)
    print(f"Using port {PORT}")

homepage_url = f"http://{HOST}:{PORT}/api/homepage"
r = requests.get(homepage_url)
print(f"Test homepage")
print(json.dumps(r.json(), indent=4))
print()

### ask chatbot for dasbhoard
chatbotreq = f"http://{HOST}:{PORT}/api/chatbot/ask"
r = requests.post(
    chatbotreq,
    json={
        "input": "Mở dashboard vtc ngày hôm nay",
    },
)
print(f"Test chatbot request: dashboard")
print(json.dumps(r.json(), indent=4))
print()

### ask chatbot for chart
chatbotreq = f"http://{HOST}:{PORT}/api/chatbot/ask"
r = requests.post(
    chatbotreq,
    json={
        "input": "Mở biểu đồ thị phần tập đoàn tháng này",
    },
)
print(f"Test chatbot request: chart")
print(json.dumps(r.json(), indent=4))
print()

### ask chatbot for function command
chatbotreq = f"http://{HOST}:{PORT}/api/chatbot/ask"
r = requests.post(
    chatbotreq,
    json={
        "input": "Mở trang chủ",
    },
)
print(f"Test chatbot request: function command")
print(json.dumps(r.json(), indent=4))
print()