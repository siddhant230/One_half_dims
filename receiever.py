import http.client
import json

conn = http.client.HTTPSConnection("127.0.0.1", 8000)

headersList = {
    "Accept": "*/*",
    "User-Agent": "Thunder Client (https://www.thunderclient.com)",
    "Content-Type": "application/json"
}

payload = json.dumps({"prompt": "yooooo"})

conn.request("POST", "/gentext2img", payload, headersList)
response = conn.getresponse()
result = response.read()

print(result.decode("utf-8"))
