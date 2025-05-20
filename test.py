import requests

with open("image copy.png", "rb") as f:
    response = requests.post("http://localhost:5000/detect", files={"image": f})

with open("result.jpg", "wb") as f:
    f.write(response.content)
