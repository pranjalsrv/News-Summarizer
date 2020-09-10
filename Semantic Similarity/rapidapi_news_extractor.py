import requests
from pprint import pprint

url = "https://google-news.p.rapidapi.com/v1/topic_headlines"

topics = ["WORLD", "NATION", "BUSINESS", "TECHNOLOGY", "ENTERTAINMENT", "SCIENCE", "SPORTS", "HEALTH"]

for i in topics:
    querystring = {"lang": "en", "topic": i, "country": "uk"}
    headers = {
        'x-rapidapi-host': "google-news.p.rapidapi.com",
        'x-rapidapi-key': "e3f738353fmshc7fd28eb79a4b05p1e6731jsn8e523b3529fc"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)

    for j in response.json()['articles']:
        print(j['title'])
