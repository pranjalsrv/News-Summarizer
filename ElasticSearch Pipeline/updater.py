import uuid
import torch
import requests
import datetime
import xmltodict
import transformers
import elasticsearch

from bs4 import BeautifulSoup

maps = {
    "INDIA":
        "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    "WORLD": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
    "NRI": "https://timesofindia.indiatimes.com/rssfeeds/7098551.cms",
    "BUSINESS": "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
    "CRICKET": "https://timesofindia.indiatimes.com/rssfeeds/54829575.cms",
    "SPORTS": "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms",
    "SCIENCE":
        "http://timesofindia.indiatimes.com/rssfeeds/-2128672765.cms",
    "ENVIRONMENT":
        "http://timesofindia.indiatimes.com/rssfeeds/2647163.cms",
    "TECH": "http://timesofindia.indiatimes.com/rssfeeds/66949542.cms",
    "EDUCATION":
        "http://timesofindia.indiatimes.com/rssfeeds/913168846.cms",
    "ENTERTAINMENT":
        "http://timesofindia.indiatimes.com/rssfeeds/1081479906.cms",
    "LIFESTYLE": "http://timesofindia.indiatimes.com/rssfeeds/2886704.cms",
    "ASTROLOGY":
        "https://timesofindia.indiatimes.com/rssfeeds/65857041.cms",
    "AUTO": "https://timesofindia.indiatimes.com/rssfeeds/74317216.cms",
    "MUMBAI":
        "http://timesofindia.indiatimes.com/rssfeeds/-2128838597.cms",
    "DELHI": "http://timesofindia.indiatimes.com/rssfeeds/-2128839596.cms",
    "BANGALORE":
        "http://timesofindia.indiatimes.com/rssfeeds/-2128833038.cms",
    "HYDERABAD":
        "http://timesofindia.indiatimes.com/rssfeeds/-2128816011.cms",
    "CHENNAI": "http://timesofindia.indiatimes.com/rssfeeds/2950623.cms",
    "AHEMDABAD":
        "http://timesofindia.indiatimes.com/rssfeeds/-2128821153.cms",
    "ALLAHBAD": "http://timesofindia.indiatimes.com/rssfeeds/3947060.cms",
    "BHUBANESHWAR":
        "http://timesofindia.indiatimes.com/rssfeeds/4118235.cms",
    "COIMBATORE":
        "http://timesofindia.indiatimes.com/rssfeeds/7503091.cms",
    "GURGAON": "http://timesofindia.indiatimes.com/rssfeeds/6547154.cms",
    "GUWAHATI": "http://timesofindia.indiatimes.com/rssfeeds/4118215.cms",
    "HUBLI": "http://timesofindia.indiatimes.com/rssfeeds/3942695.cms",
    "KANPUR": "http://timesofindia.indiatimes.com/rssfeeds/3947067.cms",
    "KOLKATA":
        "http://timesofindia.indiatimes.com/rssfeeds/-2128830821.cms",
    "LUDHIANA": "http://timesofindia.indiatimes.com/rssfeeds/3947051.cms",
    "MANGALORE": "http://timesofindia.indiatimes.com/rssfeeds/3942690.cms",
    "MYSORE": "http://timesofindia.indiatimes.com/rssfeeds/3942693.cms",
    "NOIDA": "http://timesofindia.indiatimes.com/rssfeeds/8021716.cms",
    "PUNE": "http://timesofindia.indiatimes.com/rssfeeds/-2128821991.cms",
    "GOA": "http://timesofindia.indiatimes.com/rssfeeds/3012535.cms",
    "CHANDIGARH":
        "http://timesofindia.indiatimes.com/rssfeeds/-2128816762.cms",
    "LUCKNOW":
        "http://timesofindia.indiatimes.com/rssfeeds/-2128819658.cms",
    "PATNA": "http://timesofindia.indiatimes.com/rssfeeds/-2128817995.cms",
    "JAIPUR": "http://timesofindia.indiatimes.com/rssfeeds/3012544.cms",
    "NAGPUR": "http://timesofindia.indiatimes.com/rssfeeds/442002.cms",
    "RAJKOT": "http://timesofindia.indiatimes.com/rssfeeds/3942663.cms",
    "RANCHI": "http://timesofindia.indiatimes.com/rssfeeds/4118245.cms",
    "SURAT": "http://timesofindia.indiatimes.com/rssfeeds/3942660.cms",
    "VADODARA": "http://timesofindia.indiatimes.com/rssfeeds/3942666.cms",
    "VARANASI": "http://timesofindia.indiatimes.com/rssfeeds/3947071.cms",
    "THANE": "http://timesofindia.indiatimes.com/rssfeeds/3831863.cms",
    "THIRUVANANTHAPURAM":
        "http://timesofindia.indiatimes.com/rssfeeds/878156304.cms",
    "USA": "http://timesofindia.indiatimes.com/rssfeeds/30359486.cms",
    "SOUTH_ASIA":
        "http://timesofindia.indiatimes.com/rssfeeds/3907412.cms",
    "UK": "http://timesofindia.indiatimes.com/rssfeeds/2177298.cms",
    "EUROPE": "http://timesofindia.indiatimes.com/rssfeeds/1898274.cms",
    "CHINA": "http://timesofindia.indiatimes.com/rssfeeds/1898184.cms",
    "MIDDLE_EAST":
        "http://timesofindia.indiatimes.com/rssfeeds/1898272.cms",
    "REST_OF_WORLD":
        "http://timesofindia.indiatimes.com/rssfeeds/671314.cms"
}

es = elasticsearch.Elasticsearch()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = transformers.T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')


def preprocess_text(url, tag):
    response = requests.get(url)
    content = response.content
    soup_article = BeautifulSoup(content, "html5lib")
    body = soup_article.find_all(tag, class_=None)
    s = ""
    for i in body:
        s += i.text
    return s


def summarize(text):
    t5_prep = "summarize: " + str(text).strip().replace("\n", "")
    tokenized_text = tokenizer.encode(t5_prep, max_length=len(t5_prep), return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(tokenized_text, num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=120,
                                 max_length=350,
                                 early_stopping=True)
    output = tokenizer.decode(summary_ids[0].to(device), skip_special_tokens=True)
    return output


if __name__ == "__main__":
    for key, value in maps.items():
        maps[key] = {"link": value, "lastPub": datetime.datetime(2018, 9, 10, 6, 36, 43)}

    while True:
        for key, value in maps.items():
            app = dict(xmltodict.parse(requests.get(value["link"]).content.decode("utf-8"))["rss"].pop("channel"))
            time = value["lastPub"]
            for i in app["item"]:
                try:
                    item = dict(i)
                    item["pubDate"] = datetime.datetime.strptime(" ".join(item["pubDate"].split()[1:-1]), "%d %b %Y %X")
                    if item["pubDate"] > value["lastPub"]:
                        item["category"] = key
                        item["archived"] = False
                        item.pop("guid")

                        item["summary"] = summarize(preprocess_text(item["link"], "div")) + "......."
                        item["viewCount"] = 0

                        es.index(index="news", doc_type="news-obj", id=uuid.uuid1(), body=item)
                        if item["pubDate"] > time:
                            time = item["pubDate"]

                    del item
                except:
                    continue

            del app
            maps[key]["lastPub"] = time
            del time
