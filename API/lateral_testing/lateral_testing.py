import torch
import requests

from fastapi import FastAPI
from bs4 import BeautifulSoup
from fastapi.responses import JSONResponse
from transformers import T5Tokenizer,T5ForConditionalGeneration


app = FastAPI()

url = "https://news-api.lateral.io/documents/similar-to-text"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

def preprocess_text(url, tag):
    response = requests.get(url)
    content = response.content
    soup_article = BeautifulSoup(content, "html5lib")
    body = soup_article.find_all(tag, class_ = None)
    s = ""
    for i in body:
        s+=i.text
    return s

def summarize(text):
    t5_prep = "summarize: " + str(text).strip().replace("\n", "")
    tokenized_text = tokenizer.encode(t5_prep, max_length = len(t5_prep), return_tensors = "pt", truncation = True).to(device)
    summary_ids = model.generate(tokenized_text, num_beams = 4,
                                 no_repeat_ngram_size = 2,
                                 min_length = 120,
                                 max_length = 350,
                                 early_stopping = True)
    output = tokenizer.decode(summary_ids[0].to(device), skip_special_tokens = True)
    return output

@app.get("/get_result/")
async def query(query:str):
    payload = payload = "{\"text\":\"" + query + "\"}"
    header = {
        'subscription-key': "9d1c0756be9d18a23fd60d4c62745801",
        'content-type': "application/json"
        }
    
    response = requests.request("POST", url, data = str(payload), headers = header)


    article_headlines = []
    article_urls = []
    articles = []
    
    c = 0

    for article in response.json():
        article_url = article["url"]
        text = preprocess_text(article_url, "p")
        c += 1
        if(len(article_url)>0 and len(text.split()) > 8 and len(text.split()) < 1024):
            article_urls.append(article_url)
            article_headlines.append(article['title'])
            articles.append(text)
            break
            
            
    content = [{"url" : i, "headline" : j, "summary": summarize(k)} for i,j,k in zip(article_urls, article_headlines, articles)]
    
    for article in response.json()[c:]:
        article_url = article["url"]
        if(len(article_url)>0):
            content.append({"url" : article_url, "headline" : article['title']})
    
    headers = {'content-type': "application/json"}
    return JSONResponse(content = content, headers = headers)
        
@app.get("/get_summary_from_url/")
async def url_toi(url_toi:str):
    headers = {'content-type': "application/json"}
    return JSONResponse(content = summarize(preprocess_text(url, "div")), headers = headers)
