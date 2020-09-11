from fastapi import FastAPI
from elasticsearch import Elasticsearch
from fastapi.responses import JSONResponse


app = FastAPI()
es = Elasticsearch()

@app.get("/get_category_news/")
async def category(category:str) -> JSONResponse:
    headers = {'content-type': "application/json"}
    res = es.search(index = "news", body = {"from":0, "size":10, 
    "query":{"match": {"category": category}}, "sort": {
        "pubDate": {
        "order": "desc"
        }
    }})["hits"]["hits"]
    return JSONResponse(content = [i["_source"] for i in res], headers = headers)

@app.get("/get_news/")
async def query(query:str) -> JSONResponse:
    headers = {'content-type': "application/json"}
    res = es.search(index = "news", body = {"from":0, "size":10, 
    "query":{"match": {"title": query}}, "sort": {
        "pubDate": {
        "order": "desc"
        }
    }})["hits"]["hits"]
    return JSONResponse(content = [i["_source"] for i in res], headers = headers)