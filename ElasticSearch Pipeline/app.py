from fastapi import FastAPI
from elasticsearch import Elasticsearch
from fastapi.responses import JSONResponse

app = FastAPI()
es = Elasticsearch()
headers = {'content-type': "application/json"}


@app.get("/get_category_news/")
async def category(category: str) -> JSONResponse:
    res = es.search(index="news", body={"from": 0, "size": 10,
                                        "query": {"match": {"category": category}}, "sort": {
            "pubDate": {
                "order": "desc"
            }
        }})["hits"]["hits"]
    return JSONResponse(content=[i["_source"] for i in res], headers=headers)


@app.get("/get_news/")
async def query(query: str) -> JSONResponse:
    res = es.search(index="news", body={"from": 0, "size": 10,
                                        "query": {"match": {"title": query}}, "sort": {
            "pubDate": {
                "order": "desc"
            }
        }})["hits"]["hits"]
    return JSONResponse(content=[i["_source"] for i in res], headers=headers)


@app.post("/make_view/")
async def view(article_id: str) -> JSONResponse:
    article = es.get(index="news", id=article_id)["_source"]
    article["viewCount"] += 1  # Increase view count
    if article["pubDate"] < datetime.datetime.now() - datetime.timedelta(days=3) and \
            article["viewCount"] > 10:  # If older than 3 days and view count>10 then archive
        article["archived"] = True
    es.index(index="news", doc_type="news-obj", id=article_id, body=article)  # Performing Update
