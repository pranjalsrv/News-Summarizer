import json
from fastapi import FastAPI


app = FastAPI()
newsgroup_json = json.load(open("20_newsgroup_json.json"))

@app.get("/get_docs/")
async def get_docs(topic: int):
    return {"top_articles": newsgroup_json[str(topic)][:10]}
