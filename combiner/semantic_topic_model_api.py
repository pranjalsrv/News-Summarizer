import torch
import requests

from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import T5Tokenizer, T5ForConditionalGeneration

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
async def get_closest_docs(request: Request, query: str = Form(...)):
    # Semantic similarity
    closest_topic = requests.get('http://127.0.0.1:8081/get_topic/?query=' + query).json()["closest_topic"]

    # LDA2vec
    closest_docs = requests.get('http://localhost:8082/get_docs/?topic=' + str(closest_topic)).json()["top_articles"]
    texts = []
    for docs in closest_docs:
        texts.append(docs["text"])
    summaries = []
    for text in texts:
        preprocessed_txt = str(text).strip().replace("\n", "")
        t5_prep = "summarize: " + preprocessed_txt
        tokenized_text = tokenizer.encode(t5_prep, max_length=len(t5_prep), return_tensors="pt").to(device)
        summary_ids = model.generate(tokenized_text, num_beams=4,
                                     no_repeat_ngram_size=2,
                                     min_length=20,
                                     max_length=350,
                                     early_stopping=True)
        output = tokenizer.decode(summary_ids[0].to(device), skip_special_tokens=True)
        summaries.append(output)
    # return {"closest_docs_summs": summaries}
    return templates.TemplateResponse("output.html", {"request": request, 'summaries': summaries, 'docs': closest_docs,
                                                      "num_docs": len(closest_docs)})
