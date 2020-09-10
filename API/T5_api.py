from flask import Flask
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from flask import render_template
import torch

## Model Defn ####

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-base')


class Item(BaseModel):
    headline: str
    content: str


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")




@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/v1/summarize_call")
async def summarize(content: str = Form(...)):
    text = content
    preprocessed_txt = str(text).strip().replace("\n", "")
    t5_prep = "summarize: " + preprocessed_txt
    tokenized_text = tokenizer.encode(t5_prep, max_length=len(t5_prep), return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text, num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=20,
                                 max_length=350,
                                 early_stopping=True)
    output = tokenizer.decode(summary_ids[0].to(device), skip_special_tokens=True)
    return {"summary": output}
