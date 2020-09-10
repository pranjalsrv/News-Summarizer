from flask import Flask
from transformers import BartTokenizer,BartForConditionalGeneration
from fastapi import FastAPI,Form
from pydantic import BaseModel


## Model Defn ####
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

class Item(BaseModel):
    headline: str
    content: str




app = FastAPI()



@app.get("/")
async def home():
    return "Hello World"

@app.post("/api/v1/summarize_call")
async def summarize(heading: str = Form(...), content:str = Form(...)):
    device = "cpu"
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




