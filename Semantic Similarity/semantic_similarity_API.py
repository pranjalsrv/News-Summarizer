import torch
import scipy
import warnings

import numpy as np

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

app = FastAPI()


topic_words = ["jesus god orthodox faith verse worship belief christianity christian mormon",
               "bike tire honda battery brake ride valve aluminum intake jeep",
               "game team play win baseball score playoff mets goal phillies",
               "shuttle nasa space payload computational planetary satellite mission launch lunar",
               "armenian troop cyprus israel soldier agdam bayonet terror bullet he",
               "drive controller battery dock connector ink card slot pin vram",
               "key encrypt cryptosystem encryption cryptography secure rsa crypto eavesdrop decrypt",
               "ride revolver bike bikers go jeep i dog a tank",
               "homicide gun handgun firearm fbi unconstitutional federal smuggle drug weapon",
               "israel armenian plo cyprus troop lebanese arab palestinian syria turkish",
               "armenian bayonet arena team turkish tartar nhl playoff game defenseman",
               "widget server contrib font editor format mit pixmap toolkit binary",
               "controller card disk bios rom vram drive isa vga trident",
               "cd shuttle sale frequency antenna disc space shipping offer model",
               "game win play baseball score he team playoff goal mets",
               "shuttle homicide space payload nasa launch mission gonorrhea rocket armenian",
               "key cryptography devguide encrypt ripem encryption rsa eff cryptosystem crypto",
               "jesus messiah verse orthodox god psalm apostle faith luke divine",
               "quack blood lyme jesus infant orthodox substance god candida msg",
               "font icon window server editor format disk card terminal i"]


transformers = SentenceTransformer('distilbert-base-nli-mean-tokens').to(device)
topics = np.array(transformers.encode(topic_words)).tolist()


def get_closest_topic(query_embedding):
    distance = scipy.spatial.distance.cdist(query_embedding, topics, "cosine")[0]
    results = zip(range(len(distance)), distance)
    results = sorted(results, key=lambda x: x[1])
    return results[0][0]


@app.get("/get_topic/")
async def get_docs(query: str):
    query_embedding = np.array(transformers.encode([query])).tolist()
    closest_topic = get_closest_topic(query_embedding)
    return {"closest_topic": closest_topic}
