import json

import pyLDAvis
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from pprint import pprint

texts = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).data

npz = np.load(open('../data/topics.pyldavis.npz', 'rb'))
dat = {k: v for (k, v) in npz.items()}
dat['vocab'] = dat['vocab'].tolist()

top_n = 10
topic_to_topwords = {}
for j, topic_to_word in enumerate(dat['topic_term_dists']):
    top = np.argsort(topic_to_word)[::-1][:top_n]
    # msg = 'Topic %i '  % j
    top_words = [dat['vocab'][i].strip()[:35] for i in top]
    # msg += ' '.join(top_words)
    # print(msg)
    topic_to_topwords[j] = top_words

overall_json = {i: [] for i in range(20)}

for article_no, article in enumerate(dat['doc_topic_dists']):
    max_weight = 0
    max_topic_id = -1
    for topic_id, weight in enumerate(article):
        if weight > max_weight:
            max_weight = weight
            max_topic_id = topic_id

    # overall_json[article_no] = {'text': texts[article_no], 'topic': max_topic_id}
    # print(max_topic_id)
    overall_json[max_topic_id].append({'text': texts[article_no], 'article': article_no, 'topic': max_topic_id})

#pprint(overall_json)
with open('20_newsgroup_json.json', 'w') as fp:
    json.dump(overall_json, fp)
