import torch
import time
import pandas as pd
from bertopic import BERTopic
import os
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_and_save(topic_model, docs):
    print(f"Training model with {len(docs)} docs")
    st = time.process_time()
    topics, probs = topic_model.fit_transform(docs)
    et = time.process_time()
    print(f"CPU Execution time: {et-st} seconds")
    topic_model.save("topic_models/{}_model".format(int(len(docs))))
    return topic_model, topics, probs

def load_model(path):
    return BERTopic.load(path)

def get_coherence(topic_model, topic_words, tokens, corpus, dictionary, coherence):
    print(f"Getting {coherence} Coherence")
    max_topics = len(topic_words)
    for i in range(10, max_topics, 10):
        st = time.process_time()
        coherence_model = CoherenceModel(topics=topic_words[:i], 
                                     texts=tokens, 
                                     corpus=corpus,
                                     dictionary=dictionary, 
                                     coherence=coherence)
        coherence_score = coherence_model.get_coherence()
        et = time.process_time()
        print(f"{i} topics in {et-st:.2f} seconds: {coherence_score}")