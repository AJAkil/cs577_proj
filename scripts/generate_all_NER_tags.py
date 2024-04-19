from transformers import BatchEncoding
from prepare_dataset import create_hf_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import string 
import json


tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")

labels = model.config.id2label
ner_tags = set([label.split('-')[-1] for label in labels.values()])


with open('unique_NER_tags.txt', 'w') as f:
    for tag in ner_tags:
        f.write(tag + '\n')
    
    