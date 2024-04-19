from transformers import BatchEncoding
from prepare_dataset import create_hf_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import string 
import json

from transformers import pipeline


dataset = create_hf_dataset()
tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")


def is_empty_or_none(s):
    if s is None or s == '':
        return True
    else:
        return False
    
    
def tag_dataset(split, dataset=dataset):
    print(f'Tagging {split} split...')
    ner_tagged_sentences = []

    # Collect sentences
    sentences = [example['inputs'] for example in dataset[split]]
    #sentences = ["ClinicalTrials.gov - Question - general information. Both my parents died in [LOCATION] TX with multiple myeloma, father in 70's and mother 84 years old.   the attending oncologist advised that I should be tested every few years. I was unaware of a genetic test for MM I am 63 yr old female, in good health, and where could I get the genetic test for MM and approx cost for testing?"]
    pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')
    #sentences = ['who makes bromocriptine i am wondering what company makes the drug bromocriptine, i need it for a mass i have on my pituitary gland and the cost just keeps raising. i cannot ever buy a full prescription because of the price and i was told if i get a hold of the maker of the drug sometimes they offer coupons or something to help me afford the medicine. if i buy 10 pills in which i have to take 2 times a day it costs me 78.00. and that is how i have to buy them.  thanks."']
    for i, sentence in enumerate(sentences):
        
        if (i + 1) % 100 == 0:
            print(f'Processed {i + 1} sentences...')
        # Run the pipeline on the sentence
        results = pipe(sentence.lower())
        # Print the result
        #print(results)
    
        word_entity_mapping = {}
    
        for entity_res_dict in results:
            if entity_res_dict['word'] not in word_entity_mapping:
                word_entity_mapping[entity_res_dict['word']] = entity_res_dict['entity_group']
    
        # after getting the mapping, we tag the words in the sentence and create a new sentence
        new_sentence = []
        sentence_words = sentence.split()
        i = 0
        
            
        while i < len(sentence_words):
            word = sentence.split()[i]
            # print(word)
            found = False
            for word_in_mapping in word_entity_mapping:
                # if word.lower() == word_in_mapping or word.lower().find(word_in_mapping)!=-1:
                #     print(f'Found: {word_in_mapping}')
                #     new_sentence.append(f'<{word_entity_mapping[word_in_mapping]}> {word}')
                #     found = True
                #     break
                
                # check the full tagged word in the mapping
                tagged_words = word_in_mapping.split()
                #print('tagged words: ', tagged_words)
                tagged_words_mod = [w.strip(string.punctuation) for w in tagged_words]
                search_parts = [w.strip(string.punctuation).lower() for w in sentence_words[i:i+len(tagged_words)]] 
                #print('search parts: ', search_parts)
            
                
                if search_parts == tagged_words_mod and tagged_words_mod != []:
                    #print(f'Found: {word_in_mapping}')
                    new_sentence.append(f'<{word_entity_mapping[word_in_mapping]}> {" ".join(tagged_words)}')
                    i += len(tagged_words)
                    found = True
                    break
            
            if not found:
                new_sentence.append(word)
                i += 1
            
        # join the words to create a new sentence
        new_sentence = ' '.join(new_sentence)
        # print(word_entity_mapping)
        # print(f'Original Sentence: {sentence}')
        # print(f'New Sentence: {new_sentence}')
        # print()
        ner_tagged_sentences.append(new_sentence)
    
    return ner_tagged_sentences
    


# labels = model.config.id2label

# # Print the labels
# for id, label in labels.items():
#     print(f'ID: {id}, Label: {label}')
    
# Define batch size

def add_ner_tags_to_dataset(example, idx, ner_tagged_sentences):
    example['inputs_tagged'] = ner_tagged_sentences[idx]
    return example

# create a new dataset with the tagged sentences
ner_tagged_train_sentences = tag_dataset('train', dataset=dataset)
ner_tagged_dev_sentences = tag_dataset('validation', dataset=dataset)
ner_tagged_test_sentences = tag_dataset('test', dataset=dataset)

# add the tagged sentences to the dataset
dataset['train'] = dataset['train'].map(add_ner_tags_to_dataset, with_indices=True, fn_kwargs={'ner_tagged_sentences': ner_tagged_train_sentences})
dataset['validation'] = dataset['validation'].map(add_ner_tags_to_dataset, with_indices=True, fn_kwargs={'ner_tagged_sentences': ner_tagged_dev_sentences})
dataset['test'] = dataset['test'].map(add_ner_tags_to_dataset, with_indices=True, fn_kwargs={'ner_tagged_sentences': ner_tagged_test_sentences})

    
# save the dataset train , validation and test splits to jsonl files

with open('train_tagged.jsonl', 'w') as f:
    for example in dataset['train']:
        f.write(json.dumps(example) + '\n')
        
with open('validation_tagged.jsonl', 'w') as f:
    for example in dataset['validation']:
        f.write(json.dumps(example) + '\n')
        
with open('test_tagged.jsonl', 'w') as f:
    for example in dataset['test']:
        f.write(json.dumps(example) + '\n')


# save the dataset


# Print the vocabulary
# for token, id in vocab.items():
#     print(f'Token: {token}, ID: {id}')



# Print the decoded predictions
# for pred in decoded_predictions:
#     print(pred)

# # Tokenize sentences in batches
# for i in range(0, len(sentences), batch_size):
#     batch_sentences = sentences[i:i+batch_size]
#     batch_encoding: BatchEncoding = tokenizer(batch_sentences, truncation=True, padding=True, return_tensors="pt")

#     # Pass the batch to the model and get the outputs
#     outputs = model(**batch_encoding)

#     # The outputs are logits, get the predicted tags
#     predictions = torch.argmax(outputs.logits, dim=2)

#     # Decode the predictions
#     # Note: this assumes that the labels are in the same order as the tokenizer's IDs
#     predicted_tags = [tokenizer.convert_ids_to_tokens(id_list) for id_list in predictions]

#     print(predicted_tags)
#     break