import json
import spacy
import string
from tqdm import tqdm
from IPython import embed
from collections import Counter

class AnalyzerUtils:
    
    def __init__(self):

        self.nlp = spacy.load('en')
        self.nlp_light = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.alphabet = string.ascii_lowercase


    def load_review_data(self, filepath):

        with open(filepath) as f:
            data = json.load(f)

        return data


    def get_review_rating_pairs(self, data):

        review_rating_pairs = dict()
        for doc_id in data:
            embed()
            if('revisions' in data[doc_id]):
                for revision in data[doc_id]['revisions']:
                    for review in data[doc_id]['revisions'][revision]['reviews']:
                        if('text' in data[doc_id]['revisions'][revision]['reviews'][review] and 'decision' in data['revisions'][revision]['reviews'][review]):
                            if(len(data[doc_id]['revisions'][revision]['reviews'][review]['text']) > 0):
                                decision = data[doc_id][revision][review_number]['decision']
                                text = data[doc_id][revision][review_number]['text']
                                if(decision not in review_rating_pairs):
                                    review_rating_pairs[decision] = list()
                                review_rating_pairs[decision].append(text)
    
        return review_rating_pairs


    def get_classwise_word_count(self, data):

        word_counts = dict()
        for decision in data:
            word_counts[decision] = Counter()
            for text in tqdm(data[decision]):
                for token in self.nlp_light(text):
                    if(not self.forbidden(token.text)):
                        word_counts[decision][token.text] += 1

        return word_counts


    def get_classwise_lengths(self, data):

        lengths = dict()
        for decision in data:
            lengths[decision] = list()
            for text in tqdm(data[decision]):
                lengths[decision].append(len(self.nlp_light(text)))

        return lengths


    
    def forbidden(self, word):

        if(word.lower() in self.stop_words):
            return True
        else:
            for character in word.lower():
                if(character in self.alphabet):
                    return False
            return True

        return False






