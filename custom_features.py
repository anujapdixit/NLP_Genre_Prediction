import pickle
import re

from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


class CustomFeats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def __init__(self):
        self.feat_names = set()
        self.emotion_classifier = pickle.load(open("emotion/emotions_classifier_lr.pkl", 'rb'))

    def fit(self, x, y=None):
        return self

    @staticmethod
    def words(doc):
        return doc.split(' ')

    def get_blob(self, review):
        blob = TextBlob(review)
        return blob.sentiment

    def spacy_adj_feature(self, doc):
        processed_text = self.words(doc)
        tags_annotated_list = (pos_tag(processed_text))
        for token in tags_annotated_list:
            if len(token[0]) > 3 and re.search('(JJ|JJ.|RB)$', token[1]) and token[0].isupper():
                return 1
        return 0

    @staticmethod
    def num_lines(doc):
        return doc.count('\n')

    @staticmethod
    def length_doc(doc):
        return len(doc.split())

    def analyze_tone(self, doc):
        # Authentication via IAM
        authenticator = IAMAuthenticator('<key>')
        service = ToneAnalyzerV3(
            version='2017-09-21',
            authenticator=authenticator)
        service.set_service_url('https://gateway.watsonplatform.net/tone-analyzer/api')

        tones_output = service.tone(tone_input=doc,
                                    content_type="text/plain").get_result()
        return [i['tone_name'] for i in tones_output['document_tone']['tones']]

    def get_emotion(self, doc):
        return self.emotion_classifier.predict([doc])[0]

    def features(self, doc):
        return {
            'emotion': self.get_emotion(doc),
            'length': self.length_doc(doc),
            'num_lines': self.num_lines(doc),
        }

    def get_feature_names(self):
        return list(self.feat_names)

    def transform(self, reviews):
        feats = []
        for review in reviews:
            f = self.features(review)
            [self.feat_names.add(k) for k in f]
            feats.append(f)
        return feats


def get_custom_vectorizer():
    #return CountVectorizer()
    return TfidfVectorizer(ngram_range=(1, 2))
