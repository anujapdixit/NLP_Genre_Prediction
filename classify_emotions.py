import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import preprocessing


def train_emotions(data):
    data['cleaned_lyrics'] = data['lyrics'].apply(preprocessing.preprocessing)
    train = data['cleaned_lyrics']
    label = data.label
    lr_model = Pipeline(
            steps=[
                ("combined_features", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", LogisticRegression(solver="liblinear", multi_class="ovr")),
            ]
                )
    classifier = lr_model.fit(train, label)
    # with open("emotion/emotions_classifier_lr.pkl", 'wb') as f:
    #     pickle.dump(classifier, f)
    return classifier


if __name__ == "__main__":
    emotions = pd.read_csv("emotion/lyrics_emotions.csv")
    train_emotions(emotions)
