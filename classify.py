import sys

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import joblib

from custom_features import get_custom_vectorizer
from custom_features import CustomFeats


def get_custom_features():
    manual_pipeline = Pipeline(
        steps=[
            ("custom_feats", CustomFeats()),
            ("dict_vect", DictVectorizer()),
            ]
                )

    bow_pipeline = Pipeline(
        steps=[
            ("bag_of_words", get_custom_vectorizer()),
            ]
                )
    return FeatureUnion([
        ('manual', manual_pipeline),
        ('bag_of_words', bow_pipeline)
    ])


def classify(train, test, features, input_param):

    # x_train = train['cleaned_lyrics']
    x_train = train['lyrics']
    y_train = train['genre']

    # x_test = test['cleaned_lyrics']
    x_test = test['lyrics']
    y_test = test['genre']

    # get custom features
    combined_features = get_custom_features()

    if input_param == "mnb":
        print("Multinomial Naive Bayes Classifier")
        mnb_model = Pipeline(
            steps=[
                ("combined_features", combined_features),
                ("classifier", MultinomialNB()),
            ]
                )
        mnb_model.fit(x_train, y_train)
        y_pred = mnb_model.predict(x_test)
        print("Classification report: %s" % (classification_report(y_test, y_pred)))
        print("accuracy for multinomial naive bayes: %s" % mnb_model.score(x_test, y_test))
        # print (confusion_matrix(y_train, mnb_predicted))

    if input_param == "lr":
        print("Logistic Regression Classifier")
        lr_model = Pipeline(
            steps=[
                ("combined_features", combined_features),
                ("classifier", LogisticRegression(solver="liblinear", multi_class="ovr")),
            ]
                )
        # lr = LogisticRegression(solver="liblinear", multi_class="ovr")
        lr_model.fit(x_train, y_train)
        y_pred = lr_model.predict(x_test)
        print("Classification report: %s" % (classification_report(y_test, y_pred)))
        print("accuracy for LogisticRegression: %s" % (lr_model.score(x_test, y_test)))

    if input_param == "dt":
        print("Decision Tree Classifier")
        dt_model = Pipeline(
            steps=[
                ("combined_features", combined_features),
                ("classifier", DecisionTreeClassifier()),
            ]
                )
        # dt = DecisionTreeClassifier()
        dt_model.fit(x_train, y_train)
        y_pred = dt_model.predict(x_test)
        print("Classification report: %s" % (classification_report(y_test, y_pred)))
        print("accuracy for Decision Tree: %s" % (dt_model.score(x_test, y_test)))

    if input_param == "rf":
        print("Random Forest Classifier")
        rf_model = Pipeline(
            steps=[
                ("combined_features", combined_features),
                ("classifier", RandomForestClassifier(n_estimators=100, max_features="sqrt")),
            ]
                )
        # dt = DecisionTreeClassifier()
        rf_model.fit(x_train, y_train)
        y_pred = rf_model.predict(x_test)
        print("Classification report: %s" % (classification_report(y_test, y_pred)))

        accuracy = rf_model.score(x_test, y_test)
        print("accuracy for Random Forest: %s" % accuracy)

    if input_param == "gbm":
        print("Gradient boosting Classifier")
        gbm_model = Pipeline(
            steps=[
                ("combined_features", combined_features),
                ("classifier", GradientBoostingClassifier()),
            ]
                )
        # dt = DecisionTreeClassifier()
        gbm_model.fit(x_train, y_train)
        y_pred = gbm_model.predict(x_test)
        print("Classification report: %s" % (classification_report(y_test, y_pred)))

        accuracy = gbm_model.score(x_test, y_test)
        print("accuracy for Gradient boosting: %s" % accuracy)


if __name__ == "__main__":

    classifier_param = sys.argv[1]
    train_data = pd.read_csv("Final data/train.csv")
    test_data = pd.read_csv("Final data/test.csv")
    custom_features = get_custom_features()

    classify(train_data, test_data, custom_features, classifier_param)

