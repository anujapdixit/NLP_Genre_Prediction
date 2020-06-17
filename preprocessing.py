import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def data_cleaning(df):
    df = df[(df.genre != 'Not Available') & (df.genre != 'Other')]
    return df


def preprocessing(text):
    # ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stop_words = stopwords.words('english')
    # tagged_tokens = [x[0] + '/' + x[1] for x in nltk.pos_tag(tokens)]
    word_tokens = word_tokenize(text)

    stems = []
    for word in word_tokens:
        stems.append(lemmatizer.lemmatize(word))

    cleaned_text_set = [word for word in stems if word not in stop_words and word not in string.punctuation]
    clean_text = ' '.join(cleaned_text_set).strip()

    return clean_text




