import pandas as pd

import preprocessing


def split_data_into_train_test(df):
    train = df.sample(frac=0.8, random_state=123)
    test = df.drop(train.index)
    return train, test


if __name__ == "__main__":
    music_df = pd.read_csv("Final data/lyrics_final.csv")
    music_df['cleaned_lyrics'] = music_df['lyrics'].apply(preprocessing.preprocessing)
    music_df.to_csv("Final data/cleaned_lyrics.csv", index_label='index')

    # removing NA and other rows
    music_df = preprocessing.data_cleaning(music_df)

    # train-test split
    train_df, test_df = split_data_into_train_test(music_df)

    train_df.to_csv("Final data/train.csv", index=False)
    test_df.to_csv("Final data/test.csv", index=False)
