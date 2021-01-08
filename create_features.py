import os

import nltk
from nltk.tag.stanford import StanfordPOSTagger
import pandas as pd
from nltk.tag.mapping import map_tag

def count_syllables(token):
    vowels = "aeiouyAEIOUY"
    counter = 0
    for char in token:
        if char in vowels:
            counter += 1
    return counter

def get_token_pos(row,):
    text = row["sentence"]
    token = row["token"]
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_tags = [map_tag('en-ptb', 'universal', tag) for token, tag in pos_tags]

    for i in range(len(tokens)):
        if token == tokens[i]:
            return pos_tags[i]



def main():
    path = r'data/train/lcp_single_train.tsv'
    tagger_dir = r"D:\University\NLP\SemEval-2020\lcp\stanford_postagger_full"
    tagger_jar_name = 'stanford-postagger-4.1.0.jar'
    tagger_model_name = 'models/english-bidirectional-distsim.tagger'
    tagger_jar_path = os.path.join(tagger_dir, tagger_jar_name)
    tagger_model_path = os.path.join(tagger_dir, tagger_model_name)

    train_df = pd.read_csv(path, sep='\t', encoding='utf-8')
    # признак того, что слово из контекста - полностью в верхнем регистре (следовательно, вероятно, аббревиатура)
    train_df["is_upper"] = train_df["token"].apply(lambda x: str(x).isupper())
    # Признак длины слова из контеста
    train_df["token_len"] = train_df["token"].apply(lambda x: len(str(x)))
    # Признак количества гласных в слове (не уверен, кстати, что y считается гласной...)
    train_df["num_syllables"] = train_df["token"].apply(lambda x: count_syllables(str(x)))
    # Часть речи
    train_df["pos_tag"] = train_df.apply(lambda row: get_token_pos(row,), axis=1)
    print(train_df)
    columns = ["is_upper", "token_len", "num_syllables", "complexity"]

    selected_columns = train_df[columns]
    correlations = selected_columns.corr()
    print(correlations)
    stats_mean = train_df.groupby("pos_tag").mean()
    print(stats_mean)
    stats_count = train_df.groupby("pos_tag").count()
    print(stats_count)
    train_df.to_csv("single_train_with_features.csv", sep='\t', index=False)

if __name__ == '__main__':
    main()