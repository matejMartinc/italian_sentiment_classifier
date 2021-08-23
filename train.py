# -*- coding: utf-8 -*-
from sentiment_analysis import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn import model_selection
import pickle
import argparse
import numpy as np

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Sentiment analysis')
    argparser.add_argument('--train_dataset', type=str,
                           default='data/egregorem_sentiment_train_codecs.csv',
                           help='Choose input trainset')
    args = argparser.parse_args()

    print("Dataset", args.train_dataset)

    df_data = pd.read_csv(args.train_dataset, delimiter=",")
    df_data = df_data[['id', 'text', 'target']]

    '''df_pos = df_data[df_data['target']=='Positive']
    print('Num pos: ', df_pos.shape[0]/df_data.shape[0])
    df_neg = df_data[df_data['target'] == 'Negative']
    print('Num neg: ', df_neg.shape[0] / df_data.shape[0])
    df_neu = df_data[df_data['target'] == 'Neutral']
    print('Num neutral: ', df_neu.shape[0] / df_data.shape[0])'''
    #for idx, row in df_data.iterrows():
    #    print(row['text'], row['sentiment'])
    #    if idx > 100:
    #        break
    print("Data shape: ", df_data.shape, df_data.columns)

    df_prep = preprocess(df_data)
    df_data = createFeatures(df_prep)

    print("Columns after preprocessing", df_data.columns.tolist())
    print("Data shape after preprocessing:", df_data.shape)

    # shuffle the corpus and optionaly choose the chunk you want to use if you don't want to use the whole thing - will be much faster
    df_data = df_data.reindex(np.random.seed(42))
    df_data = df_data.sample(frac=1, random_state=42)
    #df_data = df_data[:100]

    y = df_data['target'].values
    X = df_data.drop(['target', 'id'], axis=1)

    # build classification model

    lr = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=123)
    tfidf_unigram = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=10, max_df=0.8)
    tfidf_bigram = TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=False, min_df=20, max_df=0.5)
    tfidf_pos = TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=True, min_df=0.1, max_df=0.6, lowercase=False)
    character_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4), lowercase=False, min_df=4, max_df=0.8)
    bigram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2), lowercase=False, min_df=4, max_df=0.8)
    tfidf_ngram = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=0.1, max_df=0.8)
    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    tfidf_affix_punct = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=0.1, max_df=0.8, tokenizer=affix_punct_tokenize)

    features = [('cst', digit_col()),
        ('unigram', pipeline.Pipeline([('s1', text_col(key='text_clean')), ('tfidf_unigram', tfidf_unigram)])),
        ('bigram', pipeline.Pipeline([('s2', text_col(key='no_punctuation')), ('tfidf_bigram', tfidf_bigram)])),
        ('character', pipeline.Pipeline([('s5', text_col(key='text_clean')), ('character_vectorizer', character_vectorizer),
            ('tfidf_character', tfidf_transformer)])),
        ('affixes', pipeline.Pipeline([('s5', text_col(key='affixes')), ('tfidf_ngram', tfidf_ngram)])),
        ('affix_punct', pipeline.Pipeline([('s5', text_col(key='affix_punct')), ('tfidf_affix_punct', tfidf_affix_punct)])),
    ]
    weights = {'cst': 0.3,
        'unigram': 0.8,
        'bigram': 0.1,
        'character': 0.8,
        'affixes': 0.4,
        'affix_punct': 0.1,
    }

    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
            transformer_list=features,
            transformer_weights=weights,
            n_jobs=1
        )),
        ('scale', Normalizer()),
        ('lr', lr)])
    kfold = model_selection.KFold(n_splits=10, shuffle=False)
    results = model_selection.cross_val_score(clf, X, y, cv=kfold, verbose=20)
    print("CV score:")
    print(results.mean())

    clf.fit(X, y)
    pickle.dump(clf, open('model/lr_clf_sentiment_python3_new.pkl', 'wb'))
    print("Training completed")

