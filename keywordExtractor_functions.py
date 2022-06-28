from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
import yake
import pandas as pd

from keybert import KeyBERT
from collections import Counter
import spacy
from string import punctuation
import textrank

nlp = spacy.load("en_core_web_sm")



def extract_textrank(text, kw_top_n):
    tr4w = textrank.TextRank4Keyword()
    tr4w.analyze(text, candidate_pos=['NOUN', 'PROPN'], window_size=4, lower=False)
    orderedDict = tr4w.get_keywords()

    df = pd.DataFrame({"Feature": orderedDict.keys(), "Score": orderedDict.values()}).sort_values('Score', ascending=False)[0:kw_top_n].reset_index()[
        ['Feature', 'Score']]
    return df


def extract_spacy(text, kw_top_n):
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    doc = nlp(text)
    important_words = []

    for token in doc:
        if not token.text in punctuation:
            if token.pos_ in pos_tag:
                important_words.append(token.text)

    top_n_words = Counter(important_words).most_common(kw_top_n)

    key_dict = {pair[0]: pair[1] for pair in top_n_words}
    df = pd.DataFrame({"Feature": key_dict.keys(),
                       "Score": key_dict.values()}).sort_values('Score', ascending=False)[0:kw_top_n].reset_index()[
        ['Feature', 'Score']]
    return df


def extract_keybert(doc, kw_top_n):
    kw_model = KeyBERT()
    keywords_tuplist = kw_model.extract_keywords(doc)
    key_dict = {}
    for pair in keywords_tuplist:
        key_dict[pair[0]] = pair[1]
    df = pd.DataFrame({"Feature": key_dict.keys(),
                       "Score": key_dict.values()}).sort_values('Score',ascending=False)[0:kw_top_n].reset_index()[['Feature', 'Score']]
    return df


def extract_TFIDF(sentlist, kw_top_n):
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(sentlist)

    features_name = tfidf_vect.get_feature_names_out()
    features_score = tfidf_vect.idf_

    df = pd.DataFrame({"Feature":features_name, "Score":features_score}).sort_values('Score', ascending=False)[0:kw_top_n].reset_index()[['Feature','Score']]
    return df


def extract_rake(textlist, kw_top_n):
    r = Rake()
    r.extract_keywords_from_sentences(textlist)
    keywords_tuple = r.get_ranked_phrases_with_scores()
    keywords_dict = {pair[1]:pair[0] for pair in keywords_tuple}

    df = pd.DataFrame({"Feature": keywords_dict.keys(), "Score": keywords_dict.values()}).sort_values('Score', ascending=False)[
         0:kw_top_n].reset_index()[['Feature', 'Score']]
    return df


def extract_yake(textlist, kw_top_n):
    text = ' '.join(textlist)

    r = yake.KeywordExtractor(top=kw_top_n)
    keywords_tuplist = r.extract_keywords(text)

    key_dict = {}
    for pair in keywords_tuplist:
        key_dict[pair[0]] = pair[1]

    df = pd.DataFrame({"Feature": key_dict.keys(), "Score": key_dict.values()})
    return df

