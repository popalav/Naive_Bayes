import re
from typing import Tuple
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from nltk.tokenize import word_tokenize
from  contractions import contractions_dict


# flake8: noqa: E501

# ____________________________ Load & Clean Data ______________________________________

spam_data = pd.read_csv('spam.csv', nrows=100)
# drop last 3 columns, they are empty
spam_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

def _separate_data(spam_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns 2 separate DataFrames, one for spam and one for ham"""
    ham, spam = spam_data.copy(deep=True), spam_data.copy(deep=True)
    ham = ham.drop(ham[ham['v1'] == 'spam'].index) 
    spam = spam.drop(spam[spam['v1'] == 'ham'].index)      
    return(ham.reset_index(), spam.reset_index())

def _lower_text(text_data: pd.DataFrame) -> pd.DataFrame:
    """Convert text to lower case"""
    text_data = text_data.apply(lambda x: x.astype(str).str.lower())
    return text_data

def _remove_text_paranthesis(text_data: pd.DataFrame) -> pd.DataFrame:
    """Removes the text inside paranthesis () and []"""
    p = re.compile(r'\([^)]*\)')
    text_data['v2'] = [p.sub('', x) for x in text_data['v2'].tolist()]
    return text_data

def _remove_s(text_data: pd.DataFrame) -> pd.DataFrame:
    """Removes 's"""
    p = re.compile(r"'s\b")
    text_data['v2'] = [p.sub('', x) for x in text_data['v2'].tolist()]
    return text_data

def _remove_punctuation(text_data: pd.DataFrame) -> pd.DataFrame:
    """Remove punctuation and special characters """
    p = re.compile(r"[^a-zA-Z]")
    text_data = [p.sub('', word) for x in text_data.tolist() for word in word_tokenize(x)]
    return text_data

def _expand_contractions(text_data: pd.DataFrame) ->pd.DataFrame:
    """Expand contractions e.g. can't -> can not """
    text_data['v2'] = ' '.join([contractions_dict[i] if i in contractions_dict else i for i in word_tokenize(text_data[v2])])
    return text_data

def main():
    low = _lower_text(spam_data)
    print("LOW", low)
    ham, spam = _separate_data(spam_data)
    # print("HAM", ham)
    # print("SPAM", spam)
    paranthesis = _remove_text_paranthesis(spam_data)
    s = _remove_s(spam_data)
    print("S", s)
    punctuation = _remove_punctuation(spam_data)
    print("PUNCTUATION", punctuation)
    print("PARANTHESYS",paranthesis)
    expand = _expand_contractions(spam_data)
    print("EXPAND", expand)
    expand_pun = _remove_punctuation(expand)
    print("RIGT PUNCTUATION", expand_pun)


if __name__=="__main__":
    main()