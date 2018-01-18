#!/bin/python

from nltk.util import ngrams
import string
import re
from nltk.stem.porter import *

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    fNameF = open('./data/lexicon/firstname.5k', 'r')
    lNameF = open('./data/lexicon/lastname.5000', 'r')
    fName = fNameF.read().splitlines()
    fName = map(str.lower, fName)
    lName = lNameF.read().splitlines()
    lName = map(str.lower, lName)
    global f_Name, l_Name
    f_Name = set(fName)
    l_Name = set(lName)

    file_tv = open('./data/lexicon/tv.tv_program', 'r')
    tv_data = file_tv.read().splitlines()
    tv_data = map(str.lower, tv_data)
    global tv_set
    tv_set = set(tv_data)

    file_stop = open('./data/lexicon/english.stop', 'r')
    stop_data = file_stop.read().splitlines()
    stop_data = map(str.lower, stop_data)
    global stop_set
    stop_set = set(stop_data)

    file_country = open('./data/lexicon/location.country', 'r')
    country_data = file_country.read().splitlines()
    country_data = map(str.lower, country_data)
    global country_set
    country_set = set(country_data)

    file_sportsl = open('./data/lexicon/sports.sports_league', 'r')
    sportsl_data = file_sportsl.read().split()
    sportsl_data = map(str.lower, sportsl_data)
    global sportsl_set
    sportsl_set = set(sportsl_data)

    file_sportst = open('./data/lexicon/sports.sports_team', 'r')
    sportst_data = file_sportst.read().split()
    sportst_data = map(str.lower, sportst_data)
    global sportst_set
    sportst_set = set(sportst_data)

    file_company = open('./data/lexicon/business.consumer_company', 'r')
    company_data = file_company.read().split()
    company_data = map(str.lower, company_data)
    global company_set
    company_set = set(company_data)

    file_company = open('./data/lexicon/venture_capital.venture_funded_company', 'r')
    company_data = file_company.read().split()
    company_data = map(str.lower, company_data)
    global company1_set
    company1_set = set(company_data)

    file_news = open('./data/lexicon/book.newspaper', 'r')
    news_data = file_news.read().split()
    news_data = map(str.lower, news_data)
    global news_set
    news_set = set(news_data)

    global month_set
    month_set = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                 "november", "december", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sept", "oct", "nov", "dec"]

    global stemmer
    stemmer = PorterStemmer()

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")

    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])

    #ftrs.append("WORD=" + word)
    stem_word = stemmer.stem(word.lower())
    ftrs.append("STEM_" + stem_word)
    ftrs.append("LCASE=" + word.lower())

    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    if re.match(r'[0-9][0-9]', word):
        ftrs.append("IS_DOUBLE_DIGIT")

    if word.lower() in month_set:
        ftrs.append("IS_MONTH")

    if word.lower() in ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]:
        ftrs.append("IS_DAY")

    if word.lower().startswith("http"):
        ftrs.append("IS_URL")

    if i > 0 and word[0].isupper() and \
                    sent[i - 1] in ["in", "to", "from", "across", "around", "at", "for", "by"]:
        ftrs.append("PREV_WORD_IS_PREP")

    if len(word) > 2:
        ftrs.append("PREFIX_" + word[0:2].upper())
        ftrs.append("SUFFIX_" + word[-2:].upper())
    if len(word) > 3:
        ftrs.append("PREFIX_" + word[0:3].upper())
        ftrs.append("SUFFIX_" + word[-3:].upper())

    if "-" in word:
        ftrs.append("HAS_HYPHEN")
        subtext = word.split('-')
        ftrs.append("SUBTEXT_" + subtext[0].upper() + subtext[1].upper())

    if word[0] is "@":
        ftrs.append("IS_FIRST_AT")
    if word[0] is "#":
        ftrs.append("IS_FIRST_HASH")

    if word[0].isupper():
        ftrs.append("IS_FIRST_CAPS")

    if word.lower() in stop_set:
        ftrs.append("STOP_WORD")

    if word.lower() in f_Name:
        ftrs.append("IS_FNAME")
    if word.lower() in l_Name:
        ftrs.append("IS_LNAME")

    if word.lower() in country_set:
        ftrs.append("IS_COUNTRY")
    else:
        if i < len(sent) - 1:
            s = sent[i] + " " + sent[i+1]
            if s.lower() in tv_set:
                ftrs.append("IS_COUNTRY")
        if i > 0:
            s = sent[i-1] + " " + sent[i]
            if s.lower() in tv_set:
                ftrs.append("IS_COUNTRY")

    if word.lower() in sportsl_set or word.lower() in sportst_set:
        ftrs.append("IS_SPORTS")

    if word.lower() in company_set or word.lower() in company1_set:
        ftrs.append("IS_COMPANY")

    if i < len(sent) - 1:
        s = sent[i] + " " + sent[i+1]
        if s.lower() in tv_set:
            ftrs.append("IS_TV_SHOW")
    if i > 0:
        s = sent[i-1] + " " + sent[i]
        if s.lower() in tv_set:
            ftrs.append("IS_TV_SHOW")
    if i > 1:
        s = sent[i-2] + " " + sent[i-1] + " " + sent[i]
        if s.lower() in tv_set:
            ftrs.append("IS_TV_SHOW")
    if i < len(sent) - 2:
        s = sent[i] + " " + sent[i+1] + " " + sent[i+2]
        if s.lower() in tv_set:
            ftrs.append("IS_TV_SHOW")

    shape = ""
    for ch in word:
        if ch.islower():
            shape += "x"
        else:
            shape += "X"
    ftrs.append("SHAPE_" + shape)

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

        if "IS_SPORTS" in ftrs and "NEXT_IS_SPORTS" in ftrs:
            ftrs.append("SPORTS_CLUSTER")
        if "IS_SPORTS" in ftrs and "PREV_IS_SPORTS" in ftrs:
            ftrs.append("SPORTS_CLUSTER")

    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "am", "GoinG", "to", "Microsoft"]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
