import datetime

from config import *
import pandas as pd
from nltkDownload import download_nltk
from Utils import *
from Unigram import *
from Bigram import *
from Lda import *
from NRCLex import *
from LdaNrclexUnigram import *
from LdaNrclexBigram import *
import datetime
from pytz import timezone



def main():
    print(SEPERATOR)
    print(PROJECT_NAME)
    print(SEPERATOR)
    print(PLEASE_WAIT)
    download_nltk()
    print(SEPERATOR)
    print("Current Date & Time : " + str(datetime.datetime.now(timezone('Asia/Kolkata'))))
    print(SEPERATOR)
    print(READING_CSV)

    data = pd.read_csv(DATA_FILE, names=[SENTIMENT, ID, DATE, QUERY, USER, TWEET])
    data.shape
    data[SENTIMENT].value_counts()
    data[SENTIMENT] = data[SENTIMENT].replace(4, 1)
    data = data.drop(columns=[ID, DATE, QUERY, USER])
    data.head()
    data[TWEET] = data[TWEET].str.lower()

    print(SEPERATOR)
    print(REMOVING_URLS)

    # Removing URLs
    data[TWEET] = data[TWEET].apply(lambda x: remove_url(x))
    data[TWEET].head()

    print(SEPERATOR)
    print(CLEANING_STOPWORDS)

    # Cleaning Stopwords
    data[TWEET] = data[TWEET].apply(lambda text: cleaning_stopwords(text))
    data[TWEET].head()

    print(SEPERATOR)
    print(REMOVING_PUNCTUATIONS)

    # Removing Punctuations
    data[TWEET] = data[TWEET].apply(lambda x: remove_punctuations(x))
    data[TWEET].head()

    print(SEPERATOR)
    print(TOKENIZING_DATA)

    # Tokenizing the tweets
    data[TWEET] = data[TWEET].apply(lambda x: tokenizer(x))
    data[TWEET].head()

    print(SEPERATOR)
    print(LEMMETAZING_DATA)

    data[TWEET] = data[TWEET].apply(lambda x: lematizing(x))
    data[TWEET].head()

    print(SEPERATOR)
    print(STEMMING_DATA)

    data[TWEET] = data[TWEET].apply(lambda x: stemming(x))
    data[TWEET].head()

    # Joining the tweets into strings
    data[TWEET] = data[TWEET].apply(lambda x: ' '.join([w for w in x]))
    data[TWEET] = data[TWEET].apply(lambda x: ' '.join([w for w in x.split()]))

    x = get_x(data)
    y = get_y(data)

    print(SEPERATOR)
    print(PROCESSING_UNIGRAM)

    unigramData = unigram(x,y)

    print(SEPERATOR)
    print(PROCESSING_BIGRAM)

    bigramData = bigram(x,y)

    print(SEPERATOR)
    print(PROCESSING_LDA)

    ldaData = lda(x,y)

    print(SEPERATOR)
    print(PROCESSING_LIWC_NRC)

    nrcData = nrc_liwc(x,y)

    print(SEPERATOR)
    print(PROCESSING_LDA_NRC_UNIGRAM)

    LdaNrcUnigramData = lda_nrc_unigram(x, y, nrcData['SentiScoreList'], unigramData['Unigram'])

    print(SEPERATOR)
    print(PROCESSING_LDA_NRC_BIGRAM)

    LdaNrcBigramData = lda_nrc_bigram(x, y,nrcData['SentiScoreList'], bigramData['Bigram'])
    
    print(SEPERATOR)
    print(PLOTTING_DATAS)

    print(SEPERATOR)
    print(UNIGRAM_PLOTTING)

    unigram_plotting(unigramData['ClassName'], unigramData['Accuracy'])

    print(SEPERATOR)
    print(BIGRAM_PLOTTING)

    bigram_plotting(bigramData['ClassName'],bigramData['Accuracy'])

    print(SEPERATOR)
    print(LDA_PLOTTING)

    lda_plotting(ldaData['ClassName'],ldaData['Accuracy'])

    print(SEPERATOR)
    print(NRC_PLOTTING)

    nrc_plotting(nrcData['ClassName'], nrcData['Accuracy'])

    print(SEPERATOR)
    print(LDA_NRC_UNIGRAM_PLOTTING)

    lda_nrc_unigram_plotting(LdaNrcUnigramData['ClassName'],LdaNrcUnigramData['Accuracy'])

    print(SEPERATOR)
    print(LDA_NRC_BIGRAM_PLOTTING)

    lda_nrc_bigram_plotting(LdaNrcBigramData['ClassName'],LdaNrcBigramData['Accuracy'])

    print(SEPERATOR)
    print("Current Date & Time : " + str(datetime.datetime.now(timezone('Asia/Kolkata'))))
    print(SEPERATOR)
    print(THANK_YOU)
    print(SEPERATOR)



if __name__ == '__main__':
    main()