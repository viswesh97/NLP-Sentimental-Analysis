import os
from nltk.corpus import stopwords
import datetime
from pytz import timezone

# ------------------------------------------Path Management ------------------------------#
DIRE_NAME = os.path.dirname(os.path.realpath(__file__))
# DATA_FILE = DIRE_NAME + "/train.csv"
DATA_FILE = DIRE_NAME + "/train_data.csv"
# -------------------------------------------Terminologies -------------------------------#
ENGLISH = "english"
SENTIMENT = "Sentiment"
ID = "id"
DATE = "date"
QUERY = "query"
USER = "user"
TWEET = "tweet"
NEGATIVE = "negative"
FEAR = "fear"
ANGER = "anger"
DISGUST = "disgust"
SADNESS = "sadness"
POSITIVE = "positive"
ANTCIPATION = "anticipation"
TRUST = "trust"
JOY = "joy"
SURPRISE = "surprise"

# --------------------------------------------Common Functionalities ----------------------#
STOPWORDS = set(stopwords.words('english'))

# --------------------------------------------Printing seperator --------------------------#
SEPERATOR= "\n##########################\n\t"
INSIDESEPERATOR = "\n--------------------------\n\t"
PLEASE_WAIT = "Please wait till the process finishes,\nNote : It will take much time please be patient."
PROJECT_NAME= "Detection of Depression-Related Posts in Social Media Forum ( Twitter )"
READING_CSV = "Reading the datasets"
REMOVING_URLS= "Removing URLs"
CLEANING_STOPWORDS = "Cleaning StopWords"
REMOVING_PUNCTUATIONS = "Removing Punctuations"
TOKENIZING_DATA = "Tokenizing the Data"
LEMMETAZING_DATA = "Lemmatizing the Data"
STEMMING_DATA = "Stemming the Data"
PROCESSING_UNIGRAM = "Processing Unigram"
PROCESSING_BIGRAM = "Processing Bigram"
PROCESSING_LDA = "Processing Latent Dirichlet Allocation"
PROCESSING_LIWC_NRC = "Processing Linguistic Inquiry and Word Count / NRC Emotion Lexicon Model"
PROCESSING_LDA_NRC_UNIGRAM = "Processing Latent Dirichlet Allocation NRC Unigram"
PROCESSING_LDA_NRC_BIGRAM ="Processing Latent Dirichlet Allocation NRC Bigram"
PLOTTING_DATAS = "Plotting the Data and Analysis"
UNIGRAM_PLOTTING = "Plotting Unigram Accuracy"
BIGRAM_PLOTTING = "Plotting Bigram Accuracy"
LDA_PLOTTING = "Plotting Latent Dirichlet Allocation Accuracy"
NRC_PLOTTING = "Plotting Linguistic Inquiry and Word Count / NRC Emotion Lexicon Model"
LDA_NRC_UNIGRAM_PLOTTING = "Plotting Latent Dirichlet Allocation Accuracy + " \
                           "Linguistic Inquiry and Word Count / NRC Emotion Lexicon Model + " \
                           "Unigram"
LDA_NRC_BIGRAM_PLOTTING = "Plotting Latent Dirichlet Allocation Accuracy + " \
                           "Linguistic Inquiry and Word Count / NRC Emotion Lexicon Model + " \
                           "Bigram"
DOWNLOADING_DEPENDENCIES = "Please Wait, We are Downloading Dependencies \n"
PACKAGE_DOWNLOAD = "If you are running this project first time please run NLTK Download for smooth running"
THANK_YOU = "You can Exit Now \n\n Thank You !!!"