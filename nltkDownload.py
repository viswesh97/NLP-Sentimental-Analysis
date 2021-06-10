import nltk
from config import *

def download_nltk():
    print(SEPERATOR)
    print(DOWNLOADING_DEPENDENCIES)
    try:
        nltk.download('wordnet')
    except:
        pass