# Library imports
import nltk
## Uncomment below line if punkt or such error pops up
# nltk.download('punkt', quiet=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import re
import warnings

warnings.filterwarnings("ignore")


# Scene creation function
def scene_creation(text, scene_len):
    '''
    Method Params:
    text: The text script for which the scenes are to be created
    scene_len: string short for short scene and long for long scene

    Function returns: A list of textual data i.e., scenes
    '''

    # Preprocessing original text data before generating keywords
    # Removing punctuation from string data and
    # converting the text into lowercase
    new_text = re.sub(r'[^\w\s]', '', text).lower()

    # Creating list of sentences
    sen_list = nltk.tokenize.sent_tokenize(text)

    # # Counting number of sentences
    sen_count = len(sen_list)

    # Candidate Keywords/Keyphrases
    n_gram_range = (1, 1)
    stop_words = list(en_stop)

    # Extract candidate words/phrases
    tfidf = TfidfVectorizer(ngram_range=n_gram_range, stop_words=stop_words)
    t = tfidf.fit_transform([new_text])
    keywords = tfidf.get_feature_names()

    # Setting number of keywords and threshold for number of words as per long or short scene
    # For short scene, 5 keywords are used and word threshold is set to 30
    # For long scene, 9 keywords are used and word threshold is set to 60
    if scene_len == "short":
        num_key_scene = 5
        word_thresh = 30
    elif scene_len == "long":
        num_key_scene = 10
        word_thresh = 60

    # Checking the sentences for the keywords

    count = 0
    scenes = []
    keyword_list = []
    text_temp = ""
    keyword_temp = []

    for i in range(len(sen_list)):

        # If number of keywords or number of words exceed the threshold or its the last iteration
        # Append the text in the scene and set count as zero and empty temp_text
        if (count > num_key_scene or len(re.findall(r'\w+', text_temp)) >= word_thresh or len(
                re.findall(r'\w+', text_temp + sen_list[i])) >= word_thresh):
            scenes.append(text_temp[:-1])
            count = 0
            keyword_list.append(keyword_temp)
            keyword_temp = []
            text_temp = ""

        # No sentence would be omitted
        if sen_list[i] not in text_temp:
            text_temp += sen_list[i] + " "

        # Checking if keyword is present in the sentence
        # If yes, then increasing the count of keywords encountered
        for j in range(len(keywords)):
            if ((keywords[j] in sen_list[i].lower()) and (count <= num_key_scene) and (
                    keywords[j] not in keyword_temp)):
                count += 1
                # keyword_temp += keywords[j] +", "
                keyword_temp.append(keywords[j])
            if count > num_key_scene:
                break

        # If it is the last iteration i.e., last sentence in the script
        # Then, append the text in the scene
        if i == (len(sen_list) - 1):
            scenes.append(text_temp[:-1])
            keyword_list.append(keyword_temp)
            keyword_temp = []

    # scenes = list(filter(None, scenes))
    # keyword_list = list(filter(None, keyword_list))

    d = {"sentences": {ind: scenes[ind] for ind in range(len(scenes))},
         "keywords": {ind: keyword_list[ind] for ind in range(len(scenes))}}

    return d