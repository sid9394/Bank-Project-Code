import nltk
from nltk.tag.stanford import StanfordNERTagger
import pandas as pd

url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Test.csv")
documents = pd.read_csv(url)

array = documents.values

x = array[0:, 1]

text = ' '.join(x)

st = StanfordNERTagger('C:\\Users\\sidharth.m\\Desktop\\StandfordNER\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\classifiers\\english.all.3class.distsim.crf.ser.gz',
                       'C:\\Users\\sidharth.m\\Desktop\\StandfordNER\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\stanford-ner.jar')

locn = []

for sent in nltk.sent_tokenize(text):

    tokens = nltk.tokenize.word_tokenize(sent)
    tags = st.tag(tokens)
    for tag in tags:
        if tag[1] in ["LOCATION"]:
            print(tag)
            locn += tag

print(locn)






