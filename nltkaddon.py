import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tag.stanford import StanfordNERTagger
import pandas as pd

url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Test.csv")
documents = pd.read_csv(url)

array = documents.values


x = array[0:, 1]

string = ' '.join(x)

# string = """
# Hey,
# This is my account number - 164805000237.Hope it gets done soon as the Accenture office in Dubai needs a report on it as soon as possible.
# Send me a confirmation on sidharth_mnn@yahoo.co.in.
#
# Regards,
# Sidharth Menon
# 9663993784
# """

def extract_phone_numbers(string):
    r = re.compile(r'(\b91[0-9]{10}\b|\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}$|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}$|\d{3}[-\.\s]??\d{4}$)')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]

def extract_account_numbers(string):
    r = re.compile(r'(?!\b91[0-9]{10}\b)(\d{11,12})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]

def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    # print(sentences)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # print(sentences)
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    # print(sentences)
    return sentences

def extract_names(string):
    st = StanfordNERTagger(
        'C:\\Users\\sidharth.m\\Desktop\\StandfordNER\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\classifiers\\english.all.3class.distsim.crf.ser.gz',
        'C:\\Users\\sidharth.m\\Desktop\\StandfordNER\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\stanford-ner.jar')

    names = []

    for sent in nltk.sent_tokenize(string):

        tokens = nltk.tokenize.word_tokenize(sent)
        tags = st.tag(tokens)


        for tag in tags:
            if tag[1] in ["PERSON"]:
                names += tag

    return names

def extract_org(string):
    st = StanfordNERTagger(
        'C:\\Users\\sidharth.m\\Desktop\\StandfordNER\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\classifiers\\english.all.3class.distsim.crf.ser.gz',
        'C:\\Users\\sidharth.m\\Desktop\\StandfordNER\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\stanford-ner.jar')

    org = []

    for sent in nltk.sent_tokenize(string):

        tokens = nltk.tokenize.word_tokenize(sent)
        tags = st.tag(tokens)
        for tag in tags:
            if tag[1] in ["ORGANIZATION"]:
                org += tag

    return org

def extract_locn(string):
    st = StanfordNERTagger(
        'C:\\Users\\sidharth.m\\Desktop\\StandfordNER\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\classifiers\\english.all.3class.distsim.crf.ser.gz',
        'C:\\Users\\sidharth.m\\Desktop\\StandfordNER\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\stanford-ner.jar')

    locn = []

    for sent in nltk.sent_tokenize(string):

        tokens = nltk.tokenize.word_tokenize(sent)
        tags = st.tag(tokens)
        for tag in tags:
            if tag[1] in ["LOCATION"]:
                locn += tag

    return locn

def extract_entities(string):
     for sent in nltk.sent_tokenize(string):
         for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
             if hasattr(chunk, 'node'):
                 print(chunk.node, ' '.join(c[0] for c in chunk.leaves()))

if __name__ == '__main__':
    numbers = extract_phone_numbers(string)
    emails = extract_email_addresses(string)
    names = extract_names(string)
    accnum = extract_account_numbers(string)
    org = extract_org(string)
    locn = extract_locn(string)

    print("Name - ",names)
    print("Account Number - ",accnum)
    print("Organization - ",org)
    print("Location - ",locn)
    print("Phone Number - ", numbers)
    print("Email Address - ", emails)