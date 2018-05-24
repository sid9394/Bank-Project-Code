import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tag.stanford import StanfordNERTagger
import pandas as pd

def extractspecificdata(url1):

    documents = pd.read_csv(url1)

    array = documents.values

    x = array[0:, 1]

    string = ' '.join(x)

    def extract_phone_numbers(string):
        r = re.compile(
            r'(\b08[0-9][9]$|\d{8}$|\b91[0-9]{10}\b$|\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}$|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}$|\d{3}[-\.\s]??\d{4}$)')
        phone_numbers = r.findall(string)
        return [re.sub(r'\D', '', number) for number in phone_numbers]

    def extract_account_numbers(string):
        r = re.compile(r'(?!\b91[0-9]{10}\b)(\d{11,12})')
        phone_numbers = r.findall(string)
        return [re.sub(r'\D', '', number) for number in phone_numbers]

    def extract_email_addresses(string):
        r = re.compile(r'[\w\.-]+@[\w\.-]+')
        return r.findall(string)

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

    numbers = extract_phone_numbers(string)
    emails = extract_email_addresses(string)
    names = extract_names(string)
    accnum = extract_account_numbers(string)
    org = extract_org(string)
    locn = extract_locn(string)

    print("*Details extracted from each file*")
    print("______________________________________________________________")
    print("Name - ", names)
    print("--------------------------------------------------------------")
    print("Account Number - ", accnum)
    print("--------------------------------------------------------------")
    print("Organization - ", org)
    print("--------------------------------------------------------------")
    print("Location - ", locn)
    print("--------------------------------------------------------------")
    print("Phone Number - ", numbers)
    print("--------------------------------------------------------------")
    print("Email Address - ", emails)
    print("--------------------------------------------------------------")