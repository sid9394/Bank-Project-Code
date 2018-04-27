import re
from bs4 import BeautifulSoup
from bs4.element import Comment
import sys
import os
import glob
import csv

#Prog. to extract raw textual data from the body of an html file containing multiple emails and then extract them into a csv file


#To give us body of email
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

#To give Label Data

print()
mypath = 'C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Iciciprojectdata (original)(all)'

#root, dirs, files = os.walk('C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Icici proj').__next__()
#print(dirs)

#To get body content
print()
print()
for root, dirs, files in os.walk(mypath):
     print("dirs---------",dirs)

     for name in dirs:
        print("name----",name)
        htmllink=os.path.join(mypath,name)

        for filename in glob.glob(os.path.join(htmllink, '*.html')):
            print(filename)
            html2=text_from_html(open(filename))
            #print(html2)
            string_input = html2
            input_list = string_input.split() #splits the input string on spaces
            # process string elements in the list and make them integers
            input_list = [str(a) for a in input_list]

            str1=" ".join(input_list)

            findallObj = re.findall(r'Subject:(.*?)(From:|Delivery has failed to these recipients or groups:|Properties)', str1, re.M)

            label=name.split(",")
            print(label)
            xyz=findallObj
            print(xyz)

            with open('C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Final.csv', 'a', encoding="utf-8") as f:

                writer = csv.writer(f)
                writer.writerows(zip(label,xyz))



