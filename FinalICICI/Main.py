from Extracttxt import extracttxt
from Classifytxt import clasifytxt
from SpecificData import extractspecificdata

#Folder containing test data HTML files
filepath='C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\FinalICICI'

#To extract text from HTML files and write to CSV
extracttxt(filepath)

#CSV file
url1 = 'C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\FinalICICI\\AlphaOne.csv'

#To classify the type of grievance being communicated
clasifytxt(url1)

#To extract specific details pertaining to each email
extractspecificdata(url1)




