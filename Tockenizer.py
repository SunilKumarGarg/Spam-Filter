
import PyPDF2
import nltk
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import re
from nltk.stem.lancaster import LancasterStemmer


class PDFFileTockenizer:

    def tockenizeDictionary(self, fileToTockenize):

        d = open(fileToTockenize, 'rb')

        pdfReader = PyPDF2.PdfFileReader(d) 
        pageObj = pdfReader.getPage(0)
        a =  pageObj.extractText()
        a = re.sub(r"^.*\n", "", a)
        tokens = nltk.word_tokenize(a)
        filtered_words = [word for word in tokens if word not in stopwords.words('english')]
        f = []
        LS = LancasterStemmer()
        for ff in filtered_words:
            f.append(LS.stem(ff))


        return list(set(f))

    def tockenizeDocument(self, folderToTockenize):

        onlyfiles = [f for f in listdir(folderToTockenize) if isfile(join(folderToTockenize, f))]

        a = []
        for file in onlyfiles:
            d = open(join(folderToTockenize, file), 'rb')
            pdfReader = PyPDF2.PdfFileReader(d)            
            pageObj = pdfReader.getPage(0)
            aa =  pageObj.extractText()
            tokens = nltk.word_tokenize(aa)
            filtered_words = [word for word in tokens if word not in stopwords.words('english')]

            f = []
            LS = LancasterStemmer()
            for ff in filtered_words:
                f.append(LS.stem(ff))

            a.append(f)
            
        return (a, len(onlyfiles))
