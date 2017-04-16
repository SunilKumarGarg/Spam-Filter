import numpy as np
import math

from Tockenizer import PDFFileTockenizer 


def countItemCount(dict, doc): 
    count = 0   
    for d in doc:
        d = d.lower()
        if d == dict:
            count += 1
    return count

def CountWordFrequency(dictionary, documents, numFiles):
    freq = np.empty(shape=(numFiles, len(dictionary)))
    i = 0
    for docNum in range(0,numFiles):
        j = 0
        for dict in dictionary:
            c = countItemCount(dictionary[j], documents[i])
            freq[i][j] = c
            j += 1
        
        i += 1

    return freq

def calculateTF(freq, dictionary,  numFiles):
    tf = np.empty(shape=(numFiles, len(dictionary)))

    i = 0
    for docNum in range(0,numFiles):
        j = 0
        for dict in dictionary:
            if freq[i][j] > 0:
                c = 1 + math.log10(1+math.log10(freq[i][j])) 
            else:
                c = 0
            tf[i][j] = c
            j += 1
        
        i += 1

    return tf

def calculateIDF(freq, dictionary,  numFiles):
    idt = np.empty(shape=len(dictionary))

    i = 0
    for dictItem in range(0,len(dictionary)):
        dictTermFreq = freq[:,i]
        dictItem = len([c for c in dictTermFreq if c > 0])

        if dictItem > 0:
            idt[i] = math.log10((1+numFiles)/dictItem)
        else:
            idt[i] = 0
        i += 1

    return idt

def calculateTFMatrix(dictionary, tf, idt, numFiles):
    idtTFMatrix = np.empty(shape=(numFiles, len(dictionary)))
    idtTF = np.empty(shape=numFiles)
    i = 0
    for docNum in range(0,numFiles):
        j = 0
        idtTF[i] = 0
        for dict in dictionary:        
            idtTF[i] += tf[i][j]*idt[j] 
            idtTFMatrix[i][j] = tf[i][j]*idt[j]
            j += 1
        
        i += 1

    return (idtTFMatrix, idtTF)



pdfFileTockenizer = PDFFileTockenizer()

dictionary =  pdfFileTockenizer.tockenizeDictionary("Data/Spam Dictionary.pdf")
documents, numFiles =  pdfFileTockenizer.tockenizeDocument("Data/Document")
freq = CountWordFrequency(dictionary, documents, numFiles)
tf = calculateTF(freq, dictionary,  numFiles)
idt = calculateIDF(freq, dictionary,  numFiles)
idtTFMatrix, idtTF = calculateTFMatrix(dictionary, tf, idt, numFiles)


np.set_printoptions(precision=3)

print "Term Frequency\n"
print tf

print "\n\n\n\nIDT:\n"
print idt

print "\n\n\n\nTF-IDT Matrix:\n"
print idtTFMatrix

print "\n\n\n\n\n Document Sum of TF-IDT"
print idtTF

