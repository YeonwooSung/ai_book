
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = float(len(bow))
    for word, count in wordDict.items():
        tfDict[word] = count / bowCount
    return tfDict


def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        v = float(val)
        idf_val = N / v
        idfDict[word] = math.log10(idf_val)
        
    return idfDict


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf
