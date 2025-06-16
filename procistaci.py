import re

def makniSeparator(izvor, separator):
    if(len(izvor.strip().split(separator))==1):
        return izvor

    else:
        polje =[]
        for one in izvor.strip().split(separator):
            polje.append(one.strip())
        return polje
    
def lanacSeparatora(izvor, poljeSepartora):
    if izvor:
        rezPrvi = [izvor.strip().lower()]
        rezDrugi = []

        for separator in poljeSepartora:
            # Ako `rezPrvi` nije prazan, obrađujemo `rezPrvi`
            while rezPrvi:
                element = rezPrvi.pop()
                # Dijelimo element po trenutnom separatoru
                dijelovi = element.split(separator)

                for dio in dijelovi:
                    dio = dio.strip()
                    # Ako je duljina dijela veća od 2, dodaj ga u `rezDrugi`
                    if len(dio) > 2:
                        rezDrugi.append(dio)

            # Prebacujemo rezultate natrag u `rezPrvi`
            rezPrvi, rezDrugi = rezDrugi, []

        return rezPrvi

    else:
        return []
    
def lanacSeparatoraUdk(izvor, poljeSepartora):
    rezulat = []
    
    while (izvor):
        izvor = izvor.strip()
        rezulat.append(izvor)
        tren = re.split(poljeSepartora, izvor).pop()
        izvor = izvor[:-len(tren)-1].strip()

    return rezulat
    
def lanacUklanjanjeKvantifikatora(izvor, poljeSepartora):
    if(izvor != None):
        izvor =izvor.strip().lower()

        for separator in poljeSepartora:
            if len(izvor.split(separator))>1:
                izvor = (izvor.split(separator))[-1].strip()

        return izvor
    
    return []

def jedinstvenost(polje, rjecnikToInd, rjecnikToStr, index):
    for one in polje:
        if (one not in rjecnikToInd):
            rjecnikToInd[one] = index
            rjecnikToStr[index] = one
            index = index +1
    
    return index