from procistaci import *
import numpy as np
from scipy.spatial.distance import cosine
import json
from scipy.sparse import coo_matrix, save_npz, load_npz
import traceback

import os
os.add_dll_directory(r'C:\Program Files\IBM\IBM DATA SERVER DRIVER\bin')
import ibm_db

#SVE S POVEZIVANJEM NA BAZU
def bazaConnect():

    conn_str = (
        "podatci o konkciji, zamjniti pravim podtacima"
    )

    # Povezivanje na bazu
    try:
        conn = ibm_db.connect(conn_str, "", "")
        print("Uspješno povezivanje!")

        bagOfWordsMatrix(conn)

        
    except Exception as e:
        print(f"Greška: {ibm_db.conn_errormsg()}")
        print(e)
        traceback.print_exc() 

    #pozivi koji atributa koji trebaju bazu
    ibm_db.close(conn)

#reliziran NIJE korišten u radu
def bagOfWordsMatrix(conn):
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.OPIS, P.PREDMET FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV LEFT JOIN BAZA.PREDMET p ON n.RECID = p.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"
    stmt = ibm_db.exec_immediate(conn, query)

    rows = []  # indeksi naslova
    cols = []  # indeksi atributa
    data = []  # vrijednosti (1 za binarne atribute)

    indexWIn = dict()
    atrbutiRJ = dict()
    indexWOut = []
    indexAtr = 0

    # Dohvaćanje rezultata
    row = ibm_db.fetch_assoc(stmt)
    br = -1
    autor = ''
    naslov = ''
    tren = set()

    while row:
        d = dict(row.items())

        #provjera je li sljedeci redak za isto djelo 
        if(d['AUTOR'] != autor or d['NASLOV'] != naslov):
            if(br>=0):
                tren.discard('npr')
                tren.discard('itd')
                tren.discard('tzv')

                #spremanje vektora
                for atr in tren:
                    if(atr not in atrbutiRJ):
                        atrbutiRJ[atr] = indexAtr
                        indexAtr = indexAtr + 1
                    
                    cols.append(atrbutiRJ[atr])
                    data.append(1)
                    rows.append(br)

                #spremanje podataka
                if(autor != None):
                    indexWOut.append(str(naslov + '@@' + autor))
                    indexWIn[str(naslov+autor)] = br
                else:
                    indexWOut.append(str(naslov + '@@'))
                    indexWIn[str(naslov)] = br

            br = br + 1
            tren = set()
            autor = d['AUTOR']
            naslov = d['NASLOV']

            #dodaavanje atributa Autora 
            separator = ['priredila', 'priredio', 'priredli', 'priredle', 'uredio', 'uredila', 'uredili', 'uredile',
                'pripremila', 'pripremio', 'pripremili', 'pripremile', 'napisao', 'napisale', 'napisali', 'napisala'
                'pogovor', 'osmislile', 'osmislili', 'osmislio', 'osmislila', 'nacrtao', 'nacrtala', 'nacrtele', 'nacrtale',
                'ilustrirali', 'ilustrirale', 'ilustrirao', 'ilustrirala', ' by ', 'by ']
            ime = lanacUklanjanjeKvantifikatora(d['AUTOR'], separator)

            separator = ['[et al.]', '...', 'et al.', 'et al', '<', '>','[', ']']
            autorAtr = lanacSeparatora(ime, separator)
            tren.update(autorAtr)
        
        #dodavnaje atributa OPISa
        separator = ['.', 'uključujući:', ' - ', '(', ')', '[', ']', ',', ':'] 
        opisAtr = lanacSeparatora(d['OPIS'], separator) 
        tren.update(opisAtr)

        #dodavnaje atributa PREDMETa
        separator = ['.', ' - ', '(', ')', '[', ']', ',', ':']
        opisAtr = lanacSeparatora(d['PREDMET'], separator) 
        tren.update(opisAtr)

        row = ibm_db.fetch_assoc(stmt)

    print(br)

    #-----------------------------------------
    #zadnji prolaz
    tren.discard('npr')
    tren.discard('itd')
    tren.discard('tzv')

    #spremanje vektora
    for atr in tren:
        if(atr not in atrbutiRJ):
            atrbutiRJ[atr] = indexAtr
            indexAtr = indexAtr + 1
                    
        cols.append(atrbutiRJ[atr])
        data.append(1)
        rows.append(br)

    #spremanje podataka
    if(autor != None):
        indexWOut.append(str(naslov + '@@' + autor))
        indexWIn[str(naslov+autor)] = br
    else:
        indexWOut.append(str(naslov + '@@'))
        indexWIn[str(naslov)] = br


    #-----------------------------------------
    #stvaranje matrice 
    # Stvaramo COO matricu
    broj_naslova = max(rows) + 1
    broj_atributa = len(atrbutiRJ)
    sparse_matrix = coo_matrix((data, (rows, cols)), 
                              shape=(broj_naslova, broj_atributa), 
                              dtype=np.bool_)
    save_npz('sparse_matrix.npz', sparse_matrix)


    #spremanje pomoćnih 
    with open('indexWIn.json', 'w') as file:
        json.dump(indexWIn, file)
    with open('indexWOut.json', 'w') as file:
        json.dump(indexWOut, file)
    with open('AtributiRjBagOfWords.json', 'w') as file:
        json.dump(atrbutiRJ, file)

    
def bagOfWordsUDKMatrix(conn):
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.UDK FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"
    
    stmt = ibm_db.exec_immediate(conn, query)

    rows = []  # indeksi naslova
    cols = []  # indeksi atributa
    data = []  # vrijednosti (1 za binarne atribute)

    indexWIn = dict()
    atrbutiRJ = dict()
    indexWOut = []
    indexAtr = 0

    # Dohvaćanje rezultata
    row = ibm_db.fetch_assoc(stmt)
    br = -1
    autor = ''
    naslov = ''
    tren = set()

    while row:
        d = dict(row.items())

        #provjera je li sljedeci redak za isto djelo 
        if(d['AUTOR'] != autor or d['NASLOV'] != naslov):
            if(br>=0):

                #spremanje vektora
                for atr in tren:
                    if(atr not in atrbutiRJ):
                        atrbutiRJ[atr] = indexAtr
                        indexAtr = indexAtr + 1
                    
                    cols.append(atrbutiRJ[atr])
                    data.append(1)
                    rows.append(br)

                #spremanje podataka
                if(autor != None):
                    indexWOut.append(str(naslov + '@@' + autor))
                    indexWIn[str(naslov+autor)] = br
                else:
                    indexWOut.append(str(naslov + '@@'))
                    indexWIn[str(naslov)] = br

            br = br + 1
            tren = set()
            autor = d['AUTOR']
            naslov = d['NASLOV']

        #dodavnaje atributa UDK
        separator = r'\.|-|\:|\(|\)|\"|\/|\+|\,'
        opisAtr = lanacSeparatoraUdk(d['UDK'], separator) 
        tren.update(opisAtr)

        row = ibm_db.fetch_assoc(stmt)

    print(br)

    #-------------- zadnji prolaz ------------------
    #spremanje vektora
    for atr in tren:
        if(atr not in atrbutiRJ):
            atrbutiRJ[atr] = indexAtr
            indexAtr = indexAtr + 1
                    
        cols.append(atrbutiRJ[atr])
        data.append(1)
        rows.append(br)

    #spremanje podataka
    if(autor != None):
        indexWOut.append(str(naslov + '@@' + autor))
        indexWIn[str(naslov+autor)] = br
    else:
        indexWOut.append(str(naslov + '@@'))
        indexWIn[str(naslov)] = br


    #-----------------------------------------
    #stvaranje matrice 
    # Stvaramo COO matricu
    broj_naslova = max(rows) + 1
    broj_atributa = len(atrbutiRJ)
    sparse_matrix = coo_matrix((data, (rows, cols)), 
                              shape=(broj_naslova, broj_atributa), 
                              dtype=np.bool_)
    save_npz('sparse_matrix_UDK.npz', sparse_matrix)


    #spremanje pomoćnih 
    with open('indexWInUDK.json', 'w') as file:
        json.dump(indexWIn, file)
    with open('indexWOutUDK.json', 'w') as file:
        json.dump(indexWOut, file)
    with open('AtributiRjBagOfWordsUDK.json', 'w') as file:
        json.dump(atrbutiRJ, file)


def bagOfWordsAutoriMatrix(conn):
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"
    stmt = ibm_db.exec_immediate(conn, query)

    rows = []  # indeksi naslova
    cols = []  # indeksi atributa
    data = []  # vrijednosti (1 za binarne atribute)

    indexWIn = dict()
    atrbutiRJ = dict()
    indexWOut = []
    indexAtr = 0

    # Dohvaćanje rezultata
    row = ibm_db.fetch_assoc(stmt)
    br = -1
    autor = ''
    naslov = ''
    tren = set()

    while row:
        d = dict(row.items())

        #provjera je li sljedeci redak za isto djelo 
        if(d['AUTOR'] != autor or d['NASLOV'] != naslov):
            if(br>=0):

                #spremanje vektora
                for atr in tren:
                    if(atr not in atrbutiRJ):
                        atrbutiRJ[atr] = indexAtr
                        indexAtr = indexAtr + 1
                    
                    cols.append(atrbutiRJ[atr])
                    data.append(1)
                    rows.append(br)

                #spremanje podataka
                if(autor != None):
                    indexWOut.append(str(naslov + '@@' + autor))
                    indexWIn[str(naslov+autor)] = br
                else:
                    indexWOut.append(str(naslov + '@@'))
                    indexWIn[str(naslov)] = br

            br = br + 1
            tren = set()
            autor = d['AUTOR']
            naslov = d['NASLOV']

            #dodaavanje atributa Autora 
            separator = ['priredila', 'priredio', 'priredli', 'priredle', 'uredio', 'uredila', 'uredili', 'uredile',
                'pripremila', 'pripremio', 'pripremili', 'pripremile', 'napisao', 'napisale', 'napisali', 'napisala'
                'pogovor', 'osmislile', 'osmislili', 'osmislio', 'osmislila', 'nacrtao', 'nacrtala', 'nacrtele', 'nacrtale',
                'ilustrirali', 'ilustrirale', 'ilustrirao', 'ilustrirala', ' by ', 'by ']
            ime = lanacUklanjanjeKvantifikatora(d['AUTOR'], separator)

            separator = ['[et al.]', '...', 'et al.', 'et al', '<', '>','[', ']']
            autorAtr = lanacSeparatora(ime, separator)
            tren.update(autorAtr)

        row = ibm_db.fetch_assoc(stmt)

    print(br)

    #-------------- zadnji prolaz ------------------
    #spremanje vektora
    for atr in tren:
        if(atr not in atrbutiRJ):
            atrbutiRJ[atr] = indexAtr
            indexAtr = indexAtr + 1
                    
        cols.append(atrbutiRJ[atr])
        data.append(1)
        rows.append(br)

    #spremanje podataka
    if(autor != None):
        indexWOut.append(str(naslov + '@@' + autor))
        indexWIn[str(naslov+autor)] = br
    else:
        indexWOut.append(str(naslov + '@@'))
        indexWIn[str(naslov)] = br


    #-----------------------------------------
    #stvaranje matrice 
    # Stvaramo COO matricu
    broj_naslova = max(rows) + 1
    broj_atributa = len(atrbutiRJ)
    sparse_matrix = coo_matrix((data, (rows, cols)), 
                              shape=(broj_naslova, broj_atributa), 
                              dtype=np.bool_)
    save_npz('sparse_matrix_Autor.npz', sparse_matrix)


    #spremanje pomoćnih 
    with open('indexWInAutor.json', 'w') as file:
        json.dump(indexWIn, file)
    with open('indexWOutAutor.json', 'w') as file:
        json.dump(indexWOut, file)
    with open('AtributiRjBagOfWordsAutor.json', 'w') as file:
        json.dump(atrbutiRJ, file)
