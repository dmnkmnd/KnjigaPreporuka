import os
os.add_dll_directory(r'C:\Program Files\IBM\IBM DATA SERVER DRIVER\bin')
import ibm_db
import traceback

from procistaci import *
import json
import numpy as np
import fasttext
from scipy import sparse
from sentence_transformers import SentenceTransformer
import random

#SVE S POVEZIVANJEM NA BAZU
def bazaConnect():

    conn_str = (
        "podatci o konkciji, zamjniti pravim podtacima"
    )

    # Povezivanje na bazu
    try:
        conn = ibm_db.connect(conn_str, "", "")
        print("Uspješno povezivanje!")

        pripremaPodatciSamoOpisi(conn)
        
        
    except Exception as e:
        print(f"Greška: {ibm_db.conn_errormsg()}")
        print(e)
        traceback.print_exc() 

    #pozivi koji atributa koji trebaju bazu
    ibm_db.close(conn)


#PRIPREMA PODATAKA IZ KORISNIČKIH POSUDBI ZA "leave one out"
def pripremaPodatciUDK(conn):
    # punjenje korisnika
    #---------------------------------------------
    query = """SELECT DISTINCT c.RECID, n.NASLOV, n.AUTOR
            FROM baza.CLANOVI c 
                JOIN baza.POSUDBE h ON h.clan = c.RECID
                JOIN baza.PRIMJERCI p ON h.DOKUMENT = p.RECID
                JOIN baza.NASLOVI n ON p.NASLOV = n.RECID 
                JOIN BAZA.PREDMET pre ON n.RECID = pre.NASLOV 
            WHERE 
                n.VRSTA ='Knjiga' AND n.godina>=1940 AND n.NASLOV IS NOT NULL
                AND (SELECT COUNT(DISTINCT p1.DOKUMENT) 
                        FROM baza.POSUDBE p1 
                        JOIN baza.primjerci o1 ON p1.dokument = o1.RECID 
                        JOIN baza.naslovi n1 ON o1.naslov = n1.RECID
                        WHERE n1.VRSTA ='Knjiga' AND n1.godina>=1940 AND n1.NASLOV IS NOT NULL
                        AND p1.clan = c.RECID) BETWEEN 10 AND 100
            ORDER BY c.RECID """
    
    stmt = ibm_db.exec_immediate(conn, query)

    # Dohvaćanje rezultata
    row = ibm_db.fetch_assoc(stmt)
    clan =""
    indexKorisnik=-1

    indeksiranje = dict()
    korisnik = []
    indexDjela = 0

    while row:
        d = dict(row.items())
        
        if(d['RECID'] != clan):
            indexKorisnik = indexKorisnik+1
            korisnik.append([])
            clan = d['RECID']

        if((str(d['NASLOV']) + str(d['AUTOR'])) not in indeksiranje):
            indeksiranje[str(d['NASLOV']) + str(d['AUTOR'])] = indexDjela
            indexDjela = indexDjela + 1
        
        korisnik[indexKorisnik].append(indeksiranje.get(str(d['NASLOV']) +str(d['AUTOR'])))

        row = ibm_db.fetch_assoc(stmt)
    
    
    # punjenje posuDjela
    #---------------------------------------------
    posuDjela = [None] * indexDjela
    djelaPosuDjela = [None] * indexDjela

    # PROMJENA query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.UDK, N.RECID FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL"
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.UDK FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"

    stmt = ibm_db.exec_immediate(conn, query)

    row = ibm_db.fetch_assoc(stmt)
    br = -1
    autor = ''
    naslov = ''
    tren = set()

    while row:
        d = dict(row.items())

        if((str(d['NASLOV']) + str(d['AUTOR'])) in indeksiranje):
            
            #provjera je li sljedeci redak za isto djelo 
            if(d['AUTOR'] != autor or d['NASLOV'] != naslov):
                if(br>0):
                    #spremanje podataka
                    posuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = list(tren)
                    djelaPosuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = (str(d['NASLOV']) + str(d['AUTOR']))

                br = br + 1
                tren = set()
                autor = d['AUTOR']
                naslov = d['NASLOV']

        #dodavnaje atributa UDK
        separator = r'\.|-|\:|\(|\)|\"|\/|\+|\,'
        opisAtr = lanacSeparatoraUdk(d['UDK'], separator) 
        tren.update(opisAtr)
            
        row = ibm_db.fetch_assoc(stmt)

    #provjera zadnjeg djela
    if((str(d['NASLOV']) + str(d['AUTOR'])) in indeksiranje and 
        indeksiranje.get(str(d['NASLOV']) + str(d['AUTOR'])) != None and len(tren)>0):
        
        #spremanje podataka
        posuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = list(tren)
        djelaPosuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = (str(d['NASLOV'])  + str(d['AUTOR']))

    #print(indexDjela)
    #print(indeksiranje)
    print(br)
    
    #spremanje u memoriju
    # spremanje pomoćnih 
    with open('posuDjelaUDK.json', 'w') as file:
        json.dump(posuDjela, file)
    with open('korisniciProvjera.json', 'w') as file:
        json.dump(korisnik, file)
    with open('djelaPosuDjelaUDK.json', 'w') as file:
        json.dump(djelaPosuDjela, file)

def pripremaPodatciAutori(conn):
    # punjenje korisnika
    #---------------------------------------------
    query = """SELECT DISTINCT c.RECID, n.NASLOV, n.AUTOR
            FROM baza.CLANOVI c 
                JOIN baza.POSUDBE h ON h.clan = c.RECID
                JOIN baza.PRIMJERCI p ON h.DOKUMENT = p.RECID
                JOIN baza.NASLOVI n ON p.NASLOV = n.RECID 
                JOIN BAZA.PREDMET pre ON n.RECID = pre.NASLOV 
            WHERE 
                n.VRSTA ='Knjiga' AND n.godina>=1940 AND n.NASLOV IS NOT NULL
                AND (SELECT COUNT(DISTINCT p1.DOKUMENT) 
                        FROM baza.POSUDBE p1 
                        JOIN baza.primjerci o1 ON p1.dokument = o1.RECID 
                        JOIN baza.naslovi n1 ON o1.naslov = n1.RECID
                        WHERE n1.VRSTA ='Knjiga' AND n1.godina>=1940 AND n1.NASLOV IS NOT NULL
                        AND p1.clan = c.RECID) BETWEEN 10 AND 100
            ORDER BY c.RECID """
    
    stmt = ibm_db.exec_immediate(conn, query)

    # Dohvaćanje rezultata
    row = ibm_db.fetch_assoc(stmt)
    clan =""
    indexKorisnik=-1

    indeksiranje = dict()
    korisnik = []
    indexDjela = 0

    while row:
        d = dict(row.items())
        
        if(d['RECID'] != clan):
            indexKorisnik = indexKorisnik+1
            korisnik.append([])
            clan = d['RECID']

        if((str(d['NASLOV']) + str(d['AUTOR'])) not in indeksiranje):
            indeksiranje[str(d['NASLOV']) + str(d['AUTOR'])] = indexDjela
            indexDjela = indexDjela + 1
        
        korisnik[indexKorisnik].append(indeksiranje.get(str(d['NASLOV']) +str(d['AUTOR'])))

        row = ibm_db.fetch_assoc(stmt)
    
    
    # punjenje posuDjela
    #---------------------------------------------
    posuDjela = [None] * indexDjela
    djelaPosuDjela = [None] * indexDjela

    # ZAMJENE query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.UDK, N.RECID FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL"
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"

    stmt = ibm_db.exec_immediate(conn, query)

    row = ibm_db.fetch_assoc(stmt)
    br = -1
    autor = ''
    naslov = ''
    tren = set()

    while row:
        d = dict(row.items())

        if((str(d['NASLOV']) + str(d['AUTOR'])) in indeksiranje):
            
            #provjera je li sljedeci redak za isto djelo 
            if(d['AUTOR'] != autor or d['NASLOV'] != naslov):
                if(br>0):
                    #spremanje podataka
                    posuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = list(tren)
                    djelaPosuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = (str(d['NASLOV']) + str(d['AUTOR']))

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

    #provjera zadnjeg djela
    if((str(d['NASLOV']) + str(d['AUTOR'])) in indeksiranje and 
        indeksiranje.get(str(d['NASLOV']) + str(d['AUTOR'])) != None and len(tren)>0):
        
        #spremanje podataka
        posuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = list(tren)
        djelaPosuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = (str(d['NASLOV'])  + str(d['AUTOR']))

    print(br)
    #print(indexDjela)
    #print(indeksiranje)
    
    #spremanje u memoriju
    # spremanje pomoćnih 
    with open('posuDjelaAutor.json', 'w') as file:
        json.dump(posuDjela, file)
    with open('korisniciProvjera.json', 'w') as file:
        json.dump(korisnik, file)
    with open('djelaPosuDjelaAutor.json', 'w') as file:
        json.dump(djelaPosuDjela, file)

def pripremaPodatciSamoOpisi(conn):
    # punjenje korisnika
    #---------------------------------------------
    query = """SELECT DISTINCT c.RECID, n.NASLOV, n.AUTOR
            FROM baza.CLANOVI c 
                JOIN baza.POSUDBE h ON h.clan = c.RECID
                JOIN baza.PRIMJERCI p ON h.DOKUMENT = p.RECID
                JOIN baza.NASLOVI n ON p.NASLOV = n.RECID 
                JOIN BAZA.PREDMET pre ON n.RECID = pre.NASLOV 
            WHERE 
                n.VRSTA ='Knjiga' AND n.godina>=1940 AND n.NASLOV IS NOT NULL
                AND (SELECT COUNT(DISTINCT p1.DOKUMENT) 
                        FROM baza.POSUDBE p1 
                        JOIN baza.primjerci o1 ON p1.dokument = o1.RECID 
                        JOIN baza.naslovi n1 ON o1.naslov = n1.RECID
                        WHERE n1.VRSTA ='Knjiga' AND n1.godina>=1940 AND n1.NASLOV IS NOT NULL
                        AND p1.clan = c.RECID) BETWEEN 10 AND 100
            ORDER BY c.RECID """
    
    stmt = ibm_db.exec_immediate(conn, query)

    # Dohvaćanje rezultata
    row = ibm_db.fetch_assoc(stmt)
    clan =""
    indexKorisnik=-1

    indeksiranje = dict()
    korisnik = []
    indexDjela = 0

    while row:
        d = dict(row.items())
        
        if(d['RECID'] != clan):
            indexKorisnik = indexKorisnik+1
            korisnik.append([])
            clan = d['RECID']

        if((str(d['NASLOV']) + str(d['AUTOR'])) not in indeksiranje):
            indeksiranje[str(d['NASLOV']) + str(d['AUTOR'])] = indexDjela
            indexDjela = indexDjela + 1
        
        korisnik[indexKorisnik].append(indeksiranje.get(str(d['NASLOV']) +str(d['AUTOR'])))

        row = ibm_db.fetch_assoc(stmt)
    
    
    # punjenje posuDjela
    #---------------------------------------------
    posuDjela = [None] * indexDjela
    djelaPosuDjela = [None] * indexDjela

    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.OPIS, P.PREDMET FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV LEFT JOIN BAZA.PREDMET p ON n.RECID = p.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"

    stmt = ibm_db.exec_immediate(conn, query)

    row = ibm_db.fetch_assoc(stmt)
    br = -1
    autor = ''
    naslov = ''
    tren = set()

    while row:
        d = dict(row.items())

        if((str(d['NASLOV']) + str(d['AUTOR'])) in indeksiranje):
            
            #provjera je li sljedeci redak za isto djelo 
            if(d['AUTOR'] != autor or d['NASLOV'] != naslov):
                if(br>0):
                    tren.discard('npr')
                    tren.discard('itd')
                    tren.discard('tzv')

                    #spremanje podataka
                    posuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = list(tren)
                    djelaPosuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = (str(d['NASLOV']) + str(d['AUTOR']))

                br = br + 1
                tren = set()
                autor = d['AUTOR']
                naslov = d['NASLOV']
            
            #dodavnaje atributa OPISa
            separator = ['.', 'uključujući:', ' - ', '(', ')', '[', ']', ',', ':'] 
            opisAtr = lanacSeparatora(d['OPIS'], separator) 
            tren.update(opisAtr)

            #dodavnaje atributa PREDMETa
            separator = ['.', ' - ', '(', ')', '[', ']', ',', ':']
            opisAtr = lanacSeparatora(d['PREDMET'], separator) 
            tren.update(opisAtr)
        row = ibm_db.fetch_assoc(stmt)

    #provjera zadnjeg djela
    if((str(d['NASLOV']) + str(d['AUTOR'])) in indeksiranje and 
        indeksiranje.get(str(d['NASLOV']) + str(d['AUTOR'])) != None and len(tren)>0):

        tren.discard('npr')
        tren.discard('itd')
        tren.discard('tzv')
        
        #spremanje podataka
        posuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = list(tren)
        djelaPosuDjela[indeksiranje[(str(d['NASLOV']) + str(d['AUTOR']))]] = (str(d['NASLOV'])  + str(d['AUTOR']))

    print(indexDjela)
    print(indeksiranje)
    
    #spremanje u memoriju
    # spremanje pomoćnih 
    with open('posuDjelaSamoOpisi.json', 'w') as file:
        json.dump(posuDjela, file)
    with open('korisniciProvjera.json', 'w') as file:
        json.dump(korisnik, file)
    with open('djelaPosuDjelaSamoOpisi.json', 'w') as file:
        json.dump(djelaPosuDjela, file)


#KOSINUS EVALUACIJE
def fastTextEvaluacijaCos(model, matrica, inMatrix, outMatrix, posuDjela, korisnik, djelaPosuDjela):
    with open(posuDjela, 'r') as file:
        posuDjela = json.load(file)
    with open(outMatrix, 'r') as file:
        outMatrix = json.load(file)
    with open(inMatrix, 'r') as file:
        inMatrix = json.load(file)
    with open(korisnik, 'r') as file:
        korisnik = json.load(file)
    with open(djelaPosuDjela, 'r') as file:
        djelaPosuDjela = json.load(file)

    fastMatrix = np.load(matrica)
    norm_matrix = fastMatrix / np.linalg.norm(fastMatrix, axis=1, keepdims=True)
    del fastMatrix

    rezultatModela = 0.0  
    preporucena = set()
    brPogodaka = 0

    for opciKorisnik in korisnik:
  
        for djeloA in opciKorisnik:
            sumaDjelaKorisnika = set()
            
            for element in opciKorisnik:
                if element != djeloA and posuDjela[element] != None:
                    sumaDjelaKorisnika.update(posuDjela[element])

            # Vektori i normalizacija 
            vektori = np.array([model.get_word_vector(word) for word in sumaDjelaKorisnika])
            
            if len(vektori) == 0:
                continue  
            elif vektori.ndim == 1:
                norm_vektor = vektori / np.linalg.norm(vektori) #normaliziraj direktno
            else:
                norm_vektor = vektori / np.linalg.norm(vektori, axis=1, keepdims=True)
            
            
            if(djelaPosuDjela[djeloA]==None):
                continue 

            indexdjeloA = inMatrix[djelaPosuDjela[djeloA][:-4] if djelaPosuDjela[djeloA][-4:] == "None" else  djelaPosuDjela[djeloA]]
            
            # Računanje kosinusne sličnosti
            if vektori.ndim == 1:
                slicnosti = np.dot(norm_matrix, norm_vektor)
            else:
                slicnosti = np.dot(norm_matrix, norm_vektor.mean(axis=0))
            
            # Filtriranje postojećih stavki
            mask = np.ones_like(slicnosti, dtype=bool)
            for element in opciKorisnik:
                if element != djeloA and djelaPosuDjela[element] != None:
                    mask[inMatrix[djelaPosuDjela[element][:-4] if djelaPosuDjela[element][-4:] == "None" else  djelaPosuDjela[element]]] = False
                    
            slicnosti[~mask] = -np.inf

            # Pronalaženje top 20 
            top_k = 20
            partitioned_indices = np.argpartition(-slicnosti, top_k)[:top_k] #argpartition radi brže jer radi dobro za prvih 2p0 a kasnije ne sortira baš sve dobro ali oni nisu važni
            top_indices = partitioned_indices[np.argsort(-slicnosti[partitioned_indices])]
            
            #za izracun opsega djela
            preporucena.update(top_indices)

            # DCG izračun
            dcg = 0.0
            if indexdjeloA in top_indices:
                position = np.where(top_indices == indexdjeloA)[0][0] + 1  # 1-based index
                dcg = 1.0 / np.log2(position + 1)
                brPogodaka = brPogodaka +1
            
            rezultatModela += dcg

    print(len(preporucena))
    print(brPogodaka)
    return rezultatModela

def BagOFWordsEvaluacijaCos(matrica, indexWOut, indexWIn, atrbutiRJ, posuDjela, korisnik, djelaPosuDjela):
    with open(posuDjela, 'r') as file:
        posuDjela = json.load(file)
    with open(indexWOut, 'r') as file:
        indexWOut = json.load(file)
    with open(indexWIn, 'r') as file:
        indexWIn = json.load(file)
    with open(atrbutiRJ, 'r') as file:
        atrbutiRJ = json.load(file)
    with open(korisnik, 'r') as file:
        korisnik = json.load(file)
    with open(djelaPosuDjela, 'r') as file:
        djelaPosuDjela = json.load(file)

    bagMatrix = sparse.load_npz(matrica)

    # Normalizacija matrice
    norm = np.sqrt(bagMatrix.multiply(bagMatrix).sum(axis=1).A1)
    norm[norm == 0] = 1.0 # za izbjegavanje dijeljenja s nulom
    
    norm_matrix = bagMatrix.multiply(1.0 / norm[:, np.newaxis])

    preporucena = set()
    rezultatModela = 0.0  
    brPogodaka = 0


    for opciKorisnik in korisnik:   
        for djeloA in opciKorisnik:
            sumaDjelaKorisnika = set()
            
            for element in opciKorisnik:
                if element != djeloA and posuDjela[element] != None:
                    sumaDjelaKorisnika.update(posuDjela[element])

            # Vektori (stvaranje spatse)
            broj_atributa = bagMatrix.shape[1]  # Dobivanje broja atributa 
            sparse_vektor = np.zeros(broj_atributa)

            for atribut in sumaDjelaKorisnika:
                indeks = atrbutiRJ[atribut]
                sparse_vektor[indeks] = 1 

            if(np.linalg.norm(sparse_vektor) != 0):
                norm_sparse_vektor = sparse_vektor / np.linalg.norm(sparse_vektor)
            else:
                norm_sparse_vektor = sparse_vektor
            
            if(djelaPosuDjela[djeloA]==None):
                continue 

            indexdjeloA = indexWIn[djelaPosuDjela[djeloA][:-4] if djelaPosuDjela[djeloA][-4:] == "None" else  djelaPosuDjela[djeloA]]
            
            # Računanje kosinusne sličnosti
            slicnosti = norm_matrix.dot(norm_sparse_vektor)
            
            # Filtriranje postojećih stavki
            mask = np.ones(norm_matrix.shape[0], dtype=bool)
            for element in opciKorisnik:
                if element != djeloA and djelaPosuDjela[element] != None:
                    mask[indexWIn[djelaPosuDjela[element][:-4] if djelaPosuDjela[element][-4:] == "None" else  djelaPosuDjela[element]]] = False
                    
            slicnosti[~mask] = -np.inf

            # Pronalaženje top 20 
            top_k = 20
            partitioned_indices = np.argpartition(-slicnosti, top_k)[:top_k] #argpartition radi brže jer radi dobro za prvih 2p0 a kasnije ne sortira baš sve dobro ali oni nisu važni
            top_indices = partitioned_indices[np.argsort(-slicnosti[partitioned_indices])]

            #za izracun opsega djela
            preporucena.update(top_indices)
            
            # DCG izračun
            dcg = 0.0
            if indexdjeloA in top_indices:
                position = np.where(top_indices == indexdjeloA)[0][0] + 1  # 1-based index
                dcg = 1.0 / np.log2(position + 1)
                brPogodaka += 1
            
            rezultatModela += dcg

    print(brPogodaka)
    print(len(preporucena))
    return rezultatModela

def DualBagOFWordsEvaluacijaCos(
    matricaA, indexWOutA, indexWInA, atrbutiRJA, posuDjelaA, djelaPosuDjelaA,
    matricaB, indexWOutB, indexWInB, atrbutiRJB, posuDjelaB, djelaPosuDjelaB,
    korisnik, faktorA, faktorB
):

    # Učitavanje podataka za model A
    with open(posuDjelaA, 'r') as file:
        posuDjelaA = json.load(file)
    with open(indexWOutA, 'r') as file:
        indexWOutA = json.load(file)
    with open(indexWInA, 'r') as file:
        indexWInA = json.load(file)
    with open(atrbutiRJA, 'r') as file:
        atrbutiRJA = json.load(file)
    with open(djelaPosuDjelaA, 'r') as file:
        djelaPosuDjelaA = json.load(file)

    # Učitavanje podataka za model B
    with open(posuDjelaB, 'r') as file:
        posuDjelaB = json.load(file)
    with open(indexWOutB, 'r') as file:
        indexWOutB = json.load(file)
    with open(indexWInB, 'r') as file:
        indexWInB = json.load(file)
    with open(atrbutiRJB, 'r') as file:
        atrbutiRJB = json.load(file)
    with open(djelaPosuDjelaB, 'r') as file:
        djelaPosuDjelaB = json.load(file)

    # Učitavanje korisnika
    with open(korisnik, 'r') as file:
        korisnik = json.load(file)

    # Učitavanje i normalizacija matrica
    bagMatrixA = sparse.load_npz(matricaA)
    normA = np.sqrt(bagMatrixA.multiply(bagMatrixA).sum(axis=1).A1)
    normA[normA == 0] = 1.0
    norm_matrixA = bagMatrixA.multiply(1.0 / normA[:, np.newaxis])

    bagMatrixB = sparse.load_npz(matricaB)
    normB = np.sqrt(bagMatrixB.multiply(bagMatrixB).sum(axis=1).A1)
    normB[normB == 0] = 1.0
    norm_matrixB = bagMatrixB.multiply(1.0 / normB[:, np.newaxis])

    rezultatModela = 0.0
    preporucena = set()
    brPogodaka = 0

    for opciKorisnik in korisnik:

        for djeloA in opciKorisnik:
            sumaDjelaKorisnikaA = set()
            sumaDjelaKorisnikaB = set()

            #dodavanje atributa
            for element in opciKorisnik:
                if element != djeloA and posuDjelaA[element] is not None:
                    sumaDjelaKorisnikaA.update(posuDjelaA[element])
                if element != djeloA and posuDjelaB[element] is not None:
                    sumaDjelaKorisnikaB.update(posuDjelaB[element])

            # Vektor korisnika za model A
            broj_atributaA = bagMatrixA.shape[1]
            sparse_vektorA = np.zeros(broj_atributaA)
            for atribut in sumaDjelaKorisnikaA:
                indeks = atrbutiRJA[atribut]
                sparse_vektorA[indeks] = 1
            #normalizacija vektora korisnika za model A
            if np.linalg.norm(sparse_vektorA) != 0:
                norm_sparse_vektorA = sparse_vektorA / np.linalg.norm(sparse_vektorA)
            else:
                norm_sparse_vektorA = sparse_vektorA

            # Vektor korisnika za model B
            broj_atributaB = bagMatrixB.shape[1]
            sparse_vektorB = np.zeros(broj_atributaB)
            for atribut in sumaDjelaKorisnikaB:
                indeks = atrbutiRJB[atribut]
                sparse_vektorB[indeks] = 1
            #normalizacija vektora korisnika za model B
            if np.linalg.norm(sparse_vektorB) != 0:
                norm_sparse_vektorB = sparse_vektorB / np.linalg.norm(sparse_vektorB)
            else:
                norm_sparse_vektorB = sparse_vektorB

            # Provjera je li djelo u oba modela
            if djelaPosuDjelaA[djeloA] is None or djelaPosuDjelaB[djeloA] is None:
                continue

            indexdjeloA_A = indexWInA[djelaPosuDjelaA[djeloA][:-4] if djelaPosuDjelaA[djeloA][-4:] == "None" else djelaPosuDjelaA[djeloA]]

            # Računanje kosinusne sličnosti za oba modela
            slicnostiA = norm_matrixA.dot(norm_sparse_vektorA) # rezultat izracuna jednodimenzionalnim NumPy polja (zapravo jednodimenzionalni vektor)
            slicnostiB = norm_matrixB.dot(norm_sparse_vektorB)

            # Filtriranje djela koja tvore osnovu modela (posudbe - djeloA (ono je leave-out)) za oba modela
            maskA = np.ones(norm_matrixA.shape[0], dtype=bool) #vraca "jednodimenzionalni vektor" pun 0 velicine matrice
            maskB = np.ones(norm_matrixB.shape[0], dtype=bool)
            for element in opciKorisnik:
                if element != djeloA and djelaPosuDjelaA[element] is not None:
                    idxA = indexWInA[djelaPosuDjelaA[element][:-4] if djelaPosuDjelaA[element][-4:] == "None" else djelaPosuDjelaA[element]]
                    maskA[idxA] = False
                if element != djeloA and djelaPosuDjelaB[element] is not None:
                    idxB = indexWInB[djelaPosuDjelaB[element][:-4] if djelaPosuDjelaB[element][-4:] == "None" else djelaPosuDjelaB[element]]
                    maskB[idxB] = False

            slicnostiA[~maskA] = -np.inf #označava sve stavke na kojima je model treniran s -beskonacno
            slicnostiB[~maskB] = -np.inf #označava sve stavke na kojima je model treniran s -beskonacno

            # Kombinacija sličnosti
            combined_slicnosti = faktorA * slicnostiA + faktorB * slicnostiB 

            # Pronalaženje top 20
            topN = 20
            partitioned_indices = np.argpartition(-combined_slicnosti, topN)[:topN]  #fja koja pronalazi indekse topN najmanjih (minus zato ispred da bi radila obrnuto) vrijednosti u niz.
            top_indices = partitioned_indices[np.argsort(-combined_slicnosti[partitioned_indices])]

            #za izracun opsega djela
            preporucena.update(top_indices)
            
            # DCG izračun
            dcg = 0.0
            if indexdjeloA_A in top_indices:
                position = np.where(top_indices == indexdjeloA_A)[0][0] + 1  # 1-based index
                dcg = 1.0 / np.log2(position + 1)
                brPogodaka += 1


            rezultatModela += dcg

    print(len(preporucena))
    print(brPogodaka)
    return rezultatModela

def DualBagOFWordsFastTextEvaluacijaCosKaznena(
    matricaA, indexWOutA, indexWInA, atrbutiRJA, posuDjelaA, djelaPosuDjelaA,
    modelB, matricaB, inMatrixB, outMatrixB, posuDjelaB, djelaPosuDjelaB,
    korisnik, faktorA, faktorB, alfa
):

    # Učitavanje podataka za model A
    with open(posuDjelaA, 'r') as file:
        posuDjelaA = json.load(file)
    with open(indexWOutA, 'r') as file:
        indexWOutA = json.load(file)
    with open(indexWInA, 'r') as file:
        indexWInA = json.load(file)
    with open(atrbutiRJA, 'r') as file:
        atrbutiRJA = json.load(file)
    with open(djelaPosuDjelaA, 'r') as file:
        djelaPosuDjelaA = json.load(file)

    # Učitavanje podataka za model B
    with open(posuDjelaB, 'r') as file:
        posuDjelaB = json.load(file)
    with open(outMatrixB, 'r') as file:
        outMatrixB = json.load(file)
    with open(inMatrixB, 'r') as file:
        inMatrixB = json.load(file)
    with open(djelaPosuDjelaB, 'r') as file:
        djelaPosuDjelaB = json.load(file)
    fastMatrix = np.load(matricaB)
    norm_matrixB = fastMatrix / np.linalg.norm(fastMatrix, axis=1, keepdims=True)
    del fastMatrix

    # Učitavanje korisnika
    with open(korisnik, 'r') as file:
        korisnik = json.load(file)

    # Učitavanje i normalizacija matrica
    bagMatrixA = sparse.load_npz(matricaA)
    normA = np.sqrt(bagMatrixA.multiply(bagMatrixA).sum(axis=1).A1)
    normA[normA == 0] = 1.0
    norm_matrixA = bagMatrixA.multiply(1.0 / normA[:, np.newaxis])

    rezultatModela = 0.0
    preporucena = set()
    brPogodaka = 0

    for opciKorisnik in korisnik:
        for djeloA in opciKorisnik:
            sumaDjelaKorisnikaA = set()
            sumaDjelaKorisnikaB = set()

            #dodavanje atributa
            for element in opciKorisnik:
                if element != djeloA and posuDjelaA[element] is not None:
                    sumaDjelaKorisnikaA.update(posuDjelaA[element])
                if element != djeloA and posuDjelaB[element] is not None:
                    sumaDjelaKorisnikaB.update(posuDjelaB[element])

            # Vektor korisnika za model A
            broj_atributaA = bagMatrixA.shape[1]
            sparse_vektorA = np.zeros(broj_atributaA)
            for atribut in sumaDjelaKorisnikaA:
                indeks = atrbutiRJA[atribut]
                sparse_vektorA[indeks] = 1
            
            # Normalizacija vektora korisnika za model 
            norm_sparse_vektorA = sparse_vektorA.copy()
            vektor_norm_A = np.linalg.norm(sparse_vektorA)
            if vektor_norm_A > 0:
                norm_sparse_vektorA = sparse_vektorA / vektor_norm_A

            # Vektor korisnika za model B
            vektori = np.array([modelB.get_word_vector(word) for word in sumaDjelaKorisnikaB])
            
            # Normalizacija vektora korisnika za model N
            if len(vektori) == 0:
                continue  
            elif vektori.ndim == 1:
                norm_vektorB = vektori / np.linalg.norm(vektori) #normaliziraj direktno
            else:
                norm_vektorB = vektori / np.linalg.norm(vektori, axis=1, keepdims=True)
            
            #provjera ima li smilsa radirti daljunju provjeru
            if djelaPosuDjelaA[djeloA] is None or djelaPosuDjelaB[djeloA] is None:
                continue

            indexdjeloA_A = indexWInA[djelaPosuDjelaA[djeloA][:-4] if djelaPosuDjelaA[djeloA][-4:] == "None" else djelaPosuDjelaA[djeloA]]

            # Računanje kosinusne sličnosti za oba modela
            slicnostiA = norm_matrixA.dot(norm_sparse_vektorA)
            if vektori.ndim == 1:
                slicnostiB = np.dot(norm_matrixB, norm_vektorB)
            else:
                slicnostiB = np.dot(norm_matrixB, norm_vektorB.mean(axis=0))

            # Filtriranje djela koja tvore osnovu modela (posudbe - djeloA (ono je leave-out)) za oba modela
            maskA = np.ones(norm_matrixA.shape[0], dtype=bool)
            maskB = np.ones(norm_matrixB.shape[0], dtype=bool)
            for element in opciKorisnik:
                if element != djeloA and djelaPosuDjelaA[element] is not None:
                    idxA = indexWInA[djelaPosuDjelaA[element][:-4] if djelaPosuDjelaA[element][-4:] == "None" else djelaPosuDjelaA[element]]
                    maskA[idxA] = False
                if element != djeloA and djelaPosuDjelaB[element] is not None:
                    idxB = inMatrixB[djelaPosuDjelaB[element][:-4] if djelaPosuDjelaB[element][-4:] == "None" else djelaPosuDjelaB[element]]
                    maskB[idxB] = False      

            slicnostiA[~maskA] = -np.inf
            slicnostiB[~maskB] = -np.inf

            # power kažnjavanje koje je brže od eksponencijalne funkcije
            slicnostiA_penalized = np.power(np.maximum(slicnostiA, 0), alfa)
            # alfa = 1.0 nema kažnjavanja, alfa = 2.0 umjereno kažnavanje, alfa = 3.0 jako kažnjavanje (eksponiranje)

            combined_slicnosti = faktorA * slicnostiA_penalized + faktorB * slicnostiB

            # Pronalaženje top 20
            topN = 20
            partitioned_indices = np.argpartition(-combined_slicnosti, topN)[:topN]
            top_indices = partitioned_indices[np.argsort(-combined_slicnosti[partitioned_indices])]

            preporucena.update(top_indices)

            # DCG izračun
            dcg = 0.0
            if indexdjeloA_A in top_indices:
                position = np.where(top_indices == indexdjeloA_A)[0][0] + 1
                dcg = 1.0 / np.log2(position + 1)
                brPogodaka += 1

            rezultatModela += dcg

    print(len(preporucena))
    print(brPogodaka)
    return rezultatModela


#REFERENTNI MODEL 
def randomNDCG20BaselineSum(indexWOut, korisnik, djelaPosuDjela):
    # Učitavanje podataka
    with open(indexWOut, 'r') as file:
        indexWOut = json.load(file)
    with open(korisnik, 'r') as file:
        korisnik = json.load(file)
    with open(djelaPosuDjela, 'r') as file:
        djelaPosuDjela = json.load(file)

    random.seed(4734)
    ndcg_sum = 0.0
    preporucena = set()
    brPogodaka = 0

    for opciKorisnik in korisnik:
        # Generiraj jednu random listu od 20 preporuka za korisnika
        sluc = []
        while len(sluc) < 20:
            jedan = random.choice(indexWOut)
            jedan = jedan.replace("@@", "")
            if jedan not in sluc:
                sluc.append(jedan)
                preporucena.add(jedan)

        # Računaj DCG@20 za ovog korisnika
        dcg = 0.0
        for i, preporuka in enumerate(sluc):
            for djelo in opciKorisnik:
                if djelaPosuDjela[djelo] == preporuka:
                    dcg += 1.0 / np.log2(i + 2)
                    brPogodaka += 1

        # IDCG@20 = maksimalni mogući DCG (sve relevantne stavke na vrhu)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(20, len(opciKorisnik))))

        # NDCG@20 za ovog korisnika
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_sum += ndcg

    print(len(preporucena))
    print(brPogodaka)
    return ndcg_sum
