from procistaci import *
import fasttext
import numpy as np
from scipy.spatial.distance import cosine
import json
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

        model = fasttext.load_model("C:\\Users\\domin\\faks\\predmeti\\6_SEMESTAR\\ZavrsniRad\\kod\\vektorizacijaIzBaze\\fastTextOpisi2KGZ.bin")
        vektoriIzModelaSamoOpisiUMatricu(conn, model)

        #model = fasttext.load_model("C:\\Users\\domin\\faks\\predmeti\\6_SEMESTAR\\ZavrsniRad\\kod\\vektorizacijaIzBaze\\fastTextUDKKGZ.bin")
        #vektoriIzModelaUDKUMatricu(conn, model)

        #pripremaTekstualnihPodatakaSamoOpisiFastText(conn)
        
        
    except Exception as e:
        print(f"Greška: {ibm_db.conn_errormsg()}")
        print(e)

    #pozivi koji atributa koji trebaju bazu
    ibm_db.close(conn)


#reliziran NIJE korišten u radu
def pripremaTekstualnihPodatakaFastText(conn):
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.OPIS, P.PREDMET FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV LEFT JOIN BAZA.PREDMET p ON n.RECID = p.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"  
    stmt = ibm_db.exec_immediate(conn, query)

    # Podatci
    atrbutiRJ = dict()

    # Dohvaćanje rezultata
    row = ibm_db.fetch_assoc(stmt)
    br = -1
    autor = ''
    naslov = ''

    while row:
        d = dict(row.items())

        #provjera je li sljedeci redak za isto djelo 
        if(d['AUTOR'] != autor or d['NASLOV'] != naslov):
            if(br>0):
                (atrbutiRJ.get(br)).discard('npr')
                (atrbutiRJ.get(br)).discard('itd')
                (atrbutiRJ.get(br)).discard('tzv')

            br = br + 1
            atrbutiRJ[br] = set()
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
            (atrbutiRJ.get(br)).update(autorAtr)
        
        #dodavnaje atributa OPISa
        separator = ['.', 'uključujući:', ' - ', '(', ')', '[', ']', ',', ':'] 
        opisAtr = lanacSeparatora(d['OPIS'], separator) 
        (atrbutiRJ.get(br)).update(opisAtr)

        #dodavnaje atributa PREDMETa
        separator = ['.', ' - ', '(', ')', '[', ']', ',', ':']
        opisAtr = lanacSeparatora(d['PREDMET'], separator) 
        (atrbutiRJ.get(br)).update(opisAtr)

        row = ibm_db.fetch_assoc(stmt)
    
    with open('opisi_djelaFastText1.txt', 'w', encoding='utf-8') as f:
        jedno = ''
        for id_djela, opisi in atrbutiRJ.items():
            if opisi:
                opisi_s_podvlakama = [opis.replace(" ", "_") for opis in opisi]
                opis_djela = " ".join(opisi_s_podvlakama)

                if(opis_djela!=jedno):
                    f.write(f"{opis_djela}\n")  #provjera različitih djela s istim opisom (ako se radi o knigama u nizu)
                jedno = opis_djela
    print(br)
    

def pripremaTekstualnihPodatakaSamoOpisiFastText(conn):
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.OPIS, P.PREDMET FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV LEFT JOIN BAZA.PREDMET p ON n.RECID = p.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"  
    stmt = ibm_db.exec_immediate(conn, query)

    # Podatci
    atrbutiRJ = dict()

    # Dohvaćanje rezultata
    row = ibm_db.fetch_assoc(stmt)
    br = -1
    autor = ''
    naslov = ''

    while row:
        d = dict(row.items())

        #provjera je li sljedeci redak za isto djelo 
        if(d['AUTOR'] != autor or d['NASLOV'] != naslov):
            if(br>0):
                (atrbutiRJ.get(br)).discard('npr')
                (atrbutiRJ.get(br)).discard('itd')
                (atrbutiRJ.get(br)).discard('tzv')

            br = br + 1
            atrbutiRJ[br] = set()
            autor = d['AUTOR']
            naslov = d['NASLOV']
        
        #dodavnaje atributa OPISa
        separator = ['.', 'uključujući:', ' - ', '(', ')', '[', ']', ',', ':'] 
        opisAtr = lanacSeparatora(d['OPIS'], separator) 
        (atrbutiRJ.get(br)).update(opisAtr)

        #dodavnaje atributa PREDMETa
        separator = ['.', ' - ', '(', ')', '[', ']', ',', ':']
        opisAtr = lanacSeparatora(d['PREDMET'], separator) 
        (atrbutiRJ.get(br)).update(opisAtr)

        row = ibm_db.fetch_assoc(stmt)
    
    with open('opisi_djelaFastTextBezAutora.txt', 'w', encoding='utf-8') as f:
        jedno = ''
        for id_djela, opisi in atrbutiRJ.items():
            if opisi:
                opisi_s_podvlakama = [opis.replace(" ", "_") for opis in opisi]
                opis_djela = " ".join(opisi_s_podvlakama)

                if(opis_djela!=jedno):
                    f.write(f"{opis_djela}\n")  #provjera različitih djela s istim opisom (ako se radi o knigama u nizu)
                jedno = opis_djela
    print(br)


def pripremaTekstualnihUDKFastText(conn):
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.UDK FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"
    
    stmt = ibm_db.exec_immediate(conn, query)

    # Podatci
    atrbutiRJ = dict()

    # Dohvaćanje rezultata
    row = ibm_db.fetch_assoc(stmt)
    br = -1
    autor = ''
    naslov = ''

    while row:
        d = dict(row.items())

        #provjera je li sljedeci redak za isto djelo 
        if(d['AUTOR'] != autor or d['NASLOV'] != naslov):
            br = br + 1
            atrbutiRJ[br] = set()
            autor = d['AUTOR']
            naslov = d['NASLOV']
        
        #dodavnaje atributa UDK
        separator = r'\.|-|\:|\(|\)|\"|\/|\+|\,'
        opisAtr = lanacSeparatoraUdk(d['UDK'], separator) 
        (atrbutiRJ.get(br)).update(opisAtr)

        row = ibm_db.fetch_assoc(stmt)

    with open('opisi_djelaUDKFastText.txt', 'w', encoding='utf-8') as f:
        jedno = ''
        for id_djela, opisi in atrbutiRJ.items():
            if opisi:
                opisi_s_podvlakama = [opis.replace(" ", "_") for opis in opisi]
                opis_djela = " ".join(opisi_s_podvlakama)

                if(opis_djela!=jedno):
                    f.write(f"{opis_djela}\n")  #provjera različitih djela s istim opisom (ako se radi o knigama u nizu)
                jedno = opis_djela
    print(br)


def fastText_model(input_file, naziv):
    model = fasttext.train_unsupervised(
        input_file,
        model='skipgram',  # skipgram model za bolju semantičku sličnost
        dim=280,           # veličina vektora
        epoch=90,          # broj epoha
        lr=0.05,           # stopa učenja
        ws=7,              # veličina kontekstualnog prozora
        minCount=2,        # minimalni broj pojavljivanja riječi
        minn=3,            # minimalna duljina n-grama znakova
        maxn=5,            # maksimalna duljina n-grama znakova
        wordNgrams=3,      # maksimalna duljina n-grama riječi
        bucket=1500000       # broj "kanti" za hashiranje n-grama
    )

    model.save_model(naziv)
    return 1

#reliziran NIJE korišten u radu
def vektoriIzModelaUMatricu(conn, model):
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.OPIS, P.PREDMET FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV LEFT JOIN BAZA.PREDMET p ON n.RECID = p.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"
    
    stmt = ibm_db.exec_immediate(conn, query)

    # Podatci
    inMatrix = dict()
    OutMatrix = []
    listaVektora = []

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

                #dobivanje vektora iz modela
                vektori = [model.get_word_vector(word) for word in tren]
                if len(vektori) == 0:
                    listaVektora.append(np.zeros(model.get_dimension()))
                else: 
                    listaVektora.append(np.mean(vektori, axis=0))

                #spremanje podataka
                if(autor != None):
                    OutMatrix.append(str(naslov + '@@' + autor))
                    inMatrix[str(naslov+autor)] = br
                else:
                    OutMatrix.append(str(naslov + '@@'))
                    inMatrix[str(naslov)] = br

            br = br + 1
            tren = set()
            autor = d['AUTOR']
            naslov = d['NASLOV']

            #dodaavanje atributa Autora (zamjenio prvotno bilo iznad, što nema nekog smila jer se autori ubacuju u prazno onda)
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

    #dobivanje vektora iz modela
    vektori = [model.get_word_vector(word) for word in tren]
    if len(vektori) == 0:
        listaVektora.append(np.zeros(model.get_dimension()))
    else: 
        listaVektora.append(np.mean(vektori, axis=0))

    #spremanje podataka
    if(autor != None):
        OutMatrix.append(str(naslov + '@@' + autor))
        inMatrix[str(naslov+autor)] = br
    else:
        OutMatrix.append(str(naslov + '@@'))
        inMatrix[str(naslov)] = br

    #-----------------------------------------
    #stvaranje matrice 
    fastMatrix = np.array(listaVektora, dtype=np.float32)
    np.save('fastMatrix.npy', fastMatrix)

    #spremanje pomoćnih 
    with open('inMatrixFastText.json', 'w') as file:
        json.dump(inMatrix, file)
    with open('outMatrixFastText.json', 'w') as file:
        json.dump(OutMatrix, file)


def vektoriIzModelaSamoOpisiUMatricu(conn, model):
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.OPIS, P.PREDMET FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV LEFT JOIN BAZA.PREDMET p ON n.RECID = p.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"
    
    stmt = ibm_db.exec_immediate(conn, query)

    # Podatci
    inMatrix = dict()
    OutMatrix = []
    listaVektora = []

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

                #dobivanje vektora iz modela
                vektori = [model.get_word_vector(word) for word in tren]
                if len(vektori) == 0:
                    listaVektora.append(np.zeros(model.get_dimension()))
                else: 
                    listaVektora.append(np.mean(vektori, axis=0))

                #spremanje podataka
                if(autor != None):
                    OutMatrix.append(str(naslov + '@@' + autor))
                    inMatrix[str(naslov+autor)] = br
                else:
                    OutMatrix.append(str(naslov + '@@'))
                    inMatrix[str(naslov)] = br

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

    print(br)

    #-----------------------------------------
    #zadnji prolaz
    tren.discard('npr')
    tren.discard('itd')
    tren.discard('tzv')

    #dobivanje vektora iz modela
    vektori = [model.get_word_vector(word) for word in tren]
    if len(vektori) == 0:
        listaVektora.append(np.zeros(model.get_dimension()))
    else: 
        listaVektora.append(np.mean(vektori, axis=0))

    #spremanje podataka
    if(autor != None):
        OutMatrix.append(str(naslov + '@@' + autor))
        inMatrix[str(naslov+autor)] = br
    else:
        OutMatrix.append(str(naslov + '@@'))
        inMatrix[str(naslov)] = br

    #-----------------------------------------
    #stvaranje matrice 
    fastMatrix = np.array(listaVektora, dtype=np.float32)
    np.save('fastMatrixSamoOpisi.npy', fastMatrix)

    #spremanje pomoćnih 
    with open('inMatrixFastTextSamoOpisi.json', 'w') as file:
        json.dump(inMatrix, file)
    with open('outMatrixFastTextSamoOpisi.json', 'w') as file:
        json.dump(OutMatrix, file)


def vektoriIzModelaUDKUMatricu(conn, model):
    query = "SELECT DISTINCT n.NASLOV, n.AUTOR, k.UDK FROM BAZA.NASLOVI n JOIN baza.KLASIFIKACIJA k ON n.RECID = k.NASLOV WHERE n.VRSTA='Knjiga' AND godina>=1940 AND k.OPIS NOT LIKE '%#%' AND n.NASLOV IS NOT NULL ORDER BY n.NASLOV, n.AUTOR"
    
    stmt = ibm_db.exec_immediate(conn, query)

    # Podatci
    inMatrix = dict()
    OutMatrix = []
    listaVektora = []

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
                #dobivanje vektora iz modela
                vektori = [model.get_word_vector(word) for word in tren]
                if len(vektori) == 0:
                    listaVektora.append(np.zeros(model.get_dimension()))
                else: 
                    listaVektora.append(np.mean(vektori, axis=0))

                #spremanje podataka
                if(autor != None):
                    OutMatrix.append(str(naslov + '@@' + autor))
                    inMatrix[str(naslov+autor)] = br
                else:
                    OutMatrix.append(str(naslov + '@@'))
                    inMatrix[str(naslov)] = br


            br = br + 1
            tren = set()
            autor = d['AUTOR']
            naslov = d['NASLOV']
        
        #dodavnaje atributa UDK
        separator = r'\.|-|\:|\(|\)|\"|\/|\+|\,'
        opisAtr = lanacSeparatoraUdk(d['UDK'], separator) 
        tren.update(opisAtr)

        row = ibm_db.fetch_assoc(stmt)

    #dodavnje zadnjeg
    if(br>=0):
                #dobivanje vektora iz modela
                vektori = [model.get_word_vector(word) for word in tren]
                if len(vektori) == 0:
                    listaVektora.append(np.zeros(model.get_dimension()))
                else: 
                    listaVektora.append(np.mean(vektori, axis=0))

                #spremanje podataka
                if(autor != None):
                    OutMatrix.append(str(naslov + '@@' + autor))
                    inMatrix[str(naslov+autor)] = br
                else:
                    OutMatrix.append(str(naslov + '@@'))
                    inMatrix[str(naslov)] = br

        
    print(br)

    #stvaranje matrice 
    fastMatrix = np.array(listaVektora, dtype=np.float32)
    np.save('fastMatrixUDK.npy', fastMatrix)

    #spremanje pomoćnih 
    with open('inMatrixFastTextUDK.json', 'w') as file:
        json.dump(inMatrix, file)
    with open('outMatrixFastTextUDK.json', 'w') as file:
        json.dump(OutMatrix, file)

#reliziran NIJE korišten u radu
def topPreporukaDjelaFastText(djelo, autor, broj, matrica, inmatrix, outmatrix):

    with open(inmatrix, 'r') as file:
        inMatrix = json.load(file)
    with open(outmatrix, 'r') as file:
        outMatrix = json.load(file)
    fastMatrix = np.load(matrica) 

    if(autor != None):
        index = inMatrix[djelo+autor]
    else:
        index = inMatrix[djelo]
    cilj =  fastMatrix[index]


    # Normalizacija vektora za kosinusnu sličnost
    norm_vektor = cilj / np.linalg.norm(cilj)
    norm_matrix = fastMatrix / np.linalg.norm(fastMatrix, axis=1, keepdims=True)

    # Računanje kosinusne sličnosti između ciljnog djela i svih ostalih
    slicnosti = np.dot(norm_matrix, norm_vektor)
    
    # Postavljanje sličnosti samog djela na -1 kako ga ne bismo uključili u rezultate
    slicnosti[index] = -1
    
    # Pronalaženje indeksa 100 najsličnijih djela
    indeksi_najslicnijih = np.argsort(slicnosti)[-broj:][::-1]
    
    # Stvaranje liste rezultata (sličnost, naslov)
    rezultati = [(slicnosti[i], outMatrix[i]) for i in indeksi_najslicnijih]

    print(rezultati)
