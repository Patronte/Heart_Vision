
# HEARTVISION - GESTIONE DATASET, MACHINE LEARNING E PREVISIONI SU RISPOSTE DATE DA UTENTI A SCOPO DI PREVENZIONE


# IMPORTAZIONE DELLE LIBRERIE

from sklearn.neighbors import KNeighborsClassifier  # Importa il classificatore KNN
from sklearn.preprocessing import StandardScaler  # Importa lo standardizzatore
from sklearn.tree import DecisionTreeClassifier  # Importa l'albero decisionale
from sklearn.ensemble import RandomForestClassifier  # Importa la Random Forest
from sklearn.calibration import CalibratedClassifierCV  # Calibra le probabilità
from sklearn.pipeline import make_pipeline  # Gestisce lo scaler all’interno di KNN
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict  # Valida i modelli
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, fbeta_score
)

import matplotlib.pyplot as plt  # Gestisce i grafici
import pandas as pd  # Gestisce i DataFrame
import numpy as np  # Gestisce le operazioni numeriche


# DEFINIZIONE DELLE COLONNE

colonne = [  # Definisce le caratteristiche comuni ai due CSV
    "sesso",
    "eta",
    "fumatore",
    "sigarette_giorno",
    "farmaci_pressione",
    "ictus_precedente",
    "ipertensione",
    "diabete",
    "colesterolo_alto",
    "peso_kg",
    "altezza_cm",
    "frequenza_cardiaca",
    "alcol",
    "sere_alcol_settimana",
    "sport"
]

target = "evento_coronarico_10_anni"  # Variabile da prevedere


# CONVERSIONE DEL QUESTIONARIO

def converti_questionario(df):  # Converte le risposte di Google Forms

    df = df.copy()  # Crea una copia del DataFrame

    df.columns = df.columns.str.strip()  # Elimina gli spazi dalle intestazioni

    metadati = [  # Elenca le colonne non utilizzate dal modello
        "timestamp",
        "username",
        "email",
        "email address",
        "indirizzo email"
    ]

    da_eliminare = [
        c for c in df.columns
        if c.lower() in metadati  # Individua le colonne contenenti metadati
    ]

    df = df.drop(columns=da_eliminare)  # Rimuove data ed email

    if len(df.columns) != len(colonne):  # Verifica che siano presenti 15 risposte
        raise ValueError(
            "Il questionario deve contenere 15 colonne!"
        )

    if set(df.columns) == set(colonne):  # Controlla se i nomi delle colonne sono corretti
        df = df[colonne]  # Riordina le colonne
    else:
        df.columns = colonne  # Assegna i nomi secondo l'ordine del questionario

    conversioni = {  # Definisce le sostituzioni delle risposte testuali
        "maschio": "1",
        "femmina": "0",
        "si": "1",
        "sì": "1",
        "s": "1",
        "no": "0",
        "n": "0"
    }

    for colonna in colonne:  # Scorre tutte le caratteristiche

        valori = (
            df[colonna]
            .astype("string")  # Converte i valori in stringhe
            .str.strip()  # Elimina gli spazi
            .str.lower()  # Trasforma le lettere in minuscolo
        )

        valori = valori.replace(conversioni)  # Sostituisce le risposte con 0 e 1

        valori = valori.str.replace(
            ",", ".", regex=False  # Converte le virgole decimali in punti
        )

        df[colonna] = pd.to_numeric(
            valori,
            errors="coerce"  # Trasforma i valori non convertibili in NaN
        )

    print("\nQuestionario convertito correttamente!")

    return df  # Restituisce il questionario convertito


# CONTROLLO E PULIZIA DEI DATASET

def controllo_csv(df, tipo="storico", riferimento=None):

    df = df.copy()  # Crea una copia del DataFrame

    df.columns = df.columns.str.strip()  # Elimina gli spazi nelle intestazioni

    print("\n" + "=" * 55)
    print(f"       HEARTVISION - CONTROLLO {tipo.upper()}")
    print("=" * 55)

    print("Righe:", df.shape[0])  # Mostra il numero di righe
    print("Colonne:", df.shape[1])  # Mostra il numero di colonne

    if tipo not in ["storico", "questionario"]:  # Verifica il tipo di dataset
        raise ValueError("Tipo di dataset non valido!")

    if df.empty:  # Verifica che il dataset non sia vuoto
        raise ValueError("Il DataFrame è vuoto!")

    if df.columns.duplicated().any():  # Controlla eventuali intestazioni duplicate
        raise ValueError(
            "Sono presenti colonne duplicate!"
        )


    # CONTROLLO DELLE COLONNE OBBLIGATORIE

    if tipo == "storico":  # Il dataset storico comprende anche il target
        richieste = colonne + [target]

    else:
        richieste = colonne  # Il questionario contiene soltanto le caratteristiche

        if target in df.columns:  # Verifica che il questionario non contenga il target
            raise ValueError(
                "Il questionario non deve contenere il target!"
            )

    mancanti = [
        c for c in richieste
        if c not in df.columns  # Individua le colonne obbligatorie assenti
    ]

    if mancanti:
        raise ValueError(
            f"Colonne mancanti: {mancanti}"
        )


    # CONVERSIONE NUMERICA

    for colonna in richieste:  # Converte ogni colonna in numeri

        df[colonna] = pd.to_numeric(
            df[colonna],
            errors="coerce"  # Converte i valori non validi in NaN
        ).astype(float)

    df[richieste] = df[richieste].replace(
        [np.inf, -np.inf],
        np.nan  # Sostituisce i valori infiniti con NaN
    )


    # SELEZIONE DEL DATASET DI RIFERIMENTO

    if riferimento is not None:  # Verifica se è disponibile un dataset di riferimento
        base = riferimento  # Utilizza il dataset fornito
    else:
        base = df  # Utilizza il DataFrame corrente

    correzioni = {}  # Memorizza il numero delle correzioni


    # CONTROLLO DELLE VARIABILI BINARIE

    binarie = [  # Definisce le colonne che devono contenere 0 oppure 1
        "sesso",
        "fumatore",
        "farmaci_pressione",
        "ictus_precedente",
        "ipertensione",
        "diabete",
        "colesterolo_alto",
        "alcol",
        "sport"
    ]

    for colonna in binarie:  # Controlla ogni variabile binaria

        errati = ~df[colonna].isin([0, 1])  # Individua i valori diversi da 0 e 1

        if errati.any():  # Procede se sono presenti errori

            validi = pd.to_numeric(
                base[colonna],
                errors="coerce"  # Converte i valori in numeri
            )

            validi = validi[validi.isin([0, 1])]  # Mantiene soltanto i valori validi

            if validi.empty:  # Verifica la presenza di valori utilizzabili
                raise ValueError(
                    f"Nessun valore valido in {colonna}"
                )

            moda = validi.mode().iloc[0]  # Calcola il valore più frequente

            df.loc[errati, colonna] = moda  # Sostituisce i valori errati con la moda

            correzioni[colonna] = int(errati.sum())  # Conta le sostituzioni


    # CONTROLLO DELLA VARIABILE TARGET

    if tipo == "storico":  # Controlla il target soltanto nello storico

        if not df[target].isin([0, 1]).all():  # Verifica che gli esiti siano 0 oppure 1
            raise ValueError(
                "Il target contiene valori mancanti o non validi!"
            )


    # CONTROLLO DEGLI INTERVALLI NUMERICI

    intervalli = {  # Definisce gli intervalli ammessi
        "eta": (5, 100),
        "sigarette_giorno": (0, 60),
        "peso_kg": (20, 200),
        "altezza_cm": (80, 200),
        "frequenza_cardiaca": (25, 130),
        "sere_alcol_settimana": (0, 7)
    }

    intere = [  # Definisce le colonne che richiedono numeri interi
        "eta",
        "sigarette_giorno",
        "sere_alcol_settimana"
    ]

    for colonna, (minimo, massimo) in intervalli.items():  # Controlla ogni intervallo

        valori = df[colonna]  # Recupera i valori della colonna

        errati = (
            valori.isna() |  # Individua i valori mancanti
            ~valori.between(minimo, massimo)  # Individua i valori fuori intervallo
        )

        if colonna in intere:  # Verifica se sono richiesti numeri interi
            errati = errati | (valori % 1 != 0)  # Individua i valori decimali

        if errati.any():  # Procede se trova valori da correggere

            validi = pd.to_numeric(
                base[colonna],
                errors="coerce"  # Converte i valori di riferimento
            )

            validi = validi.replace(
                [np.inf, -np.inf],
                np.nan  # Esclude gli infiniti
            )

            validi = validi[
                validi.between(minimo, massimo)  # Mantiene i valori nell'intervallo
            ]

            if colonna in intere:  # Controlla le colonne intere
                validi = validi[validi % 1 == 0]  # Esclude i valori decimali

            if validi.empty:  # Verifica che esistano dati validi
                raise ValueError(
                    f"Impossibile calcolare la media di {colonna}"
                )

            media = validi.mean()  # Calcola la media dei valori validi

            if colonna in intere:  # Controlla se la colonna richiede interi
                media = int(round(media))  # Arrotonda la media

            df.loc[errati, colonna] = media  # Sostituisce gli errori con la media

            correzioni[colonna] = int(errati.sum())  # Registra le correzioni


    # CONTROLLO DELLA COERENZA DEL FUMO

    errore_fumo = (
        (df["fumatore"] == 0) &  # Individua i non fumatori
        (df["sigarette_giorno"] > 0)  # Che indicano sigarette giornaliere
    )

    if errore_fumo.any():  # Verifica la presenza di incoerenze

        df.loc[errore_fumo, "sigarette_giorno"] = 0  # Imposta zero sigarette

        correzioni["coerenza_fumo"] = int(
            errore_fumo.sum()  # Conta le correzioni
        )


    # CONTROLLO DELLA COERENZA DELL'ALCOL

    errore_alcol = (
        (df["alcol"] == 0) &  # Individua chi dichiara di non bere
        (df["sere_alcol_settimana"] > 0)  # Ma indica sere di consumo
    )

    if errore_alcol.any():  # Verifica la presenza di incoerenze

        df.loc[
            errore_alcol,
            "sere_alcol_settimana"
        ] = 0  # Imposta zero sere di consumo

        correzioni["coerenza_alcol"] = int(
            errore_alcol.sum()  # Conta le correzioni
        )


    # VERIFICA FINALE DEI VALORI MANCANTI

    if df[richieste].isnull().any().any():  # Controlla la presenza di NaN
        raise ValueError(
            "Sono rimasti valori mancanti nel dataset!"
        )


    # CONTROLLO DELLE RIGHE DUPLICATE

    duplicati = df[richieste].duplicated().sum()  # Conta le righe identiche

    print("\nRighe con valori identici:", duplicati)


    # CALCOLO AUTOMATICO DEL BMI

    df["bmi"] = (
        df["peso_kg"] /  # Recupera il peso in chilogrammi
        (df["altezza_cm"] / 100) ** 2  # Converte l'altezza in metri
    ).round(2)  # Arrotonda il BMI a due decimali


    # RIPRISTINO DEI NUMERI INTERI

    colonne_intere = binarie + intere  # Unisce le variabili binarie e intere

    if tipo == "storico":  # Verifica se stiamo controllando Framingham
        colonne_intere.append(target)  # Aggiunge il target

    for colonna in colonne_intere:  # Scorre le colonne intere
        df[colonna] = df[colonna].astype(int)  # Converte i valori in interi


    # RIEPILOGO DELLA PULIZIA

    print("\n" + "-" * 55)
    print("                RISULTATO PULIZIA")
    print("-" * 55)

    if correzioni:  # Controlla se sono state effettuate correzioni

        print("\nCorrezioni effettuate:")

        for nome, quantita in correzioni.items():  # Scorre le correzioni registrate
            print(f"  {nome}: {quantita}")

    else:
        print("\nNessuna correzione necessaria.")

    print("\nValori nulli rimasti: 0")
    print("BMI calcolato correttamente.")
    print("\nDataset pronto per l'analisi!")
    print("=" * 55)

    return df  # Restituisce il dataset pulito


# CARICAMENTO DEL DATASET STORICO E DEI QUESTIONARI

def carica_dataset():  # Importa e prepara entrambi i dataset

    global df_X, df_y, risultati, email_questionari, confronto  # Condivide dati e confronto

    risultati = None  # Azzera le previsioni del CSV precedente
    email_questionari = None  # Azzera le email del CSV precedente
    confronto = None  # Impedisce di riusare una selezione precedente

    df_X = pd.read_csv(
        "Dataset_storico/HEARTVISION_dataset_completo.csv"  # Legge Framingham
    )

    df_X = controllo_csv(
        df_X,
        tipo="storico"  # Controlla e pulisce il dataset storico
    )


    # CARICAMENTO DEL CSV SCELTO DALL'UTENTE

    percorso_csv = input("\nInserisci il percorso del CSV da analizzare: ").strip()  # Legge il percorso inserito

    percorso_csv = percorso_csv.strip('"').strip("'")  # Rimuove eventuali virgolette copiate

    if not percorso_csv:  # Controlla che sia stato inserito un percorso
        raise ValueError("Devi inserire il percorso di un file CSV!")

    df_y = pd.read_csv(percorso_csv)  # Carica il CSV indicato dall'utente


    # RECUPERO DEGLI IDENTIFICATIVI EMAIL

    colonna_email = None  # Memorizza il nome della colonna con gli identificativi

    nomi_email = [  # Elenca le possibili intestazioni
        "email",
        "indirizzo email",
        "email address",
        "username"
    ]

    for nome_cercato in nomi_email:  # Cerca una colonna contenente le email

        for nome_colonna in df_y.columns:

            if nome_colonna.strip().lower() == nome_cercato:
                colonna_email = nome_colonna
                break

        if colonna_email is not None:
            break

    if colonna_email is not None:  # Conserva gli identificativi prima della conversione

        email_questionari = (
            df_y[colonna_email]
            .astype("string")
            .str.strip()
            .str.lower()
            .reset_index(drop=True)
        )  # Mantiene le email nello stesso ordine delle righe

        print("\nIdentificativi email acquisiti correttamente.")

    else:
        print("\nNessuna colonna email trovata: analisi individuale non disponibile.")

    
    print(f"\nCSV caricato: {percorso_csv}")  # Mostra il file selezionato
    print(f"Partecipanti presenti: {len(df_y)}")  # Conta le righe del CSV


    df_y = converti_questionario(df_y)  # Converte le risposte testuali

    df_y = controllo_csv(
        df_y,
        tipo="questionario",
        riferimento=df_X  # Utilizza Framingham come riferimento
    )

    print("\n" + "=" * 55)
    print("              DATASET FRAMINGHAM")
    print("=" * 55)
    print(df_X.head())  # Mostra le prime cinque righe

    print("\n" + "=" * 55)
    print("              DATASET QUESTIONARI")
    print("=" * 55)
    print(df_y.head())  # Mostra le prime cinque risposte

    risultati = None  # Impedisce di mostrare risultati riferiti a un altro CSV

    print("\nHEARTVISION - Operazioni completate!")
    print("\nCaricamento completato.")
    
    

# ANALISI INDIVIDUALE DEL PARTECIPANTE

def analizza_partecipante():  # Analizza un partecipante tramite la sua email

    # CONTROLLO DEI DATI DISPONIBILI

    if "risultati" not in globals() or risultati is None or risultati.empty:
        print("Prima carica il CSV ed esegui le previsioni (opzioni 1, 2 e 4).")
        return

    if "email_questionari" not in globals() or email_questionari is None:
        print("Il CSV caricato non contiene una colonna email o Username.")
        return

    if len(email_questionari) != len(df_y) or len(risultati) != len(df_y):
        print("I dati non corrispondono: ricarica il CSV ed esegui le previsioni.")
        return


    # RICERCA DEL PARTECIPANTE TRAMITE EMAIL

    email_cercata = input(
        "\nInserisci l'email del partecipante: "
    ).strip().lower()  # Legge l'identificativo inserito

    if not email_cercata:
        print("Devi inserire un indirizzo email.")
        return

    indici_trovati = email_questionari[
        email_questionari == email_cercata
    ].index.tolist()  # Cerca le righe associate all'email

    if not indici_trovati:
        print("Email non trovata nel CSV attualmente caricato.")
        return

    if len(indici_trovati) > 1:  # Gestisce più risposte della stessa email

        numeri = [indice + 1 for indice in indici_trovati]

        print(f"L'email è presente più volte, agli ID: {numeri}")

        scelta_id = input(
            "Inserisci l'ID del partecipante da analizzare: "
        ).strip()

        if not scelta_id.isdigit() or int(scelta_id) not in numeri:
            print("ID non valido.")
            return

        indice = int(scelta_id) - 1

    else:
        indice = indici_trovati[0]  # Recupera la posizione della riga


    # RECUPERO DEI DATI DEL PARTECIPANTE

    soggetto = df_y.iloc[indice]  # Recupera le caratteristiche individuali

    previsione = risultati.iloc[indice]  # Recupera la previsione già calcolata

    identificativo = int(previsione["utente"])  # Recupera l'ID numerico

    classe = int(previsione["previsione"])  # Recupera la classe prevista

    percentuale = float(
        previsione["probabilita_evento_percentuale"]
    )  # Recupera la percentuale stimata


    # RIEPILOGO DELLA PREVISIONE

    print("\n" + "=" * 55)
    print("          HEARTVISION - ANALISI INDIVIDUALE")
    print("=" * 55)

    print(f"\nID partecipante: {identificativo}")
    print(f"Email: {email_cercata}")
    print(f"Classe prevista: {classe}")
    print(f"Probabilità stimata della classe 1: {percentuale:.2f}%")


    # VISUALIZZAZIONE DI TUTTE LE CARATTERISTICHE

    print("\nCARATTERISTICHE DEL PARTECIPANTE:")

    for colonna in colonne + ["bmi"]:  # Mostra le 15 caratteristiche e il BMI

        print(f"{colonna}: {soggetto[colonna]}")


    # DEFINIZIONE DELLE VARIABILI NUMERICHE (TUTTE INCLUSE)

    variabili_numeriche = {
        "eta": "Età (anni)",
        "sigarette_giorno": "Sigarette/giorno",
        "peso_kg": "Peso (kg)",
        "altezza_cm": "Altezza (cm)",
        "bmi": "BMI",
        "frequenza_cardiaca": "Frequenza cardiaca",
        "sere_alcol_settimana": "Sere con alcol/settimana"
    }

    nomi_variabili = []  # Memorizza le etichette del grafico
    ranghi_percentuali = []  # Memorizza le posizioni percentili
    colori_numerici = []  # Distingue i valori superiori al riferimento
    valori_reali = []  # Memorizza i valori originali
    variabili_elevate = []  # Registra le caratteristiche elevate


    # CONTROLLO DEI VALORI ELEVATI

    print("\nCONFRONTO CON IL DATASET STORICO - 90° PERCENTILE:")

    for colonna, nome in variabili_numeriche.items():

        valore = float(soggetto[colonna])  # Legge il valore individuale

        storico = pd.to_numeric(
            df_X[colonna],
            errors="coerce"
        ).dropna()  # Recupera i valori storici validi

        limite = float(
            storico.quantile(0.90)
        )  # Calcola il 90° percentile del dataset storico

        rango = float(
            (storico <= valore).mean() * 100
        )  # Calcola la posizione percentile individuale

        elevato = valore > limite  # Controlla se supera il riferimento

        nomi_variabili.append(nome)
        ranghi_percentuali.append(rango)
        valori_reali.append(valore)

        # Evitiamo il viola: usiamo un blu acciaio per i valori normali e un rosso corallo/arancio elegante per i valori elevati
        colori_numerici.append(
            "#E85D75" if elevato else "#3A86FF"
        )

        if elevato:

            variabili_elevate.append(nome)

            stato = "SUPERIORE AL 90° PERCENTILE"

        else:
            stato = "Non supera il 90° percentile"

        print(
            f"{nome}: {valore:.2f} | "
            f"90° percentile storico: {limite:.2f} | {stato}"
        )

    if not variabili_elevate:
        print("\nNessuna delle caratteristiche selezionate supera il 90° percentile.")


    # GRAFICO INDIVIDUALE DEL PARTECIPANTE (SFONDO SCURO LEGGERO E TESTI BIANCHI)

    figura, asse_numerico = plt.subplots(
        figsize=(12, 8)
    )

    # Impostiamo uno sfondo scuro ma leggero (es. grigio scuro/ardesia: #1E1E2F o #222222)
    figura.patch.set_facecolor("#1E1E2F")
    asse_numerico.set_facecolor("#252538")


    # GRAFICO DELLE CARATTERISTICHE NUMERICHE

    barre = asse_numerico.barh(
        nomi_variabili,
        ranghi_percentuali,
        color=colori_numerici,
        height=0.55
    )

    # Linea di riferimento verticale
    asse_numerico.axvline(
        90,
        color="#FFD166",  # Giallo chiaro visibile sullo sfondo scuro
        linestyle="--",
        linewidth=2,
        label="Soglia di riferimento (Dataset Framingham)"
    )

    asse_numerico.set_xlim(0, 115)

    # Configurazione colori assi e etichette in bianco
    asse_numerico.set_xlabel(
        "Posizione percentile rispetto a Framingham", fontsize=11, fontweight="bold", color="white"
    )
    
    asse_numerico.set_title(
        f"Confronto completo delle feature - Utente ID {identificativo}",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=15
    )

    # Colore delle etichette sugli assi (tick labels) in bianco
    asse_numerico.tick_params(axis='x', colors='white')
    asse_numerico.tick_params(axis='y', colors='white')

    # Modifica colore dei bordi del grafico (spines) per renderli chiari/armoniosi
    for spine in asse_numerico.spines.values():
        spine.set_edgecolor("#555566")

    asse_numerico.invert_yaxis()
    
    # Legenda con testo bianco e sfondo coordinato
    legenda = asse_numerico.legend(loc="lower right", facecolor="#252538", edgecolor="#555566")
    for text in legenda.get_texts():
        text.set_color("white")

    for barra, valore in zip(barre, valori_reali):

        asse_numerico.text(
            barra.get_width() + 1,
            barra.get_y() + barra.get_height() / 2,
            f"Valore: {valore:g}",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white"  # Testo bianco accanto alle barre
        )


    # TITOLO GLOBALE DELLA FIGURA IN BIANCO

    figura.suptitle(
        f"HEARTVISION - Analisi Dettagliata | Utente {identificativo} | "
        f"Classe {classe} | "
        f"Stima classe 1: {percentuale:.2f}%",
        fontsize=14,
        fontweight="bold",
        color="white"
    )

    figura.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()  # Visualizza il grafico finale


# CONFRONTO DEGLI ALGORITMI MACHINE LEARNING

def crea_modello(configurazione):  # Costruisce il modello dalla configurazione scelta

    algoritmo = configurazione["algoritmo"]  # Recupera l'algoritmo

    if algoritmo == "KNN":
        return make_pipeline(
            StandardScaler(),  # Standardizza le caratteristiche nel singolo fold
            KNeighborsClassifier(n_neighbors=int(configurazione["k"]))
        )

    profondita = configurazione["max_depth"]  # Recupera la profondità
    profondita = None if pd.isna(profondita) else int(profondita)
    peso = configurazione["peso_classi"]  # Recupera il bilanciamento
    peso = None if peso == "nessuno" else "balanced"

    if algoritmo == "Decision Tree":
        return DecisionTreeClassifier(
            criterion=configurazione["criterio"],  # Utilizza Gini oppure Entropy
            max_depth=profondita,  # Imposta la profondità
            min_samples_leaf=20,  # Limita le foglie troppo piccole
            class_weight=peso,  # Bilancia le classi quando richiesto
            random_state=42
        )

    if algoritmo == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(configurazione["n_estimators"]),  # Numero di alberi
            max_depth=profondita,  # Profondità massima
            min_samples_leaf=5,  # Riduce l'eccessivo adattamento ai dati
            class_weight=peso,  # Bilancia le classi quando richiesto
            random_state=42,
            n_jobs=-1
        )

    raise ValueError("Algoritmo non riconosciuto!")


# VALUTAZIONE E CONFRONTO DEI MODELLI

def esegui_confronto_modelli():  # Confronta i modelli senza utilizzare i questionari

    global X_train, y_train, X_test, cv, risultati_cv, confronto
    global metriche, parametri, indici_migliori, migliori_modelli
    global X_sviluppo, X_verifica, y_sviluppo, y_verifica

    X_train = df_X[colonne + ["bmi"]]  # Seleziona le caratteristiche storiche
    y_train = df_X[target]  # Seleziona gli eventi registrati
    X_test = df_y[colonne + ["bmi"]]  # Memorizza il questionario da analizzare

    # SUDDIVISIONE DEL DATASET STORICO

    X_sviluppo, X_verifica, y_sviluppo, y_verifica = train_test_split(
        X_train, y_train,
        test_size=0.20,  # Riserva il 20% per una verifica indipendente
        stratify=y_train,  # Mantiene le proporzioni delle classi
        random_state=42
    )

    cv = StratifiedKFold(
        n_splits=5,  # Effettua cinque validazioni sui soli dati di sviluppo
        shuffle=True,
        random_state=42
    )

    # PREPARAZIONE DELLE CONFIGURAZIONI

    configurazioni = []  # Elenca tutti i modelli da confrontare

    for k in [3, 5, 7, 9, 11, 15, 21]:
        configurazioni.append({
            "algoritmo": "KNN", "configurazione": f"K={k}",
            "k": k, "criterio": None, "max_depth": None,
            "n_estimators": None, "peso_classi": "nessuno"
        })

    for criterio in ["gini", "entropy"]:
        for profondita in [3, 5, 10, None]:
            for peso in ["nessuno", "balanced"]:
                configurazioni.append({
                    "algoritmo": "Decision Tree",
                    "configurazione": f"{criterio} - profondità {profondita} - {peso}",
                    "k": None, "criterio": criterio,
                    "max_depth": profondita, "n_estimators": None,
                    "peso_classi": peso
                })

    for numero in [100, 200]:
        for profondita in [5, 10, None]:
            for peso in ["nessuno", "balanced"]:
                configurazioni.append({
                    "algoritmo": "Random Forest",
                    "configurazione": f"{numero} alberi - profondità {profondita} - {peso}",
                    "k": None, "criterio": None,
                    "max_depth": profondita, "n_estimators": numero,
                    "peso_classi": peso
                })

    # CROSS-VALIDATION E CALCOLO DELLE METRICHE

    risultati_cv = []  # Memorizza le metriche fuori campione

    for configurazione in configurazioni:
        modello = crea_modello(configurazione)  # Crea il modello corrente

        probabilita_oof = cross_val_predict(
            modello, X_sviluppo, y_sviluppo,
            cv=cv,  # Ogni paziente è previsto da un modello che non lo ha visto
            method="predict_proba"
        )[:, 1]  # Recupera le probabilità della classe positiva

        classi_oof = (probabilita_oof >= 0.50).astype(int)  # Soglia iniziale comune

        riga = configurazione.copy()  # Mantiene tutti i parametri
        riga.update({
            "accuracy": accuracy_score(y_sviluppo, classi_oof),
            "precision": precision_score(y_sviluppo, classi_oof, zero_division=0),
            "recall": recall_score(y_sviluppo, classi_oof, zero_division=0),
            "f1_score": f1_score(y_sviluppo, classi_oof, zero_division=0),
            "roc_auc": roc_auc_score(y_sviluppo, probabilita_oof),
            "pr_auc": average_precision_score(y_sviluppo, probabilita_oof),
            "brier": brier_score_loss(y_sviluppo, probabilita_oof)
        })
        risultati_cv.append(riga)  # Salva il confronto

    # ORDINAMENTO E SCELTA DEL MODELLO

    metriche = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "brier"]
    confronto = pd.DataFrame(risultati_cv)  # Costruisce la tabella completa
    confronto = confronto.sort_values(
        by=["pr_auc", "roc_auc", "brier"],
        ascending=[False, False, True]
    ).reset_index(drop=True)  # Privilegia la capacità di distinguere i positivi

    indici_migliori = confronto.groupby("algoritmo")["pr_auc"].idxmax()
    migliori_modelli = confronto.loc[indici_migliori].reset_index(drop=True)
    parametri = confronto[[
        "algoritmo", "configurazione", "k", "criterio",
        "max_depth", "n_estimators", "peso_classi"
    ]]  # Mantiene le configurazioni provate

    confronto[metriche] = confronto[metriche].round(3)  # Formatta i risultati
    migliori_modelli[metriche] = migliori_modelli[metriche].round(3)

    print("\nConfronto dei modelli completato correttamente!")
    print(f"Configurazioni analizzate: {len(confronto)}")
    print("Criterio di selezione: PR-AUC (Average Precision).")
    print("Il 20% dello storico resta escluso dalla scelta dei modelli.")
    print("Premi 3 nel menu per visualizzare tutti i risultati.")


# ADDESTRAMENTO DEFINITIVO E PREVISIONI

def esegui_previsioni():  # Calibra il modello selezionato e prevede i questionari

    global X_test, modello_finale, scaler_finale, risultati, previsioni
    global probabilita_evento, soglia_finale

    if "confronto" not in globals() or confronto is None or confronto.empty:
        print("Prima esegui il confronto dei modelli (opzione 2).")
        return

    X_test = df_y[colonne + ["bmi"]]  # Utilizza il questionario corrente
    scelta = confronto.iloc[0]  # Recupera la configurazione migliore per PR-AUC
    modello_scelto = scelta["algoritmo"]  # Identifica il modello selezionato

    print("\nMODELLO SCELTO AUTOMATICAMENTE:", modello_scelto)
    print("CRITERIO: PR-AUC nella cross-validation")
    print("PR-AUC:", scelta["pr_auc"])
    print("CONFIGURAZIONE:", scelta["configurazione"])

    # CALIBRAZIONE E SCELTA DELLA SOGLIA SUI DATI DI SVILUPPO

    modello_calibrato = CalibratedClassifierCV(
        estimator=crea_modello(scelta),  # Calibra il classificatore scelto
        method="sigmoid",  # Applica la calibrazione di Platt
        cv=3  # Mantiene distinte le parti di addestramento e calibrazione
    )

    probabilita_oof = cross_val_predict(
        modello_calibrato, X_sviluppo, y_sviluppo,
        cv=cv, method="predict_proba"
    )[:, 1]  # Ottiene probabilità calibrate fuori dal fold

    
    # SELEZIONE AUTOMATICA DELLA SOGLIA TRAMITE F1-SCORE

    soglia_finale = 0.50  # Soglia iniziale
    migliore_f1 = -1.0  # Memorizza il miglior F1-score
    migliore_precision = 0.0  # Memorizza la precision
    migliore_recall = 0.0  # Memorizza il recall

    for soglia in np.arange(0.10, 0.601, 0.01):

        classi_oof = (probabilita_oof >= soglia).astype(int)

        precisione = precision_score(
            y_sviluppo, classi_oof, zero_division=0
        )

        sensibilita = recall_score(
            y_sviluppo, classi_oof, zero_division=0
        )

        punteggio = f1_score(
            y_sviluppo, classi_oof, zero_division=0
        )

        if punteggio > migliore_f1:
            migliore_f1 = punteggio
            soglia_finale = float(soglia)
            migliore_precision = precisione
            migliore_recall = sensibilita

    print(f"SOGLIA SELEZIONATA: {soglia_finale:.3f}")
    print(f"F1-SCORE DI SVILUPPO: {migliore_f1:.3f}")
    print(f"PRECISION DI SVILUPPO: {migliore_precision:.3f}")
    print(f"RECALL DI SVILUPPO: {migliore_recall:.3f}")
    # VERIFICA SU PAZIENTI STORICI MAI USATI PER LA SCELTA

    modello_verifica = CalibratedClassifierCV(
        estimator=crea_modello(scelta), method="sigmoid", cv=3
    )
    modello_verifica.fit(X_sviluppo, y_sviluppo)  # Addestra solo sullo sviluppo

    probabilita_verifica = modello_verifica.predict_proba(X_verifica)[:, 1]
    classi_verifica = (probabilita_verifica >= soglia_finale).astype(int)

    print("\nVERIFICA INDIPENDENTE SUL 20% DEL DATASET STORICO:")
    print(f"Precision: {precision_score(y_verifica, classi_verifica, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_verifica, classi_verifica, zero_division=0):.3f}")
    print(f"F2-score: {fbeta_score(y_verifica, classi_verifica, beta=2, zero_division=0):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_verifica, probabilita_verifica):.3f}")
    print(f"PR-AUC: {average_precision_score(y_verifica, probabilita_verifica):.3f}")
    print(f"Brier score: {brier_score_loss(y_verifica, probabilita_verifica):.3f}")

    # ADDESTRAMENTO DEFINITIVO SU TUTTO LO STORICO

    modello_finale = CalibratedClassifierCV(
        estimator=crea_modello(scelta), method="sigmoid", cv=5
    )  # Mantiene la calibrazione anche nel modello definitivo
    modello_finale.fit(X_train, y_train)  # Addestra sullo storico completo
    scaler_finale = None  # Lo scaler KNN è gestito nella Pipeline

    # PREVISIONI SUL QUESTIONARIO CARICATO

    probabilita_evento = modello_finale.predict_proba(X_test)[:, 1] * 100
    previsioni = ((probabilita_evento / 100) >= soglia_finale).astype(int)

    risultati = pd.DataFrame({
        "utente": range(1, len(X_test) + 1),  # Conserva l'ID progressivo
        "previsione": previsioni,  # Applica la soglia ottimizzata
        "probabilita_evento_percentuale": probabilita_evento.round(2)
    })

    print("\nMODELLO UTILIZZATO:", modello_scelto)
    print(f"SOGLIA DI CLASSIFICAZIONE: {soglia_finale:.3f}")
    print("Le percentuali sono stime del modello, non diagnosi cliniche.")
    print("\nRISULTATI HEART VISION:")
    print(risultati)  # Visualizza tutte le previsioni


# VISUALIZZAZIONE DEI GRAFICI

def mostra_grafici():  # Visualizza il grafico dei fattori di rischio in stile scuro

    if "df_X" not in globals():  # Verifica che lo storico sia stato caricato
        print("Prima carica i dataset (opzione 1).")
        return

    # GRAFICO DEI FATTORI DI RISCHIO

    fattori_rischio = [  # Seleziona i fattori da confrontare
        "fumatore",
        "diabete",
        "colesterolo_alto",
        "alcol",
        "ictus_precedente",
        "ipertensione",
    ]

    medie_rischio = []  # Memorizza le percentuali calcolate

    for f in fattori_rischio:  # Scorre i fattori selezionati

        rischio_si = df_X[df_X[f] == 1][target].mean() * 100  # Calcola la percentuale di eventi

        medie_rischio.append({
            "Fattore": f.upper(),
            "Rischio percentuale (%)": rischio_si
        })  # Salva la percentuale del fattore

    df_fattori_plot = pd.DataFrame(medie_rischio).sort_values(
        by="Rischio percentuale (%)",
        ascending=False  # Ordina i fattori per percentuale decrescente
    )

    # CREAZIONE DELLA FIGURA CON SFONDO SCURO LEGGERO
    figura, asse = plt.subplots(figsize=(11, 6))
    
    figura.patch.set_facecolor("#1E1E2F")  # Sfondo esterno scuro
    asse.set_facecolor("#252538")      # Sfondo interno dell'area grafico

    # Disegna le barre (usiamo un colore elegante in linea con il tema, es. un viola/magenta scuro ma luminoso o un blu/rosa coerente)
    bars = asse.bar(
        df_fattori_plot["Fattore"],
        df_fattori_plot["Rischio percentuale (%)"],
        color="#E85D75",  # Colore ripreso dal tema dell'app (rosso corallo/rosa)
        alpha=0.9,
        width=0.55
    )

    # Titolo ed etichette con testo bianco
    asse.set_title(
        "HEARTVISION - Incidenza dei Fattori di Rischio a 10 Anni",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=15
    )

    asse.set_xlabel("Fattore di Rischio", fontsize=12, fontweight="bold", color="white", labelpad=10)
    asse.set_ylabel("Percentuale di Evento Coronarico (%)", fontsize=12, fontweight="bold", color="white")

    asse.set_ylim(
        0,
        max(df_fattori_plot["Rischio percentuale (%)"]) * 1.2
    )

    # Configurazione colori degli assi e dei tick in bianco
    asse.tick_params(axis='x', colors='white', labelsize=10)
    asse.tick_params(axis='y', colors='white', labelsize=10)

    # Modifica il colore dei bordi del grafico (spines) per armonizzarli con lo sfondo scuro
    for spine in asse.spines.values():
        spine.set_edgecolor("#555566")

    # Aggiunge le percentuali sopra ogni barra con testo bianco
    for bar in bars:
        yval = bar.get_height()
        asse.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.5,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="white"
        )

    plt.xticks(rotation=0, fontweight="bold")
    plt.tight_layout()
    plt.show()


# VISUALIZZAZIONE DEI RISULTATI

def mostra_risultati():  # Mostra la tabella e il grafico delle previsioni

    if "risultati" not in globals() or risultati is None or risultati.empty:  # Verifica le previsioni correnti
        print("Prima esegui le previsioni (opzione 4).")
        return

    print("\nRISULTATI HEARTVISION:")
    print(risultati.to_string(index=False))  # Mostra tutti i partecipanti
    mostra_grafico_questionari()  # Visualizza le probabilità dell'ultimo CSV


# GRAFICO DELLE PREVISIONI SUI QUESTIONARI

def mostra_grafico_questionari():  # Visualizza il grafico a doppio anello in stile scuro coordinato

    if "risultati" not in globals() or risultati is None or risultati.empty:
        print("Prima esegui le previsioni sul CSV (opzione 4).")
        return

    df = risultati.copy()
    totale_utenti = len(df)

    # CLASSI PREVISTE E FASCE DI PROBABILITÀ

    c0 = df[df["previsione"] == 0]  # Partecipanti classificati come 0
    c1 = df[df["previsione"] == 1]  # Partecipanti classificati come 1
    p = df["probabilita_evento_percentuale"]  # Stime percentuali

    n_b1 = int((p <= 25).sum())  # Prima fascia: 0-25%
    n_b2 = int(((p > 25) & (p <= 50)).sum())  # Seconda fascia: 25-50%
    n_b3 = int(((p > 50) & (p <= 75)).sum())  # Terza fascia: 50-75%
    n_b4 = int((p > 75).sum())  # Quarta fascia: 75-100%

    # Costruzione dinamica anello interno (Macro Classi) con palette scura/coordinata
    valori_interno = []
    colori_interno = []
    etichette_interno_legenda = []

    if len(c0) > 0:
        valori_interno.append(len(c0))
        colori_interno.append("#3A86FF")  # Blu coordinato stile web
        etichette_interno_legenda.append(f"Classe 0 - Non segnalato ({len(c0)} utenti)")

    if len(c1) > 0:
        valori_interno.append(len(c1))
        colori_interno.append("#E85D75")  # Rosso corallo coordinato
        etichette_interno_legenda.append(f"Classe 1 - Segnalato ({len(c1)} utenti)")

    # Costruzione dinamica anello esterno (Fasce di probabilità) con sfumature coordinate
    fasce_config = [
        (n_b1, "#1D3557", "Fascia 0% - 25%"),
        (n_b2, "#457B9D", "Fascia 25% - 50%"),
        (n_b3, "#D90429", "Fascia 50% - 75%"),
        (n_b4, "#EF476F", "Fascia 75% - 100%")
    ]

    valori_esterno = []
    colori_esterno = []
    etichette_esterno_legenda = []

    for conteggio, colore, nome_fascia in fasce_config:
        if conteggio > 0:
            valori_esterno.append(conteggio)
            colori_esterno.append(colore)
            etichette_esterno_legenda.append(f"{nome_fascia} ({conteggio} utenti)")

    # Funzione helper per mostrare sia la % sul totale sia il numero esatto di utenti
    def formatta_etichetta_spicchio(pct):
        n_persone = int(round(pct / 100.0 * totale_utenti))
        if pct < 5:  # Nasconde il testo se lo spicchio è troppo piccolo
            return ""
        return f"{pct:.1f}%\n({n_persone} ut.)"

    # Creazione della figura con sfondo scuro leggero
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor("#1E1E2F")
    ax.set_facecolor("#252538")

    # 1. ANELLO ESTERNO (Fasce di probabilità)
    wedges_ext, _, autotexts_ext = ax.pie(
        valori_esterno,
        radius=1.15,
        colors=colori_esterno,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.35, edgecolor="#1E1E2F", linewidth=2),
        autopct=formatta_etichetta_spicchio,
        pctdistance=0.82,
        textprops=dict(color="white", fontsize=9, fontweight="bold", ha="center", va="center")
    )

    # 2. ANELLO INTERNO (Macro Classi)
    wedges_int, _, autotexts_int = ax.pie(
        valori_interno,
        radius=0.80,
        colors=colori_interno,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.35, edgecolor="#1E1E2F", linewidth=2),
        autopct=formatta_etichetta_spicchio,
        pctdistance=0.74,
        textprops=dict(color="white", fontsize=10, fontweight="bold", ha="center", va="center")
    )

    # Testo centrale nel foro della ciambella con il totale dei partecipanti (testo bianco)
    ax.text(
        0, 0,
        f"Totale\n{totale_utenti} Utenti",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="white"
    )

    # Legenda laterale con sfondo scuro e testi bianchi
    tutti_wedges = wedges_int + wedges_ext
    tutte_etichette = etichette_interno_legenda + etichette_esterno_legenda

    legenda = ax.legend(
        tutti_wedges,
        tutte_etichette,
        title="Legenda Classi e Fasce di Rischio",
        title_fontsize=10,
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="#252538",
        edgecolor="#555566"
    )
    
    # Rende bianchi i testi della legenda
    plt.setp(legenda.get_texts(), color="white")
    legenda.get_title().set_color("white")

    # Titolo del grafico in bianco
    ax.set_title(
        "HEARTVISION - Distribuzione Utenti per Classe e Fascia di Rischio",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=15
    )

    plt.tight_layout()
    plt.show()


# MENU PRINCIPALE

def menu_principale():  # Gestisce la navigazione del programma

    while True:  # Mantiene attivo il menu fino all'uscita

        print("\n" + "=" * 55)
        print("             HEARTVISION - MENU PRINCIPALE")
        print("=" * 55)

        print("1. Caricamento e pulizia dei dataset")
        print("2. Confronto KNN, Decision Tree e Random Forest")
        print("3. Visualizzazione dei risultati del confronto")
        print("4. Addestramento automatico e previsione questionari")
        print("5. Grafico dei fattori di rischio")
        print("6. Risultati e grafico delle probabilità")
        print("7. Studio singolo paziente")
        print("0. Esci")

        scelta = input("\nSeleziona un'opzione: ").strip()  # Legge l'opzione selezionata

        try:

            if scelta == "1":  # Carica e pulisce i dataset
                carica_dataset()

            elif scelta == "2":  # Confronta gli algoritmi

                if "df_X" not in globals() or "df_y" not in globals():
                    print("Prima carica i dataset (opzione 1).")
                else:
                    esegui_confronto_modelli()

            elif scelta == "3":  # Visualizza i risultati del confronto

                if "confronto" in globals() and confronto is not None and not confronto.empty:

                    print("\nCONFRONTO COMPLETO DEGLI ALGORITMI:")
                    print(confronto.to_string(index=False))

                    print("\nMIGLIORE CONFIGURAZIONE PER ALGORITMO:")
                    print(
                        migliori_modelli[
                            ["algoritmo", "configurazione"] + metriche
                        ].to_string(index=False)
                    )

                else:
                    print("Prima esegui il confronto dei modelli (opzione 2).")

            elif scelta == "4":  # Addestra il modello ed effettua le previsioni
                esegui_previsioni()

            elif scelta == "5":  # Visualizza i fattori di rischio
                mostra_grafici()  # Mostra soltanto il grafico fucsia

            elif scelta == "6":  # Mostra le previsioni e il relativo grafico
                mostra_risultati()  # Mostra la tabella e il grafico delle probabilità

            elif scelta == "7":  # Analizza il singolo partecipante
                analizza_partecipante()  # Cerca l'email e mostra l'analisi individuale
            
            elif scelta == "0":  # Termina il programma
                print("Chiusura di HEARTVISION. Arrivederci!")
                break

            else:
                print("Opzione non valida. Inserisci un numero da 0 a 7.")

        except FileNotFoundError as errore:  # Gestisce i file mancanti

            print(f"\nFile non trovato: {errore}")
            print(
                "Controlla che le cartelle Dataset_storico e "
                "Dataset_questionari siano accanto allo script."
            )

        except Exception as errore:  # Mostra eventuali altri errori

            print(f"\nSi è verificato un errore: {errore}")


# AVVIO DEL PROGRAMMA

if __name__ == "__main__":  # Avvia il programma soltanto se eseguito direttamente
    menu_principale()
