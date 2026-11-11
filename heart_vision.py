
# HEARTVISION - GESTIONE DATASET, MACHINE LEARNING E PREVISIONI SU RISPOSTE DATE DA UTENTI A SCOPO DI PREVENZIONE


# IMPORTAZIONE DELLE LIBRERIE

from sklearn.neighbors import KNeighborsClassifier  # Importa il classificatore KNN
from sklearn.preprocessing import StandardScaler  # Importa lo standardizzatore
from sklearn.tree import DecisionTreeClassifier  # Importa l'albero decisionale
from sklearn.ensemble import RandomForestClassifier  # Importa la Random Forest
from sklearn.model_selection import StratifiedKFold  # Importa la cross-validation stratificata
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

    global df_X, df_y, risultati  # Rende accessibili i dataset e i risultati

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


# CONFRONTO DEGLI ALGORITMI MACHINE LEARNING

def esegui_confronto_modelli():  # Valuta KNN, Decision Tree e Random Forest

    global X_train, y_train, X_test, cv, risultati_cv, confronto
    global metriche, parametri, indici_migliori, migliori_modelli

    X_train = df_X[colonne + ["bmi"]]  # Seleziona le caratteristiche dello storico
    y_train = df_X[target]  # Seleziona gli eventi coronarici osservati
    X_test = df_y[colonne + ["bmi"]]  # Seleziona le caratteristiche dei questionari


    # PREPARAZIONE DELLA CROSS-VALIDATION

    cv = StratifiedKFold(
        n_splits=5,  # Divide i dati in cinque gruppi
        shuffle=True,  # Mescola le osservazioni
        random_state=42  # Rende riproducibile la suddivisione
    )

    risultati_cv = []  # Raccoglie i risultati delle configurazioni


    # CROSS-VALIDATION KNN

    valori_k = [3, 5, 7, 9, 11, 15]  # Definisce i numeri di vicini da provare

    for k in valori_k:  # Prova ogni valore di K

        for indice_train, indice_val in cv.split(X_train, y_train):  # Scorre i fold

            X_fold_train = X_train.iloc[indice_train]  # Estrae il training
            X_fold_val = X_train.iloc[indice_val]  # Estrae la validazione

            y_fold_train = y_train.iloc[indice_train]  # Estrae le classi di training
            y_fold_val = y_train.iloc[indice_val]  # Estrae le classi di validazione

            scaler_fold = StandardScaler()  # Crea lo standardizzatore

            X_fold_train = scaler_fold.fit_transform(X_fold_train)  # Standardizza il training
            X_fold_val = scaler_fold.transform(X_fold_val)  # Standardizza la validazione

            modello = KNeighborsClassifier(
                n_neighbors=k  # Imposta il valore di K corrente
            )

            modello.fit(X_fold_train, y_fold_train)  # Addestra il KNN

            previsioni = modello.predict(X_fold_val)  # Prevede le classi
            probabilita = modello.predict_proba(X_fold_val)  # Calcola le probabilità

            indice_evento = list(modello.classes_).index(1)  # Individua la classe 1
            probabilita_evento = probabilita[:, indice_evento]  # Estrae le probabilità positive

            risultati_cv.append({  # Registra le metriche del fold

                "algoritmo": "KNN",
                "configurazione": f"K={k}",
                "k": k,
                "criterio": None,
                "max_depth": None,
                "n_estimators": None,

                "accuracy": accuracy_score(
                    y_fold_val, previsioni  # Calcola l'accuracy
                ),

                "precision": precision_score(
                    y_fold_val, previsioni, zero_division=0  # Calcola la precision
                ),

                "recall": recall_score(
                    y_fold_val, previsioni, zero_division=0  # Calcola il recall
                ),

                "f1_score": f1_score(
                    y_fold_val, previsioni, zero_division=0  # Calcola l'F1-score
                ),

                "roc_auc": roc_auc_score(
                    y_fold_val, probabilita_evento  # Calcola la ROC-AUC
                )

            })


    # CROSS-VALIDATION DECISION TREE

    criteri = ["gini", "entropy"]  # Definisce i criteri da confrontare
    profondita_alberi = [3, 5, 10, None]  # Definisce le profondità da confrontare

    for criterio in criteri:  # Prova ogni criterio

        for profondita in profondita_alberi:  # Prova ogni profondità

            for indice_train, indice_val in cv.split(X_train, y_train):  # Scorre i fold

                X_fold_train = X_train.iloc[indice_train]  # Estrae il training
                X_fold_val = X_train.iloc[indice_val]  # Estrae la validazione

                y_fold_train = y_train.iloc[indice_train]  # Estrae le classi di training
                y_fold_val = y_train.iloc[indice_val]  # Estrae le classi di validazione

                modello = DecisionTreeClassifier(
                    criterion=criterio,  # Utilizza Gini oppure Entropy
                    max_depth=profondita,  # Imposta la profondità
                    min_samples_leaf=20,  # Richiede almeno 20 pazienti per foglia
                    random_state=42  # Rende riproducibile il modello
                )

                modello.fit(X_fold_train, y_fold_train)  # Addestra l'albero

                previsioni = modello.predict(X_fold_val)  # Prevede le classi
                probabilita = modello.predict_proba(X_fold_val)  # Calcola le probabilità

                indice_evento = list(modello.classes_).index(1)  # Individua la classe 1
                probabilita_evento = probabilita[:, indice_evento]  # Estrae le probabilità positive

                risultati_cv.append({  # Registra le metriche del fold

                    "algoritmo": "Decision Tree",

                    "configurazione": (
                        f"{criterio} - profondità {profondita}"
                    ),

                    "k": None,
                    "criterio": criterio,
                    "max_depth": profondita,
                    "n_estimators": None,

                    "accuracy": accuracy_score(
                        y_fold_val, previsioni  # Calcola l'accuracy
                    ),

                    "precision": precision_score(
                        y_fold_val, previsioni, zero_division=0  # Calcola la precision
                    ),

                    "recall": recall_score(
                        y_fold_val, previsioni, zero_division=0  # Calcola il recall
                    ),

                    "f1_score": f1_score(
                        y_fold_val, previsioni, zero_division=0  # Calcola l'F1-score
                    ),

                    "roc_auc": roc_auc_score(
                        y_fold_val, probabilita_evento  # Calcola la ROC-AUC
                    )

                })


    # CROSS-VALIDATION RANDOM FOREST

    numero_alberi = [100, 200, 300]  # Definisce il numero di alberi
    profondita_foresta = [5, 10, None]  # Definisce le profondità

    for numero in numero_alberi:  # Prova ogni numero di alberi

        for profondita in profondita_foresta:  # Prova ogni profondità

            for indice_train, indice_val in cv.split(X_train, y_train):  # Scorre i fold

                X_fold_train = X_train.iloc[indice_train]  # Estrae il training
                X_fold_val = X_train.iloc[indice_val]  # Estrae la validazione

                y_fold_train = y_train.iloc[indice_train]  # Estrae le classi di training
                y_fold_val = y_train.iloc[indice_val]  # Estrae le classi di validazione

                modello = RandomForestClassifier(
                    n_estimators=numero,  # Imposta il numero di alberi
                    max_depth=profondita,  # Imposta la profondità
                    random_state=42,  # Rende riproducibile il modello
                    n_jobs=-1  # Utilizza tutti i processori disponibili
                )

                modello.fit(X_fold_train, y_fold_train)  # Addestra la foresta

                previsioni = modello.predict(X_fold_val)  # Prevede le classi
                probabilita = modello.predict_proba(X_fold_val)  # Calcola le probabilità

                indice_evento = list(modello.classes_).index(1)  # Individua la classe 1
                probabilita_evento = probabilita[:, indice_evento]  # Estrae le probabilità positive

                risultati_cv.append({  # Registra le metriche del fold

                    "algoritmo": "Random Forest",

                    "configurazione": (
                        f"{numero} alberi - profondità {profondita}"
                    ),

                    "k": None,
                    "criterio": None,
                    "max_depth": profondita,
                    "n_estimators": numero,

                    "accuracy": accuracy_score(
                        y_fold_val, previsioni  # Calcola l'accuracy
                    ),

                    "precision": precision_score(
                        y_fold_val, previsioni, zero_division=0  # Calcola la precision
                    ),

                    "recall": recall_score(
                        y_fold_val, previsioni, zero_division=0  # Calcola il recall
                    ),

                    "f1_score": f1_score(
                        y_fold_val, previsioni, zero_division=0  # Calcola l'F1-score
                    ),

                    "roc_auc": roc_auc_score(
                        y_fold_val, probabilita_evento  # Calcola la ROC-AUC
                    )

                })


    # CONFRONTO DELLE METRICHE

    df_risultati = pd.DataFrame(risultati_cv)  # Trasforma le valutazioni in un DataFrame

    metriche = [  # Elenca le metriche da confrontare
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    confronto = (
        df_risultati
        .groupby(["algoritmo", "configurazione"])[metriche]  # Raggruppa le configurazioni
        .mean()  # Calcola le medie sui cinque fold
        .reset_index()  # Ripristina l'indice
    )

    parametri = (
        df_risultati[
            [
                "algoritmo",
                "configurazione",
                "k",
                "criterio",
                "max_depth",
                "n_estimators"
            ]
        ]
        .drop_duplicates()  # Elimina le configurazioni ripetute
    )

    confronto = confronto.merge(
        parametri,
        on=["algoritmo", "configurazione"]  # Unisce metriche e parametri
    )

    confronto[metriche] = confronto[metriche].round(3)  # Arrotonda le metriche

    confronto = confronto.sort_values(
        by="f1_score",  # Ordina secondo l'F1-score
        ascending=False  # Mostra prima i valori più alti
    )

    # SELEZIONE DELLE MIGLIORI CONFIGURAZIONI

    indici_migliori = (
        confronto
        .groupby("algoritmo")["f1_score"]
        .idxmax()
    )  # Trova il miglior F1-score per ciascun algoritmo

    migliori_modelli = confronto.loc[
        indici_migliori
    ].reset_index(drop=True)  # Salva i parametri migliori

    # CONFERMA DEL CONFRONTO

    print("\nConfronto dei modelli completato correttamente!")
    print(f"Configurazioni analizzate: {len(confronto)}")
    print("Premi 3 nel menu per visualizzare tutti i risultati.")


# ADDESTRAMENTO DEFINITIVO E PREVISIONI

def esegui_previsioni():  # Addestra automaticamente il modello con F1-score migliore

    if "confronto" not in globals():  # Verifica che il confronto sia stato eseguito
        print("Prima esegui il confronto dei modelli (opzione 2).")
        return

    # AGGIORNAMENTO DEL CSV DA ANALIZZARE

    global X_test  # Permette di aggiornare i questionari utilizzati

    X_test = df_y[colonne + ["bmi"]]  # Recupera sempre l'ultimo CSV caricato

    # SCELTA AUTOMATICA DEL MODELLO MIGLIORE

    global modello_finale, scaler_finale, risultati, previsioni, probabilita_evento

    indice_migliore = confronto["f1_score"].idxmax()  # Individua il punteggio F1 medio più alto
    riga_scelta = confronto.loc[indice_migliore]  # Recupera i parametri della configurazione migliore
    modello_scelto = riga_scelta["algoritmo"]  # Seleziona automaticamente KNN, Decision Tree o Random Forest

    print("\nMODELLO SCELTO AUTOMATICAMENTE:", modello_scelto)
    print("CRITERIO: F1-score medio nella cross-validation")
    print("F1-SCORE MEDIO:", riga_scelta["f1_score"])
    print("CONFIGURAZIONE:", riga_scelta["configurazione"])


    # CREAZIONE DEL MODELLO DEFINITIVO

    if modello_scelto == "KNN":  # Prepara il classificatore KNN

        scaler_finale = StandardScaler()  # Crea lo scaler finale

        X_train_finale = scaler_finale.fit_transform(X_train)  # Standardizza lo storico
        X_test_finale = scaler_finale.transform(X_test)  # Standardizza i questionari

        modello_finale = KNeighborsClassifier(
            n_neighbors=int(riga_scelta["k"])  # Utilizza il K migliore
        )

    elif modello_scelto == "Decision Tree":  # Prepara il Decision Tree

        X_train_finale = X_train  # Utilizza le caratteristiche originali
        X_test_finale = X_test  # Utilizza i questionari originali

        profondita = riga_scelta["max_depth"]  # Recupera la profondità migliore

        if pd.isna(profondita):  # Verifica se la profondità è illimitata
            profondita = None
        else:
            profondita = int(profondita)  # Converte la profondità in intero

        modello_finale = DecisionTreeClassifier(
            criterion=riga_scelta["criterio"],  # Utilizza il criterio migliore
            max_depth=profondita,  # Utilizza la profondità migliore
            min_samples_leaf=20,  # Richiede almeno 20 pazienti per foglia
            random_state=42  # Rende riproducibile il modello
        )

    elif modello_scelto == "Random Forest":  # Prepara la Random Forest

        X_train_finale = X_train  # Utilizza le caratteristiche originali
        X_test_finale = X_test  # Utilizza i questionari originali

        profondita = riga_scelta["max_depth"]  # Recupera la profondità migliore

        if pd.isna(profondita):  # Verifica se la profondità è illimitata
            profondita = None
        else:
            profondita = int(profondita)  # Converte la profondità in intero

        modello_finale = RandomForestClassifier(
            n_estimators=int(riga_scelta["n_estimators"]),  # Utilizza il numero di alberi migliore
            max_depth=profondita,  # Utilizza la profondità migliore
            random_state=42,  # Rende riproducibile il modello
            n_jobs=-1  # Utilizza tutti i processori disponibili
        )


    # ADDESTRAMENTO DEL MODELLO

    modello_finale.fit(
        X_train_finale,  # Utilizza le caratteristiche dei pazienti storici
        y_train  # Utilizza gli eventi coronarici osservati
    )


    # PREVISIONI SUI QUESTIONARI

    previsioni = modello_finale.predict(X_test_finale)  # Prevede le classi dei questionari

    probabilita = modello_finale.predict_proba(X_test_finale)  # Calcola le probabilità delle classi

    indice_evento = list(modello_finale.classes_).index(1)  # Individua la classe positiva

    probabilita_evento = (
        probabilita[:, indice_evento] * 100  # Converte le probabilità in percentuali
    )


    # RISULTATI DELLE PREVISIONI

    risultati = pd.DataFrame({

        "utente": range(1, len(X_test) + 1),  # Numera le righe dei risultati

        "previsione": previsioni,  # Mostra la classe prevista

        "probabilita_evento_percentuale": (
            probabilita_evento.round(2)  # Arrotonda le probabilità
        )

    })

    print("\nMODELLO UTILIZZATO:", modello_scelto)
    print("\nRISULTATI HEART VISION:")
    print(risultati)  # Visualizza le previsioni


# VISUALIZZAZIONE DEI GRAFICI

def mostra_grafici():  # Visualizza il grafico fucsia dei fattori di rischio

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

    plt.figure(figsize=(11, 6))  # Crea una nuova figura

    bars = plt.bar(
        df_fattori_plot["Fattore"],
        df_fattori_plot["Rischio percentuale (%)"],
        color="#FF27BE",
        alpha=0.85,
        width=0.55
    )  # Disegna le barre dei fattori

    plt.title(
        "HEARTVISION - Incidenza dei Fattori di Rischio a 10 Anni",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    plt.xlabel("Fattore di Rischio", fontsize=12, labelpad=10)
    plt.ylabel("Percentuale di Evento Coronarico (%)", fontsize=12)

    plt.ylim(
        0,
        max(df_fattori_plot["Rischio percentuale (%)"]) * 1.2
    )  # Lascia spazio sopra le barre

    for bar in bars:  # Aggiunge le percentuali sopra ogni barra

        yval = bar.get_height()  # Recupera l'altezza della barra

        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.5,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.xticks(rotation=0, fontsize=10, fontweight="bold")
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

def mostra_grafico_questionari():  # Visualizza le previsioni dell'ultimo CSV

    if "risultati" not in globals() or risultati is None or risultati.empty:
        print("Prima esegui le previsioni sul CSV (opzione 4).")
        return

    etichette = [f"Utente {n}" for n in risultati["utente"]]  # Crea le etichette dei partecipanti

    percentuali = risultati["probabilita_evento_percentuale"]  # Recupera le probabilità stimate
    classi = risultati["previsione"]  # Recupera le classi previste

    colori = [
        "#E85D75" if classe == 1 else "#4682B4"
        for classe in classi
    ]  # Rosso per la classe 1 e blu per la classe 0

    altezza = max(6, len(risultati) * 0.45)  # Adatta il grafico al numero di utenti

    plt.figure(figsize=(11, altezza))  # Crea una figura con altezza dinamica

    barre = plt.barh(
        etichette,
        percentuali,
        color=colori,
        alpha=0.85
    )  # Disegna una barra per ogni partecipante

    plt.axvline(
        x=50,
        color="gray",
        linestyle="--",
        label="Riferimento 50%"
    )  # Mostra la soglia percentuale di riferimento

    for barra, percentuale, classe in zip(barre, percentuali, classi):

        posizione_y = barra.get_y() + barra.get_height() / 2  # Trova il centro della barra

        if percentuale >= 90:
            posizione_x = percentuale - 2  # Sposta il testo dentro le barre lunghe
            allineamento = "right"
        else:
            posizione_x = percentuale + 1  # Sposta il testo dopo le barre corte
            allineamento = "left"

        plt.text(
            posizione_x,
            posizione_y,
            f"{percentuale:.2f}% - Classe {classe}",
            va="center",
            ha=allineamento,
            fontsize=9
        )  # Scrive percentuale e classe accanto alla barra

    plt.title("HEARTVISION - Previsioni sul CSV analizzato")  # Imposta il titolo
    plt.xlabel("Probabilità stimata della classe 1 (%)")  # Imposta l'asse orizzontale
    plt.ylabel("Partecipanti del questionario")  # Imposta l'asse verticale
    plt.xlim(0, 100)  # Imposta la scala percentuale
    plt.gca().invert_yaxis()  # Mantiene il primo partecipante in alto
    plt.grid(axis="x", linestyle="--", alpha=0.3)  # Mostra la griglia
    plt.legend()  # Mostra la legenda
    plt.tight_layout()  # Ottimizza la disposizione
    plt.show()  # Visualizza il grafico


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

                if "confronto" in globals():

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

            elif scelta == "0":  # Termina il programma
                print("Chiusura di HEARTVISION. Arrivederci!")
                break

            else:
                print("Opzione non valida. Inserisci un numero da 0 a 6.")

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
