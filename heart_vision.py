
# ALGORITMI MACHINE LEARNING
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# CROSS VALIDATION
from sklearn.model_selection import StratifiedKFold


# ============================================================
#                  HEARTVISION - GESTIONE DATI
# ============================================================


# ------------------------------------------------------------
# 1. DEFINIZIONE DELLE COLONNE
# ------------------------------------------------------------

# Colonne comuni al dataset storico e al questionario.
# L'ordine deve corrispondere a quello delle domande del Form.

colonne = [
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

# Variabile da prevedere, presente solo nel dataset storico.
target = "evento_coronarico_10_anni"


# ============================================================
# 2. FUNZIONE CONVERSIONE QUESTIONARIO
# ============================================================

def converti_questionario(df):

    # Copiamo il DataFrame per non modificare l'originale.
    df = df.copy()

    # Eliminiamo eventuali spazi nelle intestazioni.
    df.columns = df.columns.str.strip()

    # --------------------------------------------------------
    # 2.1 ELIMINAZIONE DEI METADATI
    # --------------------------------------------------------

    # Data ed email non servono al Machine Learning.
    metadati = [
        "timestamp",
        "username",
        "email",
        "email address",
        "indirizzo email"
    ]

    # Individuiamo le colonne da eliminare.
    da_eliminare = [
        c for c in df.columns
        if c.lower() in metadati
    ]

    # Eliminiamo le colonne individuate.
    df = df.drop(columns=da_eliminare)

    # --------------------------------------------------------
    # 2.2 CONTROLLO DEL NUMERO DI COLONNE
    # --------------------------------------------------------

    # Il questionario deve contenere esattamente 15 risposte.
    if len(df.columns) != len(colonne):
        raise ValueError(
            "Il questionario deve contenere 15 colonne!"
        )

    # Se i nomi sono già corretti, ordiniamo le colonne.
    # Altrimenti assegniamo i nomi in base all'ordine del Form.
    if set(df.columns) == set(colonne):
        df = df[colonne]
    else:
        df.columns = colonne

    # --------------------------------------------------------
    # 2.3 CONVERSIONE DELLE RISPOSTE
    # --------------------------------------------------------

    # Dizionario per trasformare le risposte in 0 e 1.
    conversioni = {
        "maschio": "1",
        "femmina": "0",
        "si": "1",
        "sì": "1",
        "s": "1",
        "no": "0",
        "n": "0"
    }

    # Scorriamo tutte le colonne del questionario.
    for colonna in colonne:

        # Convertiamo in stringa, eliminiamo gli spazi
        # e trasformiamo le lettere in minuscolo.
        valori = (
            df[colonna]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # Sostituiamo le risposte testuali.
        valori = valori.replace(conversioni)

        # Convertiamo eventuali virgole decimali in punti.
        valori = valori.str.replace(
            ",", ".", regex=False
        )

        # Trasformiamo i valori in numeri.
        # Gli errori diventano NaN e saranno corretti dopo.
        df[colonna] = pd.to_numeric(
            valori,
            errors="coerce"
        )

    print("\nQuestionario convertito correttamente!")

    return df


# ============================================================
# 3. FUNZIONE CONTROLLO E PULIZIA CSV
# ============================================================

def controllo_csv(df, tipo="storico", riferimento=None):

    # Creiamo una copia del DataFrame.
    df = df.copy()

    # Eliminiamo gli spazi dai nomi delle colonne.
    df.columns = df.columns.str.strip()

    # --------------------------------------------------------
    # 3.1 CONTROLLO DIMENSIONI
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print(f"       HEARTVISION - CONTROLLO {tipo.upper()}")
    print("=" * 55)

    print("Righe:", df.shape[0])
    print("Colonne:", df.shape[1])

    # Controlliamo che il tipo sia corretto.
    if tipo not in ["storico", "questionario"]:
        raise ValueError("Tipo di dataset non valido!")

    # Controlliamo che il DataFrame non sia vuoto.
    if df.empty:
        raise ValueError("Il DataFrame è vuoto!")

    # Controlliamo eventuali nomi di colonne duplicati.
    if df.columns.duplicated().any():
        raise ValueError(
            "Sono presenti colonne duplicate!"
        )

    # --------------------------------------------------------
    # 3.2 CONTROLLO COLONNE OBBLIGATORIE
    # --------------------------------------------------------

    # Lo storico contiene anche la variabile target.
    if tipo == "storico":
        richieste = colonne + [target]

    # Il questionario contiene soltanto le 15 feature.
    else:
        richieste = colonne

        if target in df.columns:
            raise ValueError(
                "Il questionario non deve contenere il target!"
            )

    # Individuiamo eventuali colonne mancanti.
    mancanti = [
        c for c in richieste
        if c not in df.columns
    ]

    if mancanti:
        raise ValueError(
            f"Colonne mancanti: {mancanti}"
        )

    # --------------------------------------------------------
    # 3.3 CONVERSIONE NUMERICA
    # --------------------------------------------------------

    # Convertiamo tutte le colonne richieste in numeri.
    # Usiamo temporaneamente float per consentire
    # l'inserimento delle medie durante la pulizia.
    for colonna in richieste:

        df[colonna] = pd.to_numeric(
            df[colonna],
            errors="coerce"
        ).astype(float)

    # Sostituiamo gli infiniti con NaN.
    df[richieste] = df[richieste].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # 3.4 DATASET DI RIFERIMENTO
    # --------------------------------------------------------

    # Per lo storico utilizziamo i suoi valori validi.
    # Per i questionari possiamo utilizzare il dataset
    # Framingham già controllato, passato come riferimento.
    if riferimento is not None:
        base = riferimento
    else:
        base = df

    # Dizionario per registrare tutte le correzioni.
    correzioni = {}

    # --------------------------------------------------------
    # 3.5 CONTROLLO VARIABILI BINARIE
    # --------------------------------------------------------

    # Queste colonne possono contenere soltanto 0 oppure 1.
    binarie = [
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

    for colonna in binarie:

        # Individuiamo valori diversi da 0 e 1.
        errati = ~df[colonna].isin([0, 1])

        if errati.any():

            # Recuperiamo i valori validi dal riferimento.
            validi = pd.to_numeric(
                base[colonna],
                errors="coerce"
            )

            validi = validi[validi.isin([0, 1])]

            if validi.empty:
                raise ValueError(
                    f"Nessun valore valido in {colonna}"
                )

            # Per le variabili binarie utilizziamo la moda.
            moda = validi.mode().iloc[0]

            df.loc[errati, colonna] = moda

            # Registriamo il numero di correzioni.
            correzioni[colonna] = int(errati.sum())

    # --------------------------------------------------------
    # 3.6 CONTROLLO DEL TARGET
    # --------------------------------------------------------

    # Il target non deve essere corretto con una media:
    # rappresenta l'esito realmente registrato.
    if tipo == "storico":

        if not df[target].isin([0, 1]).all():
            raise ValueError(
                "Il target contiene valori mancanti o non validi!"
            )

    # --------------------------------------------------------
    # 3.7 CONTROLLO INTERVALLI NUMERICI
    # --------------------------------------------------------

    # Definiamo gli intervalli accettati dal programma.
    intervalli = {
        "eta": (5, 100),
        "sigarette_giorno": (0, 60),
        "peso_kg": (20, 200),
        "altezza_cm": (80, 200),
        "frequenza_cardiaca": (25, 130),
        "sere_alcol_settimana": (0, 7)
    }

    # Colonne che devono contenere numeri interi.
    intere = [
        "eta",
        "sigarette_giorno",
        "sere_alcol_settimana"
    ]

    # Controlliamo ogni colonna numerica.
    for colonna, (minimo, massimo) in intervalli.items():

        valori = df[colonna]

        # Individuiamo valori nulli o fuori intervallo.
        errati = (
            valori.isna() |
            ~valori.between(minimo, massimo)
        )

        # Se la colonna richiede interi,
        # consideriamo errati anche i numeri decimali.
        if colonna in intere:
            errati = errati | (valori % 1 != 0)

        # Se ci sono errori, procediamo alla correzione.
        if errati.any():

            # Prendiamo i valori dal dataset di riferimento.
            validi = pd.to_numeric(
                base[colonna],
                errors="coerce"
            )

            # Eliminiamo infiniti e valori non validi.
            validi = validi.replace(
                [np.inf, -np.inf],
                np.nan
            )

            # Conserviamo soltanto i valori nell'intervallo.
            validi = validi[
                validi.between(minimo, massimo)
            ]

            # Per le colonne intere escludiamo i decimali.
            if colonna in intere:
                validi = validi[validi % 1 == 0]

            # Verifichiamo che sia possibile calcolare la media.
            if validi.empty:
                raise ValueError(
                    f"Impossibile calcolare la media di {colonna}"
                )

            # Calcoliamo la media dei valori validi.
            media = validi.mean()

            # Arrotondiamo la media per le colonne intere.
            if colonna in intere:
                media = int(round(media))

            # Sostituiamo i valori errati con la media.
            df.loc[errati, colonna] = media

            # Registriamo le correzioni effettuate.
            correzioni[colonna] = int(errati.sum())

    # --------------------------------------------------------
    # 3.8 CONTROLLO COERENZA DEL FUMO
    # --------------------------------------------------------

    # Un non fumatore non può avere sigarette giornaliere
    # maggiori di zero.
    errore_fumo = (
        (df["fumatore"] == 0) &
        (df["sigarette_giorno"] > 0)
    )

    if errore_fumo.any():

        # Per i non fumatori il numero coerente
        # di sigarette giornaliere è zero.
        df.loc[errore_fumo, "sigarette_giorno"] = 0

        correzioni["coerenza_fumo"] = int(
            errore_fumo.sum()
        )

    # --------------------------------------------------------
    # 3.9 CONTROLLO COERENZA DELL'ALCOL
    # --------------------------------------------------------

    # Chi dichiara di non bere deve avere zero
    # sere settimanali di consumo.
    errore_alcol = (
        (df["alcol"] == 0) &
        (df["sere_alcol_settimana"] > 0)
    )

    if errore_alcol.any():

        df.loc[
            errore_alcol,
            "sere_alcol_settimana"
        ] = 0

        correzioni["coerenza_alcol"] = int(
            errore_alcol.sum()
        )

    # --------------------------------------------------------
    # 3.10 CONTROLLO FINALE DEI VALORI NULLI
    # --------------------------------------------------------

    # Verifichiamo che non rimangano valori mancanti
    # nelle colonne necessarie.
    if df[richieste].isnull().any().any():
        raise ValueError(
            "Sono rimasti valori mancanti nel dataset!"
        )

    # --------------------------------------------------------
    # 3.11 CONTROLLO DELLE RIGHE DUPLICATE
    # --------------------------------------------------------

    # Contiamo i duplicati senza eliminarli.
    # Due persone possono avere le stesse caratteristiche.
    duplicati = df[richieste].duplicated().sum()

    print("\nRighe con valori identici:", duplicati)

    # --------------------------------------------------------
    # 3.12 CALCOLO AUTOMATICO DEL BMI
    # --------------------------------------------------------

    # Calcoliamo il BMI dopo aver corretto altezza e peso.
    df["bmi"] = (
        df["peso_kg"] /
        (df["altezza_cm"] / 100) ** 2
    ).round(2)

    # --------------------------------------------------------
    # 3.13 RIPRISTINO DEI NUMERI INTERI
    # --------------------------------------------------------

    # Dopo le correzioni riportiamo le variabili intere
    # al loro tipo numerico corretto.
    colonne_intere = binarie + intere

    if tipo == "storico":
        colonne_intere.append(target)

    for colonna in colonne_intere:
        df[colonna] = df[colonna].astype(int)

    # --------------------------------------------------------
    # 3.14 RIEPILOGO DELLA PULIZIA
    # --------------------------------------------------------

    print("\n" + "-" * 55)
    print("                RISULTATO PULIZIA")
    print("-" * 55)

    if correzioni:

        print("\nCorrezioni effettuate:")

        for nome, quantita in correzioni.items():
            print(f"  {nome}: {quantita}")

    else:
        print("\nNessuna correzione necessaria.")

    print("\nValori nulli rimasti: 0")
    print("BMI calcolato correttamente.")
    print("\nDataset pronto per l'analisi!")

    print("=" * 55)

    return df


# ============================================================
# 4. CARICAMENTO DEL DATASET STORICO
# ============================================================

# Importiamo il dataset Framingham.
df_X = pd.read_csv(
    "Dataset_storico/HEARTVISION_dataset_completo.csv"
)

# Eseguiamo il controllo e la pulizia.
df_X = controllo_csv(
    df_X,
    tipo="storico"
)


# ============================================================
# 5. CARICAMENTO DEL DATASET QUESTIONARI
# ============================================================

# Importiamo le risposte esportate da Google Forms.
df_y = pd.read_csv(
    "Dataset_questionari/Framingham_form.csv"
)

# Convertiamo le risposte testuali in valori numerici.
df_y = converti_questionario(df_y)

# Controlliamo il nuovo dataset.
# Le medie vengono prese dal Framingham già pulito.
df_y = controllo_csv(
    df_y,
    tipo="questionario",
    riferimento=df_X
)


# ============================================================
# 6. VISUALIZZAZIONE FINALE
# ============================================================

# Mostriamo le prime righe del dataset storico.
print("\n" + "=" * 55)
print("              DATASET FRAMINGHAM")
print("=" * 55)
print(df_X.head())

# Mostriamo le prime righe dei questionari.
print("\n" + "=" * 55)
print("              DATASET QUESTIONARI")
print("=" * 55)
print(df_y.head())

print("\nHEARTVISION - Operazioni completate!")



# METRICHE DI VALUTAZIONE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# 7. MACHINE LEARNING - HEART VISION
# ============================================================


# ============================================================
# 7.1 SEPARAZIONE DEI DATI
# ============================================================

# X_train contiene le caratteristiche dei pazienti storici.
# Utilizziamo le 15 variabili originali e il BMI calcolato.

X_train = df_X[colonne + ["bmi"]]

# y_train contiene i risultati storici conosciuti:
# 0 = nessun evento coronarico registrato
# 1 = evento coronarico registrato entro 10 anni

y_train = df_X[target]

# X_test contiene le caratteristiche dei nuovi utenti.
# Non utilizziamo questi dati per addestrare o confrontare
# gli algoritmi.

X_test = df_y[colonne + ["bmi"]]


# ============================================================
# 7.2 PREPARAZIONE DELLA CROSS VALIDATION
# ============================================================

# Dividiamo temporaneamente i dati storici in 5 gruppi.
#
# A ogni iterazione:
# - 4 gruppi vengono utilizzati per addestrare il modello.
# - 1 gruppo viene utilizzato per valutarlo.
#
# La procedura viene ripetuta 5 volte, cambiando il gruppo
# utilizzato per la valutazione.
#
# In questo modo tutti i pazienti storici vengono valutati.
# I nuovi questionari (X_test) restano esclusi.

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Qui raccogliamo i risultati di tutte le configurazioni.

risultati_cv = []


# ============================================================
# 7.3 K NEAREST NEIGHBORS (KNN)
# ============================================================

# Proviamo diversi valori di K.
# K indica il numero di pazienti storici più vicini
# utilizzati per classificare un nuovo paziente.

valori_k = [3, 5, 7, 9, 11, 15]

for k in valori_k:

    # Ripetiamo la valutazione sui 5 gruppi.

    for indice_train, indice_val in cv.split(X_train, y_train):

        # Selezioniamo i dati di addestramento e valutazione.

        X_fold_train = X_train.iloc[indice_train]
        X_fold_val = X_train.iloc[indice_val]

        y_fold_train = y_train.iloc[indice_train]
        y_fold_val = y_train.iloc[indice_val]


        # NORMALIZZAZIONE

        # KNN utilizza le distanze tra i pazienti.
        # Per questo motivo dobbiamo normalizzare le variabili.

        scaler_fold = StandardScaler()

        # Impariamo media e deviazione standard
        # esclusivamente dai dati di addestramento.

        X_fold_train = scaler_fold.fit_transform(X_fold_train)

        # Applichiamo la stessa trasformazione ai dati
        # utilizzati per la valutazione.

        X_fold_val = scaler_fold.transform(X_fold_val)


        # CREAZIONE DEL MODELLO

        modello = KNeighborsClassifier(
            n_neighbors=k
        )


        # ADDESTRAMENTO

        modello.fit(X_fold_train, y_fold_train)


        # PREVISIONI

        previsioni = modello.predict(X_fold_val)

        probabilita = modello.predict_proba(X_fold_val)

        indice_evento = list(modello.classes_).index(1)

        probabilita_evento = probabilita[:, indice_evento]


        # CALCOLO DELLE METRICHE

        risultati_cv.append({

            "algoritmo": "KNN",

            "configurazione": f"K={k}",

            "k": k,

            "criterio": None,

            "max_depth": None,

            "n_estimators": None,

            "accuracy": accuracy_score(
                y_fold_val, previsioni
            ),

            "precision": precision_score(
                y_fold_val, previsioni, zero_division=0
            ),

            "recall": recall_score(
                y_fold_val, previsioni, zero_division=0
            ),

            "f1_score": f1_score(
                y_fold_val, previsioni, zero_division=0
            ),

            "roc_auc": roc_auc_score(
                y_fold_val, probabilita_evento
            )

        })


# ============================================================
# 7.4 DECISION TREE CLASSIFIER
# ============================================================

# Proviamo due criteri di divisione:
# Gini ed Entropy.
#
# Proviamo inoltre diverse profondità dell'albero.
# None significa che non imponiamo un limite alla profondità.

criteri = ["gini", "entropy"]

profondita_alberi = [3, 5, 10, None]


for criterio in criteri:

    for profondita in profondita_alberi:

        for indice_train, indice_val in cv.split(X_train, y_train):

            # SEPARAZIONE DEI GRUPPI

            X_fold_train = X_train.iloc[indice_train]
            X_fold_val = X_train.iloc[indice_val]

            y_fold_train = y_train.iloc[indice_train]
            y_fold_val = y_train.iloc[indice_val]


            # CREAZIONE DEL MODELLO

            modello = DecisionTreeClassifier(
                criterion=criterio,
                max_depth=profondita,
                random_state=42
            )


            # ADDESTRAMENTO

            modello.fit(X_fold_train, y_fold_train)


            # PREVISIONI

            previsioni = modello.predict(X_fold_val)

            probabilita = modello.predict_proba(X_fold_val)

            indice_evento = list(modello.classes_).index(1)

            probabilita_evento = probabilita[:, indice_evento]


            # CALCOLO DELLE METRICHE

            risultati_cv.append({

                "algoritmo": "Decision Tree",

                "configurazione": (
                    f"{criterio} - profondità {profondita}"
                ),

                "k": None,

                "criterio": criterio,

                "max_depth": profondita,

                "n_estimators": None,

                "accuracy": accuracy_score(
                    y_fold_val, previsioni
                ),

                "precision": precision_score(
                    y_fold_val, previsioni, zero_division=0
                ),

                "recall": recall_score(
                    y_fold_val, previsioni, zero_division=0
                ),

                "f1_score": f1_score(
                    y_fold_val, previsioni, zero_division=0
                ),

                "roc_auc": roc_auc_score(
                    y_fold_val, probabilita_evento
                )

            })


# ============================================================
# 7.5 RANDOM FOREST CLASSIFIER
# ============================================================

# Proviamo diverse quantità di alberi:
# 100, 200 e 300.
#
# Proviamo anche diverse profondità.
# Ogni configurazione viene valutata sui 5 gruppi.

numero_alberi = [100, 200, 300]

profondita_foresta = [5, 10, None]


for numero in numero_alberi:

    for profondita in profondita_foresta:

        for indice_train, indice_val in cv.split(X_train, y_train):

            # SEPARAZIONE DEI GRUPPI

            X_fold_train = X_train.iloc[indice_train]
            X_fold_val = X_train.iloc[indice_val]

            y_fold_train = y_train.iloc[indice_train]
            y_fold_val = y_train.iloc[indice_val]


            # CREAZIONE DEL MODELLO

            modello = RandomForestClassifier(
                n_estimators=numero,
                max_depth=profondita,
                random_state=42,
                n_jobs=-1
            )


            # ADDESTRAMENTO

            modello.fit(X_fold_train, y_fold_train)


            # PREVISIONI

            previsioni = modello.predict(X_fold_val)

            probabilita = modello.predict_proba(X_fold_val)

            indice_evento = list(modello.classes_).index(1)

            probabilita_evento = probabilita[:, indice_evento]


            # CALCOLO DELLE METRICHE

            risultati_cv.append({

                "algoritmo": "Random Forest",

                "configurazione": (
                    f"{numero} alberi - profondità {profondita}"
                ),

                "k": None,

                "criterio": None,

                "max_depth": profondita,

                "n_estimators": numero,

                "accuracy": accuracy_score(
                    y_fold_val, previsioni
                ),

                "precision": precision_score(
                    y_fold_val, previsioni, zero_division=0
                ),

                "recall": recall_score(
                    y_fold_val, previsioni, zero_division=0
                ),

                "f1_score": f1_score(
                    y_fold_val, previsioni, zero_division=0
                ),

                "roc_auc": roc_auc_score(
                    y_fold_val, probabilita_evento
                )

            })


# ============================================================
# 7.6 CONFRONTO DEI RISULTATI
# ============================================================

# Trasformiamo tutti i risultati in un DataFrame.

df_risultati = pd.DataFrame(risultati_cv)

# Elenco delle metriche da confrontare.

metriche = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc"
]

# Calcoliamo la media delle 5 valutazioni
# per ogni configurazione.

confronto = (
    df_risultati
    .groupby(["algoritmo", "configurazione"])[metriche]
    .mean()
    .reset_index()
)

# Recuperiamo i parametri utilizzati da ogni configurazione.

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
    .drop_duplicates()
)

confronto = confronto.merge(
    parametri,
    on=["algoritmo", "configurazione"]
)

# Arrotondiamo i risultati.

confronto[metriche] = confronto[metriche].round(3)

# Ordiniamo le configurazioni in base a F1 Score.

confronto = confronto.sort_values(
    by="f1_score",
    ascending=False
)

print("\nCONFRONTO COMPLETO DEGLI ALGORITMI:")
print(confronto.to_string(index=False))


# ============================================================
# 7.7 MIGLIOR CONFIGURAZIONE PER OGNI ALGORITMO
# ============================================================

# Per ogni algoritmo individuiamo la configurazione
# con il miglior F1 Score medio.

indici_migliori = (
    confronto
    .groupby("algoritmo")["f1_score"]
    .idxmax()
)

migliori_modelli = confronto.loc[
    indici_migliori
].reset_index(drop=True)

print("\nMIGLIORE CONFIGURAZIONE PER OGNI ALGORITMO:")
print(
    migliori_modelli[
        ["algoritmo", "configurazione"] + metriche
    ].to_string(index=False)
)

"""
# ____________________________________________________________________________________
# ============================================================
# 7.8 GRAFICO KNN - CONFRONTO DEI VALORI DI K
# ============================================================

knn_grafico = confronto[
    confronto["algoritmo"] == "KNN"
].sort_values("k")

plt.figure(figsize=(10, 6))

plt.plot(
    knn_grafico["k"],
    knn_grafico["f1_score"],
    marker="o",
    label="F1 Score"
)

plt.plot(
    knn_grafico["k"],
    knn_grafico["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    knn_grafico["k"],
    knn_grafico["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    knn_grafico["k"],
    knn_grafico["roc_auc"],
    marker="o",
    label="ROC-AUC"
)

plt.title("KNN - Prestazioni al variare di K")
plt.xlabel("Numero di vicini (K)")
plt.ylabel("Punteggio medio")
plt.xticks(knn_grafico["k"])
plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

"""

"""
# ============================================================
# 7.9 GRAFICO DECISION TREE - PROFONDITA'
# ============================================================

decision_grafico = confronto[
    confronto["algoritmo"] == "Decision Tree"
].copy()

# Prepariamo le etichette delle profondità.
# None viene indicato come "Illimitata".

decision_grafico["profondita"] = (
    decision_grafico["max_depth"]
    .apply(
        lambda valore:
        "Illimitata" if pd.isna(valore) else str(int(valore))
    )
)

# Creiamo una tabella con F1 Score per criterio
# e profondità.

tabella_decision = decision_grafico.pivot(
    index="profondita",
    columns="criterio",
    values="f1_score"
)

tabella_decision = tabella_decision.reindex(
    ["3", "5", "10", "Illimitata"]
)

plt.figure(figsize=(10, 6))

for criterio in tabella_decision.columns:

    plt.plot(
        tabella_decision.index,
        tabella_decision[criterio],
        marker="o",
        label=criterio
    )

plt.title("Decision Tree - Prestazioni per profondità")
plt.xlabel("Profondità massima")
plt.ylabel("F1 Score medio")
plt.ylim(0, 1)
plt.grid()
plt.legend(title="Criterio")
plt.tight_layout()
plt.show()
"""


"""
# ============================================================
# 7.10 GRAFICO RANDOM FOREST - NUMERO DI ALBERI
# ============================================================

random_grafico = confronto[
    confronto["algoritmo"] == "Random Forest"
].copy()

# Prepariamo le etichette delle profondità.

random_grafico["profondita"] = (
    random_grafico["max_depth"]
    .apply(
        lambda valore:
        "Illimitata" if pd.isna(valore) else str(int(valore))
    )
)

# Creiamo una tabella con F1 Score per numero di alberi
# e profondità della foresta.

tabella_random = random_grafico.pivot(
    index="n_estimators",
    columns="profondita",
    values="f1_score"
)

plt.figure(figsize=(10, 6))

for profondita in tabella_random.columns:

    plt.plot(
        tabella_random.index,
        tabella_random[profondita],
        marker="o",
        label=f"Profondità {profondita}"
    )

plt.title("Random Forest - Prestazioni per numero di alberi")
plt.xlabel("Numero di alberi")
plt.ylabel("F1 Score medio")
plt.ylim(0, 1)
plt.xticks(numero_alberi)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

"""
# ============================================================
# 7.11 GRAFICO CONFRONTO FINALE
# ============================================================

plt.figure(figsize=(10, 6))

posizioni = np.arange(len(migliori_modelli))

larghezza = 0.35

plt.bar(
    posizioni - larghezza / 2,
    migliori_modelli["f1_score"],
    larghezza,
    label="F1 Score"
)

plt.bar(
    posizioni + larghezza / 2,
    migliori_modelli["recall"],
    larghezza,
    label="Recall"
)

plt.title("Confronto finale dei migliori algoritmi")
plt.xlabel("Algoritmo")
plt.ylabel("Punteggio medio")
plt.xticks(
    posizioni,
    migliori_modelli["algoritmo"]
)

plt.ylim(0, 1)
plt.grid(axis="y")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7.12 SCELTA DEL MODELLO FINALE
# ============================================================

# Dopo aver osservato la tabella e i grafici,
# scegliamo quale algoritmo utilizzare.
#
# Possibili valori:
# "KNN"
# "Decision Tree"
# "Random Forest"

modello_scelto = "KNN"


# Recuperiamo la migliore configurazione F1
# dell'algoritmo selezionato.

riga_scelta = migliori_modelli[
    migliori_modelli["algoritmo"] == modello_scelto
].iloc[0]

print("\nMODELLO SELEZIONATO:", modello_scelto)

print("CONFIGURAZIONE:", riga_scelta["configurazione"])


# ============================================================
# 7.13 ADDESTRAMENTO DEFINITIVO
# ============================================================

# Creiamo il modello utilizzando i parametri migliori
# individuati durante il confronto.

if modello_scelto == "KNN":

    # Normalizziamo tutti i dati storici.

    scaler_finale = StandardScaler()

    X_train_finale = scaler_finale.fit_transform(X_train)

    X_test_finale = scaler_finale.transform(X_test)

    modello_finale = KNeighborsClassifier(
        n_neighbors=int(riga_scelta["k"])
    )


elif modello_scelto == "Decision Tree":

    X_train_finale = X_train

    X_test_finale = X_test

    profondita = riga_scelta["max_depth"]

    if pd.isna(profondita):
        profondita = None
    else:
        profondita = int(profondita)

    modello_finale = DecisionTreeClassifier(
        criterion=riga_scelta["criterio"],
        max_depth=profondita,
        random_state=42
    )


elif modello_scelto == "Random Forest":

    X_train_finale = X_train

    X_test_finale = X_test

    profondita = riga_scelta["max_depth"]

    if pd.isna(profondita):
        profondita = None
    else:
        profondita = int(profondita)

    modello_finale = RandomForestClassifier(
        n_estimators=int(riga_scelta["n_estimators"]),
        max_depth=profondita,
        random_state=42,
        n_jobs=-1
    )


# Addestriamo il modello scelto su tutti i pazienti storici.

modello_finale.fit(
    X_train_finale,
    y_train
)


# ============================================================
# 7.14 PREVISIONI SUI NUOVI UTENTI
# ============================================================

# Applichiamo il modello ai nuovi questionari.

previsioni = modello_finale.predict(X_test_finale)

probabilita = modello_finale.predict_proba(X_test_finale)

# Individuiamo la colonna relativa alla classe 1.

indice_evento = list(modello_finale.classes_).index(1)

probabilita_evento = (
    probabilita[:, indice_evento] * 100
)


# ============================================================
# 7.15 RISULTATI HEART VISION
# ============================================================

risultati = pd.DataFrame({

    "utente": range(1, len(X_test) + 1),

    "previsione": previsioni,

    "probabilita_evento_percentuale": (
        probabilita_evento.round(2)
    )

})

print("\nMODELLO UTILIZZATO:", modello_scelto)

print("\nRISULTATI HEART VISION:")

print(risultati)

