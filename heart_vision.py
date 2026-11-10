
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
