
# HEARTVISION - MACHINE LEARNING, GRAFICI E PREVISIONI

# IMPORTAZIONE DELLE LIBRERIE

from sklearn.neighbors import KNeighborsClassifier  # Importa il classificatore KNN
from sklearn.preprocessing import StandardScaler  # Standardizza le caratteristiche numeriche
from sklearn.tree import DecisionTreeClassifier  # Importa l'albero decisionale
from sklearn.ensemble import RandomForestClassifier  # Importa la foresta casuale
from sklearn.model_selection import StratifiedKFold  # Divide i dati in fold stratificati
from sklearn.metrics import (  # Importa le metriche di valutazione
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt  # Crea i grafici
import pandas as pd  # Gestisce i DataFrame
import numpy as np  # Gestisce i valori e le posizioni numeriche

# COLLEGAMENTO AL PROGRAMMA HEARTVISION

import heart_vision as hv  # Importa il modulo principale di HEARTVISION

hv.carica_dataset()  # Carica, controlla e pulisce entrambi i dataset

df_X = hv.df_X  # Recupera il dataset storico Framingham
df_y = hv.df_y  # Recupera il dataset dei questionari

colonne = hv.colonne  # Recupera le 15 caratteristiche
target = hv.target  # Recupera la variabile da prevedere


# SEPARAZIONE DEI DATI

X_train = df_X[colonne + ["bmi"]]  # Seleziona le caratteristiche del dataset storico
y_train = df_X[target]  # Seleziona gli eventi coronarici registrati
X_test = df_y[colonne + ["bmi"]]  # Seleziona le caratteristiche dei questionari


# SEPARAZIONE DEI DATI

X_train = df_X[colonne + ["bmi"]]  # Seleziona le caratteristiche del dataset storico
y_train = df_X[target]  # Seleziona gli esiti storici conosciuti: 0 oppure 1
X_test = df_y[colonne + ["bmi"]]  # Seleziona i questionari, esclusi dalla validazione


# PREPARAZIONE DELLA CROSS-VALIDATION

cv = StratifiedKFold(
    n_splits=5,  # Divide lo storico in cinque gruppi
    shuffle=True,  # Mescola le osservazioni
    random_state=42  # Rende riproducibile la suddivisione
)

risultati_cv = []  # Raccoglie le metriche di ogni configurazione e fold


# CONFRONTO K-NEAREST NEIGHBORS

valori_k = [3, 5, 7, 9, 11, 15]  # Definisce i numeri di vicini da provare

for k in valori_k:  # Valuta ogni valore di K
    for indice_train, indice_val in cv.split(X_train, y_train):  # Scorre i cinque fold
        X_fold_train = X_train.iloc[indice_train]  # Seleziona le caratteristiche di training
        X_fold_val = X_train.iloc[indice_val]  # Seleziona le caratteristiche di validazione
        y_fold_train = y_train.iloc[indice_train]  # Seleziona gli esiti del training
        y_fold_val = y_train.iloc[indice_val]  # Seleziona gli esiti della validazione

        scaler_fold = StandardScaler()  # Crea uno scaler indipendente per il fold
        X_fold_train = scaler_fold.fit_transform(X_fold_train)  # Impara le scale dal solo training
        X_fold_val = scaler_fold.transform(X_fold_val)  # Applica le stesse scale alla validazione

        modello = KNeighborsClassifier(n_neighbors=k)  # Crea il KNN con K vicini
        modello.fit(X_fold_train, y_fold_train)  # Addestra il KNN

        previsioni = modello.predict(X_fold_val)  # Prevede le classi di validazione
        probabilita = modello.predict_proba(X_fold_val)  # Stima le probabilità per classe
        indice_evento = list(modello.classes_).index(1)  # Trova la posizione della classe positiva
        probabilita_evento = probabilita[:, indice_evento]  # Estrae la probabilità della classe 1

        risultati_cv.append({  # Salva parametri e metriche del fold
            "algoritmo": "KNN",
            "configurazione": f"K={k}",
            "k": k,
            "criterio": None,
            "max_depth": None,
            "n_estimators": None,
            "accuracy": accuracy_score(y_fold_val, previsioni),  # Percentuale di previsioni corrette
            "precision": precision_score(y_fold_val, previsioni, zero_division=0),  # Precisione della classe 1
            "recall": recall_score(y_fold_val, previsioni, zero_division=0),  # Positivi effettivamente individuati
            "f1_score": f1_score(y_fold_val, previsioni, zero_division=0),  # Equilibrio tra precision e recall
            "roc_auc": roc_auc_score(y_fold_val, probabilita_evento)  # Capacità di distinguere le classi
        })


# CONFRONTO DECISION TREE

criteri = ["gini", "entropy"]  # Definisce i criteri di separazione
profondita_alberi = [3, 5, 10, None]  # None indica profondità non limitata

for criterio in criteri:  # Prova ciascun criterio
    for profondita in profondita_alberi:  # Prova ciascuna profondità
        for indice_train, indice_val in cv.split(X_train, y_train):  # Scorre i cinque fold
            X_fold_train = X_train.iloc[indice_train]  # Seleziona le caratteristiche di training
            X_fold_val = X_train.iloc[indice_val]  # Seleziona le caratteristiche di validazione
            y_fold_train = y_train.iloc[indice_train]  # Seleziona gli esiti del training
            y_fold_val = y_train.iloc[indice_val]  # Seleziona gli esiti della validazione

            modello = DecisionTreeClassifier(
                criterion=criterio,  # Utilizza Gini oppure Entropy
                max_depth=profondita,  # Imposta la profondità massima
                random_state=42  # Rende riproducibile il risultato
            )
            modello.fit(X_fold_train, y_fold_train)  # Addestra l'albero sui dati originali

            previsioni = modello.predict(X_fold_val)  # Prevede le classi di validazione
            probabilita = modello.predict_proba(X_fold_val)  # Stima le probabilità per classe
            indice_evento = list(modello.classes_).index(1)  # Trova la posizione della classe positiva
            probabilita_evento = probabilita[:, indice_evento]  # Estrae la probabilità della classe 1

            risultati_cv.append({  # Salva parametri e metriche del fold
                "algoritmo": "Decision Tree",
                "configurazione": f"{criterio} - profondità {profondita}",
                "k": None,
                "criterio": criterio,
                "max_depth": profondita,
                "n_estimators": None,
                "accuracy": accuracy_score(y_fold_val, previsioni),  # Percentuale di previsioni corrette
                "precision": precision_score(y_fold_val, previsioni, zero_division=0),  # Precisione della classe 1
                "recall": recall_score(y_fold_val, previsioni, zero_division=0),  # Positivi effettivamente individuati
                "f1_score": f1_score(y_fold_val, previsioni, zero_division=0),  # Equilibrio tra precision e recall
                "roc_auc": roc_auc_score(y_fold_val, probabilita_evento)  # Capacità di distinguere le classi
            })


# CONFRONTO RANDOM FOREST

numero_alberi = [100, 200, 300]  # Definisce il numero di alberi da provare
profondita_foresta = [5, 10, None]  # Definisce le profondità da provare

for numero in numero_alberi:  # Prova ogni quantità di alberi
    for profondita in profondita_foresta:  # Prova ogni profondità
        for indice_train, indice_val in cv.split(X_train, y_train):  # Scorre i cinque fold
            X_fold_train = X_train.iloc[indice_train]  # Seleziona le caratteristiche di training
            X_fold_val = X_train.iloc[indice_val]  # Seleziona le caratteristiche di validazione
            y_fold_train = y_train.iloc[indice_train]  # Seleziona gli esiti del training
            y_fold_val = y_train.iloc[indice_val]  # Seleziona gli esiti della validazione

            modello = RandomForestClassifier(
                n_estimators=numero,  # Imposta il numero di alberi
                max_depth=profondita,  # Imposta la profondità massima
                random_state=42,  # Rende riproducibile il risultato
                n_jobs=-1  # Usa tutti i processori disponibili
            )
            modello.fit(X_fold_train, y_fold_train)  # Addestra la foresta sui dati originali

            previsioni = modello.predict(X_fold_val)  # Prevede le classi di validazione
            probabilita = modello.predict_proba(X_fold_val)  # Stima le probabilità per classe
            indice_evento = list(modello.classes_).index(1)  # Trova la posizione della classe positiva
            probabilita_evento = probabilita[:, indice_evento]  # Estrae la probabilità della classe 1

            risultati_cv.append({  # Salva parametri e metriche del fold
                "algoritmo": "Random Forest",
                "configurazione": f"{numero} alberi - profondità {profondita}",
                "k": None,
                "criterio": None,
                "max_depth": profondita,
                "n_estimators": numero,
                "accuracy": accuracy_score(y_fold_val, previsioni),  # Percentuale di previsioni corrette
                "precision": precision_score(y_fold_val, previsioni, zero_division=0),  # Precisione della classe 1
                "recall": recall_score(y_fold_val, previsioni, zero_division=0),  # Positivi effettivamente individuati
                "f1_score": f1_score(y_fold_val, previsioni, zero_division=0),  # Equilibrio tra precision e recall
                "roc_auc": roc_auc_score(y_fold_val, probabilita_evento)  # Capacità di distinguere le classi
            })


# CONFRONTO DELLE METRICHE

df_risultati = pd.DataFrame(risultati_cv)  # Riunisce tutte le prove in una tabella
metriche = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]  # Definisce le metriche da confrontare

confronto = (
    df_risultati
    .groupby(["algoritmo", "configurazione"])[metriche]  # Raggruppa le prove della stessa configurazione
    .mean()  # Calcola la media sui cinque fold
    .reset_index()  # Ripristina un indice numerico
)

parametri = (
    df_risultati[["algoritmo", "configurazione", "k", "criterio", "max_depth", "n_estimators"]]
    .drop_duplicates()  # Mantiene una riga per ogni configurazione
)

confronto = confronto.merge(parametri, on=["algoritmo", "configurazione"])  # Associa le metriche ai parametri
confronto[metriche] = confronto[metriche].round(3)  # Arrotonda i punteggi a tre decimali
confronto = confronto.sort_values(by="f1_score", ascending=False)  # Ordina secondo il punteggio F1

print("\nCONFRONTO COMPLETO DEGLI ALGORITMI:")  # Introduce la tabella completa
print(confronto.to_string(index=False))  # Mostra le configurazioni senza indice


# MIGLIORE CONFIGURAZIONE PER ALGORITMO

indici_migliori = (
    confronto
    .groupby("algoritmo")["f1_score"]  # Raggruppa per tipo di algoritmo
    .idxmax()  # Trova la configurazione con F1 più alto
)

migliori_modelli = confronto.loc[indici_migliori].reset_index(drop=True)  # Riunisce le configurazioni migliori

print("\nMIGLIORE CONFIGURAZIONE PER OGNI ALGORITMO:")  # Introduce la sintesi
print(migliori_modelli[["algoritmo", "configurazione"] + metriche].to_string(index=False))  # Mostra le metriche migliori


# GRAFICO KNN - CONFRONTO DEI VALORI DI K

knn_grafico = confronto[confronto["algoritmo"] == "KNN"].sort_values("k")  # Ordina i risultati per K
plt.figure(figsize=(10, 6))  # Crea una nuova figura
plt.plot(knn_grafico["k"], knn_grafico["f1_score"], marker="o", label="F1 Score")  # Disegna l'F1
plt.plot(knn_grafico["k"], knn_grafico["recall"], marker="o", label="Recall")  # Disegna il recall
plt.plot(knn_grafico["k"], knn_grafico["precision"], marker="o", label="Precision")  # Disegna la precision
plt.plot(knn_grafico["k"], knn_grafico["roc_auc"], marker="o", label="ROC-AUC")  # Disegna la ROC-AUC
plt.title("KNN - Prestazioni al variare di K")  # Imposta il titolo
plt.xlabel("Numero di vicini (K)")  # Imposta l'asse orizzontale
plt.ylabel("Punteggio medio")  # Imposta l'asse verticale
plt.xticks(knn_grafico["k"])  # Mostra ogni K sull'asse orizzontale
plt.ylim(0, 1)  # Imposta l'intervallo dei punteggi
plt.grid()  # Mostra la griglia
plt.legend()  # Mostra la legenda
plt.tight_layout()  # Evita sovrapposizioni
plt.show()  # Visualizza il grafico


# FUNZIONE PER FORMATTARE LA PROFONDITÀ

def formatta_profondita(valore):  # Sostituisce le lambda dei grafici
    if pd.isna(valore):  # Verifica se la profondità è illimitata
        return "Illimitata"  # Converte None o NaN in un'etichetta leggibile
    return str(int(valore))  # Converte la profondità numerica in testo


# GRAFICO DECISION TREE - PROFONDITÀ

decision_grafico = confronto[confronto["algoritmo"] == "Decision Tree"].copy()  # Seleziona i risultati dell'albero
decision_grafico["profondita"] = decision_grafico["max_depth"].apply(formatta_profondita)  # Formatta le profondità

tabella_decision = decision_grafico.pivot(
    index="profondita",  # Mette le profondità sulle righe
    columns="criterio",  # Mette Gini ed Entropy sulle colonne
    values="f1_score"  # Utilizza il punteggio F1
)
tabella_decision = tabella_decision.reindex(["3", "5", "10", "Illimitata"])  # Ordina le profondità

plt.figure(figsize=(10, 6))  # Crea una nuova figura
for criterio in tabella_decision.columns:  # Disegna una curva per criterio
    plt.plot(tabella_decision.index, tabella_decision[criterio], marker="o", label=criterio)  # Mostra gli F1 medi
plt.title("Decision Tree - Prestazioni per profondità")  # Imposta il titolo
plt.xlabel("Profondità massima")  # Imposta l'asse orizzontale
plt.ylabel("F1 Score medio")  # Imposta l'asse verticale
plt.ylim(0, 1)  # Imposta l'intervallo dei punteggi
plt.grid()  # Mostra la griglia
plt.legend(title="Criterio")  # Distingue Gini ed Entropy
plt.tight_layout()  # Evita sovrapposizioni
plt.show()  # Visualizza il grafico


# GRAFICO RANDOM FOREST - NUMERO DI ALBERI

random_grafico = confronto[confronto["algoritmo"] == "Random Forest"].copy()  # Seleziona i risultati della foresta
random_grafico["profondita"] = random_grafico["max_depth"].apply(formatta_profondita)  # Formatta le profondità

tabella_random = random_grafico.pivot(
    index="n_estimators",  # Mette il numero di alberi sulle righe
    columns="profondita",  # Mette le profondità sulle colonne
    values="f1_score"  # Utilizza il punteggio F1
)

plt.figure(figsize=(10, 6))  # Crea una nuova figura
for profondita in tabella_random.columns:  # Disegna una curva per profondità
    plt.plot(tabella_random.index, tabella_random[profondita], marker="o", label=f"Profondità {profondita}")  # Mostra gli F1 medi
plt.title("Random Forest - Prestazioni per numero di alberi")  # Imposta il titolo
plt.xlabel("Numero di alberi")  # Imposta l'asse orizzontale
plt.ylabel("F1 Score medio")  # Imposta l'asse verticale
plt.ylim(0, 1)  # Imposta l'intervallo dei punteggi
plt.xticks(numero_alberi)  # Mostra tutte le quantità di alberi
plt.grid()  # Mostra la griglia
plt.legend()  # Distingue le profondità
plt.tight_layout()  # Evita sovrapposizioni
plt.show()  # Visualizza il grafico


# GRAFICO DEL CONFRONTO FINALE

plt.figure(figsize=(10, 6))  # Crea una nuova figura
posizioni = np.arange(len(migliori_modelli))  # Calcola le posizioni dei tre algoritmi
larghezza = 0.35  # Definisce la larghezza delle barre
plt.bar(posizioni - larghezza / 2, migliori_modelli["f1_score"], larghezza, label="F1 Score")  # Mostra l'F1 medio
plt.bar(posizioni + larghezza / 2, migliori_modelli["recall"], larghezza, label="Recall")  # Mostra il recall medio
plt.title("Confronto finale dei migliori algoritmi")  # Imposta il titolo
plt.xlabel("Algoritmo")  # Imposta l'asse orizzontale
plt.ylabel("Punteggio medio")  # Imposta l'asse verticale
plt.xticks(posizioni, migliori_modelli["algoritmo"])  # Mostra i nomi degli algoritmi
plt.ylim(0, 1)  # Imposta l'intervallo dei punteggi
plt.grid(axis="y")  # Mostra la griglia orizzontale
plt.legend()  # Mostra la legenda
plt.tight_layout()  # Evita sovrapposizioni
plt.show()  # Visualizza il grafico


# SELEZIONE DEL MODELLO FINALE

modello_scelto = "KNN"  # Sceglie l'algoritmo da utilizzare sui questionari
riga_scelta = migliori_modelli[migliori_modelli["algoritmo"] == modello_scelto].iloc[0]  # Recupera i parametri migliori
print("\nMODELLO SELEZIONATO:", modello_scelto)  # Mostra l'algoritmo scelto
print("CONFIGURAZIONE:", riga_scelta["configurazione"])  # Mostra la configurazione scelta


# ADDESTRAMENTO DEFINITIVO

if modello_scelto == "KNN":  # Prepara il KNN definitivo
    scaler_finale = StandardScaler()  # Crea lo scaler finale
    X_train_finale = scaler_finale.fit_transform(X_train)  # Standardizza tutto lo storico
    X_test_finale = scaler_finale.transform(X_test)  # Standardizza i questionari con lo stesso scaler
    modello_finale = KNeighborsClassifier(n_neighbors=int(riga_scelta["k"]))  # Usa il K migliore

elif modello_scelto == "Decision Tree":  # Prepara l'albero decisionale definitivo
    X_train_finale = X_train  # Mantiene le caratteristiche originali
    X_test_finale = X_test  # Mantiene i questionari originali
    profondita = riga_scelta["max_depth"]  # Recupera la profondità selezionata
    if pd.isna(profondita):  # Controlla se la profondità è illimitata
        profondita = None  # Non impone un limite
    else:
        profondita = int(profondita)  # Converte la profondità in intero
    modello_finale = DecisionTreeClassifier(
        criterion=riga_scelta["criterio"],  # Usa il criterio selezionato
        max_depth=profondita,  # Usa la profondità selezionata
        random_state=42  # Rende riproducibile il risultato
    )

elif modello_scelto == "Random Forest":  # Prepara la foresta definitiva
    X_train_finale = X_train  # Mantiene le caratteristiche originali
    X_test_finale = X_test  # Mantiene i questionari originali
    profondita = riga_scelta["max_depth"]  # Recupera la profondità selezionata
    if pd.isna(profondita):  # Controlla se la profondità è illimitata
        profondita = None  # Non impone un limite
    else:
        profondita = int(profondita)  # Converte la profondità in intero
    modello_finale = RandomForestClassifier(
        n_estimators=int(riga_scelta["n_estimators"]),  # Usa il numero di alberi selezionato
        max_depth=profondita,  # Usa la profondità selezionata
        random_state=42,  # Rende riproducibile il risultato
        n_jobs=-1  # Usa tutti i processori disponibili
    )

modello_finale.fit(X_train_finale, y_train)  # Addestra sui soggetti storici


# PREVISIONI SUI QUESTIONARI

previsioni = modello_finale.predict(X_test_finale)  # Prevede la classe 0 o 1
probabilita = modello_finale.predict_proba(X_test_finale)  # Calcola le probabilità stimate
indice_evento = list(modello_finale.classes_).index(1)  # Trova la colonna della classe positiva
probabilita_evento = probabilita[:, indice_evento] * 100  # Converte le probabilità in percentuale


# TABELLA DEI RISULTATI

risultati = pd.DataFrame({  # Costruisce la tabella delle previsioni
    "utente": range(1, len(X_test) + 1),  # Numera le righe del risultato
    "previsione": previsioni,  # Riporta la classe prevista
    "probabilita_evento_percentuale": probabilita_evento.round(2)  # Riporta la percentuale stimata
})

print("\nMODELLO UTILIZZATO:", modello_scelto)  # Mostra il modello definitivo
print("\nRISULTATI HEART VISION:")  # Introduce la tabella
print(risultati)  # Mostra tutte le previsioni
