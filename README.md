# Heart Vision ♥

**Software di analisi e previsione degli eventi coronarici attraverso il Machine Learning**

## 📖 Descrizione

**Heart Vision** è un software sviluppato in Python che utilizza tecniche di Machine Learning per analizzare i dati degli utenti e stimare la probabilità di un evento coronarico nei successivi 10 anni.

Il progetto nasce con l'obiettivo di integrare il mondo delle biotecnologie mediche e quello dell'intelligenza artificiale, sfruttando l'analisi dei dati e gli algoritmi di apprendimento automatico per individuare correlazioni tra le caratteristiche di un individuo e la presenza di fattori associati al rischio cardiovascolare.

Il sistema utilizza un dataset storico di riferimento, contenente informazioni relative a migliaia di soggetti, per addestrare e confrontare diversi modelli di Machine Learning. Successivamente, attraverso l'elaborazione di nuovi questionari, è in grado di analizzare i dati degli utenti e generare previsioni individuali.

Heart Vision integra un programma Python per l'elaborazione dei dati e una web application sviluppata con Flask, che permette di accedere alle funzionalità del progetto attraverso un'interfaccia web.

*Il software è un progetto prototipale a scopo didattico e sperimentale. Le previsioni non costituiscono una diagnosi medica né una valutazione clinica individuale validata.*

## 🎯 Obiettivi del progetto

Il progetto è stato realizzato con i seguenti obiettivi:

- Integrare competenze di biotecnologie mediche, programmazione e intelligenza artificiale in un unico progetto.
- Sviluppare un software capace di elaborare dataset contenenti informazioni anagrafiche, abitudini di vita e fattori associati alla salute cardiovascolare.
- Applicare differenti algoritmi di Machine Learning a un problema di classificazione.
- Confrontare le prestazioni dei modelli attraverso metriche statistiche e tecniche di Cross-Validation.
- Individuare le configurazioni più adatte al problema analizzato.
- Elaborare nuovi questionari e generare previsioni individuali.
- Rappresentare i risultati attraverso tabelle e grafici per facilitarne la consultazione e l'interpretazione.
- Realizzare un sistema organizzato, modulare ed estendibile attraverso un'interfaccia web.

## ✨ Funzionalità principali

- 📂 **Caricamento e gestione dei dataset:** importazione del dataset storico e dei nuovi questionari in formato CSV, con gestione delle principali eccezioni durante la lettura dei file.
- 🧹 **Data Cleaning e Preprocessing:** controllo della struttura dei dati, conversione dei valori, gestione dei dati mancanti e dei valori fuori intervallo, verifica della coerenza delle risposte e calcolo automatico del BMI.
- 🤖 **Confronto tra algoritmi di Machine Learning:** implementazione e confronto di K-Nearest Neighbors (KNN), Decision Tree Classifier e Random Forest Classifier.
- 📊 **Cross-Validation e valutazione:** utilizzo della Cross-Validation stratificata a 5 fold e confronto dei modelli attraverso Accuracy, Precision, Recall, F1-Score e ROC-AUC.
- ⚙️ **Ottimizzazione dei parametri:** sperimentazione di differenti configurazioni per individuare quelle con le prestazioni migliori.
- 📈 **Visualizzazione dei risultati:** generazione di tabelle e grafici per confrontare algoritmi, configurazioni e metriche di valutazione.
- 🧬 **Addestramento e previsione:** addestramento del modello selezionato sull'intero dataset storico e applicazione ai nuovi questionari.
- 👥 **Analisi degli utenti:** elaborazione di più questionari e restituzione delle previsioni individuali, con classe prevista e probabilità associate alle due classi.

## 🛠️ Stack tecnologico

Il progetto utilizza Python e le principali librerie dell'ecosistema Data Science e Machine Learning. La componente web è realizzata attraverso Flask.

| Componente                        | Tecnologia             |
| --------------------------------- | ---------------------- |
| Linguaggio principale             | Python                 |
| Backend e applicazione web        | Flask                  |
| Frontend                          | HTML                   |
| Gestione e manipolazione dei dati | Pandas                 |
| Calcolo numerico                  | NumPy                  |
| Visualizzazione dei dati          | Matplotlib             |
| Machine Learning                  | Scikit-learn           |
| Formato dei dataset               | CSV                    |
| Gestione delle dipendenze         | pip e requirements.txt |

## ▶️ Come avviare il progetto in locale

Per eseguire Heart Vision in locale è necessario disporre di Python 3 e installare le librerie richieste dal progetto.

### 📦 Installazione delle dipendenze

Prima del primo avvio, accedere alla cartella `Sito_Web` e installare le librerie necessarie.

**Windows**

1. Aprire il Prompt dei comandi (CMD).

2. Accedere alla cartella `Sito_Web` utilizzando il comando `cd` seguito dal percorso della cartella.

   ```bash
   cd "C:\Users\Nome\Desktop\HeartVision\Sito_Web"
   ```

3. Installare le dipendenze:

   ```bash
   py -m pip install -r requirements.txt
   ```

**macOS**

1. Aprire il Terminale.

2. Accedere alla cartella `Sito_Web` utilizzando il comando `cd` seguito dal percorso della cartella.

   ```bash
   cd "/Users/Nome/Desktop/HeartVision/Sito_Web"
   ```

3. Installare le dipendenze:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

### 🚀 Avvio dell'applicazione

Una volta installate le dipendenze, è possibile avviare l'applicazione.

**Windows**

```bash
py app.py
```

**macOS**

```bash
python3 app.py
```

Attendere l'avvio del server. Copiare l'indirizzo URL mostrato nel terminale e incollarlo nel browser per accedere all'interfaccia web di Heart Vision.

## 📁 Struttura del progetto

```text
HeartVision/
│
├── Dataset_questionari/       # Risposte raccolte dai questionari
│   ├── Framingham_form.csv
│   ├── HeartVision_50_ute...
│   └── HeartVision_50_ute...
│
├── Dataset_storico/           # Dataset di riferimento per il Machine Learning
│   └── HEARTVISION_data...
│
├── Grafici Utili/             # Grafici e immagini del progetto
│   ├── image 1.png
│   └── image2.png
│
├── Relazioni/                 # Documentazione del progetto
│   ├── Relazione_HeartVisi...
│   └── Relazione_HeartVisi...
│
├── Sito_Web/                  # Interfaccia web sviluppata con Flask
│   ├── __pycache__
│   │   └── app.cpython-312.pyc
│   ├── Dataset_storico
│   │   └── HEARTVISION_dataset_completo.csv
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│   ├── heart_vision.py
│   └── requirements.txt
│
├── heart_vision.py            # Programma Python principale
└── README.md
```

## 👨‍💻 Il Team

Progetto realizzato da **FuturVision Labs**.

Ogni componente del team ha contribuito allo sviluppo del progetto assumendo una responsabilità principale in una specifica area tecnica:

- **Luca Marchese — Machine Learning:** sviluppo, implementazione e confronto degli algoritmi di apprendimento automatico, valutazione delle prestazioni dei modelli e analisi dei risultati.
- **Giuseppe Giacco — Data Analysis:** analisi, organizzazione e interpretazione dei dati utilizzati dal progetto, con particolare attenzione alla loro struttura e alla loro preparazione per le successive elaborazioni.
- **Matteo Criscuolo — Grafici e funzionalità:** sviluppo delle funzionalità del software e realizzazione delle componenti di visualizzazione grafica dei dati e dei risultati.
- **Valentina Avitabile — Menu e interfaccia grafica:** progettazione e sviluppo del menu di navigazione e dell'interfaccia grafica, con attenzione all'organizzazione delle sezioni e alla semplicità di utilizzo.

### Collaborazione e sviluppo condiviso

La suddivisione dei ruoli ha rappresentato un'organizzazione del lavoro e non una separazione delle competenze. Tutti i componenti del team hanno collaborato attivamente e si sono supportati reciprocamente durante le diverse fasi dello sviluppo, contribuendo anche agli aspetti tecnici al di fuori delle rispettive aree di responsabilità.

Il gruppo ha inoltre condiviso la gestione della repository GitHub, il ragionamento progettuale, la definizione dell'architettura e la strutturazione del programma. Le scelte implementative sono state discusse e sviluppate insieme, con l'obiettivo di realizzare un software chiaro, semplice da utilizzare e ben organizzato, mantenendo allo stesso tempo un'impostazione tecnica, funzionale e orientata alla qualità del codice.

**FuturVision Labs — Un progetto sviluppato attraverso competenze complementari, collaborazione e visione condivisa.**
