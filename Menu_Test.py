nome_file = input("Inserisci il nome del file CSV: ")  # chiediamo all'utente il nome del file da caricare

df = pd.read_csv(nome_file)  # leggiamo il file CSV e salviamo i dati nel DataFrame

print("File caricato correttamente!")  # comunichiamo che il caricamento è avvenuto con successo


# MENU PRINCIPALE

while True:  # ripetiamo il menu fino a quando l'utente decide di uscire

    print("\n--- MENU PRINCIPALE ---")
    print("1. Analisi del Dataset")  # visualizziamo l'opzione per analizzare il dataset
    print("2. Studio e Pulizia del CSV")  # visualizziamo l'opzione per controllare e pulire i dati
    print("3. Analisi del Campione ML")  # visualizziamo l'opzione per analizzare e aggiungere nuovi prodotti
    print("4. Grafici")  # visualizziamo l'opzione per creare il grafico delle vendite
    print("0. Esci")  # visualizziamo l'opzione per terminare il programma

    scelta = input("Seleziona un'opzione: ")  # chiediamo all'utente di scegliere un'opzione

    if scelta == "1":  # se viene scelta l'opzione 1, avviamo l'analisi del dataset

       pass  # inseriremo qui la funzione dedicata all'analisi

    elif scelta == "2":  # se viene scelta l'opzione 2, avviamo la pulizia del CSV

        pass  # inseriremo qui le operazioni per pulire e aggiornare il DataFrame

    elif scelta == "3":  # se viene scelta l'opzione 3, avviamo l'analisi del campione ML

        pass  # inseriremo qui le operazioni per aggiungere un nuovo prodotto

    elif scelta == "4":  # se viene scelta l'opzione 4, visualizziamo il grafico delle vendite

        pass  # inseriremo qui la funzione per creare il grafico

    elif scelta == "0":  # se viene scelta l'opzione 0, usciamo dal programma

        print("Programma terminato!")  # comunichiamo la chiusura del programma

        break  # interrompiamo il ciclo e terminiamo il menu

    else:  # se viene inserita un'opzione non presente nel menu

        print("Scelta non valida, riprova!")  # chiediamo all'utente di effettuare una nuova scelta