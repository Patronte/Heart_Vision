from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# 
# ============================================================
# 7. MACHINE LEARNING - K NEAREST NEIGHBORS
# ============================================================

# SEPARAZIONE DEI DATI

X_train = df_X[colonne + ["bmi"]]
y_train = df_X[target]

X_test = df_y[colonne + ["bmi"]]


# NORMALIZZAZIONE DEI DATI

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# CREAZIONE DEL MODELLO

modello = KNeighborsClassifier(n_neighbors=5)


# ADDESTRAMENTO

modello.fit(X_train_scaled, y_train)


# PREVISIONI

previsioni = modello.predict(X_test_scaled)

probabilita = modello.predict_proba(X_test_scaled)

indice_evento = list(modello.classes_).index(1)

probabilita_evento = probabilita[:, indice_evento] * 100


# RISULTATI

risultati = pd.DataFrame({
    "utente": range(1, len(X_test) + 1),
    "previsione": previsioni,
    "probabilita_evento_percentuale": probabilita_evento.round(2)
})

print("\nRISULTATI HEARTVISION:")
print(risultati)