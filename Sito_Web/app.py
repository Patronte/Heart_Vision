# HEARTVISION WEB - backend Flask
# Riusa SENZA modifiche colonne, target, converti_questionario e controllo_csv di heart_vision.py
# ed espone le funzioni del menu (1-6) come API per il sito.

import ast, os, json, functools, threading
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

BASE = os.path.dirname(os.path.abspath(__file__))
HV = os.path.join(BASE, "heart_vision.py")
P_STORICO = os.path.join(BASE, "Dataset_storico", "HEARTVISION_dataset_completo.csv")

_ns = {}
_tree = ast.parse(open(HV, encoding="utf-8").read())
_keep = [n for n in _tree.body if
         isinstance(n, (ast.Import, ast.ImportFrom)) or
         (isinstance(n, ast.FunctionDef) and n.name in ("converti_questionario", "controllo_csv")) or
         (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("colonne", "target"))]
exec(compile(ast.Module(body=_keep, type_ignores=[]), HV, "exec"), _ns)
colonne, target = _ns["colonne"], _ns["target"]
converti_questionario, controllo_csv = _ns["converti_questionario"], _ns["controllo_csv"]

metriche = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
S = {}  # stato dell'applicazione (le variabili globali dello script)
app = Flask(__name__)


def rec(df):  # DataFrame -> lista di dict (NaN -> null)
    return json.loads(df.to_json(orient="records"))


def api(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except FileNotFoundError as e:
            return jsonify(errore=f"File non trovato: {e}. Controlla che la cartella Dataset_storico "
                                  "sia accanto ad app.py oppure carica il dataset storico dal sito."), 400
        except Exception as e:
            return jsonify(errore=str(e)), 400
    return w


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/stato")
def stato():
    g = lambda k: rec(S[k]) if k in S else None
    return jsonify(dataset="X" in S, confronto=g("confronto"), migliori=g("migliori"),
                   risultati=g("risultati"), modello=S.get("modello"), calcolo=P["stato"])


# 1. Caricamento e pulizia
@app.post("/api/carica")
@api
def carica():
    f1, f2 = request.files.get("storico"), request.files.get("questionari")
    if not (f2 and f2.filename):
        raise ValueError("Devi inserire un file CSV con le risposte del questionario!")
    dx = controllo_csv(pd.read_csv(f1 if f1 and f1.filename else P_STORICO), tipo="storico")
    dy = converti_questionario(pd.read_csv(f2))
    dy = controllo_csv(dy, tipo="questionario", riferimento=dx)
    S.pop("risultati", None); S.pop("modello", None)  # niente risultati di un altro CSV
    if f1 and f1.filename:  # nuovo storico: il vecchio confronto non è più valido
        for k in ("confronto", "migliori"):
            S.pop(k, None)
    S.update(X=dx, Y=dy)
    return jsonify(storico=dict(righe=len(dx), eventi=round(float(dx[target].mean() * 100), 1),
                                head=rec(dx.head())),
                   questionari=dict(righe=len(dy), head=rec(dy.head())))


# 2-3. Confronto KNN / Decision Tree / Random Forest
P = dict(stato="idle", fatto=0, totale=115, errore=None)  # avanzamento del confronto


@app.post("/api/confronto")
@api
def confronto():  # avvia il calcolo in background: il sito ne segue l'avanzamento
    if "X" not in S:
        raise ValueError("Prima carica i dataset.")
    if P["stato"] == "running":
        return jsonify(avviato=False)
    P.update(stato="running", fatto=0, errore=None)
    threading.Thread(target=_job, daemon=True).start()
    return jsonify(avviato=True)


@app.get("/api/progresso")
def progresso():
    pc = round(P["fatto"] / P["totale"] * 100)
    return jsonify(stato=P["stato"], percentuale=100 if P["stato"] == "done" else min(pc, 99), errore=P["errore"])


def _job():
    try:
        _calcola()
        P.update(stato="done", fatto=P["totale"])
    except Exception as e:
        P.update(stato="error", errore=str(e))


def _calcola():
    cols = colonne + ["bmi"]
    Xtr, ytr = S["X"][cols], S["X"][target]
    S["Xtr"], S["ytr"] = Xtr, ytr
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cfgs = []  # (algoritmo, configurazione, parametri, scaling, costruttore)
    for k in [3, 5, 7, 9, 11, 15]:
        cfgs.append(("KNN", f"K={k}", dict(k=k), True, lambda k=k: KNeighborsClassifier(n_neighbors=k)))
    for c in ["gini", "entropy"]:
        for d in [3, 5, 10, None]:
            cfgs.append(("Decision Tree", f"{c} - profondità {d}", dict(criterio=c, max_depth=d), False,
                         lambda c=c, d=d: DecisionTreeClassifier(criterion=c, max_depth=d,
                                                                 min_samples_leaf=20, random_state=42)))
    for n in [100, 200, 300]:
        for d in [5, 10, None]:
            cfgs.append(("Random Forest", f"{n} alberi - profondità {d}", dict(n_estimators=n, max_depth=d),
                         False, lambda n=n, d=d: RandomForestClassifier(n_estimators=n, max_depth=d,
                                                                        random_state=42, n_jobs=-1)))
    P["totale"] = len(cfgs) * 5  # configurazioni x 5 fold
    righe = []
    for algo, nome, par, scala, mk in cfgs:
        for i_tr, i_va in cv.split(Xtr, ytr):
            a, b = Xtr.iloc[i_tr], Xtr.iloc[i_va]
            if scala:
                sc = StandardScaler()
                a, b = sc.fit_transform(a), sc.transform(b)
            mod = mk().fit(a, ytr.iloc[i_tr])
            yv, p = ytr.iloc[i_va], mod.predict(b)
            pr = mod.predict_proba(b)[:, list(mod.classes_).index(1)]
            r = dict(algoritmo=algo, configurazione=nome, k=None, criterio=None, max_depth=None,
                     n_estimators=None)
            r.update(par)
            r.update(accuracy=accuracy_score(yv, p), precision=precision_score(yv, p, zero_division=0),
                     recall=recall_score(yv, p, zero_division=0), f1_score=f1_score(yv, p, zero_division=0),
                     roc_auc=roc_auc_score(yv, pr))
            righe.append(r)
            P["fatto"] += 1

    dfr = pd.DataFrame(righe)
    conf = dfr.groupby(["algoritmo", "configurazione"])[metriche].mean().reset_index()
    par = dfr[["algoritmo", "configurazione", "k", "criterio", "max_depth", "n_estimators"]].drop_duplicates()
    conf = conf.merge(par, on=["algoritmo", "configurazione"])
    conf[metriche] = conf[metriche].round(3)
    conf = conf.sort_values(by="f1_score", ascending=False)
    migl = conf.loc[conf.groupby("algoritmo")["f1_score"].idxmax()].reset_index(drop=True)
    S["confronto"], S["migliori"] = conf, migl
    S.pop("risultati", None); S.pop("modello", None)


# 4. Modello migliore (F1 medio) e previsioni
@app.post("/api/previsioni")
@api
def previsioni():
    if "confronto" not in S:
        raise ValueError("Prima esegui il confronto dei modelli.")
    cols = colonne + ["bmi"]
    Xtr, ytr, Xte = S["Xtr"], S["ytr"], S["Y"][cols]
    conf = S["confronto"]
    riga = conf.loc[conf["f1_score"].idxmax()]
    nome = riga["algoritmo"]
    prof = None if pd.isna(riga["max_depth"]) else int(riga["max_depth"])
    if nome == "KNN":
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
        mod = KNeighborsClassifier(n_neighbors=int(riga["k"]))
    elif nome == "Decision Tree":
        mod = DecisionTreeClassifier(criterion=riga["criterio"], max_depth=prof,
                                     min_samples_leaf=20, random_state=42)
    else:
        mod = RandomForestClassifier(n_estimators=int(riga["n_estimators"]), max_depth=prof,
                                     random_state=42, n_jobs=-1)
    mod.fit(Xtr, ytr)
    prev = mod.predict(Xte)
    prob = mod.predict_proba(Xte)[:, list(mod.classes_).index(1)] * 100
    ris = pd.DataFrame({"utente": range(1, len(Xte) + 1), "previsione": prev,
                        "probabilita_evento_percentuale": prob.round(2)})
    S["risultati"] = ris
    S["modello"] = dict(algoritmo=nome, configurazione=riga["configurazione"], f1_score=float(riga["f1_score"]))
    return jsonify(modello=S["modello"], risultati=rec(ris))


# 5. Fattori di rischio
@app.get("/api/fattori")
@api
def fattori():
    if "X" not in S:
        raise ValueError("Prima carica i dataset (scheda Dataset).")
    df, out = S["X"], []
    for f in ["fumatore", "diabete", "colesterolo_alto", "alcol", "ictus_precedente", "ipertensione"]:
        out.append(dict(fattore=f.upper(), rischio=round(float(df[df[f] == 1][target].mean() * 100), 2)))
    return jsonify(sorted(out, key=lambda x: -x["rischio"]))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
