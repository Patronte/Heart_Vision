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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, fbeta_score)

BASE = os.path.dirname(os.path.abspath(__file__))
HV = os.path.join(BASE, "heart_vision.py")
P_STORICO = os.path.join(BASE, "Dataset_storico", "HEARTVISION_dataset_completo.csv")

_ns = {}
_tree = ast.parse(open(HV, encoding="utf-8").read())
_keep = [n for n in _tree.body if
         isinstance(n, (ast.Import, ast.ImportFrom)) or
         (isinstance(n, ast.FunctionDef) and n.name in ("converti_questionario", "controllo_csv", "crea_modello")) or
         (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("colonne", "target"))]
exec(compile(ast.Module(body=_keep, type_ignores=[]), HV, "exec"), _ns)
colonne, target = _ns["colonne"], _ns["target"]
converti_questionario, controllo_csv = _ns["converti_questionario"], _ns["controllo_csv"]
crea_modello = _ns["crea_modello"]  # costruzione dei modelli identica al programma originale

metriche = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "brier"]
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
                   risultati=g("risultati"), modello=S.get("modello"), calcolo=P["stato"], emails=S.get("email"))


# 1. Caricamento e pulizia
@app.post("/api/carica")
@api
def carica():
    if P["stato"] == "running":
        raise ValueError("Attendi la fine del confronto in corso.")
    f1, f2 = request.files.get("storico"), request.files.get("questionari")
    if not (f2 and f2.filename):
        raise ValueError("Devi inserire un file CSV con le risposte del questionario!")
    dx = controllo_csv(pd.read_csv(f1 if f1 and f1.filename else P_STORICO), tipo="storico")
    raw = pd.read_csv(f2)  # le email servono per l'analisi del singolo paziente
    col = next((c for n in ["email", "indirizzo email", "email address", "username"]
                for c in raw.columns if c.strip().lower() == n), None)
    emails = raw[col].astype("string").str.strip().str.lower().fillna("").tolist() if col is not None else None
    dy = converti_questionario(raw)
    dy = controllo_csv(dy, tipo="questionario", riferimento=dx)
    for k in ("risultati", "modello", "confronto", "migliori", "split", "Xtr", "ytr"):
        S.pop(k, None)  # impedisce di riusare una selezione precedente
    S.update(X=dx, Y=dy, email=emails)
    return jsonify(storico=dict(righe=len(dx), eventi=round(float(dx[target].mean() * 100), 1),
                                head=rec(dx.head())),
                   questionari=dict(righe=len(dy), head=rec(dy.head())), emails=emails)


# 2-3. Confronto KNN / Decision Tree / Random Forest
P = dict(stato="idle", fatto=0, totale=35, errore=None)  # avanzamento del confronto


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
    # 20% dello storico escluso dalla scelta dei modelli (verifica indipendente)
    Xs, Xv, ys, yv = train_test_split(Xtr, ytr, test_size=0.20, stratify=ytr, random_state=42)
    S["Xtr"], S["ytr"], S["split"] = Xtr, ytr, (Xs, Xv, ys, yv)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cfgs = []
    for k in [3, 5, 7, 9, 11, 15, 21]:
        cfgs.append(dict(algoritmo="KNN", configurazione=f"K={k}", k=k, criterio=None, max_depth=None,
                         n_estimators=None, peso_classi="nessuno"))
    for c in ["gini", "entropy"]:
        for d in [3, 5, 10, None]:
            for w in ["nessuno", "balanced"]:
                cfgs.append(dict(algoritmo="Decision Tree", configurazione=f"{c} - profondità {d} - {w}", k=None,
                                 criterio=c, max_depth=d, n_estimators=None, peso_classi=w))
    for n in [100, 200]:
        for d in [5, 10, None]:
            for w in ["nessuno", "balanced"]:
                cfgs.append(dict(algoritmo="Random Forest", configurazione=f"{n} alberi - profondità {d} - {w}",
                                 k=None, criterio=None, max_depth=d, n_estimators=n, peso_classi=w))
    P["totale"] = len(cfgs)
    righe = []
    for cfg in cfgs:
        po = cross_val_predict(crea_modello(cfg), Xs, ys, cv=cv, method="predict_proba")[:, 1]
        co = (po >= 0.50).astype(int)
        r = dict(cfg)
        r.update(accuracy=accuracy_score(ys, co), precision=precision_score(ys, co, zero_division=0),
                 recall=recall_score(ys, co, zero_division=0), f1_score=f1_score(ys, co, zero_division=0),
                 roc_auc=roc_auc_score(ys, po), pr_auc=average_precision_score(ys, po),
                 brier=brier_score_loss(ys, po))
        righe.append(r)
        P["fatto"] += 1

    conf = pd.DataFrame(righe).sort_values(by=["pr_auc", "roc_auc", "brier"],
                                           ascending=[False, False, True]).reset_index(drop=True)
    migl = conf.loc[conf.groupby("algoritmo")["pr_auc"].idxmax()].reset_index(drop=True)
    conf[metriche] = conf[metriche].round(3)
    migl[metriche] = migl[metriche].round(3)
    S["confronto"], S["migliori"] = conf, migl
    S.pop("risultati", None); S.pop("modello", None)


# 4. Modello scelto per PR-AUC, calibrazione, soglia, verifica indipendente e previsioni
@app.post("/api/previsioni")
@api
def previsioni():
    if "confronto" not in S:
        raise ValueError("Prima esegui il confronto dei modelli.")
    cols = colonne + ["bmi"]
    Xtr, ytr, Xte = S["Xtr"], S["ytr"], S["Y"][cols]
    Xs, Xv, ys, yv = S["split"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scelta = S["confronto"].iloc[0]  # migliore per PR-AUC
    mk = lambda folds: CalibratedClassifierCV(estimator=crea_modello(scelta), method="sigmoid", cv=folds)

    po = cross_val_predict(mk(3), Xs, ys, cv=cv, method="predict_proba")[:, 1]
    soglia, bf, bp, br = 0.50, -1.0, 0.0, 0.0  # soglia scelta massimizzando l'F1 sullo sviluppo
    for t in np.arange(0.10, 0.601, 0.01):
        c = (po >= t).astype(int)
        f = f1_score(ys, c, zero_division=0)
        if f > bf:
            bf, soglia = f, float(t)
            bp, br = precision_score(ys, c, zero_division=0), recall_score(ys, c, zero_division=0)

    pv = mk(3).fit(Xs, ys).predict_proba(Xv)[:, 1]  # verifica sul 20% mai usato
    cvv = (pv >= soglia).astype(int)
    ver = dict(precision=precision_score(yv, cvv, zero_division=0), recall=recall_score(yv, cvv, zero_division=0),
               f2=fbeta_score(yv, cvv, beta=2, zero_division=0), roc_auc=roc_auc_score(yv, pv),
               pr_auc=average_precision_score(yv, pv), brier=brier_score_loss(yv, pv))

    finale = mk(5).fit(Xtr, ytr)  # addestramento definitivo su tutto lo storico
    prob = finale.predict_proba(Xte)[:, 1] * 100
    prev = ((prob / 100) >= soglia).astype(int)
    ris = pd.DataFrame({"utente": range(1, len(Xte) + 1), "previsione": prev,
                        "probabilita_evento_percentuale": prob.round(2)})
    S["risultati"] = ris
    S["modello"] = dict(algoritmo=scelta["algoritmo"], configurazione=scelta["configurazione"],
                        pr_auc=float(scelta["pr_auc"]), soglia=round(soglia, 3),
                        sviluppo=dict(f1=round(bf, 3), precision=round(bp, 3), recall=round(br, 3)),
                        verifica={k: round(float(v), 3) for k, v in ver.items()})
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


# 7. Studio singolo paziente (ricerca tramite email)
@app.get("/api/partecipante")
@api
def partecipante():
    if "risultati" not in S:
        raise ValueError("Prima carica il CSV ed esegui le previsioni.")
    em, dy, ris = S.get("email"), S["Y"], S["risultati"]
    if em is None:
        raise ValueError("Il CSV caricato non contiene una colonna email o Username.")
    if len(em) != len(dy) or len(ris) != len(dy):
        raise ValueError("I dati non corrispondono: ricarica il CSV ed esegui le previsioni.")
    q = request.args.get("email", "").strip().lower()
    if not q:
        raise ValueError("Devi inserire un indirizzo email.")
    trovati = [i for i, e in enumerate(em) if e == q]
    if not trovati:
        raise ValueError("Email non trovata nel CSV attualmente caricato.")
    if len(trovati) > 1:  # stessa email presente più volte: serve l'ID
        sid = request.args.get("id", "")
        if not sid.isdigit() or int(sid) - 1 not in trovati:
            return jsonify(ids=[i + 1 for i in trovati])
        i = int(sid) - 1
    else:
        i = trovati[0]
    sogg, prev, df = dy.iloc[i], ris.iloc[i], S["X"]
    num = []
    for c, nome in [("eta", "Età (anni)"), ("sigarette_giorno", "Sigarette/giorno"), ("peso_kg", "Peso (kg)"),
                    ("altezza_cm", "Altezza (cm)"), ("bmi", "BMI"), ("frequenza_cardiaca", "Frequenza cardiaca"),
                    ("sere_alcol_settimana", "Sere con alcol/settimana")]:
        v = float(sogg[c])
        st = pd.to_numeric(df[c], errors="coerce").dropna()
        lim = float(st.quantile(0.90))  # 90° percentile dello storico
        num.append(dict(nome=nome, valore=round(v, 2), limite=round(lim, 2),
                        rango=round(float((st <= v).mean() * 100), 1), elevato=bool(v > lim)))
    return jsonify(id=int(prev["utente"]), email=q, classe=int(prev["previsione"]),
                   percentuale=float(prev["probabilita_evento_percentuale"]),
                   caratteristiche={c: round(float(sogg[c]), 2) for c in colonne + ["bmi"]},
                   numeriche=num)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
