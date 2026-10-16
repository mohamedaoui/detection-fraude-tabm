# Détection de fraude bancaire par TabM – de l'étude au déploiement

PFA Sujet 09 – EMSI Marrakech. Détection des transactions frauduleuses par carte bancaire (jeu ULB, 284 807
transactions, dont 0,17 % de fraudes). Le projet compare une régression logistique, un MLP et **TabM** (un ensemble
de réseaux à poids partagés, ICLR 2025). Le modèle retenu est ensuite déployé derrière une API REST, avec une
interface de démonstration.

```
Notebook (étude) ──► MLflow (suivi des expériences)
       │
       └──► modele/ (poids + mise à l'échelle + seuil) ──► API FastAPI ──► Démo Streamlit
                                                              └── Docker Compose
```

## Résultats (test : 52 fraudes sur 42 721 transactions, seuil figé sur la validation)

| Modèle | AP | Précision | Rappel | F1 | Fausses alertes |
|---|---|---|---|---|---|
| Régression logistique | 0,707 | 0,94 | 0,63 | 0,76 | 2 |
| MLP 30-64-64-1 | 0,771 | 0,95 | 0,69 | 0,80 | 2 |
| **TabM pondéré (déployé)** | 0,755 | **1,00** | **0,69** | **0,82** | **0** |
| TabM non pondéré (ablation) | **0,782** | 1,00 | 0,63 | 0,78 | 0 |

Au seuil 0,5, les modèles pondérés déclenchent entre 306 et 986 fausses alertes : le choix du seuil sur la
validation est déterminant.

## Structure

| Élément | Rôle |
|---|---|
| `PFA_Sujet09_Fraude_TabM.ipynb` | Étude complète : Comprendre → Préparer → Construire → Comparer → Restituer, puis export du modèle |
| `fraude/` | Code d'inférence partagé : mise à l'échelle du train, TabM, seuil |
| `modele/` | Modèle exporté : `tabm_pondere.pt` et `modele.json` (seuil, scaler, métriques) |
| `api/main.py` | API FastAPI : `/predict`, `/predict/batch`, `/modele`, `/health` |
| `app/streamlit_app.py` | Démo : analyser une transaction ou un fichier CSV, consulter les résultats |
| `tests/` | Tests de l'API et de la reproductibilité du modèle exporté (pytest) |
| `resultats/` | Figures et tableaux produits par le notebook |
| `Dockerfile.api`, `Dockerfile.app`, `docker-compose.yml` | Conteneurisation de l'API et de la démo |

## Installation

```bash
pip install -r requirements.txt
```

Données : placer `creditcard.csv` dans `data/`, depuis [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
ou depuis la copie publique de TensorFlow
(`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`, contenu identique).

## Utilisation

**1. Étude et entraînement** : exécuter le notebook en local (environ 4 minutes sur CPU) ou sur Google Colab
(données et checkpoints sur Google Drive). L'entraînement reprend depuis le dernier checkpoint si la session
s'interrompt. La dernière section exporte le modèle dans `modele/`.

**2. Suivi des expériences (MLflow)**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ouvrir http://localhost:5000, expérience `PFA-fraude-TabM`. Chaque modèle y a un run avec ses paramètres, la courbe
d'AP de validation par époque et les métriques de test. Le run déployé porte le tag `deploye=oui` et contient le
modèle en artefact.

**3. API**

```bash
uvicorn api.main:app --reload
```

La documentation interactive (Swagger) est sur http://localhost:8000/docs. Exemple d'appel :

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Time": 160000, "Amount": 120.5, "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0}'
```

Réponse : `probabilite_fraude`, `fraude` (alerte si la probabilité atteint le seuil), `niveau_risque`
(élevé / à surveiller / faible) et `seuil`.

**4. Démo Streamlit** (l'API doit tourner)

```bash
streamlit run app/streamlit_app.py
```

**5. Tests**

```bash
python -m pytest -q
```

Les tests vérifient les réponses de l'API, le rejet des entrées invalides et la cohérence entre `/predict` et
`/predict/batch`. Ils vérifient aussi que le modèle exporté retrouve exactement les métriques de test du notebook.

**6. Docker**

```bash
docker compose up --build
```

L'API est servie sur le port 8000 et la démo sur le port 8501. `data/` et `resultats/` sont montés en lecture seule
dans la démo.

## Protocole

- **Coupure chronologique** 70/15/15 sur `Time`, sans répartir un même timestamp entre deux partitions.
- **Mise à l'échelle** et `pos_weight` calculés sur le **train** uniquement. La fréquence naturelle de fraude est
  conservée en validation et en test.
- **Sélection** des modèles par Average Precision de validation (arrêt anticipé, patience de 8 époques).
- **Seuil** figé sur la validation (maximisation du F1). Les résultats au seuil 0,5 sont aussi rapportés.
- **Test** utilisé une seule fois pour le tableau final.
- **Reproductibilité** : seed 42, checkpoints, configuration enregistrée dans `resultats/config.json`. Deux
  entraînements complets successifs donnent des résultats identiques.

## Limites

- Le test ne contient que 52 fraudes : une fraude représente environ 2 points de rappel. Un seul entraînement par
  modèle a été réalisé.
- Le jeu de données couvre deux jours et ses variables sont anonymisées (composantes PCA fournies par l'éditeur).
- L'API est une démonstration : elle n'a ni authentification ni supervision de la dérive des données.

## Technologies

Python · pandas · NumPy · scikit-learn · PyTorch · TabM · Matplotlib · MLflow · FastAPI · Pydantic · Streamlit ·
pytest · Docker

## Références

- TabM : *Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*, ICLR 2025 – https://github.com/yandex-research/tabm
- Dataset : Machine Learning Group, ULB – https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
