"""API REST de détection de fraude : sert le TabM exporté par le notebook.

Lancement : uvicorn api.main:app --reload  (depuis la racine du projet)
Documentation interactive : http://localhost:8000/docs
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from fraude import ScoreurFraude

MODELE_DIR = Path(os.environ.get('MODELE_DIR', Path(__file__).resolve().parent.parent / 'modele'))
TAILLE_MAX_LOT = 10_000

# Schéma d'une transaction : Time, Amount et les 28 composantes anonymisées V1..V28
_champs = {
    'Time': (float, Field(..., ge=0, description='Secondes écoulées depuis la première transaction du jeu')),
    'Amount': (float, Field(..., ge=0, description='Montant de la transaction')),
}
_champs.update({f'V{i}': (float, Field(..., description='Composante PCA anonymisée')) for i in range(1, 29)})
_exemple = {'Time': 160000.0, 'Amount': 120.5, **{f'V{i}': 0.0 for i in range(1, 29)}}
Transaction = create_model('Transaction', __config__=ConfigDict(json_schema_extra={'examples': [_exemple]}),
                           **_champs)


class Prediction(BaseModel):
    probabilite_fraude: float = Field(..., description='Moyenne des probabilités des k sous-modèles TabM')
    fraude: bool = Field(..., description='Alerte : probabilité supérieure ou égale au seuil figé')
    niveau_risque: str = Field(..., description='élevé (alerte), à surveiller (>= 0,5) ou faible')
    seuil: float


class Lot(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1)


class ResultatLot(BaseModel):
    nb_transactions: int
    nb_alertes: int
    duree_ms: float
    predictions: List[Prediction]


etat = {}


@asynccontextmanager
async def cycle_de_vie(_app: FastAPI):
    # Le modèle est chargé une seule fois au démarrage
    etat['scoreur'] = ScoreurFraude(MODELE_DIR)
    yield
    etat.clear()


app = FastAPI(
    title='API Détection de fraude bancaire – TabM',
    description='PFA Sujet 09 : score de fraude d\'une transaction carte bancaire par un TabM compact.',
    version='1.0.0',
    lifespan=cycle_de_vie,
)


@app.get('/health', tags=['Supervision'])
def health():
    return {'statut': 'ok', 'modele_charge': 'scoreur' in etat}


@app.get('/modele', tags=['Supervision'])
def modele():
    meta = etat['scoreur'].meta
    cles = ('nom', 'version', 'architecture', 'seuil', 'regle_seuil', 'metriques_test', 'partition_test_time_min',
            'features')
    return {k: meta[k] for k in cles}


@app.post('/predict', response_model=Prediction, tags=['Prédiction'])
def predict(transaction: Transaction):
    return etat['scoreur'].predire([transaction.model_dump()])[0]


@app.post('/predict/batch', response_model=ResultatLot, tags=['Prédiction'])
def predict_batch(lot: Lot):
    if len(lot.transactions) > TAILLE_MAX_LOT:
        raise HTTPException(413, f'Lot limité à {TAILLE_MAX_LOT} transactions')
    debut = time.perf_counter()
    predictions = etat['scoreur'].predire([t.model_dump() for t in lot.transactions])
    return {
        'nb_transactions': len(predictions),
        'nb_alertes': sum(p['fraude'] for p in predictions),
        'duree_ms': round(1000 * (time.perf_counter() - debut), 2),
        'predictions': predictions,
    }
