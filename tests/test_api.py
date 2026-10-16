"""Tests de l'API et de la reproductibilité du modèle exporté.

Lancement : python -m pytest -q  (depuis la racine du projet)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import average_precision_score

from api.main import app
from fraude import ScoreurFraude

RACINE = Path(__file__).resolve().parent.parent
CSV = RACINE / 'data' / 'creditcard.csv'
TRANSACTION = {'Time': 160000.0, 'Amount': 120.5, **{f'V{i}': 0.0 for i in range(1, 29)}}


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:   # le with déclenche le chargement du modèle au démarrage
        yield c


@pytest.fixture(scope='module')
def scoreur():
    return ScoreurFraude(RACINE / 'modele')


def test_health(client):
    assert client.get('/health').json() == {'statut': 'ok', 'modele_charge': True}


def test_modele_expose_le_seuil_et_les_metriques(client, scoreur):
    meta = client.get('/modele').json()
    assert meta['seuil'] == pytest.approx(scoreur.seuil)
    assert meta['architecture']['k'] == 4
    assert len(meta['features']) == 30


def test_predict_une_transaction(client):
    r = client.post('/predict', json=TRANSACTION)
    assert r.status_code == 200
    p = r.json()
    assert 0.0 <= p['probabilite_fraude'] <= 1.0
    assert p['fraude'] == (p['probabilite_fraude'] >= p['seuil'])


def test_champ_manquant_refuse(client):
    incomplete = {k: v for k, v in TRANSACTION.items() if k != 'V14'}
    assert client.post('/predict', json=incomplete).status_code == 422


def test_montant_negatif_refuse(client):
    assert client.post('/predict', json={**TRANSACTION, 'Amount': -5}).status_code == 422


def test_batch_coherent_avec_predict(client):
    autre = {**TRANSACTION, 'V14': -8.0, 'V4': 5.0}
    lot = client.post('/predict/batch', json={'transactions': [TRANSACTION, autre]}).json()
    assert lot['nb_transactions'] == 2
    for transaction, prediction in zip([TRANSACTION, autre], lot['predictions']):
        seule = client.post('/predict', json=transaction).json()
        assert prediction['probabilite_fraude'] == pytest.approx(seule['probabilite_fraude'], abs=1e-6)


def test_lot_vide_refuse(client):
    assert client.post('/predict/batch', json={'transactions': []}).status_code == 422


@pytest.mark.skipif(not CSV.exists(), reason='data/creditcard.csv absent')
def test_modele_exporte_reproduit_les_resultats_du_notebook(scoreur):
    # Le scoreur seul (hors notebook) doit retrouver les métriques de test enregistrées à l'export
    df = pd.read_csv(CSV)
    test = df[df['Time'] >= scoreur.meta['partition_test_time_min']]
    p = scoreur.probabilites(test[scoreur.features].to_numpy())
    y = test['Class'].to_numpy()
    attendu = scoreur.meta['metriques_test']
    assert average_precision_score(y, p) == pytest.approx(attendu['AP'], abs=1e-4)
    alertes = p >= scoreur.seuil
    assert int((alertes & (y == 1)).sum()) == attendu['TP']
    assert int((alertes & (y == 0)).sum()) == attendu['FP']
