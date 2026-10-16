"""Chargement du modèle exporté par le notebook et prédiction de la probabilité de fraude."""
import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from tabm import TabM


class TabMBinaire(nn.Module):
    # Même enveloppe que dans le notebook : sortie (batch, k, 1) -> (batch, k)
    def __init__(self, d_in, **params):
        super().__init__()
        self.tabm = TabM.make(n_num_features=d_in, **params)

    def forward(self, x):
        return self.tabm(x).squeeze(-1)


class ScoreurFraude:
    """Applique la mise à l'échelle du train, le TabM exporté et le seuil figé sur la validation."""

    def __init__(self, dossier: Union[str, Path]):
        dossier = Path(dossier)
        with open(dossier / 'modele.json', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.features: List[str] = self.meta['features']
        self.seuil: float = self.meta['seuil']
        # Calcul en float64 puis conversion, comme StandardScaler dans le notebook
        self._moyenne = np.asarray(self.meta['scaler_mean'], dtype=np.float64)
        self._echelle = np.asarray(self.meta['scaler_scale'], dtype=np.float64)
        self.model = TabMBinaire(self.meta['d_in'], **self.meta['architecture'])
        poids = torch.load(dossier / self.meta['fichier_poids'], map_location='cpu')
        self.model.load_state_dict(poids)
        self.model.eval()

    @torch.no_grad()
    def probabilites(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        # X : tableau (n, 30) dans l'ordre de self.features, valeurs brutes
        X = ((np.asarray(X, dtype=np.float64) - self._moyenne) / self._echelle).astype(np.float32)
        x = torch.from_numpy(X)
        # Moyenne des probabilités des k sous-modèles, comme dans le notebook
        sorties = [torch.sigmoid(self.model(x[i:i + batch_size])).mean(dim=1) for i in range(0, len(x), batch_size)]
        return torch.cat(sorties).numpy()

    def predire(self, transactions: List[Dict[str, float]]) -> List[Dict]:
        X = np.array([[t[f] for f in self.features] for t in transactions], dtype=np.float64)
        return [self.decision(float(p)) for p in self.probabilites(X)]

    def decision(self, p: float) -> Dict:
        if p >= self.seuil:
            niveau = 'élevé'
        elif p >= 0.5:
            niveau = 'à surveiller'
        else:
            niveau = 'faible'
        return {'probabilite_fraude': p, 'fraude': p >= self.seuil, 'niveau_risque': niveau, 'seuil': self.seuil}
