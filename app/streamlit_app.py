"""Démo Streamlit : interface d'analyse des transactions, branchée sur l'API FastAPI.

Lancement : streamlit run app/streamlit_app.py  (l'API doit tourner, voir README)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get('API_URL', 'http://localhost:8000')
RACINE = Path(__file__).resolve().parent.parent
CSV = RACINE / 'data' / 'creditcard.csv'
RES_DIR = RACINE / 'resultats'
FEATURES = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
TAILLE_LOT = 5000

st.set_page_config(page_title='Détection de fraude – TabM', page_icon='💳', layout='wide')


def appeler_api(methode, chemin, **kwargs):
    try:
        r = requests.request(methode, f'{API_URL}{chemin}', timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"L'API ne répond pas sur {API_URL} : {e}")
        st.stop()


@st.cache_data
def charger_partition_test(time_min):
    df = pd.read_csv(CSV)
    return df[df['Time'] >= time_min].reset_index(drop=True)


def scorer_lot(df):
    # Envoie le tableau par paquets à /predict/batch et renvoie les prédictions alignées sur df
    resultats = []
    barre = st.progress(0.0, text='Analyse en cours…')
    for i in range(0, len(df), TAILLE_LOT):
        paquet = df[FEATURES].iloc[i:i + TAILLE_LOT].to_dict(orient='records')
        resultats += appeler_api('POST', '/predict/batch', json={'transactions': paquet})['predictions']
        barre.progress(min(1.0, (i + TAILLE_LOT) / len(df)), text='Analyse en cours…')
    barre.empty()
    return pd.DataFrame(resultats, index=df.index)


def afficher_decision(p):
    col1, col2, col3 = st.columns(3)
    col1.metric('Probabilité de fraude', f"{p['probabilite_fraude']:.4f}")
    col2.metric('Seuil d\'alerte', f"{p['seuil']:.4f}")
    col3.metric('Niveau de risque', p['niveau_risque'])
    if p['fraude']:
        st.error('🚨 ALERTE : transaction suspecte, à bloquer ou vérifier.')
    elif p['niveau_risque'] == 'à surveiller':
        st.warning('⚠️ Sous le seuil d\'alerte, mais score élevé : à surveiller.')
    else:
        st.success('✅ Transaction considérée comme normale.')


# --- Barre latérale : état de l'API et du modèle --------------------------------------------
meta = appeler_api('GET', '/modele')
with st.sidebar:
    st.header('Modèle déployé')
    st.success(f"API connectée : {API_URL}")
    st.write(f"**{meta['nom']}** – version {meta['version']}")
    st.write(f"Architecture : k={meta['architecture']['k']}, {meta['architecture']['n_blocks']} blocs "
             f"de {meta['architecture']['d_block']}")
    st.write(f"Seuil : **{meta['seuil']:.4f}**")
    st.caption(f"Règle : {meta['regle_seuil']}")
    m = meta['metriques_test']
    st.subheader('Performances sur le test')
    st.write(f"AP {m['AP']:.3f} · Précision {m['precision']:.2f} · Rappel {m['rappel']:.2f} · F1 {m['F1']:.2f}")
    st.caption(f"{int(m['TP'])} fraudes détectées, {int(m['FN'])} manquées, {int(m['FP'])} fausses alertes")

st.title('💳 Détection de fraude bancaire par TabM')
st.caption('PFA Sujet 09 – EMSI Marrakech · Les données de démonstration sont prises dans la période de test, '
           'jamais vue à l\'entraînement.')

onglet_tx, onglet_lot, onglet_res = st.tabs(['🔎 Analyser une transaction', '📂 Analyser un fichier',
                                              '📊 Résultats du modèle'])

# --- Onglet 1 : une transaction -------------------------------------------------------------
with onglet_tx:
    # Les champs du formulaire sont pilotés par session_state (clé champ_<variable>)
    for f in FEATURES:
        st.session_state.setdefault(f'champ_{f}', 0.0)
    st.session_state.setdefault('classe', None)

    if CSV.exists():
        test = charger_partition_test(meta['partition_test_time_min'])
        c1, c2, _ = st.columns([1, 1, 2])
        tirage = None
        if c1.button('Tirer une fraude réelle'):
            tirage = test[test['Class'] == 1]
        if c2.button('Tirer une transaction normale'):
            tirage = test[test['Class'] == 0]
        if tirage is not None:
            ligne = tirage.sample(1, random_state=np.random.randint(1_000_000)).iloc[0]
            st.session_state.classe = int(ligne['Class'])
            for f in FEATURES:
                st.session_state[f'champ_{f}'] = float(ligne[f])
    else:
        st.info('data/creditcard.csv absent : saisissez une transaction manuellement.')

    c1, c2 = st.columns(2)
    valeurs = {
        'Time': c1.number_input('Time (s)', min_value=0.0, key='champ_Time'),
        'Amount': c2.number_input('Montant', min_value=0.0, key='champ_Amount'),
    }
    with st.expander('Composantes anonymisées V1 à V28'):
        colonnes = st.columns(4)
        for i in range(1, 29):
            f = f'V{i}'
            valeurs[f] = colonnes[(i - 1) % 4].number_input(f, key=f'champ_{f}', format='%.4f')

    if st.button('Analyser la transaction', type='primary'):
        afficher_decision(appeler_api('POST', '/predict', json=valeurs))
        if st.session_state.classe is not None:
            st.caption(f"Classe réelle de cette transaction : "
                       f"{'fraude' if st.session_state.classe == 1 else 'normale'}")

# --- Onglet 2 : un fichier ------------------------------------------------------------------
with onglet_lot:
    st.write('Fichier CSV avec les colonnes `Time`, `Amount`, `V1`…`V28` ; la colonne `Class` est facultative '
             '(si elle est présente, la qualité des alertes est calculée).')
    fichier = st.file_uploader('Fichier de transactions', type='csv')
    utiliser_test = CSV.exists() and st.button('Ou utiliser toute la période de test (42 721 transactions)')

    df = None
    if fichier is not None:
        df = pd.read_csv(fichier)
    elif utiliser_test:
        df = charger_partition_test(meta['partition_test_time_min'])

    if df is not None:
        manquantes = [f for f in FEATURES if f not in df.columns]
        if manquantes:
            st.error(f'Colonnes manquantes : {manquantes}')
            st.stop()
        preds = scorer_lot(df)
        res = df.join(preds)
        alertes = res[res['fraude']].sort_values('probabilite_fraude', ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric('Transactions analysées', f'{len(res):,}'.replace(',', ' '))
        c2.metric('Alertes', len(alertes))
        c3.metric('À surveiller', int((res['niveau_risque'] == 'à surveiller').sum()))

        if 'Class' in res.columns:
            tp = int((res['fraude'] & (res['Class'] == 1)).sum())
            fp = int((res['fraude'] & (res['Class'] == 0)).sum())
            fn = int((~res['fraude'] & (res['Class'] == 1)).sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Fraudes détectées', tp)
            c2.metric('Fraudes manquées', fn)
            c3.metric('Fausses alertes', fp)
            c4.metric('Rappel', f'{tp / (tp + fn):.2f}' if tp + fn else 'n/a')

        st.subheader('Transactions en alerte')
        colonnes = ['Time', 'Amount', 'probabilite_fraude', 'niveau_risque'] + (['Class'] if 'Class' in res else [])
        st.dataframe(alertes[colonnes], width='stretch')
        st.download_button('Télécharger toutes les prédictions (CSV)', res.to_csv(index=False).encode('utf-8'),
                           'predictions_fraude.csv', 'text/csv')

# --- Onglet 3 : résultats de l'étude --------------------------------------------------------
with onglet_res:
    if (RES_DIR / 'resultats_test.csv').exists():
        st.subheader('Comparaison des modèles sur le test')
        st.dataframe(pd.read_csv(RES_DIR / 'resultats_test.csv').round(4), width='stretch')
    figures = [('pr_validation.png', 'Courbes précision-rappel (validation)'),
               ('matrices_confusion.png', 'Matrices de confusion (test, seuil figé)'),
               ('courbes_entrainement.png', 'Courbes d\'entraînement'),
               ('effet_ponderation.png', 'Effet de la pondération de la loss')]
    for nom, titre in figures:
        if (RES_DIR / nom).exists():
            st.subheader(titre)
            st.image(str(RES_DIR / nom))
    if not (RES_DIR / 'resultats_test.csv').exists():
        st.info('Exécutez le notebook pour produire le dossier resultats/.')
