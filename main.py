import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

# -------------------------------
# Fonctions utilitaires
# -------------------------------

def rolling_window_features(df, window=60, stride=10):
    """
    Extrait des features simples par fenêtre glissante
    (moyenne, std, min, max, pente)
    """
    features, idx = [], []
    for start in range(0, len(df) - window, stride):
        chunk = df.iloc[start:start+window]
        feat = []
        # Stats basiques
        feat.extend(chunk.mean().values)
        feat.extend(chunk.std().values)
        feat.extend(chunk.min().values)
        feat.extend(chunk.max().values)
        # Approximation de pente (linéaire)
        t = np.arange(len(chunk))
        for col in df.columns:
            slope = np.polyfit(t, chunk[col].values, 1)[0]
            feat.append(slope)
        features.append(feat)
        idx.append(chunk.index[-1])  # timestamp de fin de fenêtre
    cols = [f"{c}_mean" for c in df.columns] + \
           [f"{c}_std" for c in df.columns] + \
           [f"{c}_min" for c in df.columns] + \
           [f"{c}_max" for c in df.columns] + \
           [f"{c}_slope" for c in df.columns]
    return pd.DataFrame(features, columns=cols, index=idx)

def hotelling_T2(X, pca):
    """Calcule la statistique Hotelling T² à partir d'un PCA fitted"""
    check_is_fitted(pca)
    X_proj = pca.transform(X)
    # Variances par composante
    var = np.var(X_proj, axis=0)
    T2 = np.sum((X_proj ** 2) / (var + 1e-6), axis=1)
    return T2

def zscore_univariate(df, window=60):
    """Z-score simple sur chaque capteur, max par timestamp"""
    zscores = (df - df.rolling(window).mean()) / (df.rolling(window).std() + 1e-6)
    return np.abs(zscores).max(axis=1)

# -------------------------------
# App Streamlit
# -------------------------------

st.title("🔍 Détection d'anomalies IoT (IsolationForest + PCA + Z-score)")

st.sidebar.header("⚙️ Paramètres")
window = st.sidebar.slider("Taille de fenêtre (points)", 30, 300, 60, 10)
stride = st.sidebar.slider("Stride (décalage fenêtrage)", 1, 50, 10, 1)
contamination = st.sidebar.slider("Taux contamination (IF)", 0.001, 0.1, 0.01)

uploaded_file = st.sidebar.file_uploader("Uploader un CSV (capteurs)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données")
    st.write(df.head())

    # Supposons que première colonne = temps si présente
    if not np.issubdtype(df.iloc[:,0].dtype, np.number):
        df.index = pd.to_datetime(df.iloc[:,0])
        df = df.iloc[:,1:]
    else:
        df.index = pd.RangeIndex(len(df))

    st.write(f"Shape: {df.shape}")

    # -------------------------------
    # Prétraitement
    # -------------------------------
    st.subheader("📊 Prétraitement et Features")
    X_feat = rolling_window_features(df, window=window, stride=stride)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # -------------------------------
    # Isolation Forest
    # -------------------------------
    iforest = IsolationForest(n_estimators=200, contamination=contamination, random_state=0)
    iforest.fit(X_scaled)
    score_if = -iforest.score_samples(X_scaled)

    thr_if = np.quantile(score_if, 0.995)

    # -------------------------------
    # PCA Hotelling T²
    # -------------------------------
    pca = PCA(n_components=0.95).fit(X_scaled)
    score_t2 = hotelling_T2(X_scaled, pca)
    thr_t2 = np.quantile(score_t2, 0.995)

    # -------------------------------
    # Z-score univarié
    # -------------------------------
    score_uni = zscore_univariate(df, window=window).reindex(X_feat.index)
    thr_uni = np.quantile(score_uni.dropna(), 0.995)

    # -------------------------------
    # Score global & anomalies
    # -------------------------------
    score_global = np.maximum.reduce([
        score_if / (thr_if+1e-6),
        score_t2 / (thr_t2+1e-6),
        score_uni / (thr_uni+1e-6)
    ])

    anomalies = (score_global > 1).astype(int)

    results = pd.DataFrame({
        "score_if": score_if,
        "score_t2": score_t2,
        "score_uni": score_uni,
        "score_global": score_global,
        "anomaly": anomalies
    }, index=X_feat.index)

    st.subheader("📈 Scores et anomalies")
    st.line_chart(results[["score_global"]])

    st.subheader("🚨 Anomalies détectées")
    st.write(results[results["anomaly"]==1].head(20))

    # Visualisation d’un capteur + anomalies
    cap = st.selectbox("Choisir un capteur à visualiser", df.columns)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df[cap], label=cap)
    ax.scatter(results.index[results["anomaly"]==1],
               df.loc[results.index[results["anomaly"]==1], cap],
               color="red", marker="x", label="Anomalie")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("➡️ Uploade un fichier CSV pour commencer (colonnes = capteurs, index = temps).")
