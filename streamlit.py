# streamlit_app.py (versión mínima)
import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Movie Visual Explorer", layout="wide")

DATA_DIR = "data"
meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
X = np.load(os.path.join(DATA_DIR, "features.npy"))
labels = pd.read_csv(os.path.join(DATA_DIR, "cluster_labels.csv"))["cluster"].to_numpy()

def safe_img(p):
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return Image.new("RGB", (320,480), (225,225,225))

st.title("🎬 Movie Visual Explorer (Lite)")

# Filtros
col1, col2, col3 = st.columns([2,1,1])
with col1:
    q = st.text_input("🔎 Buscar título", "")
with col2:
    years = meta["year"].dropna()
    y_min = int(years.min()) if len(years) else 1900
    y_max = int(years.max()) if len(years) else 2025
    yr = st.slider("Año", y_min, y_max, (y_min, y_max))
with col3:
    all_genres = sorted({g.strip() for s in meta["genres"].fillna("") for g in s.split("|") if g})
    sel_gen = st.multiselect("Géneros", all_genres, default=[])

mask = pd.Series([True]*len(meta))
if q.strip():
    mask &= meta["title"].fillna("").str.contains(q.strip(), case=False)
mask &= meta["year"].between(yr[0], yr[1], inclusive="both")
if sel_gen:
    mask &= meta["genres"].fillna("").apply(lambda s: any(g in s.split("|") for g in sel_gen))

view = meta[mask].copy()
idx_view = view.index.to_numpy()

# (a) Búsqueda por similitud (elegir película)
st.subheader("a) Búsqueda por similitud visual")
if len(view):
    pick = st.selectbox("Película de referencia", view["title"].tolist())
    sel_idx = view.index[view["title"] == pick][0]
    nn = NearestNeighbors(n_neighbors=15, metric="cosine").fit(X)
    _, neigh = nn.kneighbors(X[sel_idx].reshape(1,-1))
    neigh_idx = neigh[0]

    ncol = 5
    rows = (len(neigh_idx) + ncol - 1)//ncol
    for r in range(rows):
        cols = st.columns(ncol)
        for j, c in enumerate(cols):
            i = r*ncol + j
            if i >= len(neigh_idx): break
            k = neigh_idx[i]
            row = meta.iloc[k]
            with c:
                st.image(safe_img(row["poster_path"]), caption=f"{row['title']} ({row.get('year','')})", use_container_width=True)
else:
    st.info("No hay resultados con los filtros actuales.")

# (b) Representantes por cluster
st.subheader("b) Representantes por cluster")
K = len(np.unique(labels))
topn = st.slider("Top-N por cluster", 3, 10, 5)
for c in sorted(np.unique(labels)):
    idx = np.where(labels == c)[0]
    centroid = X[idx].mean(axis=0, keepdims=True)
    nnc = NearestNeighbors(n_neighbors=min(topn, len(idx)), metric="cosine").fit(X[idx])
    _, local = nnc.kneighbors(centroid)
    gids = idx[local[0]]
    st.markdown(f"**Cluster {c}**")
    cols = st.columns(len(gids))
    for j, gi in enumerate(gids):
        with cols[j]:
            row = meta.iloc[gi]
            st.image(safe_img(row["poster_path"]), caption=row["title"], use_container_width=True)

# (c) Distribución 2D con PCA
st.subheader("c) Distribución 2D (PCA)")
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X)
dfz = meta.copy()
dfz["x"] = Z[:,0]; dfz["y"] = Z[:,1]; dfz["cluster"] = labels
dfz = dfz.loc[idx_view]
fig = px.scatter(dfz, x="x", y="y", color="cluster", hover_data=["title","year","genres"], height=600)
fig.update_traces(marker=dict(size=7, opacity=0.85))
st.plotly_chart(fig, use_container_width=True)

# (d) Tabla filtrable
st.subheader("d) Tabla filtrable")
st.dataframe(meta.loc[idx_view, ["title","year","genres"]].assign(cluster=labels[idx_view]), use_container_width=True)
