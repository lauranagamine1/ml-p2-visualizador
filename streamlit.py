# streamlit_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Recomendación de películas", layout="wide")

# === Rutas por defecto (sin barra lateral) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FEATS  = os.path.join(BASE_DIR, "data", "movies_streamlit.csv")
CSV_JOINED = os.path.join(BASE_DIR, "data", "movies_streamlit_joined.csv")

# --- Cargar CSV base (features) ---
if not os.path.exists(CSV_FEATS):
    st.error(f"No encuentro el archivo: {CSV_FEATS}")
    st.stop()

df = pd.read_csv(CSV_FEATS)
required = {"movieId", "cluster"}
missing = required - set(df.columns)
if missing:
    st.error(f"Faltan columnas requeridas en el CSV de features: {missing}")
    st.stop()

# --- Cargar CSV joined (metadata) ---
if not os.path.exists(CSV_JOINED):
    st.warning(f"No encuentro el archivo joined: {CSV_JOINED}. Se mostrará solo ID/cluster.")
    joined_meta = None
else:
    joined_meta = pd.read_csv(CSV_JOINED)
    id_candidates = ["movieId", "movieId_ms", "movieid", "movieid_ms"]
    id_meta_col = next((c for c in id_candidates if c in joined_meta.columns), None)
    if id_meta_col is None:
        st.warning(f"No se encontró columna de ID en joined. Columnas: {list(joined_meta.columns)}")
        joined_meta = None

# === Separar meta / labels / features del CSV base ===
meta = df[["movieId"]].copy()
labels = df["cluster"].to_numpy()
feat_cols = [c for c in df.columns if c.startswith("f")]
if not feat_cols:
    st.error("No se encontraron columnas de features (f0, f1, ...).")
    st.stop()
X = df[feat_cols].to_numpy(dtype=float)

# === Preparar mapeos de metadata (Title, Poster, Genre) por movieId ===
title_map = poster_map = genre_map = {}
if joined_meta is not None:
    ids_series = pd.to_numeric(joined_meta[id_meta_col], errors="coerce")
    valid = joined_meta.loc[~ids_series.isna()].copy()
    valid["_id_int"] = ids_series.dropna().astype(int).values

    title_map  = dict(zip(valid["_id_int"], valid.get("Title",  pd.Series(index=valid.index, dtype=object))))
    poster_map = dict(zip(valid["_id_int"], valid.get("Poster", pd.Series(index=valid.index, dtype=object))))
    genre_map  = dict(zip(valid["_id_int"], valid.get("Genre",  pd.Series(index=valid.index, dtype=object))))

def enrich_with_meta(df_ids: pd.DataFrame) -> pd.DataFrame:
    """Añade Title, Genre, Poster a un DataFrame que ya tiene columna 'movieId'."""
    if not title_map:
        return df_ids
    out = df_ids.copy()
    mid_int = pd.to_numeric(out["movieId"], errors="coerce").astype("Int64")
    out["Title"]  = mid_int.map(lambda x: title_map.get(int(x))  if pd.notna(x) else None)
    out["Genre"]  = mid_int.map(lambda x: genre_map.get(int(x))  if pd.notna(x) else None)
    out["Poster"] = mid_int.map(lambda x: poster_map.get(int(x)) if pd.notna(x) else None)
    return out

def title_for_id(mid):
    if not title_map: return None
    try:
        return title_map.get(int(mid))
    except Exception:
        return None

# ====== UI ======
st.title("🎬 Movie Explorer (búsqueda por ID o Título)")

# Filtros de cluster + top-N
colf1, colf2 = st.columns([1,1])
with colf1:
    clusters_disponibles = sorted(np.unique(labels).tolist())
    sel_clusters = st.multiselect("Filtrar clusters", clusters_disponibles, default=clusters_disponibles)
with colf2:
    k_sim = st.slider("N° similares a mostrar", 5, 30, 15, 1)

mask = np.isin(labels, sel_clusters)
view_idx = np.where(mask)[0]
view_ids = meta.iloc[view_idx]["movieId"].tolist()

# Lista de títulos visibles (según clusters) si hay joined
view_titles = []
if title_map:
    for mid in view_ids:
        t = title_for_id(mid)
        if t is not None:
            view_titles.append(t)
    # quitar duplicados manteniendo orden
    seen = set()
    view_titles = [t for t in view_titles if not (t in seen or seen.add(t))]

# (a) Búsqueda por similitud (ID o Título)
st.subheader("a) Búsqueda por similitud")
mode = st.radio("Modo de búsqueda", ["Por movieId", "Por Title"], horizontal=True)

query_id = None
if mode == "Por movieId":
    colid1, colid2 = st.columns([2,1])
    with colid1:
        q_id_str = st.text_input("movieId exacto (o elige de la lista):", "")
    with colid2:
        chosen_from_list = st.selectbox(
            "Elegir ID de la lista filtrada",
            view_ids, index=0 if len(view_ids) > 0 else None
        )

    if q_id_str.strip():
        try:
            q_try = int(q_id_str.strip())
            if q_try in meta["movieId"].values:
                query_id = q_try
            else:
                st.warning(f"El movieId {q_try} no existe en el CSV.")
        except ValueError:
            st.warning("El movieId ingresado no es un entero válido.")
    elif len(view_ids) > 0:
        query_id = chosen_from_list

else:  # Por Title
    colt1, colt2 = st.columns([2,1])
    with colt1:
        q_title = st.text_input("Título exacto (o elige de la lista):", "")
    with colt2:
        chosen_title = st.selectbox(
            "Elegir título (filtrado por clusters)",
            view_titles if view_titles else ["(sin títulos disponibles)"],
            index=0 if len(view_titles) > 0 else None
        )

    title_to_use = q_title.strip() if q_title.strip() else (chosen_title if view_titles else None)
    if title_to_use and title_map:
        # Encontrar un movieId cuyo título coincida (preferimos IDs dentro del filtro actual)
        candidate_ids = [mid for mid in view_ids if title_for_id(mid) == title_to_use]
        if not candidate_ids:
            # como fallback, buscar en todo el mapa
            candidate_ids = [mid for mid, t in title_map.items() if t == title_to_use]
        if candidate_ids:
            query_id = candidate_ids[0]
        else:
            st.warning("No se encontró ese título.")

# Ejecutar kNN si tenemos query_id
if query_id is not None:
    sel_idx_series = meta.index[meta["movieId"] == query_id]
    if len(sel_idx_series) == 0:
        st.warning("No se encontró el ID seleccionado.")
    else:
        sel_idx = sel_idx_series[0]
        n_neighbors = min(k_sim, len(X))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(X)
        _, neigh = nn.kneighbors(X[sel_idx].reshape(1, -1))
        neigh_idx = neigh[0]

        recs = pd.DataFrame({
            "#": np.arange(1, len(neigh_idx)+1, dtype=int),
            "movieId": meta.iloc[neigh_idx]["movieId"].values,
            "cluster": labels[neigh_idx]
        })
        recs = enrich_with_meta(recs)

        st.dataframe(
            recs,
            use_container_width=True,
            column_config={
                "Poster": st.column_config.ImageColumn(
                    "Poster",
                    help="Vista previa",
                    width="large"  # más grande en (a)
                )
            }
        )
else:
    st.info("Ingresa un movieId o un Title (o elige de las listas) para ver similares.")

# (b) Representantes por cluster (con metadata)
st.subheader("b) Representantes por cluster")
topn = st.slider("Top-N por cluster", 3, 12, 5, 1)
for c in sorted(np.unique(labels)):
    idx = np.where(labels == c)[0]
    if len(idx) == 0:
        continue
    centroid = X[idx].mean(axis=0, keepdims=True)
    nnc = NearestNeighbors(n_neighbors=min(topn, len(idx)), metric="cosine").fit(X[idx])
    _, local = nnc.kneighbors(centroid)
    gids = idx[local[0]]
    st.markdown(f"**Cluster {c}**")
    reps = pd.DataFrame({
        "#": np.arange(1, len(gids)+1, dtype=int),
        "movieId": meta.iloc[gids]["movieId"].values
    })
    reps = enrich_with_meta(reps)
    st.dataframe(
        reps,
        use_container_width=True,
        column_config={"Poster": st.column_config.ImageColumn(width="small")}
    )

# (c) Distribución 2D (PCA) con hover enriquecido
st.subheader("c) Distribución 2D (PCA)")
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X)

title_hover = None
if title_map:
    mid_int_all = pd.to_numeric(meta["movieId"], errors="coerce").astype("Int64")
    title_hover = mid_int_all.map(lambda x: title_map.get(int(x)) if pd.notna(x) else None)

dfz = pd.DataFrame({
    "x": Z[:, 0],
    "y": Z[:, 1],
    "cluster": labels,
    "movieId": meta["movieId"].values,
})
if title_hover is not None:
    dfz["Title"] = title_hover

dfz_view = dfz[dfz["cluster"].isin(sel_clusters)]
hover_cols = ["movieId", "cluster"] + (["Title"] if "Title" in dfz_view.columns else [])
fig = px.scatter(
    dfz_view, x="x", y="y", color="cluster",
    hover_data=hover_cols, height=600
)
fig.update_traces(marker=dict(size=7, opacity=0.85))
st.plotly_chart(fig, use_container_width=True)

# (d) Tabla filtrable (IDs + cluster + metadata)
st.subheader("d) Tabla filtrable (IDs en vista)")
tabla_view = pd.DataFrame({
    "movieId": meta.iloc[view_idx]["movieId"].values,
    "cluster": labels[view_idx]
})
tabla_view = enrich_with_meta(tabla_view)
st.dataframe(
    tabla_view,
    use_container_width=True,
    column_config={"Poster": st.column_config.ImageColumn(width="medium")}
)
