import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from math import radians, sin, cos, atan2, sqrt
import folium
from streamlit_folium import st_folium

# ==========================
# Funciones de apoyo
# ==========================

def distancia_haversine(lat1, lon1, lat2, lon2):
    """Distancia en km entre dos puntos usando Haversine."""
    radio_tierra = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * radio_tierra * atan2(sqrt(a), sqrt(1-a))

def construir_grafo_knn(df, k=3):
    """Construye un grafo k-NN simple a partir de lat/long."""
    G = nx.Graph()
    coords = {}

    for _, fila in df.iterrows():
        ruc = str(fila["RUC"])
        lat = float(fila["LATITUD"])
        lon = float(fila["LONGITUD"])
        G.add_node(ruc, nombre=fila["RAZON_SOCIAL"], lat=lat, lon=lon)
        coords[ruc] = (lat, lon)

    for nodo in coords:
        distancias = []
        for otro in coords:
            if otro == nodo:
                continue
            d = distancia_haversine(coords[nodo][0], coords[nodo][1],
                                    coords[otro][0], coords[otro][1])
            distancias.append((otro, d))
        distancias.sort(key=lambda x: x[1])
        for vecino, d in distancias[:k]:
            G.add_edge(nodo, vecino, weight=d)
    return G

def construir_grafo_mst(df):
    """Construye un MST sobre todas las distancias (puede ser pesado si hay demasiados nodos)."""
    G_completo = nx.Graph()
    coords = {}

    for _, fila in df.iterrows():
        ruc = str(fila["RUC"])
        lat = float(fila["LATITUD"])
        lon = float(fila["LONGITUD"])
        G_completo.add_node(ruc, nombre=fila["RAZON_SOCIAL"], lat=lat, lon=lon)
        coords[ruc] = (lat, lon)

    nodos = list(coords.keys())
    # grafo completo
    for i in range(len(nodos)):
        for j in range(i+1, len(nodos)):
            n1, n2 = nodos[i], nodos[j]
            d = distancia_haversine(coords[n1][0], coords[n1][1],
                                    coords[n2][0], coords[n2][1])
            G_completo.add_edge(n1, n2, weight=d)

    MST = nx.minimum_spanning_tree(G_completo, weight="weight", algorithm="kruskal")
    return MST

def dibujar_grafo_spring(G):
    """
    Dibujo abstracto del grafo con NetworkX.
    Usamos circular_layout para evitar cualquier dependencia con scipy.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))

    if G.number_of_nodes() > 0:
        # <<< CAMBIO CLAVE: layout que NO usa scipy >>>
        pos = nx.circular_layout(G)
    else:
        pos = {}

    nx.draw(
        G,
        pos,
        node_size=20,
        node_color="skyblue",
        edge_color="gray",
        with_labels=False,
        ax=ax,
    )
    ax.set_title(f"Grafo – {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    ax.axis("off")
    return fig



def dibujar_mapa_folium(G, camino=None, solo_ruta=False):
    """
    Mapa Folium con nodos y aristas.
    - si solo_ruta=False: muestra toda la red y resalta la ruta (si existe).
    - si solo_ruta=True: solo muestra los nodos y aristas de la ruta.
    """
    if G.number_of_nodes() == 0:
        return None

    # --- Qué nodos usar para centrar el mapa ---
    if solo_ruta and camino and len(camino) >= 1:
        nodos_centro = camino
    else:
        nodos_centro = list(G.nodes)

    lats = [G.nodes[n]["lat"] for n in nodos_centro]
    lons = [G.nodes[n]["lon"] for n in nodos_centro]
    centro = [np.mean(lats), np.mean(lons)]

    m = folium.Map(location=centro, zoom_start=12, control_scale=True)

    # --- Dibujo de aristas ---
    if solo_ruta:
        # Solo segmentos consecutivos de la ruta
        if camino and len(camino) >= 2:
            puntos = []
            for u, v in zip(camino[:-1], camino[1:]):
                lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
                lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
                folium.PolyLine(
                    [(lat1, lon1), (lat2, lon2)],
                    weight=4, color="red", opacity=0.9
                ).add_to(m)
    else:
        # Toda la red en gris
        for u, v, data in G.edges(data=True):
            lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
            lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                weight=2, opacity=0.5, color="gray"
            ).add_to(m)

    # --- Dibujo de nodos ---
    if solo_ruta and camino:
        nodos_a_mostrar = camino
    else:
        nodos_a_mostrar = list(G.nodes)

    for idx, n in enumerate(nodos_a_mostrar):
        attr = G.nodes[n]
        popup = f"<b>{attr.get('nombre','')}</b><br>RUC: {n}"
        # Origen y destino con color distinto
        if solo_ruta and camino:
            if n == camino[0]:
                fill = "green"
            elif n == camino[-1]:
                fill = "blue"
            else:
                fill = "#8FEAF3"
        else:
            fill = "#8FEAF3"

        folium.CircleMarker(
            location=[attr["lat"], attr["lon"]],
            radius=5, fill=True, fill_opacity=0.95,
            color="black", weight=0.7, fill_color=fill
        ).add_to(m).add_child(folium.Popup(popup, max_width=250))

    # Si no es solo_ruta pero hay camino, añadimos polyline roja encima
    if (not solo_ruta) and camino and len(camino) >= 2:
        puntos = [(G.nodes[r]["lat"], G.nodes[r]["lon"]) for r in camino]
        folium.PolyLine(
            puntos, weight=5, color="red", opacity=0.9
        ).add_to(m)

    return m



def calcular_ruta_dijkstra(G, origen, destino):
    try:
        camino = nx.shortest_path(G, source=origen, target=destino, weight="weight")
        longitud = nx.shortest_path_length(G, source=origen, target=destino, weight="weight")
        return camino, longitud
    except nx.NetworkXNoPath:
        return None, None

# ==========================
# Configuración de Streamlit
# ==========================

st.set_page_config(page_title="Optimizador Logístico Courier", layout="wide")
st.title("📦 Optimizador Logístico Courier con Grafos")

st.sidebar.header("⚙️ Configuración del aplicativo")

tipo_grafo = st.sidebar.selectbox(
    "Tipo de grafo",
    ["k-NN", "MST"],
    key="sb_tipo_grafo"
)

k_vecinos = st.sidebar.slider(
    "k vecinos (solo k-NN)",
    min_value=1,
    max_value=6,
    value=3,
    step=1,
    key="sb_k_vecinos"
)

submuestro = st.sidebar.checkbox(
    "Usar submuestreo visual",
    value=True,
    key="sb_submuestreo"
)

n_max = st.sidebar.slider(
    "Máx. nodos a visualizar",
    min_value=100,
    max_value=1500,
    value=400,
    step=100,
    key="sb_n_max"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Análisis disponibles")
activar_ruta = st.sidebar.checkbox("Ruta óptima (Dijkstra)", key="sb_ruta")
activar_hubs = st.sidebar.checkbox("Hubs (betweenness)", key="sb_hubs")
activar_falla = st.sidebar.checkbox("Simulación de falla", key="sb_falla")
activar_drones = st.sidebar.checkbox("Escenario con drones", key="sb_drones")

# ==========================
# Lógica principal
# ==========================

# ==========================
# Lógica principal
# ==========================

st.sidebar.markdown("### 📂 Base de datos fija")
st.sidebar.markdown("Usando archivo: **DataBase.xlsx**")

# Leer datos directamente del archivo local
DATA_PATH = "DataBase.xlsx"   # nombre de tu archivo en la misma carpeta

try:
    df = pd.read_excel(DATA_PATH)
except FileNotFoundError:
    st.error(f"No se encontró el archivo {DATA_PATH}. Asegúrate de que esté en la misma carpeta que app.py.")
    st.stop()


df = pd.read_excel(DATA_PATH)

# Nos quedamos solo con las columnas necesarias
df = df[["RUC", "RAZON_SOCIAL", "LATITUD", "LONGITUD"]].copy()

# RUC siempre como texto
df["RUC"] = df["RUC"].astype(str).str.strip()

# --- LIMPIEZA ROBUSTA DE COORDENADAS ---
for col in ["LATITUD", "LONGITUD"]:
    # Pasamos a texto, quitamos espacios y cambiamos coma por punto
    df[col] = (
        df[col]
        .astype(str)           # por si vienen números mezclados con texto
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    # Convertimos a número; lo que no se pueda se vuelve NaN
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Eliminamos filas sin coordenadas válidas
df = df.dropna(subset=["LATITUD", "LONGITUD"])

# Aseguramos tipo float y quitamos RUC duplicados
df["LATITUD"]  = df["LATITUD"].astype(float)
df["LONGITUD"] = df["LONGITUD"].astype(float)
df = df.drop_duplicates(subset=["RUC"]).reset_index(drop=True)

st.success(f"Datos cargados correctamente desde {DATA_PATH}. Registros válidos: {len(df)}")








if submuestro and len(df) > n_max:
    df_vis = df.sample(n_max, random_state=42).reset_index(drop=True)
else:
    df_vis = df.copy()

df_vis["NOMBRE_EMPRESA"] = df_vis["RAZON_SOCIAL"].astype(str)
emp_a_ruc = dict(zip(df_vis["NOMBRE_EMPRESA"], df_vis["RUC"]))

# Construir grafo según selección
if tipo_grafo == "k-NN":
    G = construir_grafo_knn(df_vis, k=k_vecinos)
else:
    st.warning("MST puede demorar si hay muchos nodos; úsalo con <= 200 nodos.")
    G = construir_grafo_mst(df_vis)



# ==========================
# Tabs de interfaz
# ==========================

tab_dataset, tab_grafo, tab_mapa, tab_rutas, tab_hubs, tab_fallas, tab_drones = st.tabs(
    ["📄 Dataset", "🕸 Grafo", "🗺 Mapa", "🧭 Rutas", "⭐ Hubs", "⚠️ Fallas", "🚁 Drones"]
)

# -------- Tab Dataset --------
with tab_dataset:
    st.subheader("Vista del dataset")
    st.dataframe(df.head(20))
    st.write(f"Total de nodos (RUC únicos): {df['RUC'].nunique()}")

# -------- Tab Grafo --------
with tab_grafo:
    st.subheader("Grafo (vista abstracta)")
    fig = dibujar_grafo_spring(G)
    st.pyplot(fig)

# -------- Tab Mapa --------
with tab_mapa:
    st.subheader("Grafo georreferenciado")
    mapa = dibujar_mapa_folium(G)
    if mapa:
        st_folium(mapa, width=900, height=600)
    else:
        st.warning("No se pudo construir el mapa.")

# -------- Tab Rutas --------
with tab_rutas:
    st.subheader("Cálculo de ruta óptima (Dijkstra)")
    if not activar_ruta:
        st.info("Activa 'Ruta óptima (Dijkstra)' en la barra lateral.")
    else:
        if G.number_of_nodes() < 2:
            st.warning("No hay suficientes nodos para calcular rutas.")
        else:
            # Opciones por nombre de empresa
            opciones_empresas = sorted(df_vis["NOMBRE_EMPRESA"].unique())

            col1, col2 = st.columns(2)
            with col1:
                origen_nombre = st.selectbox(
                    "Empresa origen", opciones_empresas, key="ruta_origen"
                )
            with col2:
                opciones_destino = [e for e in opciones_empresas if e != origen_nombre]
                destino_nombre = st.selectbox(
                    "Empresa destino", opciones_destino, key="ruta_destino"
                )

            # Convertir nombre -> RUC (que es el ID del nodo en el grafo)
            origen_ruc = str(emp_a_ruc[origen_nombre])
            destino_ruc = str(emp_a_ruc[destino_nombre])

            if st.button("Calcular ruta"):
                camino, dist_km = calcular_ruta_dijkstra(G, origen_ruc, destino_ruc)
                if camino:
                    st.success(
                        f"Camino encontrado ({len(camino)} nodos), distancia aprox: {dist_km:.2f} km"
                    )

                    # --- Info completa de origen y destino ---
                    info_origen = df_vis[df_vis["RUC"].astype(str) == str(origen_ruc)].iloc[0]
                    info_destino = df_vis[df_vis["RUC"].astype(str) == str(destino_ruc)].iloc[0]


                    col_o, col_d = st.columns(2)
                    with col_o:
                        st.markdown("### 🟢 Origen")
                        st.write(f"**Empresa:** {info_origen['RAZON_SOCIAL']}")
                        st.write(f"**RUC:** {info_origen['RUC']}")
                        st.write(
                            f"**Coordenadas:** ({info_origen['LATITUD']:.5f}, {info_origen['LONGITUD']:.5f})"
                        )
                    with col_d:
                        st.markdown("### 🔵 Destino")
                        st.write(f"**Empresa:** {info_destino['RAZON_SOCIAL']}")
                        st.write(f"**RUC:** {info_destino['RUC']}")
                        st.write(
                            f"**Coordenadas:** ({info_destino['LATITUD']:.5f}, {info_destino['LONGITUD']:.5f})"
                        )

                    st.markdown("#### Ruta (secuencia de nodos)")
                    st.write(" → ".join(camino))

                    # --- Mapa con la ruta óptima resaltada ---
                    mapa_ruta = dibujar_mapa_folium(G, camino=camino, solo_ruta=True)
                    st_folium(mapa_ruta, width=900, height=600)

                else:
                    st.error("No existe ruta entre esas empresas en el grafo.")


# -------- Tab Hubs --------
with tab_hubs:
    st.subheader("Análisis de hubs logísticos")
    if not activar_hubs:
        st.info("Activa 'Hubs (betweenness)' en la barra lateral.")
    else:
        if G.number_of_nodes() == 0:
            st.warning("No hay nodos.")
        else:
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
            df_bc = pd.DataFrame([
                {"RUC": n, "Razon_Social": G.nodes[n].get("nombre",""), "Betweenness": v}
                for n, v in bc.items()
            ])
            df_bc = df_bc.sort_values("Betweenness", ascending=False).head(10)
            st.write("Top 10 nodos por betweenness:")
            st.dataframe(df_bc)

# -------- Tab Fallas --------
with tab_fallas:
    st.subheader("Simulación de falla de nodo")
    if not activar_falla:
        st.info("Activa 'Simulación de falla' en la barra lateral.")
    else:
        nodos = list(G.nodes)
        if len(nodos) == 0:
            st.warning("No hay nodos para simular.")
        else:
            victima = st.selectbox("Nodo a desactivar (RUC)", nodos)
            if st.button("Simular falla"):
                G_fail = G.copy()
                G_fail.remove_node(victima)
                comps = list(nx.connected_components(G_fail))
                st.write(f"Componentes conectados tras la falla: {len(comps)}")
                st.write(f"Ejemplo de componente aislado (si existe): {list(comps[0])[:10]}")
                # Aquí podrías añadir un mapa post-falla si quieres.

# -------- Tab Drones --------
with tab_drones:
    st.subheader("Escenario de uso de drones")
    if not activar_drones:
        st.info("Activa 'Escenario con drones' en la barra lateral.")
    else:
        st.write("Aquí puedes estimar energía, autonomía y comparar rutas con vs sin dron.")
        # TODO: aquí enchufas tus fórmulas de Wh/km, autonomía, etc.
        st.markdown("- Ejemplo: consumo base 15 Wh/km + 1.2 Wh/km·kg por carga.")
        st.markdown("- Autonomía estimada: distancia máxima antes de recarga.")

st.sidebar.header("⚙️ Configuración del aplicativo")

tipo_grafo = st.sidebar.selectbox("Tipo de grafo", ["k-NN", "MST"])
k_vecinos = st.sidebar.slider("k vecinos (solo k-NN)", 1, 6, 3)









