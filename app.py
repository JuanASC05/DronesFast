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
PROHIBIDOS_FUERTES = {
    "CALLAO",
    "SAN ISIDRO",
    "SAN BORJA",
    "MIRAFLORES",
    "SANTIAGO DE SURCO",
    "SURCO",
}

RESTRICCION_PARCIAL = {
    "BARRANCO",
    "LA MOLINA",
    "VILLA EL SALVADOR",
    "VES",
    "VILLA MARIA DEL TRIUNFO",
    "VMT",
    "LURIN",
    "PACHACAMAC",
}



def norm_distrito(d):
    if d is None:
        return "SIN_DISTRITO"
    return str(d).strip().upper()
# Penalizaciones para las zonas
PENALIZACION_PARCIAL_KM = 20.0       # distritos de restricción parcial
PENALIZACION_FUERTE_KM  = 200.0      # distritos “prohibidos fuertes” (muy caro)

def construir_grafo_dron(df, k_vecinos=4):
    """
    Grafo para drones:
    - Usa TODOS los nodos (no se eliminan).
    - Aristas k-NN con peso = distancia + penalización
      (según si los extremos están en distritos restringidos o prohibidos).
    Esto permite a Bellman-Ford evitar esos nodos siempre que haya alternativas,
    pero el grafo no se rompe.
    """
    df_loc = df.copy()
    df_loc["DIST_NORM"] = df_loc["DISTRITO"].apply(norm_distrito)

    G = nx.Graph()
    coords = {}
    distritos = {}

    # Nodos
    for _, row in df_loc.iterrows():
        ruc = str(row["RUC"])
        lat = float(row["LATITUD"])
        lon = float(row["LONGITUD"])
        dist = row["DIST_NORM"]
        name = row["RAZON_SOCIAL"]

        G.add_node(ruc, nombre=name, lat=lat, lon=lon, distrito=dist)
        coords[ruc] = (lat, lon)
        distritos[ruc] = dist

    rucs = list(coords.keys())

    # Aristas k-NN con penalización
    for u in rucs:
        lat_u, lon_u = coords[u]
        dist_list = []
        for v in rucs:
            if v == u:
                continue
            lat_v, lon_v = coords[v]
            d = distancia_haversine(lat_u, lon_u, lat_v, lon_v)

            extra = 0.0
            du, dv = distritos[u], distritos[v]

            # Restricción parcial
            if du in RESTRICCION_PARCIAL or dv in RESTRICCION_PARCIAL:
                extra += PENALIZACION_PARCIAL_KM

            # Prohibición fuerte (muy caro, pero no imposible)
            if du in PROHIBIDOS_FUERTES or dv in PROHIBIDOS_FUERTES:
                extra += PENALIZACION_FUERTE_KM

            dist_list.append((v, d + extra))

        dist_list.sort(key=lambda x: x[1])
        for v, w in dist_list[:k_vecinos]:
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=w)

    return G

def distancia_haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

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


# mapa con restricciones
def dibujar_mapa_ruta_dron(G, camino, origen_ruc=None, destino_ruc=None):
    """
    Mapa para la pestaña de drones:
    - Dibuja SOLO la ruta (camino) y resalta:
        * origen (verde)
        * destino (azul)
        * nodos intermedios (naranja)
    - Muestra también TODOS los nodos en distritos prohibidos fuertes en rojo
      (aunque no estén en la ruta).
    origen_ruc / destino_ruc son opcionales, por si se llama solo con (G, camino).
    """
    if not camino:
        return None

    # Centro del mapa usando solo la ruta (nodos que existan en G)
    lats = [G.nodes[n]["lat"] for n in camino if n in G.nodes]
    lons = [G.nodes[n]["lon"] for n in camino if n in G.nodes]
    if not lats or not lons:
        return None

    centro = [float(np.mean(lats)), float(np.mean(lons))]
    m = folium.Map(location=centro, zoom_start=13, control_scale=True)

    # Ruta en azul
    puntos = []
    for n in camino:
        if n in G.nodes:
            puntos.append((G.nodes[n]["lat"], G.nodes[n]["lon"]))
    folium.PolyLine(puntos, weight=4, color="blue", opacity=0.8).add_to(m)

    # Nodos prohibidos fuertes (rojo)
    nodos_prohibidos = [
        n for n, data in G.nodes(data=True)
        if data.get("distrito", "").upper() in PROHIBIDOS_FUERTES
    ]

    # Nodos a mostrar = ruta + prohibidos
    nodos_a_mostrar = set(camino) | set(nodos_prohibidos)

    for n in nodos_mostrar:
        if n not in G_base.nodes:
            continue
        data = G_base.nodes[n]
        lat, lon = data["lat"], data["lon"]
        dist = data.get("distrito", "")
        nombre = data.get("nombre", "")

        # --- Colores según tipo de nodo ---
        if n in nodos_prohibidos:
            fill = "#FF7F00"      # 🔥 naranja intenso para zonas prohibidas fuertes
        elif n == origen_ruc:
            fill = "green"        # origen
        elif n == destino_ruc:
            fill = "blue"         # destino
        elif n in camino:
            fill = "orange"       # parte de la ruta
        else:
            fill = "#8FEAF3"      # nodo normal



        popup = f"<b>{nombre}</b><br>RUC: {n}<br>Distrito: {dist}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="black",
            weight=0.8,
            fill=True,
            fill_opacity=0.95,
            fill_color=fill
        ).add_to(m).add_child(folium.Popup(popup, max_width=300))

    return m




def bellman_ford(nodes, edges, origen):
    """
    Bellman-Ford clásico:
    nodes: lista de nodos
    edges: lista de tuplas (u, v, w)
    origen: nodo origen
    Devuelve:
      dist: diccionario de distancias mínimas desde origen
      padre: predecesor de cada nodo en el mejor camino
    """
    INF = float("inf")
    dist = {n: INF for n in nodes}
    padre = {n: None for n in nodes}
    dist[origen] = 0.0

    # relajamos |V|-1 veces
    for _ in range(len(nodes) - 1):
        hubo_cambio = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                padre[v] = u
                hubo_cambio = True
        if not hubo_cambio:
            break

    # (opcional) detección de ciclo negativo; aquí no debería ocurrir
    return dist, padre


def camino_bellman_ford(G, origen, destino):
    """
    Prepara lista de aristas (u,v,w) desde un grafo NetworkX no dirigido,
    ejecuta Bellman-Ford y reconstruye el camino origen->destino.
    """
    nodes = list(G.nodes())
    edges = []

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        edges.append((u, v, w))
        edges.append((v, u, w))  # grafo no dirigido

    dist, padre = bellman_ford(nodes, edges, origen)

    if dist[destino] == float("inf"):
        return None, None  # no hay camino

    # reconstruimos el camino recorriendo los padres desde el destino
    camino = []
    actual = destino
    while actual is not None:
        camino.append(actual)
        actual = padre[actual]
    camino.reverse()

    return camino, dist[destino]


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
def dibujar_mapa_ruta(G, camino):
    """
    Mapa folium con SOLO la ruta óptima:
    - nodos en la ruta
    - líneas entre ellos
    """
    if not camino:
        return None

    # centro del mapa: promedio de las coordenadas de la ruta
    lats = [G.nodes[n]["lat"] for n in camino]
    lons = [G.nodes[n]["lon"] for n in camino]
    centro = [float(np.mean(lats)), float(np.mean(lons))]

    m = folium.Map(location=centro, zoom_start=13, control_scale=True)

    # polilínea de la ruta
    puntos = [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in camino]
    folium.PolyLine(puntos, weight=4, color="red", opacity=0.8).add_to(m)

    # nodos
    for i, n in enumerate(camino):
        nodo = G.nodes[n]
        color = "green" if i == 0 else ("blue" if i == len(camino)-1 else "orange")
        popup = f"<b>{nodo.get('nombre','')}</b><br>RUC: {n}<br>Distrito: {nodo.get('distrito','')}"
        folium.CircleMarker(
            location=[nodo["lat"], nodo["lon"]],
            radius=6,
            color="black",
            weight=0.8,
            fill=True,
            fill_opacity=0.95,
            fill_color=color
        ).add_to(m).add_child(folium.Popup(popup, max_width=300))

    return m

# ==========================
# Lógica principal
# ==========================

st.sidebar.markdown("### 📂 Base de datos fija")
st.sidebar.markdown("Usando archivo: **DataBase.xlsx**")

# Leer datos directamente del archivo local
DATA_PATH = "DataBase.xlsx"

df = pd.read_excel(DATA_PATH)

# Intentamos detectar la columna de distrito
def pick_col(df, *names):
    up = {str(c).strip().upper(): c for c in df.columns}
    for n in names:
        nU = str(n).strip().upper()
        if nU in up:
            return up[nU]
    return None

c_ruc  = pick_col(df, "RUC")
c_raz  = pick_col(df, "RAZON_SOCIAL", "RAZON SO", "RAZON_SO", "RAZON")
c_lat  = pick_col(df, "LATITUD", "LAT")
c_lon  = pick_col(df, "LONGITUD", "LON", "LONG")
c_dist = pick_col(df, "DISTRITO", "NOM_SIST", "UBIGEO_DISTRITO")

if any(x is None for x in [c_ruc, c_raz, c_lat, c_lon]):
    st.error("Faltan columnas RUC, RAZON_SOCIAL, LATITUD o LONGITUD en la base.")
    st.stop()

df = df[[c_ruc, c_raz, c_lat, c_lon] + ([c_dist] if c_dist else [])].copy()
df.columns = ["RUC", "RAZON_SOCIAL", "LATITUD", "LONGITUD"] + (["DISTRITO"] if c_dist else [])

# Normalización básica
df["RUC"] = df["RUC"].astype(str).str.strip()

for col in ["LATITUD", "LONGITUD"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["LATITUD", "LONGITUD"])
df["LATITUD"]  = df["LATITUD"].astype(float)
df["LONGITUD"] = df["LONGITUD"].astype(float)
df = df.drop_duplicates(subset=["RUC"]).reset_index(drop=True)

if "DISTRITO" in df.columns:
    df["DISTRITO"] = df["DISTRITO"].astype(str).str.strip()
else:
    df["DISTRITO"] = "SIN_DISTRITO"

st.success(f"Datos cargados desde {DATA_PATH}. Registros válidos: {len(df)}")








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



# ---------------------------------------------------------
# 🔹 TAB DRONES – Bellman-Ford + zonas restringidas
# ---------------------------------------------------------
with tab_drones:
    st.subheader("Ruta óptima con restricciones para drones (Bellman-Ford)")

    # Aseguramos tener distrito normalizado en el df completo
    df["DIST_NORM"] = df["DISTRITO"].apply(norm_distrito)
    distritos_disponibles = sorted(df["DIST_NORM"].unique())

    if len(distritos_disponibles) == 0:
        st.warning("No hay distritos en la base de datos.")
    else:
        colA, colB = st.columns(2)
        with colA:
            dist_origen = st.selectbox("Distrito de origen", distritos_disponibles)
            empresas_origen = df[df["DIST_NORM"] == dist_origen]["RAZON_SOCIAL"].tolist()
            emp_origen_nombre = st.selectbox("Empresa origen", empresas_origen)

        with colB:
            dist_destino = st.selectbox(
                "Distrito de destino",
                distritos_disponibles,
                index=min(1, len(distritos_disponibles) - 1),
            )
            empresas_destino = df[df["DIST_NORM"] == dist_destino]["RAZON_SOCIAL"].tolist()
            emp_destino_nombre = st.selectbox("Empresa destino", empresas_destino)

        # Filas de origen y destino
        origen_row  = df[df["RAZON_SOCIAL"] == emp_origen_nombre].iloc[0]
        destino_row = df[df["RAZON_SOCIAL"] == emp_destino_nombre].iloc[0]
        origen_ruc  = str(origen_row["RUC"])
        destino_ruc = str(destino_row["RUC"])

        if st.button("Calcular ruta (Bellman-Ford con zonas restringidas)"):
            # 1) Construimos el grafo especial para drones
            G_dron = construir_grafo_dron(df, k_vecinos=4)

            # 2) Intentamos Bellman-Ford en ese grafo
            camino, dist_km = camino_bellman_ford(G_dron, origen_ruc, destino_ruc)
            uso_fallback = False

            # 3) Si el grafo está desconectado para ese par,
            #    usamos fallback de ruta directa (origen -> destino)
            if not camino:
                uso_fallback = True
                lat_o, lon_o = float(origen_row["LATITUD"]), float(origen_row["LONGITUD"])
                lat_d, lon_d = float(destino_row["LATITUD"]), float(destino_row["LONGITUD"])
                dist_km = distancia_haversine(lat_o, lon_o, lat_d, lon_d)
                camino = [origen_ruc, destino_ruc]

            # --- Mensajes al usuario ---
            if uso_fallback:
                st.info(
                    "La red de nodos para drones no tenía un camino conectado entre estas empresas. "
                    "Se muestra un vuelo directo aproximado origen → destino."
                )
            else:
                st.success(
                    f"Ruta encontrada considerando penalizaciones por zonas restringidas. "
                    f"Distancia aproximada: {dist_km:.2f} km"
                )

            # --- Info de origen / destino ---
            st.markdown("### 🟢 Origen")
            st.write(f"**Empresa:** {origen_row['RAZON_SOCIAL']}")
            st.write(f"**RUC:** {origen_row['RUC']}")
            st.write(f"**Distrito:** {origen_row['DISTRITO']}")
            st.write(f"**Coordenadas:** ({origen_row['LATITUD']}, {origen_row['LONGITUD']})")

            st.markdown("### 🔵 Destino")
            st.write(f"**Empresa:** {destino_row['RAZON_SOCIAL']}")
            st.write(f"**RUC:** {destino_row['RUC']}")
            st.write(f"**Distrito:** {destino_row['DISTRITO']}")
            st.write(f"**Coordenadas:** ({destino_row['LATITUD']}, {destino_row['LONGITUD']})")

            st.markdown("### Ruta (secuencia de nodos)")
            st.write(" → ".join(camino))

            # --- 4) Mapa: ruta (azul) + nodos prohibidos (rojo) ---
            mapa_ruta = dibujar_mapa_ruta_dron(G_dron, camino, origen_ruc, destino_ruc)
            if mapa_ruta:
                st_folium(mapa_ruta, width=900, height=600)



















