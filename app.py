import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from math import radians, sin, cos, atan2, sqrt
import folium
from streamlit_folium import st_folium
import json
import requests
import matplotlib.pyplot as plt
from collections import defaultdict

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


# ========================== CONSTANTES ==========================
ZONAS_RESTRINGIDAS = [
    "San Borja", "San Isidro", "Cercado de Lima", "Miraflores", "Barranco",
    "La Molina", "Santiago de Surco"
]

fallbacks = {
    "San Borja": [[-12.0838, -76.9998], [-12.0839, -76.9903], [-12.0918, -76.9899], [-12.1039, -76.9811],
                  [-12.1080, -76.9835], [-12.1140, -76.9898], [-12.1104, -76.9987], [-12.1019, -77.0038],
                  [-12.0910, -77.0051], [-12.0840, -77.0015], [-12.0838, -76.9998]],
    "San Isidro": [[-12.0858, -77.0370], [-12.0954, -77.0401], [-12.1020, -77.0363], [-12.1065, -77.0275],
                   [-12.1086, -77.0200], [-12.1041, -77.0160], [-12.0958, -77.0152], [-12.0886, -77.0175],
                   [-12.0853, -77.0259], [-12.0838, -77.0332], [-12.0858, -77.0370]],
    "Cercado de Lima": [[-12.0300, -77.0500], [-12.0400, -77.0600], [-12.0500, -77.0700], [-12.0600, -77.0800],
                        [-12.0700, -77.0900], [-12.0800, -77.1000], [-12.0900, -77.1100], [-12.1000, -77.1200],
                        [-12.0300, -77.0500]],
    "Miraflores": [[-12.1100, -77.0400], [-12.1200, -77.0300], [-12.1300, -77.0200], [-12.1400, -77.0100],
                   [-12.1500, -77.0000], [-12.1600, -76.9900], [-12.1700, -76.9800], [-12.1800, -76.9700],
                   [-12.1100, -77.0400]],
    "Barranco": [[-12.1400, -77.0200], [-12.1500, -77.0100], [-12.1600, -77.0000], [-12.1700, -76.9900],
                 [-12.1800, -76.9800], [-12.1900, -76.9700], [-12.2000, -76.9600], [-12.2100, -76.9500],
                 [-12.1400, -77.0200]],
    "La Molina": [[-12.0635, -77.0032], [-12.065, -76.995], [-12.072, -76.985], [-12.080, -76.970],
                  [-12.088, -76.958], [-12.098, -76.945], [-12.112, -76.935], [-12.130, -76.922],
                  [-12.150, -76.915], [-12.165, -76.920], [-12.165, -76.950], [-12.155, -76.980],
                  [-12.145, -77.000], [-12.125, -77.005], [-12.100, -77.005], [-12.075, -77.000],
                  [-12.0635, -77.0032]],
    "Santiago de Surco": [
        [-12.126386, -77.0026989],
        [-12.1284408, -77.0012861],
        [-12.126386, -77.0026989]
    ]
}

with open("lima_government.geojson", "r", encoding="utf-8") as f:
    gov_data = json.load(f)

with open("lima_health.geojson", "r", encoding="utf-8") as f:
    GEO_SALUD = json.load(f)

# ========================== FUNCIONES AUXILIARES ==========================
def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def is_in_district(lat, lon, polys):
    if not polys:
        return False
    outer = polys[0]
    inners = polys[1:]
    in_outer = point_in_polygon(lon, lat, outer)
    in_inners = any(point_in_polygon(lon, lat, inner) for inner in inners)
    return in_outer and not in_inners
    
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
def is_in_zona_critica(lat, lon):
    lat_penta = -12.100201829561424
    lon_penta = -76.9868184334633
    return distancia_haversine(lat, lon, lat_penta, lon_penta) <= 1.80
def obtener_limites_distrito(nombre):
    return fallbacks.get(nombre)

def obtener_boundary_osm(distrito):
    rel_ids = {
        "San Borja": 1944802,
        "San Isidro": 1944812,
        "Cercado de Lima": 1944756,
        "Miraflores": 1944770,
        "Barranco": 1944691,
        "La Molina": 1944745,
        "Santiago de Surco": 1944798
    }

    # Si no hay ID para el distrito, devolver fallback (si existe)
    if distrito not in rel_ids:
        return [fallbacks.get(distrito)] if distrito in fallbacks else None

    relation_id = rel_ids[distrito]
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:180];
    relation({relation_id});
    (._;>;);
    out body;
    """

    try:
        response = requests.post(overpass_url, data={'data': query}, timeout=30)
        # Si falla la petición, usar fallback
        if response.status_code != 200:
            return [fallbacks.get(distrito)]

        data = response.json()
        if "elements" not in data:
            return [fallbacks.get(distrito)]

        # Mapear nodos y ways
        nodes = {el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"] == "node"}
        ways = {el["id"]: el["nodes"] for el in data["elements"] if el["type"] == "way"}

        # Buscar la relation
        rels = [el for el in data["elements"] if el["type"] == "relation"]
        if not rels:
            return [fallbacks.get(distrito)]
        rel = rels[0]

        polys = []
        # Construir anillos para roles 'outer' y 'inner'
        for role in ['outer', 'inner']:
            members = [m for m in rel.get("members", []) if m.get("role") == role and m.get("type") == "way"]
            if not members:
                continue

            way_nodes = {m["ref"]: ways[m["ref"]] for m in members if m["ref"] in ways}
            if not way_nodes:
                continue

            # Armado del ring
            start_to_way = defaultdict(list)
            end_to_way = defaultdict(list)
            for w_id, nlist in way_nodes.items():
                start_to_way[nlist[0]].append(w_id)
                end_to_way[nlist[-1]].append(w_id)

            current_way = list(way_nodes.keys())[0]
            polygon_nodes = way_nodes[current_way][:]
            used = {current_way}
            current_end = polygon_nodes[-1]

            while True:
                next_way = None
                reverse = False
                # Buscar siguiente way que empiece por current_end
                for w in start_to_way.get(current_end, []):
                    if w not in used:
                        next_way = w
                        break
                if not next_way:
                    # Buscar way que termine en current_end (y hacer reverse)
                    for w in end_to_way.get(current_end, []):
                        if w not in used:
                            next_way = w
                            reverse = True
                            break
                if not next_way:
                    break

                used.add(next_way)
                next_nodes = way_nodes[next_way][:]
                if reverse:
                    next_nodes = next_nodes[::-1]
                polygon_nodes.extend(next_nodes[1:])
                current_end = polygon_nodes[-1]

            # Cerrar anillo si es necesario
            if polygon_nodes and polygon_nodes[0] != polygon_nodes[-1]:
                polygon_nodes.append(polygon_nodes[0])

            # Convertir node ids a coordenadas (lat, lon)
            coords = []
            for node_id in polygon_nodes:
                if node_id in nodes:
                    lon, lat = nodes[node_id]
                    coords.append([lat, lon])

            # Agregar solo si hay coordenadas
            if coords:
                polys.append(coords)

        # --- Validar polígonos: deben tener al menos 4 puntos (triángulo cerrado = 4) ---
        polys_validos = [p for p in polys if p and len(p) >= 4]

        # Si no hay polígonos válidos, retornar fallback (si existe)
        # Para Surco, usar fallback sí o sí
        if distrito == "Santiago de Surco":
            return [fallbacks["Santiago de Surco"]]

        if not polys_validos:
            return [fallbacks.get(distrito)]

        return polys_validos

    except Exception as e:
        try:
            st.warning(f"obtener_boundary_osm({distrito}) fallback por excepción: {str(e)}")
        except:
            pass
        return [fallbacks.get(distrito)]
        
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



def dibujar_mapa_folium(
    G,
    camino=None,
    solo_ruta=False,
    dibujar_aristas=True,   # <--- nuevo parámetro opcional
):
    """
    Mapa Folium con nodos y aristas.
    - si solo_ruta=False: muestra toda la red y resalta la ruta (si existe).
    - si solo_ruta=True: solo muestra los nodos y aristas de la ruta.
    - si dibujar_aristas=False: solo dibuja nodos (sin líneas).
    """
    if G.number_of_nodes() == 0:
        return None

    # Qué nodos usar para centrar el mapa
    if solo_ruta and camino and len(camino) >= 1:
        nodos_centro = camino
    else:
        nodos_centro = list(G.nodes)

    lats = [G.nodes[n]["lat"] for n in nodos_centro]
    lons = [G.nodes[n]["lon"] for n in nodos_centro]
    centro = [float(np.mean(lats)), float(np.mean(lons))]

    m = folium.Map(location=centro, zoom_start=12, control_scale=True)

    # --- Aristas ---
    if dibujar_aristas:
        if solo_ruta:
            # Solo tramos de la ruta
            if camino and len(camino) >= 2:
                for u, v in zip(camino[:-1], camino[1:]):
                    lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
                    lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
                    folium.PolyLine(
                        [(lat1, lon1), (lat2, lon2)],
                        weight=4, color="red", opacity=0.9
                    ).add_to(m)
        else:
            # Toda la red
            for u, v, data in G.edges(data=True):
                lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
                lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
                folium.PolyLine(
                    [(lat1, lon1), (lat2, lon2)],
                    weight=2, opacity=0.5, color="gray"
                ).add_to(m)

    # --- Nodos ---
    if solo_ruta and camino:
        nodos_a_mostrar = camino
    else:
        nodos_a_mostrar = list(G.nodes)

    for n in nodos_a_mostrar:
        attr = G.nodes[n]
        popup = f"<b>{attr.get('nombre','')}</b><br>RUC: {n}"
        folium.CircleMarker(
            location=[attr["lat"], attr["lon"]],
            radius=5,
            fill=True,
            fill_opacity=0.95,
            color="black",
            weight=0.7,
            fill_color="#8FEAF3",
        ).add_to(m).add_child(folium.Popup(popup, max_width=250))

    # Si no es solo_ruta pero hay camino, resaltamos la ruta encima
    if (not solo_ruta) and camino and len(camino) >= 2:
        puntos = [(G.nodes[r]["lat"], G.nodes[r]["lon"]) for r in camino]
        folium.PolyLine(
            puntos, weight=5, color="red", opacity=0.9
        ).add_to(m)

    return m


# mapa con restricciones
def dibujar_mapa_ruta_dron(G_base, camino, origen_ruc, destino_ruc):
    """
    Mapa Folium que muestra:
      - Solo los nodos de la ruta + nodos prohibidos fuertes.
      - Ruta en azul.
      - Nodos prohibidos fuertes en naranja.
      - Origen en verde, destino en azul, nodos intermedios en naranja.
    """
    if not camino:
        return None

    # Centro del mapa usando solo la ruta
    lats = [G_base.nodes[n]["lat"] for n in camino if n in G_base.nodes]
    lons = [G_base.nodes[n]["lon"] for n in camino if n in G_base.nodes]
    if not lats or not lons:
        return None

    centro = [float(np.mean(lats)), float(np.mean(lons))]
    m = folium.Map(location=centro, zoom_start=13, control_scale=True)

    # --- Línea de la ruta (azul) ---
    puntos = []
    for n in camino:
        if n in G_base.nodes:
            puntos.append((G_base.nodes[n]["lat"], G_base.nodes[n]["lon"]))
    folium.PolyLine(puntos, weight=4, color="blue", opacity=0.85).add_to(m)

    # --- Nodos prohibidos fuertes (según distrito) ---
    nodos_prohibidos = [
        n for n, data in G_base.nodes(data=True)
        if data.get("distrito", "").upper() in PROHIBIDOS_FUERTES
    ]

    # --- Nodos a mostrar: ruta + prohibidos ---
    nodos_mostrar = set(camino) | set(nodos_prohibidos)

    for n in nodos_mostrar:
        if n not in G_base.nodes:
            continue
        data = G_base.nodes[n]
        lat, lon = data["lat"], data["lon"]
        dist = data.get("distrito", "")
        nombre = data.get("nombre", "")

        # Colores según tipo de nodo
        if n in nodos_prohibidos:
            fill = "#FF7F00"       # naranja fuerte para zonas prohibidas
        elif n == origen_ruc:
            fill = "green"         # origen
        elif n == destino_ruc:
            fill = "blue"          # destino
        elif n in camino:
            fill = "orange"        # parte de la ruta
        else:
            fill = "#8FEAF3"       # nodo normal

        popup = f"<b>{nombre}</b><br>RUC: {n}<br>Distrito: {dist}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="black",
            weight=0.8,
            fill=True,
            fill_opacity=0.95,
            fill_color=fill,
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
def agregar_zonas_restringidas(mapa):
    # === CÍRCULO DEL PENTAGONITO (1.8 km) ===
    # (si quieres agregar círculo, puedes hacerlo aquí; lo dejé fuera por claridad)

    # === INSTALACIONES DE SALUD (generalmente puntos) ===
    for feat in GEO_SALUD["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        name = props.get("name") or props.get("nombre") or props.get("NOMBRE") or "Instalación de Salud"
        
        if geom["type"] == "Point":
            lon, lat = geom["coordinates"]
            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="#ff0000",
                fill_color="#ff6666",
                fill_opacity=0.8,
                weight=2,
                tooltip=name,
                popup=folium.Popup(f"<b style='color:red'>{name}</b><br>Zona restringida - Salud", max_width=300)
            ).add_to(mapa)
        
        elif geom["type"] in ["Polygon", "MultiPolygon"]:
            poly_list = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly_rings in poly_list:
                rings_latlng = []
                for ring in poly_rings:
                    ring_latlng = [[lat, lon] for lon, lat in ring]
                    rings_latlng.append(ring_latlng)
                
                folium.Polygon(
                    locations=rings_latlng,
                    color="#ff0000",
                    weight=2,
                    fill=True,
                    fill_color="#ff6666",
                    fill_opacity=0.45,
                    tooltip=name,
                    popup=folium.Popup(f"<b style='color:red'>{name}</b><br>Zona restringida - Salud", max_width=300)
                ).add_to(mapa)

    # === EDIFICIOS GUBERNAMENTALES ===
    for feat in gov_data["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        name = props.get("name") or props.get("nombre") or props.get("NOMBRE") or "Edificio Gubernamental"
        
        if geom["type"] == "Point":
            lon, lat = geom["coordinates"]
            folium.CircleMarker(
                location=[lat, lon],
                radius=11,
                color="#8b0000",
                fill_color="#dc143c",
                fill_opacity=0.9,
                weight=3,
                tooltip=name,
                popup=folium.Popup(f"<b style='color:#8b0000'>{name}</b><br>EDIFICIO GUBERNAMENTAL", max_width=300)
            ).add_to(mapa)
        
        elif geom["type"] in ["Polygon", "MultiPolygon"]:
            poly_list = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly_rings in poly_list:
                rings_latlng = []
                for ring in poly_rings:
                    ring_latlng = [[lat, lon] for lon, lat in ring]
                    rings_latlng.append(ring_latlng)
                
                folium.Polygon(
                    locations=rings_latlng,
                    color="#8b0000",
                    weight=4,
                    fill=True,
                    fill_color="#dc143c",
                    fill_opacity=0.55,
                    tooltip=name,
                    popup=folium.Popup(f"<b style='color:#8b0000'>{name}</b><br>EDIFICIO GUBERNAMENTAL", max_width=300)
                ).add_to(mapa)

    # === DISTRITOS RESTRINGIDOS (fondo suave) ===
    for distrito in ZONAS_RESTRINGIDAS:
        polys = obtener_boundary_osm(distrito)
        if polys:
            for poly in polys:
                folium.Polygon(
                    locations=poly,
                    color="#ff3333",
                    weight=2,
                    fill=True,
                    fill_color="#ff3333",
                    fill_opacity=0.18,
                    tooltip=f"Distrito restringido: {distrito}"
                ).add_to(mapa)
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
    st.subheader("Grafo georreferenciado con zonas reguladas")

    # Mapa base con nodos (sin aristas para que no se sature)
    mapa = dibujar_mapa_folium(G, dibujar_aristas=False)
    if mapa:
        # 🔹 NUEVO: superponer capas GEOJSON (gobierno / salud / zonas sensibles)
        mapa = agregar_zonas_restringidas(mapa)
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

# -------- Tab Fallas (versión mejorada) --------
with tab_fallas:
    st.subheader("Simulación de falla de nodo y resiliencia de la red")
    if not activar_falla:
        st.info("Activa 'Simulación de falla' en la barra lateral.")
    else:
        nodos = list(G.nodes)
        if len(nodos) == 0:
            st.warning("No hay nodos para simular.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                victima = st.selectbox("Nodo a desactivar (RUC)", nodos)
            with col2:
                st.write(
                    "Al desactivar un nodo se evalúa cómo se fragmenta la red de courier "
                    "y qué tan robusta es frente a fallas de infraestructura."
                )

            if st.button("Simular falla"):
                # Grafo sin el nodo "víctima"
                G_fail = G.copy()
                if victima in G_fail:
                    G_fail.remove_node(victima)

                # Componentes conectados
                componentes = list(nx.connected_components(G_fail))
                componentes_orden = sorted(componentes, key=len, reverse=True)

                st.write(f"Componentes conectados tras la falla: **{len(componentes_orden)}**")

                resumen = []
                for i, comp in enumerate(componentes_orden):
                    resumen.append({
                        "Componente": i + 1,
                        "N° nodos": len(comp)
                    })
                st.table(pd.DataFrame(resumen))

                if len(componentes_orden) > 1:
                    st.write(
                        "Ejemplo de componente pequeño/aislado después de la falla "
                        f"(hasta 10 nodos): {list(componentes_orden[-1])[:10]}"
                    )

                # Mapa de la red dañada con capas de regulación
                mapa_falla = dibujar_mapa_folium(G_fail, dibujar_aristas=True)
                if mapa_falla:
                    mapa_falla = agregar_zonas_restringidas(mapa_falla)
                    st_folium(mapa_falla, width=900, height=600)

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

            # 3) Fallback: si el grafo estuviera desconectado
            if not camino:
                uso_fallback = True
                lat_o, lon_o = float(origen_row["LATITUD"]), float(origen_row["LONGITUD"])
                lat_d, lon_d = float(destino_row["LATITUD"]), float(destino_row["LONGITUD"])
                dist_km = distancia_haversine(lat_o, lon_o, lat_d, lon_d)
                camino = [origen_ruc, destino_ruc]

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

            # Info de origen / destino
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

            mapa_ruta = dibujar_mapa_ruta_dron(G_dron, camino, origen_ruc, destino_ruc)
            if mapa_ruta:
                st_folium(mapa_ruta, width=900, height=600)






















