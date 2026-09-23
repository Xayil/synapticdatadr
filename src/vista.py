# src/vista.py
"""Vista compartida: la tabla de D-Tale y la zona de strike de Plotly usan los mismos filtros."""

import hashlib
import json
import os
import time
import webbrowser

import dtale
import dtale.global_state as global_state
from flask import Response, jsonify, request, send_file

from plots import construir_grafica

COLUMNAS_COMPARTIDAS = [
    "PitchUID",
    "pitcher_anon_id",
    "AutoPitchType",
    "PlateLocSide",
    "PlateLocHeight",
    "game_anon_id",
    "HorzBreak",
    "InducedVertBreak",
    "RelSpeed",
    "SpinRate",
    "RelSide",
    "RelHeight",
    "PitchCall",
    "hit_type",
]

_rutas_listas = False
_instancia = None
_url_vista = None
# Consulta que escribió el gráfico, para no pisar un filtro que el usuario escribió en D-Tale.
_consulta_grafico = {}
_consulta_usuario = {}


def preparar_dataframe(df, columnas=None):
    """Conserva las columnas elegidas y las que el gráfico necesita para ubicarse."""
    if not columnas:
        return df.reset_index(drop=True)

    cols = []
    for columna in list(columnas) + COLUMNAS_COMPARTIDAS:
        if columna in df.columns and columna not in cols:
            cols.append(columna)
    return df[cols].reset_index(drop=True)


def registrar_rutas():
    global _rutas_listas
    if _rutas_listas:
        return

    blueprint = dtale.dtale
    blueprint.add_url_rule("/vista/<data_id>", "vista_compartida", pagina_vista)
    blueprint.add_url_rule("/vista-datos/<data_id>", "vista_datos", datos_vista)
    blueprint.add_url_rule(
        "/vista-seleccion/<data_id>",
        "vista_seleccion",
        aplicar_seleccion,
        methods=["POST"],
    )
    blueprint.add_url_rule(
        "/vista-limpiar/<data_id>",
        "vista_limpiar",
        limpiar_seleccion,
        methods=["POST"],
    )
    blueprint.add_url_rule("/vista-plotly.js", "vista_plotly_js", plotly_js)
    _rutas_listas = True


def plotly_js():
    import plotly

    ruta = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    return send_file(ruta, mimetype="application/javascript")


def _dataframe_visible(data_id):
    instancia = dtale.get_instance(data_id)
    if instancia is None:
        raise ValueError("No hay datos abiertos en D-Tale.")
    return instancia.view_data


def leer_filtros(data_id):
    ajustes = global_state.get_settings(str(data_id)) or {}
    filtros = []

    for clave in ("columnFilters", "outlierFilters"):
        for columna, cfg in (ajustes.get(clave) or {}).items():
            if not isinstance(cfg, dict):
                continue
            filtros.append(
                {
                    "origen": "D-Tale",
                    "columna": columna,
                    "detalle": cfg.get("query") or "",
                }
            )

    consulta = ajustes.get("query") or ""
    if consulta:
        origen = "Plotly" if consulta == _consulta_grafico.get(str(data_id)) else "D-Tale"
        detalle = consulta if len(consulta) < 280 else consulta[:280] + "…"
        filtros.append(
            {
                "origen": origen,
                "columna": "Filtro personalizado",
                "detalle": detalle,
            }
        )

    if ajustes.get("invertFilter"):
        filtros.append(
            {
                "origen": "D-Tale",
                "columna": "Invertir filtro",
                "detalle": "Se están mostrando las filas que no cumplen el filtro.",
            }
        )

    from dtale.query import build_query

    consulta_final = build_query(str(data_id), global_state.get_query(str(data_id))) or ""
    return filtros, consulta_final


def datos_vista(data_id):
    grafica = request.args.get("grafica") or "zona"
    try:
        filtros, consulta = leer_filtros(data_id)
        df = _dataframe_visible(data_id)
        paquete = construir_grafica(df, grafica)
        firma = hashlib.sha1(
            f"{paquete['id']}|{consulta}|{paquete['total']}|{len(filtros)}".encode("utf-8")
        ).hexdigest()
        return jsonify(ok=True, filtros=filtros, firma=firma, **paquete)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc), filtros=[], series=[], graficas=[], total=0)


def _consulta_por_ids(ids):
    limpios = []
    for valor in ids:
        texto = str(valor).replace("\\", "\\\\").replace("'", "\\'")
        if texto and texto.lower() != "nan":
            limpios.append(texto)
    if not limpios:
        raise ValueError("La selección no tiene identificadores de lanzamiento.")
    if len(limpios) > 5000:
        limpios = limpios[:5000]
    valores = ", ".join(f"'{valor}'" for valor in limpios)
    return f"`PitchUID` in [{valores}]"


def _guardar_consulta(data_id, consulta):
    from dtale.query import run_query

    df = global_state.get_data(str(data_id))
    if df is None:
        raise ValueError("No hay datos abiertos en D-Tale.")
    if consulta and "PitchUID" not in df.columns:
        raise ValueError("El dataset no tiene la columna PitchUID.")
    if consulta:
        run_query(
            df,
            consulta,
            global_state.get_context_variables(str(data_id)),
            ignore_empty=True,
        )
    global_state.update_settings(str(data_id), {"query": consulta})
    _consulta_grafico[str(data_id)] = consulta


def aplicar_seleccion(data_id):
    cuerpo = request.get_json(silent=True) or {}
    ids = cuerpo.get("ids") or []
    try:
        seleccion = _consulta_por_ids(ids)
        ajustes = global_state.get_settings(str(data_id)) or {}
        actual = ajustes.get("query") or ""
        if actual != _consulta_grafico.get(str(data_id), ""):
            _consulta_usuario[str(data_id)] = actual

        base = _consulta_usuario.get(str(data_id), "")
        consulta = f"({base}) and ({seleccion})" if base else seleccion
        if consulta == actual:
            return jsonify(ok=True, recargar=False, query=consulta)

        _guardar_consulta(data_id, consulta)
        return jsonify(ok=True, recargar=True, query=consulta)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc), recargar=False)


def limpiar_seleccion(data_id):
    try:
        base = _consulta_usuario.get(str(data_id), "")
        _guardar_consulta(data_id, base)
        _consulta_grafico[str(data_id)] = base
        return jsonify(ok=True, recargar=True)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc), recargar=False)


def pagina_vista(data_id):
    if dtale.get_instance(data_id) is None:
        return Response("No hay datos abiertos en D-Tale.", status=404)

    data_id_js = json.dumps(str(data_id))
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Datos, filtros y gráficas</title>
  <script src="/dtale/vista-plotly.js"></script>
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      background: #111;
      color: #f2f2f2;
      font-family: "Segoe UI", sans-serif;
    }}
    header {{
      padding: 12px 16px 8px;
      background: #1b1b1b;
      border-bottom: 1px solid #333;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 18px;
      font-weight: 600;
    }}
    p {{
      margin: 0;
      color: #bdbdbd;
      font-size: 13px;
    }}
    .barra {{
      display: flex;
      gap: 16px;
      align-items: flex-start;
      justify-content: space-between;
      padding: 8px 16px;
      background: #161616;
      border-bottom: 1px solid #333;
    }}
    #filtros {{
      margin: 0;
      padding-left: 18px;
      font-size: 13px;
      max-height: 88px;
      overflow: auto;
    }}
    #filtros li {{ margin: 2px 0; }}
    .origen {{
      display: inline-block;
      min-width: 52px;
      margin-right: 6px;
      color: #8fd3ff;
      font-weight: 600;
    }}
    button {{
      background: #2c3e50;
      color: white;
      border: 0;
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
      white-space: nowrap;
    }}
    button:hover {{ background: #3d566e; }}
    .layout {{
      display: flex;
      height: calc(100% - 132px);
      min-height: 480px;
    }}
    iframe {{
      flex: 1.15;
      border: 0;
      background: #fff;
    }}
    #panel {{
      flex: 0.85;
      min-width: 420px;
      display: flex;
      flex-direction: column;
    }}
    #estado {{
      padding: 6px 12px;
      font-size: 13px;
      color: #d0d0d0;
    }}
    #graficas {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 8px 12px 0;
    }}
    #graficas button {{
      background: #243140;
      font-size: 12px;
    }}
    #graficas button.activa {{
      background: #1abc9c;
      color: #102027;
    }}
    #graficas button:disabled {{
      opacity: 0.35;
      cursor: default;
    }}
    #grafico {{
      flex: 1;
      min-height: 420px;
    }}
    .error {{ color: #ff8a80; }}
  </style>
</head>
<body>
  <header>
    <h1>Tabla, filtros y gráficas</h1>
    <p>Elige el tipo de gráfica en los botones. Los filtros de la tabla se aplican a todas. En un mapa de puntos, recuadra lanzamientos para dejar solo esos en la tabla.</p>
  </header>
  <div class="barra">
    <ul id="filtros"><li>Leyendo filtros…</li></ul>
    <button id="limpiar" type="button">Quitar selección del gráfico</button>
  </div>
  <div class="layout">
    <iframe id="tabla" title="Tabla D-Tale" src="/dtale/main/{data_id}"></iframe>
    <div id="panel">
      <div id="graficas"></div>
      <div id="estado"></div>
      <div id="grafico"></div>
    </div>
  </div>
  <script>
    const DATA_ID = {data_id_js};
    const grafico = document.getElementById("grafico");
    const lista = document.getElementById("filtros");
    const estado = document.getElementById("estado");
    let firma = null;
    let graficaActual = "zona";
    let menuFirma = "";
    let modoActual = "scatter";
    let aplicando = false;
    let ignorarSeleccion = false;

    function idDe(punto) {{
      const valor = punto.customdata;
      if (Array.isArray(valor)) return String(valor[0]);
      return String(valor);
    }}

    function pintarFiltros(data) {{
      lista.replaceChildren();
      if (!data.filtros || !data.filtros.length) {{
        const item = document.createElement("li");
        item.textContent = "Sin filtros. El gráfico muestra todos los lanzamientos de la tabla.";
        lista.appendChild(item);
        return;
      }}
      data.filtros.forEach((filtro) => {{
        const item = document.createElement("li");
        const origen = document.createElement("span");
        origen.className = "origen";
        origen.textContent = filtro.origen;
        item.appendChild(origen);
        item.appendChild(document.createTextNode(" " + filtro.columna + ": " + filtro.detalle));
        lista.appendChild(item);
      }});
    }}

    function pintarMenu(data) {{
      const firmaMenu = (data.graficas || []).map((g) => g.id + (g.disponible ? "1" : "0")).join("|");
      if (firmaMenu === menuFirma && document.getElementById("graficas").childElementCount) return;
      menuFirma = firmaMenu;
      const caja = document.getElementById("graficas");
      caja.replaceChildren();
      (data.graficas || []).forEach((opcion) => {{
        const boton = document.createElement("button");
        boton.type = "button";
        boton.className = "grafica" + (opcion.id === graficaActual ? " activa" : "");
        boton.textContent = opcion.nombre;
        boton.disabled = !opcion.disponible;
        boton.title = opcion.disponible ? opcion.nombre : opcion.motivo;
        boton.addEventListener("click", () => {{
          graficaActual = opcion.id;
          firma = null;
          menuFirma = "";
          actualizar();
        }});
        caja.appendChild(boton);
      }});
    }}

    function dibujar(data) {{
      modoActual = data.modo || "scatter";
      const ejes = data.ejes || {{}};
      const esBarra = modoActual === "bar";
      let trazas;
      if (esBarra) {{
        const serie = (data.series || [])[0] || {{ x: [], y: [], color: [] }};
        trazas = [{{
          type: "bar",
          x: serie.x,
          y: serie.y,
          marker: {{ color: serie.color }},
          hovertemplate: "<b>%{{x}}</b><br>Lanzamientos: %{{y}}<extra></extra>"
        }}];
      }} else {{
        trazas = (data.series || []).map((serie) => ({{
          x: serie.x,
          y: serie.y,
          customdata: serie.id,
          mode: "markers",
          type: "scatter",
          name: serie.tipo,
          marker: {{ size: 9, opacity: 0.85, color: serie.color }},
          hovertemplate: "<b>%{{fullData.name}}</b><br>ID: %{{customdata}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>"
        }}));
      }}
      const xaxis = {{ title: ejes.x || "", zeroline: false, automargin: true }};
      const yaxis = {{ title: ejes.y || "", zeroline: false, automargin: true }};
      if (ejes.x_rango) xaxis.range = ejes.x_rango;
      if (ejes.y_rango) yaxis.range = ejes.y_rango;
      if (esBarra) xaxis.tickangle = -35;
      const layout = {{
        paper_bgcolor: "#111111",
        plot_bgcolor: "#1a1a1a",
        font: {{ color: "#f2f2f2" }},
        title: data.titulo || "",
        xaxis: xaxis,
        yaxis: yaxis,
        shapes: data.forma === "zona" ? [{{
          type: "rect",
          x0: -0.83, x1: 0.83, y0: 1.5, y1: 3.5,
          line: {{ color: "white", width: 2, dash: "dash" }},
          fillcolor: "rgba(255,255,255,0.05)"
        }}] : [],
        legend: {{ title: {{ text: "Tipo de lanzamiento" }} }},
        showlegend: !esBarra,
        margin: {{ t: 36, r: 16, b: 72, l: 56 }},
        dragmode: esBarra ? false : "select",
        uirevision: data.firma
      }};
      ignorarSeleccion = true;
      Plotly.react(grafico, trazas, layout, {{ responsive: true, displaylogo: false }});
      if (!grafico.dataset.seleccion) {{
        grafico.on("plotly_selected", (evento) => {{
          if (modoActual === "bar" || ignorarSeleccion || !evento || !evento.points || !evento.points.length) return;
          const ids = [...new Set(evento.points.map(idDe))];
          enviarSeleccion(ids);
        }});
        grafico.dataset.seleccion = "1";
      }}
      setTimeout(() => {{ ignorarSeleccion = false; }}, 500);
    }}

    async function actualizar() {{
      if (aplicando) return;
      try {{
        const respuesta = await fetch("/dtale/vista-datos/" + DATA_ID + "?grafica=" + graficaActual, {{ cache: "no-store" }});
        const data = await respuesta.json();
        if (!data.ok) {{
          estado.innerHTML = "";
          estado.className = "error";
          estado.textContent = data.error || "No se pudo leer el filtro.";
          pintarFiltros(data);
          return;
        }}
        estado.className = "";
        let texto = data.total + " lanzamientos coinciden con los filtros.";
        if (data.muestra) {{
          texto += " Se dibujan " + data.mostrados + " para mantener el gráfico ligero.";
        }}
        estado.textContent = (data.titulo ? data.titulo + ". " : "") + texto;
        pintarFiltros(data);
        pintarMenu(data);
        if (data.firma !== firma) {{
          firma = data.firma;
          dibujar(data);
        }}
      }} catch (error) {{
        estado.className = "error";
        estado.textContent = "No se pudo actualizar el gráfico.";
      }}
    }}

    async function enviarSeleccion(ids) {{
      aplicando = true;
      try {{
        const respuesta = await fetch("/dtale/vista-seleccion/" + DATA_ID, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ ids: ids }})
        }});
        const data = await respuesta.json();
        if (!data.ok) {{
          estado.className = "error";
          estado.textContent = data.error || "No se pudo aplicar la selección.";
          return;
        }}
        if (data.recargar) {{
          document.getElementById("tabla").contentWindow.location.reload();
          firma = null;
        }}
      }} finally {{
        aplicando = false;
        actualizar();
      }}
    }}

    document.getElementById("limpiar").addEventListener("click", async () => {{
      aplicando = true;
      try {{
        const respuesta = await fetch("/dtale/vista-limpiar/" + DATA_ID, {{ method: "POST" }});
        const data = await respuesta.json();
        if (data.recargar) {{
          document.getElementById("tabla").contentWindow.location.reload();
          firma = null;
        }}
        if (!data.ok) {{
          estado.className = "error";
          estado.textContent = data.error || "No se pudo quitar la selección.";
        }}
      }} finally {{
        aplicando = false;
        actualizar();
      }}
    }});

    actualizar();
    setInterval(actualizar, 1500);
  </script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


def abrir_vista_compartida(df, abrir_navegador=True, reutilizar=False):
    """Abre D-Tale y la zona de strike en la misma página, compartiendo filtros."""
    global _instancia, _url_vista

    registrar_rutas()

    if reutilizar and _instancia is not None and _url_vista and _instancia.is_up():
        if abrir_navegador:
            webbrowser.open(_url_vista)
        return _instancia

    if _instancia is not None:
        try:
            _instancia.cleanup()
        except Exception:
            pass

    instancia = dtale.show(
        df,
        host="127.0.0.1",
        enable_custom_filters=True,
        ignore_duplicate=True,
        open_browser=False,
        theme="dark",
    )
    if instancia is None:
        raise RuntimeError(
            "D-Tale no pudo abrir los datos. Si ya estaban cargados, vuelve a ejecutar la opción."
        )

    for _ in range(50):
        if instancia.is_up():
            break
        time.sleep(0.1)

    _instancia = instancia
    _url_vista = f"{instancia._url}/dtale/vista/{instancia._data_id}"
    _consulta_grafico[str(instancia._data_id)] = ""
    _consulta_usuario[str(instancia._data_id)] = ""

    print("\nTabla y gráfico comparten filtros en:")
    print(_url_vista)
    print("Filtra columnas en D-Tale y el gráfico se actualiza solo.")
    print("Recuadra puntos en el gráfico para filtrar la tabla.\n")

    if abrir_navegador:
        webbrowser.open(_url_vista)
    return instancia


def cerrar_vista():
    global _instancia, _url_vista
    if _instancia is not None:
        try:
            _instancia.kill()
        except Exception:
            pass
    _instancia = None
    _url_vista = None
