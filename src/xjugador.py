import pandas as pd
import dtale
import time

# Cargar la base de datos
df = pd.read_parquet('database/stuff_model_df.parquet')

# Obtener los IDs de pitchers
ids_jugadores = df['pitcher_anon_id'].dropna()

# Contar jugadores diferentes
cantidad_jugadores = ids_jugadores.nunique()

print("Cantidad de IDs diferentes:", cantidad_jugadores)

# Obtener jugadores respetando su primera aparición
jugadores = ids_jugadores.drop_duplicates().tolist()

# Asignar un número a cada jugador
numero_a_id = {
    numero: pitcher_id
    for numero, pitcher_id in enumerate(jugadores, start=1)
}

# Relacionar cada ID con su número
id_a_numero = {
    pitcher_id: numero
    for numero, pitcher_id in numero_a_id.items()
}

# Contar apariciones
repeticiones = ids_jugadores.value_counts()

# Mostrar los primeros 20 jugadores
print("\nPrimeros 20 jugadores:")

for numero, pitcher_id in list(numero_a_id.items())[:20]:
    veces = repeticiones[pitcher_id]

    print(
        f"Jugador {numero} | "
        f"ID: {pitcher_id} | "
        f"Apariciones: {veces}"
    )


# Menú de selección de jugadores
print("\n==========================================")
print("          CONSULTA POR JUGADOR")
print("==========================================")
print("1. Un jugador")
print("2. Varios jugadores")
print("3. Rango de jugadores")
print("4. Todos los jugadores")

opcion = input("\nSelecciona una opción: ")


# Opción 1: Un jugador
if opcion == "1":
    numero_jugador = int(input("\nIngresa el número del jugador: "))

    if numero_jugador in numero_a_id:
        pitcher_id = numero_a_id[numero_jugador]

        # Obtener los registros del jugador
        datos_seleccionados = df[
            df['pitcher_anon_id'] == pitcher_id
        ]

        # Contar partidos
        partidos_jugador = datos_seleccionados['game_anon_id'].nunique()

        # Obtener años
        años_jugador = datos_seleccionados['year'].drop_duplicates().tolist()

        print(f"\nJugador {numero_jugador}")
        print(f"ID: {pitcher_id}")
        print(f"Registros encontrados: {len(datos_seleccionados)}")
        print(f"Partidos encontrados: {partidos_jugador}")
        print(f"Años: {años_jugador}")

    else:
        print("Jugador no encontrado")


# Opción 2: Varios jugadores
elif opcion == "2":
    entrada = input("\nIngresa los jugadores separados por comas: ")

    numeros_jugadores = [
        int(numero.strip())
        for numero in entrada.split(",")
    ]

    ids_seleccionados = [
        numero_a_id[numero]
        for numero in numeros_jugadores
        if numero in numero_a_id
    ]

    # Obtener los registros de los jugadores seleccionados
    datos_seleccionados = df[
        df['pitcher_anon_id'].isin(ids_seleccionados)
    ]

    print(f"\nJugadores seleccionados: {numeros_jugadores}")
    print(f"Registros encontrados: {len(datos_seleccionados)}")


# Opción 3: Rango de jugadores
elif opcion == "3":
    inicio = int(input("\nDesde el jugador: "))
    fin = int(input("Hasta el jugador: "))

    numeros_jugadores = list(range(inicio, fin + 1))

    ids_seleccionados = [
        numero_a_id[numero]
        for numero in numeros_jugadores
        if numero in numero_a_id
    ]

    # Obtener los registros del rango seleccionado
    datos_seleccionados = df[
        df['pitcher_anon_id'].isin(ids_seleccionados)
    ]

    print(f"\nJugadores seleccionados: {inicio} al {fin}")
    print(f"Registros encontrados: {len(datos_seleccionados)}")


# Opción 4: Todos los jugadores
elif opcion == "4":
    datos_seleccionados = df[
        df['pitcher_anon_id'].notna()
    ]

    print(f"\nJugadores seleccionados: {cantidad_jugadores}")
    print(f"Registros encontrados: {len(datos_seleccionados)}")


# Opción no válida
else:
    print("Opción no válida")


# Categorías de datos
categorias = {
    "1": [
        "year",
        "PitchUID",
        "game_anon_id",
        "pitcher_anon_id",
        "batter_anon_id",
        "catcher_anon_id"
    ],

    "2": [
        "PitcherThrows",
        "BatterSide",
        "altitude_category",
        "Inning",
        "Top/Bottom",
        "Outs",
        "Balls",
        "Strikes",
        "count"
    ],

    "3": [
        "AutoPitchType"
    ],

    "4": [
        "RelSpeed",
        "EffectiveVelo",
        "ZoneSpeed",
        "SpeedDrop",
        "ZoneTime"
    ],

    "5": [
        "SpinRate",
        "SpinAxis",
        "Tilt"
    ],

    "6": [
        "RelHeight",
        "RelSide",
        "Extension"
    ],

    "7": [
        "VertBreak",
        "InducedVertBreak",
        "HorzBreak"
    ],

    "8": [
        "VertRelAngle",
        "HorzRelAngle",
        "VertApprAngle",
        "HorzApprAngle"
    ],

     "9": [
        "x0",
        "y0",
        "z0",
        "vx0",
        "vy0",
        "vz0",
        "ax0",
        "ay0",
        "az0",
        "pfxx",
        "pfxz",
        "PitchTrajectoryXc0",
        "PitchTrajectoryXc1",
        "PitchTrajectoryXc2",
        "PitchTrajectoryYc0",
        "PitchTrajectoryYc1",
        "PitchTrajectoryYc2",
        "PitchTrajectoryZc0",
        "PitchTrajectoryZc1",
        "PitchTrajectoryZc2",
        "PlateLocHeight",
        "PlateLocSide",
        "in_strike_zone",
        "outside_strike_zone"
    ],

    "10": [
        "PitchCall",
        "KorBB",
        "is_swing",
        "is_whiff",
        "is_contact",
        "is_called_strike",
        "is_swinging_strike",
        "is_ball_in_play",
        "swung_outside_strike_zone"
    ],

    "11": [
        "play_result",
        "hit_type",
        "ExitSpeed",
        "Angle",
        "Direction",
        "Distance",
        "is_batted",
        "is_hit",
        "single",
        "double",
        "triple",
        "home_run",
        "base_on_balls",
        "strikeout",
        "is_hit_by_pitch",
        "RunsScored",
        "OutsOnPlay"
    ],

    "12": list(df.columns)
}

# Menú de selección de datos
print("\n==========================================")
print("          SELECCIÓN DE DATOS")
print("==========================================")
print("1. Identificación")
print("2. Contexto")
print("3. Tipo de lanzamiento")
print("4. Velocidad")
print("5. Spin")
print("6. Liberación")
print("7. Movimiento")
print("8. Ángulos")
print("9. Trayectoria")
print("10. Resultado del lanzamiento")
print("11. Resultado de bateo")
print("12. Todos los datos")

opciones_datos = input(
    "\nSelecciona una o varias opciones separadas por comas: "
)

opciones_datos = [
    opcion.strip()
    for opcion in opciones_datos.split(",")
]

# Reunir las columnas seleccionadas
columnas_seleccionadas = []

for opcion in opciones_datos:
    if opcion in categorias:
        columnas_seleccionadas.extend(categorias[opcion])

# Evitar columnas repetidas
columnas_seleccionadas = list(dict.fromkeys(columnas_seleccionadas))

# Mostrar las columnas seleccionadas
print("\nColumnas seleccionadas:")

for columna in columnas_seleccionadas:
    print(columna)

# Crear los datos finales para el análisis
datos_finales = datos_seleccionados[columnas_seleccionadas].reset_index(drop=True)

print(f"\nFilas finales: {len(datos_finales)}")
print(f"Columnas finales: {len(datos_finales.columns)}")

# Resumen de la consulta
jugadores_finales = datos_seleccionados['pitcher_anon_id'].nunique()
años_finales = sorted(datos_seleccionados['year'].dropna().unique())
partidos_finales = datos_seleccionados['game_anon_id'].nunique()

print("\n==========================================")
print("          RESUMEN DE LA CONSULTA")
print("==========================================")
print(f"Jugadores: {jugadores_finales}")
print(f"Años: {años_finales}")
print(f"Partidos: {partidos_finales}")
print(f"Registros: {len(datos_finales)}")
print(f"Columnas: {len(datos_finales.columns)}")

# Abrir los datos seleccionados en D-Tale
print("\nIniciando servidor D-Tale...")

d = dtale.show(datos_finales, host='127.0.0.1')

# Obtener la URL
url = d._url
print(f"\n¡Servidor listo! Abre este enlace en tu navegador:\n{url}\n")

# Abrir D-Tale automáticamente
dtale.instances()
d.open_browser()

# Mantener el servidor activo
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nServidor detenido.")

#Para detener el servidor: control + C 