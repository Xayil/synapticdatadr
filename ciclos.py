import pandas as pd


# Esto sirve para cargar la base de datos completa desde el archivo Parquet
df = pd.read_parquet("database/stuff_model_df.parquet")

# Esto sirve para contar cuántos partidos diferentes existen en cada año
partidos_por_año = (
    df.groupby("year")["game_anon_id"]
    .nunique()
)

# Esto sirve para mostrar cuántos partidos diferentes contiene cada temporada
print("\nPartidos por año:")
print(partidos_por_año)

# Esto sirve para comprobar cuántos partidos diferentes existen en toda la base
print("\nPartidos únicos totales:")
print(df["game_anon_id"].nunique())

# Esto sirve para obtener los partidos respetando el orden original de la base de datos
juegos = df["game_anon_id"].dropna().unique()


# Esto sirve para relacionar un número sencillo de partido con su ID original
numero_a_id = {
    numero: game_id
    for numero, game_id in enumerate(juegos, start=1)
}


# Esto sirve para determinar si un lanzamiento cierra el ciclo de un turno al bate
def termina_ciclo(fila):

    # Esto sirve para cerrar el ciclo cuando el bateador recibe base por bolas
    if fila["base_on_balls"] == 1:
        return True

    # Esto sirve para cerrar el ciclo cuando el bateador termina en strikeout
    if fila["strikeout"] == 1:
        return True

    # Esto sirve para cerrar el ciclo cuando la pelota es puesta en juego
    # Un foul no entra aquí porque no se considera bola en juego
    if fila["is_ball_in_play"] == 1:
        return True

    return False


# Esto sirve para identificar todos los ciclos o turnos al bate de un partido
def identificar_ciclos(df_partido):

    ciclos = []

    ciclo_actual = []
    numero_ciclo = 0
    ciclo_abierto = False

    # Esto sirve para revisar los lanzamientos del partido en el orden en que aparecen
    for indice, fila in df_partido.iterrows():

        balls = fila["Balls"]
        strikes = fila["Strikes"]

        # Esto sirve para detectar el inicio de un nuevo turno cuando la cuenta comienza en 0-0
        if balls == 0 and strikes == 0 and not ciclo_abierto:

            numero_ciclo += 1
            ciclo_actual = []
            ciclo_abierto = True

        # Esto sirve para guardar cada lanzamiento que pertenece al ciclo actualmente abierto
        if ciclo_abierto:

            ciclo_actual.append(indice)

            # Esto sirve para revisar si este lanzamiento termina el turno al bate
            if termina_ciclo(fila):

                ciclos.append({
                    "numero_ciclo": numero_ciclo,
                    "indices": ciclo_actual.copy()
                })

                ciclo_actual = []
                ciclo_abierto = False

    # Esto sirve para conservar un ciclo que haya quedado abierto al terminar los datos del partido
    if ciclo_abierto and ciclo_actual:

        ciclos.append({
            "numero_ciclo": numero_ciclo,
            "indices": ciclo_actual.copy()
        })

    return ciclos


# Esto sirve para mostrar los ciclos encontrados de una manera fácil de leer
def mostrar_ciclos(df_partido, ciclos):

    print("\n==========================================")
    print("          CICLOS DEL PARTIDO")
    print("==========================================")

    print(f"\nCiclos encontrados: {len(ciclos)}")

    for ciclo in ciclos:

        numero = ciclo["numero_ciclo"]
        indices = ciclo["indices"]

        datos_ciclo = df_partido.loc[indices]

        primera_fila = datos_ciclo.iloc[0]
        ultima_fila = datos_ciclo.iloc[-1]

        pitcher = primera_fila["pitcher_anon_id"]
        batter = primera_fila["batter_anon_id"]

        lanzamientos = len(datos_ciclo)

        # Esto sirve para determinar de qué manera terminó el ciclo
        if ultima_fila["base_on_balls"] == 1:
            resultado = "Base por bolas"

        elif ultima_fila["strikeout"] == 1:
            resultado = "Strikeout"

        elif ultima_fila["is_ball_in_play"] == 1:

            if ultima_fila["is_hit"] == 1:
                resultado = "Hit"

            else:
                resultado = "Bola puesta en juego"

        else:
            resultado = "Ciclo incompleto"

        print(
            f"{numero}. "
            f"Pitcher: {pitcher} | "
            f"Bateador: {batter} | "
            f"Lanzamientos: {lanzamientos} | "
            f"Resultado: {resultado}"
        )

# Esto sirve para mostrar todos los lanzamientos que forman parte de un ciclo
def mostrar_lanzamientos_ciclo(df_partido, ciclo):

    # Esto sirve para obtener el número y los índices correspondientes al ciclo seleccionado
    numero_ciclo = ciclo["numero_ciclo"]
    indices = ciclo["indices"]

    # Esto sirve para obtener únicamente los lanzamientos pertenecientes al ciclo seleccionado
    datos_ciclo = df_partido.loc[indices]

    print("\n==========================================")
    print(f"              CICLO {numero_ciclo}")
    print("==========================================")

    # Esto sirve para obtener el pitcher y el bateador que participan en este turno
    pitcher = datos_ciclo.iloc[0]["pitcher_anon_id"]
    bateador = datos_ciclo.iloc[0]["batter_anon_id"]

    print(f"\nPitcher: {pitcher}")
    print(f"Bateador: {bateador}")
    print(f"Lanzamientos: {len(datos_ciclo)}")

    print("\n------------------------------------------")
    print("        CRONOLOGÍA DEL TURNO")
    print("------------------------------------------")

    # Esto sirve para recorrer cronológicamente cada lanzamiento del ciclo
    # Esto sirve para recorrer cronológicamente cada lanzamiento del ciclo
    for numero_lanzamiento, (_, fila) in enumerate(
        datos_ciclo.iterrows(),
        start=1
    ):

        # Esto sirve para obtener la cuenta existente antes del lanzamiento
        balls = fila["Balls"]
        strikes = fila["Strikes"]

        # Esto sirve para obtener las características principales del lanzamiento
        tipo_lanzamiento = fila["AutoPitchType"]
        velocidad = fila["RelSpeed"]
        spin = fila["SpinRate"]

        # Esto sirve para obtener el movimiento de la pelota
        movimiento_vertical = fila["InducedVertBreak"]
        movimiento_horizontal = fila["HorzBreak"]

        # Esto sirve para obtener la ubicación de la pelota al llegar al plato
        ubicacion_horizontal = fila["PlateLocSide"]
        ubicacion_vertical = fila["PlateLocHeight"]
        en_zona = fila["in_strike_zone"]

        # Esto sirve para obtener el resultado individual del lanzamiento
        resultado_lanzamiento = fila["PitchCall"]

        # Esto sirve para saber si el bateador hizo swing y si consiguió contacto
        hizo_swing = fila["is_swing"]
        hizo_contacto = fila["is_contact"]

        print("\n==========================================")
        print(f"          LANZAMIENTO {numero_lanzamiento}")
        print("==========================================")

        print(f"\nCuenta: {balls}-{strikes}")

        print("\n--- LANZAMIENTO ---")
        print(f"Tipo: {tipo_lanzamiento}")
        print(f"Velocidad: {velocidad} mph")
        print(f"Spin: {spin} rpm")

        print("\n--- MOVIMIENTO ---")
        print(f"Movimiento vertical: {movimiento_vertical}")
        print(f"Movimiento horizontal: {movimiento_horizontal}")

        print("\n--- UBICACIÓN ---")
        print(f"PlateLocSide: {ubicacion_horizontal}")
        print(f"PlateLocHeight: {ubicacion_vertical}")

        # Esto sirve para mostrar de manera sencilla si el lanzamiento estuvo dentro de la zona de strike
        if en_zona == 1:
            print("En zona de strike: Sí")
        else:
            print("En zona de strike: No")

        print("\n--- RESULTADO ---")
        print(f"PitchCall: {resultado_lanzamiento}")

        # Esto sirve para mostrar de manera sencilla si el bateador realizó swing
        if hizo_swing == 1:
            print("Swing: Sí")
        else:
            print("Swing: No")

        # Esto sirve para mostrar de manera sencilla si el bateador hizo contacto con la pelota
        if hizo_contacto == 1:
            print("Contacto: Sí")
        else:
            print("Contacto: No")

        # Esto sirve para mostrar información adicional únicamente cuando la pelota fue puesta en juego
        if fila["is_ball_in_play"] == 1:

            print("\n--- CONTACTO EN JUEGO ---")

            print(f"Tipo de contacto: {fila['hit_type']}")
            print(f"Velocidad de salida: {fila['ExitSpeed']} mph")
            print(f"Ángulo de salida: {fila['Angle']}°")
            print(f"Dirección: {fila['Direction']}")
            print(f"Distancia: {fila['Distance']}")

            # Esto sirve para indicar si la bola puesta en juego terminó siendo un hit
            if fila["is_hit"] == 1:
                print("Hit: Sí")
            else:
                print("Hit: No")

            print(f"Resultado de la jugada: {fila['play_result']}")

    # Esto sirve para obtener el último lanzamiento del ciclo
    ultima_fila = datos_ciclo.iloc[-1]

    # Esto sirve para identificar cómo terminó el turno al bate
    if ultima_fila["base_on_balls"] == 1:
        resultado_final = "Base por bolas"

    elif ultima_fila["strikeout"] == 1:
        resultado_final = "Strikeout"

    elif ultima_fila["is_ball_in_play"] == 1:

        if ultima_fila["is_hit"] == 1:
            resultado_final = "Hit"

        else:
            resultado_final = "Bola puesta en juego"

    else:
        resultado_final = "Ciclo incompleto"

    print("\n------------------------------------------")
    print(f"RESULTADO FINAL: {resultado_final}")
    print("------------------------------------------")

# Esto sirve para permitir que el analista seleccione qué ciclo desea revisar
def menu_ciclos(df_partido, ciclos):

    while True:

        print("\n==========================================")
        print("          SELECCIÓN DE CICLO")
        print("==========================================")

        print("\n1. Ver un ciclo")
        print("2. Regresar")

        opcion = input("\nSelecciona una opción: ")

        # Esto sirve para seleccionar un ciclo específico del partido
        if opcion == "1":

            numero_ciclo = int(
                input("\nEscribe el número del ciclo: ")
            )

            # Esto sirve para comprobar que el ciclo seleccionado exista
            if 1 <= numero_ciclo <= len(ciclos):

                ciclo_seleccionado = ciclos[numero_ciclo - 1]

                # Esto sirve para mostrar cronológicamente los lanzamientos del ciclo seleccionado
                mostrar_lanzamientos_ciclo(
                    df_partido,
                    ciclo_seleccionado
                )

            else:
                print("\nEse ciclo no existe.")

        # Esto sirve para regresar al menú anterior
        elif opcion == "2":
            break

        # Esto sirve para detectar una opción inexistente
        else:
            print("\nOpción no válida.")

# Esto sirve para mostrar el menú principal del programa
def menu_principal():

    while True:

        print("\n==========================================")
        print("        ANÁLISIS DE CICLOS")
        print("==========================================")

        print(f"\nPartidos disponibles: {len(juegos)}")

        print("\n1. Analizar un partido")
        print("2. Salir")

        opcion = input("\nSelecciona una opción: ")

        # Esto sirve para permitir que el analista seleccione un partido
        if opcion == "1":

            numero_partido = int(
                input("\nEscribe el número del partido: ")
            )

            # Esto sirve para comprobar que el partido seleccionado exista
            if numero_partido not in numero_a_id:

                print("\nEse partido no existe.")
                continue

            game_id = numero_a_id[numero_partido]

            # Esto sirve para conservar únicamente los lanzamientos del partido seleccionado
            df_partido = df[
                df["game_anon_id"] == game_id
            ].copy()

            # Esto sirve para reiniciar el índice conservando el orden original de los lanzamientos
            df_partido = df_partido.reset_index(drop=True)

            # Esto sirve para identificar los ciclos del partido seleccionado
            ciclos = identificar_ciclos(df_partido)

            print("\n==========================================")
            print("          PARTIDO SELECCIONADO")
            print("==========================================")

            print(f"\nPartido: {numero_partido}")
            print(f"ID: {game_id}")
            print(f"Lanzamientos totales: {len(df_partido)}")

            # Esto sirve para mostrar todos los ciclos encontrados
            mostrar_ciclos(
                df_partido,
                ciclos
            )
            # Esto sirve para permitir que el analista seleccione un ciclo y revise sus lanzamientos
            menu_ciclos(
            df_partido,
            ciclos
            )
        # Esto sirve para cerrar el programa
        elif opcion == "2":

            print("\nPrograma terminado.")
            break

        # Esto sirve para detectar una opción inexistente
        else:

            print("\nOpción no válida.")


# Esto sirve para iniciar el programa
menu_principal()
