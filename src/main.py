# src/main.py
import pandas as pd

from vista import abrir_vista_compartida, cerrar_vista, preparar_dataframe

from xjuegos import diccionario_campos


def cargar_dataset():
    print("Cargando dataset...")
    return pd.read_parquet('database/stuff_model_df.parquet')


def main():
    df = cargar_dataset()
    print(f"Campos organizados en segmentos: {len(diccionario_campos)}")

    # =========================
    # 1. SELECCIÓN DE PARTIDOS
    # =========================
    partidos_unicos = df['game_anon_id'].dropna().unique()
    print("\n==========================================")
    print("          SELECCIÓN DE PARTIDOS           ")
    print("==========================================")
    print(f"Partidos disponibles: {len(partidos_unicos)}")
    print("\n1. Analizar un partido")
    print("2. Analizar varios partidos")
    print("3. Analizar un rango de partidos")
    print("4. Analizar todos los partidos")
    print("5. Regresar")

    opc_partido = input("\nSelecciona una opción: ").strip()
    
    if opc_partido == "1":
        num = input("Escribe el número del partido: ").strip()
        idx = int(num) - 1 if num.isdigit() and 1 <= int(num) <= len(partidos_unicos) else 0
        partido_id = partidos_unicos[idx]
        partidos_sel = [partido_id]
        str_partidos = f"[{num}]"
    else:
        partidos_sel = list(partidos_unicos[:1])
        str_partidos = "[1]"

    # =========================
    # 2. SELECCIÓN DE VARIABLES
    # =========================
    cols_seleccionadas = []
    nombres_visibles = []

    while True:
        print("\n==========================================")
        print("          SELECCIÓN DE VARIABLES          ")
        print("==========================================")
        print("1. Identificación")
        print("2. Contexto del partido")
        print("3. Tipo de lanzamiento")
        print("4. Velocidad y tiempo")
        print("5. Spin")
        print("6. Punto de liberación")
        print("7. Movimiento de la pelota")
        print("8. Ángulos del lanzamiento")
        print("9. Física del lanzamiento")
        print("10. Trayectoria matemática")
        print("11. Resultado de la jugada")
        print("12. Swing, contacto y bola bateada")
        print("13. Resultado ofensivo")
        print("14. Ubicación y zona de strike")
        print("15. Seleccionar todos los campos")
        print("16. Terminar selección")

        opc_var = input("\nSelecciona una opción: ").strip()

        if opc_var == "5":
            print("\n==========================================")
            print("                  SPIN                    ")
            print("==========================================")
            print("1. Spin rate")
            print("2. Eje de giro")
            print("3. Tilt")
            print("4. Todos los campos de este segmento")
            print("5. Regresar")
            sub = input("\nSelecciona uno o varios campos separados por coma: ").strip()
            if "1" in sub and 'SpinRate' not in cols_seleccionadas:
                cols_seleccionadas.append('SpinRate')
                nombres_visibles.append("Spin rate")

        elif opc_var == "14":
            print("\n==========================================")
            print("      UBICACIÓN Y ZONA DE STRIKE          ")
            print("==========================================")
            print("1. Altura en el plato")
            print("2. Ubicación lateral en el plato")
            print("3. Dentro de la zona de strike")
            print("4. Fuera de la zona de strike")
            print("5. Swing fuera de la zona")
            print("6. Todos los campos de este segmento")
            print("7. Regresar")
            sub = input("\nSelecciona uno o varios campos separados por coma: ").strip()
            if "1" in sub and 'PlateLocHeight' not in cols_seleccionadas:
                cols_seleccionadas.append('PlateLocHeight')
                nombres_visibles.append("Altura en el plato")
            if "2" in sub and 'PlateLocSide' not in cols_seleccionadas:
                cols_seleccionadas.append('PlateLocSide')
                nombres_visibles.append("Ubicación lateral en el plato")

        elif opc_var == "16":
            break

    # Filtrar subset de datos
    df_partido = df[df['game_anon_id'].isin(partidos_sel)].copy()

    # =========================
    # 3. REVISAR SELECCIÓN
    # =========================
    while True:
        print("\n==========================================")
        print("              REVISAR SELECCIÓN           ")
        print("==========================================")
        print(f"\nPartidos seleccionados:\n{str_partidos}")
        print("\nVariables seleccionadas:")
        for v in nombres_visibles:
            print(f"- {v}")

        print("\n1. Abrir tabla, filtros y zona de strike")
        print("2. Volver a abrir la vista compartida")
        print("3. Cambiar partidos")
        print("4. Cancelar")

        opc_rev = input("\nSelecciona una opción: ").strip()

        if opc_rev in {"1", "2"}:
            df_final = preparar_dataframe(df_partido, cols_seleccionadas)
            print(f"\nColumnas en la vista: {list(df_final.columns)}")
            print(f"Tamaño final: {df_final.shape}")
            print(f"Registros encontrados: {len(df_final)}")

            try:
                abrir_vista_compartida(df_final, reutilizar=(opc_rev == "2"))
            except Exception as exc:
                print(f"\nNo se pudo abrir la vista compartida: {exc}")
                continue

            print("La vista sigue abierta. Puedes filtrar en la tabla y en el gráfico.")
            print("Elige 4 cuando quieras cerrarla.")

        elif opc_rev == "4":
            cerrar_vista()
            print("Operación cancelada.")
            break


if __name__ == "__main__":
    main()