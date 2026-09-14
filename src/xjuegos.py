# Funcion: Consulta de cada juego en la base de datos
# Para guardar necesito el command + S


# Esto sirve para abrir los datos seleccionados en una interfaz visual
import dtale

import pandas as pd

# Esto sirve para cargar la base de datos del archivo Parquet
df = pd.read_parquet('database/stuff_model_df.parquet')

# Esto sirve para mostrar los nombres de todas las columnas de la base de datos
#print(df.columns)

# Esto sirve para contar cuántos juegos diferentes existen en la base de datos
#print(df['game_anon_id'].nunique())

# Esto sirve para mostrar los primeros 10 identificadores de juegos diferentes
#print(df['game_anon_id'].unique()[:10])

# Esto sirve para obtener una lista con todos los juegos disponibles en la base de datos
juegos = df['game_anon_id'].unique()

# Esto sirve para mostrar la lista completa de juegos
#print(juegos)

# Esto sirve para relacionar el número del partido con su ID original
numero_a_id = {
    numero: game_id
    for numero, game_id in enumerate(juegos, start=1)
}

# Esto sirve para relacionar el ID original con el número del partido
id_a_numero = {
    game_id: numero
    for numero, game_id in numero_a_id.items()
}

# Esto sirve para comprobar qué ID corresponde a un número de partido
#print(numero_a_id[1])

# Esto sirve para comprobar qué número de partido corresponde a un ID
#print(id_a_numero['game_001331'])

# Esto sirve para obtener el ID original usando el número del partido
def obtener_id(numero_partido):
    return numero_a_id.get(numero_partido)

# Esto sirve para obtener el número del partido usando el ID original
def obtener_numero(game_id):
    return id_a_numero.get(game_id)


# Esto sirve para probar la función que obtiene el ID de un partido
#print(obtener_id(1))

# Esto sirve para probar la función inversa
#print(obtener_numero('game_001331'))

# Esto sirve para identificar, describir y permitir buscar las variables de la base de datos
diccionario_campos = {

    # =========================
    # IDENTIFICACIÓN
    # =========================

    "year": {
        "nombre": "Año",
        "descripcion": "Año en el que se registró el lanzamiento",
        "categoria": "identificacion",
        "alias": [
            "año",
            "year"
        ]
    },

    "PitchUID": {
        "nombre": "ID del lanzamiento",
        "descripcion": "Identificador único de cada lanzamiento",
        "categoria": "identificacion",
        "alias": [
            "id del lanzamiento",
            "lanzamiento",
            "pitch id",
            "pitchuid"
        ]
    },

    "game_anon_id": {
        "nombre": "ID del partido",
        "descripcion": "Identificador anónimo original del partido",
        "categoria": "identificacion",
        "alias": [
            "id del partido",
            "game id",
            "game anon id",
            "game_anon_id"
        ]
    },

    "pitcher_anon_id": {
        "nombre": "ID del pitcher",
        "descripcion": "Identificador anónimo del pitcher que realizó el lanzamiento",
        "categoria": "identificacion",
        "alias": [
            "id del pitcher",
            "pitcher",
            "pitcher id",
            "pitcher_anon_id"
        ]
    },

    "batter_anon_id": {
        "nombre": "ID del bateador",
        "descripcion": "Identificador anónimo del bateador que enfrentó el lanzamiento",
        "categoria": "identificacion",
        "alias": [
            "id del bateador",
            "bateador",
            "batter",
            "batter id",
            "batter_anon_id"
        ]
    },

    "catcher_anon_id": {
        "nombre": "ID del catcher",
        "descripcion": "Identificador anónimo del catcher durante el lanzamiento",
        "categoria": "identificacion",
        "alias": [
            "id del catcher",
            "catcher",
            "catcher id",
            "catcher_anon_id"
        ]
    },


    # =========================
    # CONTEXTO DEL ENFRENTAMIENTO
    # =========================

    "PitcherThrows": {
        "nombre": "Mano del pitcher",
        "descripcion": "Indica si el pitcher lanza con la mano derecha o izquierda",
        "categoria": "contexto",
        "alias": [
            "mano del pitcher",
            "mano del lanzador",
            "pitcher throws",
            "pitcherthrows"
        ]
    },

    "BatterSide": {
        "nombre": "Lado del bateador",
        "descripcion": "Indica el lado desde el que batea el jugador",
        "categoria": "contexto",
        "alias": [
            "lado del bateador",
            "mano del bateador",
            "batter side",
            "batterside"
        ]
    },

    "Inning": {
        "nombre": "Entrada",
        "descripcion": "Entrada del partido en la que ocurrió el lanzamiento",
        "categoria": "contexto",
        "alias": [
            "entrada",
            "inning"
        ]
    },

    "Top/Bottom": {
        "nombre": "Parte de la entrada",
        "descripcion": "Indica si el lanzamiento ocurrió en la parte alta o baja de la entrada",
        "categoria": "contexto",
        "alias": [
            "parte de la entrada",
            "alta o baja",
            "top bottom",
            "top/bottom"
        ]
    },

    "Outs": {
        "nombre": "Outs",
        "descripcion": "Número de outs existentes al momento del lanzamiento",
        "categoria": "contexto",
        "alias": [
            "outs",
            "out"
        ]
    },

    "Balls": {
        "nombre": "Bolas",
        "descripcion": "Número de bolas en la cuenta antes del lanzamiento",
        "categoria": "contexto",
        "alias": [
            "bolas",
            "balls"
        ]
    },

    "Strikes": {
        "nombre": "Strikes",
        "descripcion": "Número de strikes en la cuenta antes del lanzamiento",
        "categoria": "contexto",
        "alias": [
            "strikes",
            "strike"
        ]
    },

    "count": {
        "nombre": "Cuenta",
        "descripcion": "Cuenta de bolas y strikes correspondiente al lanzamiento",
        "categoria": "contexto",
        "alias": [
            "cuenta",
            "conteo",
            "count"
        ]
    },


    # =========================
    # TIPO DE LANZAMIENTO
    # =========================

    "AutoPitchType": {
        "nombre": "Tipo de lanzamiento",
        "descripcion": "Clasificación automática del tipo de lanzamiento realizado",
        "categoria": "tipo_lanzamiento",
        "alias": [
            "tipo de lanzamiento",
            "tipo de pitch",
            "pitch type",
            "autopitchtype"
        ]
    },


    # =========================
    # VELOCIDAD
    # =========================

    "RelSpeed": {
        "nombre": "Velocidad de salida",
        "descripcion": "Velocidad de la pelota al momento de ser liberada por el pitcher",
        "categoria": "velocidad",
        "alias": [
            "velocidad",
            "velocidad de salida",
            "velocidad de lanzamiento",
            "release speed",
            "rel speed",
            "relspeed"
        ]
    },

    "EffectiveVelo": {
        "nombre": "Velocidad efectiva",
        "descripcion": "Velocidad efectiva del lanzamiento considerando cómo se percibe su llegada al bateador",
        "categoria": "velocidad",
        "alias": [
            "velocidad efectiva",
            "effective velocity",
            "effective velo",
            "effectivevelo"
        ]
    },

    "ZoneSpeed": {
        "nombre": "Velocidad en la zona",
        "descripcion": "Velocidad de la pelota al llegar a la zona del plato",
        "categoria": "velocidad",
        "alias": [
            "velocidad en zona",
            "velocidad en la zona",
            "velocidad al plato",
            "zone speed",
            "zonespeed"
        ]
    },

    "SpeedDrop": {
        "nombre": "Pérdida de velocidad",
        "descripcion": "Cambio o reducción de velocidad de la pelota durante su trayectoria",
        "categoria": "velocidad",
        "alias": [
            "perdida de velocidad",
            "pérdida de velocidad",
            "caida de velocidad",
            "caída de velocidad",
            "speed drop",
            "speeddrop"
        ]
    },


    # =========================
    # GIRO / SPIN
    # =========================

    "SpinRate": {
        "nombre": "Spin rate",
        "descripcion": "Tasa de giro de la pelota durante el lanzamiento",
        "categoria": "spin",
        "alias": [
            "spin rate",
            "spin",
            "tasa de giro",
            "revoluciones",
            "spinrate"
        ]
    },

    "SpinAxis": {
        "nombre": "Eje de giro",
        "descripcion": "Orientación del eje de giro de la pelota",
        "categoria": "spin",
        "alias": [
            "eje de giro",
            "eje del giro",
            "spin axis",
            "spinaxis"
        ]
    },

    "Tilt": {
        "nombre": "Tilt",
        "descripcion": "Representación de la orientación del giro de la pelota",
        "categoria": "spin",
        "alias": [
            "tilt",
            "inclinacion",
            "inclinación",
            "orientacion del giro",
            "orientación del giro"
        ]
    },

     # =========================
    # PUNTO DE LIBERACIÓN
    # =========================

    "RelHeight": {
        "nombre": "Altura de liberación",
        "descripcion": "Altura desde la que el pitcher libera la pelota",
        "categoria": "liberacion",
        "alias": [
            "altura de liberacion",
            "altura de liberación",
            "release height",
            "relheight"
        ]
    },

    "RelSide": {
        "nombre": "Posición lateral de liberación",
        "descripcion": "Posición lateral desde la que el pitcher libera la pelota",
        "categoria": "liberacion",
        "alias": [
            "posicion lateral",
            "posición lateral",
            "lado de liberacion",
            "lado de liberación",
            "release side",
            "relside"
        ]
    },

    "Extension": {
        "nombre": "Extensión",
        "descripcion": "Distancia de extensión del pitcher hacia home al momento de liberar la pelota",
        "categoria": "liberacion",
        "alias": [
            "extension",
            "extensión",
            "release extension"
        ]
    },


    # =========================
    # MOVIMIENTO DEL LANZAMIENTO
    # =========================

    "VertBreak": {
        "nombre": "Movimiento vertical",
        "descripcion": "Movimiento vertical observado de la pelota durante su trayectoria",
        "categoria": "movimiento",
        "alias": [
            "movimiento vertical",
            "vertical break",
            "vert break",
            "vertbreak"
        ]
    },

    "InducedVertBreak": {
        "nombre": "Movimiento vertical inducido",
        "descripcion": "Componente vertical del movimiento atribuida principalmente al efecto del giro de la pelota",
        "categoria": "movimiento",
        "alias": [
            "movimiento vertical inducido",
            "induced vertical break",
            "ivb",
            "inducedvertbreak"
        ]
    },

    "HorzBreak": {
        "nombre": "Movimiento horizontal",
        "descripcion": "Movimiento horizontal de la pelota durante su trayectoria",
        "categoria": "movimiento",
        "alias": [
            "movimiento horizontal",
            "horizontal break",
            "horz break",
            "horzbreak"
        ]
    },


    # =========================
    # ÁNGULOS
    # =========================

    "VertRelAngle": {
        "nombre": "Ángulo vertical de liberación",
        "descripcion": "Ángulo vertical de la trayectoria de la pelota al momento de la liberación",
        "categoria": "angulos",
        "alias": [
            "angulo vertical de liberacion",
            "ángulo vertical de liberación",
            "vertical release angle",
            "vertrelangle"
        ]
    },

    "HorzRelAngle": {
        "nombre": "Ángulo horizontal de liberación",
        "descripcion": "Ángulo horizontal de la trayectoria de la pelota al momento de la liberación",
        "categoria": "angulos",
        "alias": [
            "angulo horizontal de liberacion",
            "ángulo horizontal de liberación",
            "horizontal release angle",
            "horzrelangle"
        ]
    },

    "VertApprAngle": {
        "nombre": "Ángulo vertical de aproximación",
        "descripcion": "Ángulo vertical con el que la pelota se aproxima al plato",
        "categoria": "angulos",
        "alias": [
            "angulo vertical de aproximacion",
            "ángulo vertical de aproximación",
            "vertical approach angle",
            "vaa",
            "vertapprangle"
        ]
    },

    "HorzApprAngle": {
        "nombre": "Ángulo horizontal de aproximación",
        "descripcion": "Ángulo horizontal con el que la pelota se aproxima al plato",
        "categoria": "angulos",
        "alias": [
            "angulo horizontal de aproximacion",
            "ángulo horizontal de aproximación",
            "horizontal approach angle",
            "haa",
            "horzapprangle"
        ]
    },


    # =========================
    # TIEMPO
    # =========================

    "ZoneTime": {
        "nombre": "Tiempo hasta la zona",
        "descripcion": "Tiempo que tarda la pelota en llegar a la zona del plato",
        "categoria": "tiempo",
        "alias": [
            "tiempo hasta la zona",
            "tiempo de vuelo",
            "zone time",
            "zonetime"
        ]
    },


    # =========================
    # POSICIÓN INICIAL
    # =========================

    "x0": {
        "nombre": "Posición inicial X",
        "descripcion": "Posición inicial de la pelota sobre el eje X",
        "categoria": "fisica",
        "alias": [
            "posicion x",
            "posición x",
            "posicion inicial x",
            "posición inicial x",
            "x0"
        ]
    },

    "y0": {
        "nombre": "Posición inicial Y",
        "descripcion": "Posición inicial de la pelota sobre el eje Y",
        "categoria": "fisica",
        "alias": [
            "posicion y",
            "posición y",
            "posicion inicial y",
            "posición inicial y",
            "y0"
        ]
    },

    "z0": {
        "nombre": "Posición inicial Z",
        "descripcion": "Posición inicial de la pelota sobre el eje Z",
        "categoria": "fisica",
        "alias": [
            "posicion z",
            "posición z",
            "posicion inicial z",
            "posición inicial z",
            "z0"
        ]
    },


    # =========================
    # VELOCIDAD POR EJE
    # =========================

    "vx0": {
        "nombre": "Velocidad inicial en X",
        "descripcion": "Componente de la velocidad inicial de la pelota sobre el eje X",
        "categoria": "fisica",
        "alias": [
            "velocidad x",
            "velocidad en x",
            "vx0"
        ]
    },

    "vy0": {
        "nombre": "Velocidad inicial en Y",
        "descripcion": "Componente de la velocidad inicial de la pelota sobre el eje Y",
        "categoria": "fisica",
        "alias": [
            "velocidad y",
            "velocidad en y",
            "vy0"
        ]
    },

    "vz0": {
        "nombre": "Velocidad inicial en Z",
        "descripcion": "Componente de la velocidad inicial de la pelota sobre el eje Z",
        "categoria": "fisica",
        "alias": [
            "velocidad z",
            "velocidad en z",
            "vz0"
        ]
    },


    # =========================
    # ACELERACIÓN POR EJE
    # =========================

    "ax0": {
        "nombre": "Aceleración en X",
        "descripcion": "Componente de aceleración de la pelota sobre el eje X",
        "categoria": "fisica",
        "alias": [
            "aceleracion x",
            "aceleración x",
            "aceleracion en x",
            "aceleración en x",
            "ax0"
        ]
    },

    "ay0": {
        "nombre": "Aceleración en Y",
        "descripcion": "Componente de aceleración de la pelota sobre el eje Y",
        "categoria": "fisica",
        "alias": [
            "aceleracion y",
            "aceleración y",
            "aceleracion en y",
            "aceleración en y",
            "ay0"
        ]
    },

    "az0": {
        "nombre": "Aceleración en Z",
        "descripcion": "Componente de aceleración de la pelota sobre el eje Z",
        "categoria": "fisica",
        "alias": [
            "aceleracion z",
            "aceleración z",
            "aceleracion en z",
            "aceleración en z",
            "az0"
        ]
    },
    
    # =========================
    # CONDICIONES DEL ENTORNO
    # =========================

    "altitude_category": {
        "nombre": "Categoría de altitud",
        "descripcion": "Clasificación de la altitud correspondiente al lugar donde se realizó el lanzamiento",
        "categoria": "entorno",
        "alias": [
            "altitud",
            "categoria de altitud",
            "categoría de altitud",
            "altitude",
            "altitude category"
        ]
    },


    # =========================
    # MOVIMIENTO PFX
    # =========================

    "pfxx": {
        "nombre": "Movimiento PFX en X",
        "descripcion": "Componente horizontal del movimiento calculado de la pelota",
        "categoria": "movimiento",
        "alias": [
            "pfx x",
            "movimiento pfx x",
            "movimiento en x",
            "pfxx"
        ]
    },

    "pfxz": {
        "nombre": "Movimiento PFX en Z",
        "descripcion": "Componente vertical del movimiento calculado de la pelota",
        "categoria": "movimiento",
        "alias": [
            "pfx z",
            "movimiento pfx z",
            "movimiento en z",
            "pfxz"
        ]
    },


    # =========================
    # COEFICIENTES DE TRAYECTORIA
    # =========================

    "PitchTrajectoryXc0": {
        "nombre": "Trayectoria X coeficiente 0",
        "descripcion": "Coeficiente 0 utilizado para representar matemáticamente la trayectoria de la pelota en el eje X",
        "categoria": "trayectoria",
        "alias": [
            "trayectoria x c0",
            "coeficiente x 0",
            "pitch trajectory x c0",
            "pitchtrajectoryxc0"
        ]
    },

    "PitchTrajectoryXc1": {
        "nombre": "Trayectoria X coeficiente 1",
        "descripcion": "Coeficiente 1 utilizado para representar matemáticamente la trayectoria de la pelota en el eje X",
        "categoria": "trayectoria",
        "alias": [
            "trayectoria x c1",
            "coeficiente x 1",
            "pitch trajectory x c1",
            "pitchtrajectoryxc1"
        ]
    },

    "PitchTrajectoryXc2": {
        "nombre": "Trayectoria X coeficiente 2",
        "descripcion": "Coeficiente 2 utilizado para representar matemáticamente la trayectoria de la pelota en el eje X",
        "categoria": "trayectoria",
        "alias": [
            "trayectoria x c2",
            "coeficiente x 2",
            "pitch trajectory x c2",
            "pitchtrajectoryxc2"
        ]
    },

    "PitchTrajectoryYc0": {
        "nombre": "Trayectoria Y coeficiente 0",
        "descripcion": "Coeficiente 0 utilizado para representar matemáticamente la trayectoria de la pelota en el eje Y",
        "categoria": "trayectoria",
        "alias": [
            "trayectoria y c0",
            "coeficiente y 0",
            "pitch trajectory y c0",
            "pitchtrajectoryyc0"
        ]
    },

    "PitchTrajectoryYc1": {
        "nombre": "Trayectoria Y coeficiente 1",
        "descripcion": "Coeficiente 1 utilizado para representar matemáticamente la trayectoria de la pelota en el eje Y",
        "categoria": "trayectoria",
        "alias": [
            "trayectoria y c1",
            "coeficiente y 1",
            "pitch trajectory y c1",
            "pitchtrajectoryyc1"
        ]
    },

    "PitchTrajectoryYc2": {
        "nombre": "Trayectoria Y coeficiente 2",
        "descripcion": "Coeficiente 2 utilizado para representar matemáticamente la trayectoria de la pelota en el eje Y",
        "categoria": "trayectoria",
        "alias": [
            "trayectoria y c2",
            "coeficiente y 2",
            "pitch trajectory y c2",
            "pitchtrajectoryyc2"
        ]
    },

    "PitchTrajectoryZc0": {
        "nombre": "Trayectoria Z coeficiente 0",
        "descripcion": "Coeficiente 0 utilizado para representar matemáticamente la trayectoria de la pelota en el eje Z",
        "categoria": "trayectoria",
        "alias": [
            "trayectoria z c0",
            "coeficiente z 0",
            "pitch trajectory z c0",
            "pitchtrajectoryzc0"
        ]
    },

    "PitchTrajectoryZc1": {
        "nombre": "Trayectoria Z coeficiente 1",
        "descripcion": "Coeficiente 1 utilizado para representar matemáticamente la trayectoria de la pelota en el eje Z",
        "categoria": "trayectoria",
        "alias": [
            "trayectoria z c1",
            "coeficiente z 1",
            "pitch trajectory z c1",
            "pitchtrajectoryzc1"
        ]
    },

    "PitchTrajectoryZc2": {
        "nombre": "Trayectoria Z coeficiente 2",
        "descripcion": "Coeficiente 2 utilizado para representar matemáticamente la trayectoria de la pelota en el eje Z",
        "categoria": "trayectoria",
        "alias": [
            "trayectoria z c2",
            "coeficiente z 2",
            "pitch trajectory z c2",
            "pitchtrajectoryzc2"
        ]
    },


    # =========================
    # RESULTADO DEL LANZAMIENTO
    # =========================

    "PitchCall": {
        "nombre": "Resultado del lanzamiento",
        "descripcion": "Clasificación del resultado inmediato del lanzamiento",
        "categoria": "resultado_lanzamiento",
        "alias": [
            "resultado del lanzamiento",
            "resultado del pitch",
            "pitch call",
            "pitchcall"
        ]
    },

    "KorBB": {
        "nombre": "Strikeout o base por bolas",
        "descripcion": "Indica si el resultado de la aparición al plato terminó en ponche o base por bolas",
        "categoria": "resultado_lanzamiento",
        "alias": [
            "ponche o base por bolas",
            "strikeout o walk",
            "k or bb",
            "korbb"
        ]
    },

    "play_result": {
        "nombre": "Resultado de la jugada",
        "descripcion": "Resultado general de la jugada asociada al lanzamiento",
        "categoria": "resultado_jugada",
        "alias": [
            "resultado de la jugada",
            "resultado",
            "play result",
            "play_result"
        ]
    },

    "hit_type": {
        "nombre": "Tipo de contacto",
        "descripcion": "Clasificación del tipo de pelota bateada o contacto producido",
        "categoria": "contacto",
        "alias": [
            "tipo de contacto",
            "tipo de hit",
            "hit type",
            "hit_type"
        ]
    },


    # =========================
    # BOLA BATEADA
    # =========================

    "ExitSpeed": {
        "nombre": "Velocidad de salida del batazo",
        "descripcion": "Velocidad de la pelota después del contacto con el bate",
        "categoria": "bola_bateada",
        "alias": [
            "velocidad de salida del batazo",
            "velocidad de salida del bate",
            "exit velocity",
            "exit speed",
            "exitspeed"
        ]
    },

    "Angle": {
        "nombre": "Ángulo de salida",
        "descripcion": "Ángulo de salida de la pelota después del contacto",
        "categoria": "bola_bateada",
        "alias": [
            "angulo de salida",
            "ángulo de salida",
            "launch angle",
            "angle"
        ]
    },

    "Direction": {
        "nombre": "Dirección del batazo",
        "descripcion": "Dirección en la que salió la pelota después del contacto",
        "categoria": "bola_bateada",
        "alias": [
            "direccion",
            "dirección",
            "direccion del batazo",
            "dirección del batazo",
            "direction"
        ]
    },

    "Distance": {
        "nombre": "Distancia del batazo",
        "descripcion": "Distancia recorrida por la pelota bateada",
        "categoria": "bola_bateada",
        "alias": [
            "distancia",
            "distancia del batazo",
            "hit distance",
            "distance"
        ]
    },


    # =========================
    # ACCIÓN DEL BATEADOR
    # =========================

    "is_swing": {
        "nombre": "Swing",
        "descripcion": "Indica si el bateador realizó swing al lanzamiento",
        "categoria": "accion_bateador",
        "alias": [
            "swing",
            "hizo swing",
            "is swing",
            "is_swing"
        ]
    },

    "is_whiff": {
        "nombre": "Whiff",
        "descripcion": "Indica si el bateador hizo swing sin hacer contacto con la pelota",
        "categoria": "accion_bateador",
        "alias": [
            "whiff",
            "swing fallido",
            "abanico",
            "is whiff",
            "is_whiff"
        ]
    },

    "is_contact": {
        "nombre": "Contacto",
        "descripcion": "Indica si el bateador hizo contacto con la pelota",
        "categoria": "accion_bateador",
        "alias": [
            "contacto",
            "hizo contacto",
            "is contact",
            "is_contact"
        ]
    },

    "is_called_strike": {
        "nombre": "Strike cantado",
        "descripcion": "Indica si el lanzamiento fue marcado como strike sin swing",
        "categoria": "resultado_lanzamiento",
        "alias": [
            "strike cantado",
            "called strike",
            "is called strike",
            "is_called_strike"
        ]
    },

    "is_swinging_strike": {
        "nombre": "Strike con swing",
        "descripcion": "Indica si el lanzamiento produjo un strike debido a un swing fallido",
        "categoria": "resultado_lanzamiento",
        "alias": [
            "strike con swing",
            "swinging strike",
            "is swinging strike",
            "is_swinging_strike"
        ]
    },

    "is_ball_in_play": {
        "nombre": "Pelota en juego",
        "descripcion": "Indica si la pelota quedó en juego después del lanzamiento",
        "categoria": "resultado_lanzamiento",
        "alias": [
            "pelota en juego",
            "bola en juego",
            "ball in play",
            "is_ball_in_play"
        ]
    },

    "is_batted": {
        "nombre": "Pelota bateada",
        "descripcion": "Indica si el lanzamiento fue bateado",
        "categoria": "contacto",
        "alias": [
            "pelota bateada",
            "fue bateada",
            "batted",
            "is_batted"
        ]
    },


    # =========================
    # RESULTADO OFENSIVO
    # =========================

    "is_hit": {
        "nombre": "Hit",
        "descripcion": "Indica si la jugada resultó en un hit",
        "categoria": "resultado_ofensivo",
        "alias": [
            "hit",
            "fue hit",
            "is hit",
            "is_hit"
        ]
    },

    "single": {
        "nombre": "Sencillo",
        "descripcion": "Indica si la jugada resultó en un hit sencillo",
        "categoria": "resultado_ofensivo",
        "alias": [
            "sencillo",
            "single"
        ]
    },

    "double": {
        "nombre": "Doble",
        "descripcion": "Indica si la jugada resultó en un doble",
        "categoria": "resultado_ofensivo",
        "alias": [
            "doble",
            "double"
        ]
    },

    "triple": {
        "nombre": "Triple",
        "descripcion": "Indica si la jugada resultó en un triple",
        "categoria": "resultado_ofensivo",
        "alias": [
            "triple"
        ]
    },

    "home_run": {
        "nombre": "Home run",
        "descripcion": "Indica si la jugada resultó en un home run",
        "categoria": "resultado_ofensivo",
        "alias": [
            "home run",
            "homerun",
            "jonron",
            "jonrón",
            "cuadrangular"
        ]
    },

    "base_on_balls": {
        "nombre": "Base por bolas",
        "descripcion": "Indica si el bateador recibió una base por bolas",
        "categoria": "resultado_ofensivo",
        "alias": [
            "base por bolas",
            "walk",
            "bb",
            "base_on_balls"
        ]
    },

    "strikeout": {
        "nombre": "Ponche",
        "descripcion": "Indica si el bateador terminó ponchado",
        "categoria": "resultado_ofensivo",
        "alias": [
            "ponche",
            "strikeout",
            "strike out",
            "k"
        ]
    },

    "is_hit_by_pitch": {
        "nombre": "Golpeado por lanzamiento",
        "descripcion": "Indica si el bateador fue golpeado por el lanzamiento",
        "categoria": "resultado_ofensivo",
        "alias": [
            "golpeado por lanzamiento",
            "hit by pitch",
            "hbp",
            "is_hit_by_pitch"
        ]
    },

    "RunsScored": {
        "nombre": "Carreras anotadas",
        "descripcion": "Número de carreras anotadas como resultado de la jugada",
        "categoria": "resultado_jugada",
        "alias": [
            "carreras",
            "carreras anotadas",
            "runs",
            "runs scored",
            "runsscored"
        ]
    },

    "OutsOnPlay": {
        "nombre": "Outs en la jugada",
        "descripcion": "Número de outs registrados como resultado de la jugada",
        "categoria": "resultado_jugada",
        "alias": [
            "outs en la jugada",
            "outs de la jugada",
            "outs on play",
            "outsonplay"
        ]
    },


    # =========================
    # UBICACIÓN EN EL PLATO
    # =========================

    "PlateLocHeight": {
        "nombre": "Altura en el plato",
        "descripcion": "Ubicación vertical de la pelota al cruzar la zona del plato",
        "categoria": "ubicacion",
        "alias": [
            "altura en el plato",
            "ubicacion vertical",
            "ubicación vertical",
            "plate location height",
            "platelocheight"
        ]
    },

    "PlateLocSide": {
        "nombre": "Ubicación lateral en el plato",
        "descripcion": "Ubicación horizontal o lateral de la pelota al cruzar la zona del plato",
        "categoria": "ubicacion",
        "alias": [
            "ubicacion lateral",
            "ubicación lateral",
            "lado del plato",
            "plate location side",
            "platelocside"
        ]
    },

    "in_strike_zone": {
        "nombre": "Dentro de la zona de strike",
        "descripcion": "Indica si el lanzamiento pasó dentro de la zona de strike",
        "categoria": "zona_strike",
        "alias": [
            "dentro de la zona",
            "en zona",
            "zona de strike",
            "in strike zone",
            "in_strike_zone"
        ]
    },

    "outside_strike_zone": {
        "nombre": "Fuera de la zona de strike",
        "descripcion": "Indica si el lanzamiento pasó fuera de la zona de strike",
        "categoria": "zona_strike",
        "alias": [
            "fuera de la zona",
            "fuera de zona",
            "outside strike zone",
            "outside_strike_zone"
        ]
    },

    "swung_outside_strike_zone": {
        "nombre": "Swing fuera de la zona",
        "descripcion": "Indica si el bateador realizó swing a un lanzamiento que estaba fuera de la zona de strike",
        "categoria": "zona_strike",
        "alias": [
            "swing fuera de la zona",
            "persiguio fuera de la zona",
            "persiguió fuera de la zona",
            "chase",
            "swung outside strike zone",
            "swung_outside_strike_zone"
        ]
    }
}

# Esto sirve para comprobar cuántos campos están registrados en el diccionario
#print("Campos registrados:", len(diccionario_campos))

# Esto sirve para mostrar las diferentes formas en las que el analista puede seleccionar partidos
def seleccionar_partidos():

    print("\n==========================================")
    print("          SELECCIÓN DE PARTIDOS")
    print("==========================================")
    print(f"Partidos disponibles: {len(juegos)}")
    print()
    print("1. Analizar un partido")
    print("2. Analizar varios partidos")
    print("3. Analizar un rango de partidos")
    print("4. Analizar todos los partidos")
    print("5. Regresar")

    opcion = input("\nSelecciona una opción: ")

    # Esto sirve para seleccionar solamente un partido
    if opcion == "1":
        numero = int(input("Escribe el número del partido: "))

        if numero in numero_a_id:
            partidos_seleccionados = [numero]
        else:
            print("Ese partido no existe.")
            partidos_seleccionados = []

    # Esto sirve para seleccionar varios partidos específicos
    elif opcion == "2":
        cantidad = int(input("¿Cuántos partidos quieres analizar?: "))

        entrada = input("Escribe los números de los partidos separados por coma: ")

        partidos_seleccionados = [
            int(numero.strip())
            for numero in entrada.split(",")
        ]

        # Esto sirve para comprobar que la cantidad escrita coincida con los partidos ingresados
        if len(partidos_seleccionados) != cantidad:
            print("La cantidad de partidos no coincide con los números ingresados.")
            partidos_seleccionados = []

    # Esto sirve para seleccionar todos los partidos comprendidos entre dos números
    elif opcion == "3":
        inicio = int(input("Escribe el primer partido del rango: "))
        final = int(input("Escribe el último partido del rango: "))

        partidos_seleccionados = list(range(inicio, final + 1))

    # Esto sirve para seleccionar automáticamente todos los partidos disponibles
    elif opcion == "4":
        partidos_seleccionados = list(numero_a_id.keys())

    # Esto sirve para regresar sin seleccionar partidos
    elif opcion == "5":
        partidos_seleccionados = []

    # Esto sirve para detectar una opción que no existe
    else:
        print("Opción no válida.")
        partidos_seleccionados = []

    return partidos_seleccionados


# Esto sirve para agrupar los 84 campos de la base de datos en segmentos fáciles de consultar
segmentos_campos = {

    "Identificación": [
        "year",
        "PitchUID",
        "game_anon_id",
        "pitcher_anon_id",
        "batter_anon_id",
        "catcher_anon_id"
    ],

    "Contexto del partido": [
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

    "Tipo de lanzamiento": [
        "AutoPitchType"
    ],

    "Velocidad y tiempo": [
        "RelSpeed",
        "EffectiveVelo",
        "ZoneSpeed",
        "SpeedDrop",
        "ZoneTime"
    ],

    "Spin": [
        "SpinRate",
        "SpinAxis",
        "Tilt"
    ],

    "Punto de liberación": [
        "RelHeight",
        "RelSide",
        "Extension"
    ],

    "Movimiento de la pelota": [
        "VertBreak",
        "InducedVertBreak",
        "HorzBreak",
        "pfxx",
        "pfxz"
    ],

    "Ángulos del lanzamiento": [
        "VertRelAngle",
        "HorzRelAngle",
        "VertApprAngle",
        "HorzApprAngle"
    ],

    "Física del lanzamiento": [
        "x0",
        "y0",
        "z0",
        "vx0",
        "vy0",
        "vz0",
        "ax0",
        "ay0",
        "az0"
    ],

    "Trayectoria matemática": [
        "PitchTrajectoryXc0",
        "PitchTrajectoryXc1",
        "PitchTrajectoryXc2",
        "PitchTrajectoryYc0",
        "PitchTrajectoryYc1",
        "PitchTrajectoryYc2",
        "PitchTrajectoryZc0",
        "PitchTrajectoryZc1",
        "PitchTrajectoryZc2"
    ],

    "Resultado de la jugada": [
        "PitchCall",
        "KorBB",
        "play_result"
    ],

    "Swing, contacto y bola bateada": [
        "hit_type",
        "ExitSpeed",
        "Angle",
        "Direction",
        "Distance",
        "is_swing",
        "is_whiff",
        "is_contact",
        "is_called_strike",
        "is_swinging_strike",
        "is_ball_in_play",
        "is_batted"
    ],

    "Resultado ofensivo": [
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

    "Ubicación y zona de strike": [
        "PlateLocHeight",
        "PlateLocSide",
        "in_strike_zone",
        "outside_strike_zone",
        "swung_outside_strike_zone"
    ]
}

# Esto sirve para comprobar cuántos campos están incluidos entre todos los segmentos
total_campos_segmentados = sum(
    len(campos)
    for campos in segmentos_campos.values()
)

print("Campos organizados en segmentos:", total_campos_segmentados)


# Esto sirve para mostrar los segmentos de variables disponibles
def mostrar_menu_variables():

    print("\n==========================================")
    print("          SELECCIÓN DE VARIABLES")
    print("==========================================")

    print("1. Identificación")
    print("   Incluye IDs, año, partido, pitcher, bateador y catcher.")

    print("2. Contexto del partido")
    print("   Incluye mano del pitcher, lado del bateador, inning, outs y cuenta.")

    print("3. Tipo de lanzamiento")
    print("   Incluye la clasificación del tipo de pitch.")

    print("4. Velocidad y tiempo")
    print("   Incluye velocidad de salida, velocidad efectiva, velocidad al plato y tiempo.")

    print("5. Spin")
    print("   Incluye spin rate, eje de giro y tilt.")

    print("6. Punto de liberación")
    print("   Incluye altura, posición lateral y extensión.")

    print("7. Movimiento de la pelota")
    print("   Incluye movimiento vertical, horizontal y componentes PFX.")

    print("8. Ángulos del lanzamiento")
    print("   Incluye ángulos de liberación y aproximación.")

    print("9. Física del lanzamiento")
    print("   Incluye posición, velocidad y aceleración en X, Y y Z.")

    print("10. Trayectoria matemática")
    print("    Incluye los coeficientes de trayectoria en X, Y y Z.")

    print("11. Resultado de la jugada")
    print("    Incluye resultado del pitch, ponche/base por bolas y resultado de la jugada.")

    print("12. Swing, contacto y bola bateada")
    print("    Incluye swing, whiff, contacto, velocidad de salida, ángulo y dirección.")

    print("13. Resultado ofensivo")
    print("    Incluye hits, dobles, triples, home runs, ponches, carreras y outs.")

    print("14. Ubicación y zona de strike")
    print("    Incluye ubicación en el plato y relación con la zona de strike.")

    print("15. Seleccionar todos los campos")
    print("16. Terminar selección")

# Esto sirve para relacionar cada número del menú con un segmento de variables
opciones_segmentos = {
    "1": "Identificación",
    "2": "Contexto del partido",
    "3": "Tipo de lanzamiento",
    "4": "Velocidad y tiempo",
    "5": "Spin",
    "6": "Punto de liberación",
    "7": "Movimiento de la pelota",
    "8": "Ángulos del lanzamiento",
    "9": "Física del lanzamiento",
    "10": "Trayectoria matemática",
    "11": "Resultado de la jugada",
    "12": "Swing, contacto y bola bateada",
    "13": "Resultado ofensivo",
    "14": "Ubicación y zona de strike"
}

# Esto sirve para mostrar los campos individuales de un segmento seleccionado
def mostrar_submenu_segmento(nombre_segmento):

    print("\n==========================================")
    print(f"          {nombre_segmento.upper()}")
    print("==========================================")

    campos = segmentos_campos[nombre_segmento]

    for numero, campo in enumerate(campos, start=1):
        nombre_amigable = diccionario_campos[campo]["nombre"]
        print(f"{numero}. {nombre_amigable}")

    print(f"{len(campos) + 1}. Todos los campos de este segmento")
    print(f"{len(campos) + 2}. Regresar")

# Esto sirve para permitir que el analista seleccione uno, varios o todos los campos de un segmento
def seleccionar_campos_segmento(nombre_segmento):

    campos = segmentos_campos[nombre_segmento]

    mostrar_submenu_segmento(nombre_segmento)

    seleccion = input("\nSelecciona uno o varios campos separados por coma: ")

    opciones = [
        opcion.strip()
        for opcion in seleccion.split(",")
    ]

    campos_seleccionados = []

    # Esto sirve para definir qué número corresponde a seleccionar todos los campos
    opcion_todos = str(len(campos) + 1)

    # Esto sirve para definir qué número corresponde a regresar
    opcion_regresar = str(len(campos) + 2)

    # Esto sirve para regresar sin seleccionar campos
    if opcion_regresar in opciones:
        return []

    # Esto sirve para seleccionar todos los campos del segmento
    if opcion_todos in opciones:
        return campos

    # Esto sirve para convertir los números seleccionados en columnas reales
    for opcion in opciones:

        if opcion.isdigit():
            numero = int(opcion)

            if 1 <= numero <= len(campos):
                campo = campos[numero - 1]
                campos_seleccionados.append(campo)

            else:
                print(f"\nLa opción {opcion} no existe dentro de este segmento.")

        else:
            print(f"\nLa opción '{opcion}' no es válida.")

    return campos_seleccionados

# Esto sirve para permitir que el analista seleccione variables de varios segmentos
def seleccionar_variables():

    campos_finales = []

    while True:

        mostrar_menu_variables()

        opcion = input("\nSelecciona una opción: ")

        # Esto sirve para entrar a uno de los 14 segmentos
        if opcion in opciones_segmentos:

            nombre_segmento = opciones_segmentos[opcion]

            seleccion_segmento = seleccionar_campos_segmento(nombre_segmento)

            for campo in seleccion_segmento:
                if campo not in campos_finales:
                    campos_finales.append(campo)

        # Esto sirve para seleccionar automáticamente los 84 campos
        elif opcion == "15":

            campos_finales = list(diccionario_campos.keys())

            print("\nSe seleccionaron todos los campos.")

        # Esto sirve para terminar la selección y continuar
        elif opcion == "16":

            break

        # Esto sirve para detectar una opción que no existe
        else:

            print("\nOpción no válida.")

    return campos_finales

# Esto sirve para mostrar un resumen de los partidos y variables seleccionados
def revisar_seleccion(partidos_seleccionados, campos_seleccionados):

    print("\n==========================================")
    print("             REVISAR SELECCIÓN")
    print("==========================================")

    # Esto sirve para mostrar los números de los partidos elegidos por el analista
    print("\nPartidos seleccionados:")
    print(partidos_seleccionados)

    # Esto sirve para mostrar las variables seleccionadas con sus nombres fáciles de entender
    print("\nVariables seleccionadas:")

    for campo in campos_seleccionados:
        nombre_amigable = diccionario_campos[campo]["nombre"]
        print(f"- {nombre_amigable}")

    # Esto sirve para mostrar las opciones disponibles después de revisar la selección
    print("\n1. Abrir visualización")
    print("2. Cambiar partidos")
    print("3. Cambiar variables")
    print("4. Cancelar")

    # Esto sirve para guardar la opción elegida por el analista
    opcion = input("\nSelecciona una opción: ")

    return opcion

# Esto sirve para iniciar el programa y conectar la selección de partidos con la selección de variables
partidos_seleccionados = seleccionar_partidos()

# Esto sirve para continuar con la selección de variables después de elegir los partidos
campos_seleccionados = seleccionar_variables()

# Esto sirve para guardar la opción seleccionada en la pantalla de revisión
opcion_revision = revisar_seleccion(partidos_seleccionados, campos_seleccionados)

# Esto sirve para cambiar los partidos si el analista eligió la opción 2
if opcion_revision == "2":

    partidos_seleccionados = seleccionar_partidos()

    print("\nNueva selección de partidos:")
    print(partidos_seleccionados)
# Esto sirve para cambiar las variables si el analista eligió la opción 3
elif opcion_revision == "3":

    campos_seleccionados = seleccionar_variables()

    print("\nNueva selección de variables:")

    # Esto sirve para mostrar los nombres de las nuevas variables seleccionadas
    for campo in campos_seleccionados:
        nombre_amigable = diccionario_campos[campo]["nombre"]
        print(f"- {nombre_amigable}")

# Esto sirve para cancelar la consulta y terminar el programa
elif opcion_revision == "4":

    print("\nConsulta cancelada.")

# Esto sirve para preparar los datos cuando el analista elige abrir la visualización
elif opcion_revision == "1":

    # Esto sirve para convertir los números de los partidos seleccionados en sus IDs originales
    ids_partidos_seleccionados = [
        numero_a_id[numero]
        for numero in partidos_seleccionados
        if numero in numero_a_id
    ]

    # Esto sirve para conservar únicamente los registros de los partidos seleccionados
    df_partidos = df[
        df["game_anon_id"].isin(ids_partidos_seleccionados)
    ]

    # Esto sirve para conservar únicamente las variables seleccionadas y reiniciar el índice de los resultados
    df_filtrado = df_partidos[campos_seleccionados].reset_index(drop=True)

    # Esto sirve para comprobar qué columnas quedaron en la consulta
    print("\nColumnas seleccionadas:")
    print(df_filtrado.columns.tolist())

    # Esto sirve para comprobar el tamaño final de la consulta
    print("Tamaño final:", df_filtrado.shape)

    # Esto sirve para comprobar qué partidos se utilizarán en la consulta
    print("\nPartidos seleccionados:", partidos_seleccionados)
    print("IDs correspondientes:", ids_partidos_seleccionados)

    # Esto sirve para comprobar cuántos registros quedaron después del filtro
    print("Registros encontrados:", len(df_partidos))

   # Esto sirve para abrir los datos filtrados en D-Tale
    d = dtale.show(df_filtrado, host="127.0.0.1")

    # Esto sirve para obtener la dirección web de D-Tale
    url_dtale = d._url

    # Esto sirve para mostrar la dirección que el analista puede abrir en el navegador
    print("\nD-Tale está disponible en:")
    print(url_dtale)

    # Esto sirve para intentar abrir automáticamente D-Tale en el navegador
    d.open_browser()

    # Esto sirve para mantener activo el servidor de D-Tale mientras el analista usa la visualización
    input("\nPresiona Enter cuando quieras cerrar D-Tale...")
