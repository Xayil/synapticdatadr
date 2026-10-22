# ⚾ DRITAM — ISAC Hackathon 2026 (Reto Diablos Rojos)

Sistema de análisis sabermétrico y simulación de calidad de picheo (**Stuff+**, **Location+** y **Pitching+**) adaptado a las condiciones atmosféricas y de altitud de la Ciudad de México (Estadio Alfredo Harp Helú, 2,240m).

---

## 🛠️ Requisitos Previos

Asegúrate de contar con **Python 3.10+** instalado en tu sistema.

---

## 🚀 Configuración e Instalación del Entorno

Sigue estos pasos en tu terminal (**PowerShell** o Git Bash) dentro de la carpeta raíz del proyecto (`synapticdatadr`):

### 1. Clonar el repositorio y entrar a la carpeta
```powershell
git clone
cd synapticdatadr

# Crear el entorno virtual
python -m venv venv

# Activar en Windows (PowerShell)
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

pip install pyreadr
pip install dtale
pip install pandas pyarrow

Para correr el archivos de visualización general corre eto:
python src/ver_todo.py
