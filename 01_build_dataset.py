# 01_build_dataset.py (VERSIÓN EXTERMINADOR)
import pandas as pd
from utils import process_koi_lightcurve
from tqdm import tqdm
import os
import csv
import shutil  # <--- La herramienta para borrar carpetas de forma segura

def purge_all_caches():
    """
    Encuentra y destruye todas las cachés de lightkurve, la local y la global.
    Esta es la solución de "tierra quemada" para asegurar un comienzo limpio.
    """
    print("--- INICIANDO PROTOCOLO DE PURGA DE CACHÉ ---")
    
    # Ruta a la caché local en la carpeta del proyecto
    local_cache = './fits_cache'
    # Ruta a la caché global en la carpeta de usuario
    home_dir = os.path.expanduser('~')
    global_cache = os.path.join(home_dir, '.lightkurve')

    # Destruir la caché local
    if os.path.exists(local_cache):
        try:
            shutil.rmtree(local_cache)
            print(f"ÉXITO: Caché local en '{local_cache}' destruida.")
        except Exception as e:
            print(f"AVISO: No se pudo destruir la caché local. Razón: {e}")
    else:
        print("INFO: No se encontró caché local para destruir.")

    # Destruir la caché global
    if os.path.exists(global_cache):
        try:
            shutil.rmtree(global_cache)
            print(f"ÉXITO: Caché global en '{global_cache}' destruida.")
        except Exception as e:
            print(f"AVISO: No se pudo destruir la caché global. Razón: {e}")
    else:
        print("INFO: No se encontró caché global para destruir.")
        
    print("--- PROTOCOLO DE PURGA FINALIZADO ---")

# --- El resto del script es el que ya funcionaba ---

# --- Configuración ---
INPUT_CSV = './data/cumulative_koi.csv'
OUTPUT_CSV = './data/processed_lightcurves.csv'
N_SAMPLES_PER_CLASS = 200

def build_dataset_final_version():
    # Primero, ejecutamos la purga
    purge_all_caches()

    # Borramos el archivo de resultados anterior para un reinicio 100% limpio
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"ÉXITO: Archivo de resultados anterior '{OUTPUT_CSV}' eliminado.")

    print("\nCargando el catálogo maestro de KOIs...")
    df_koi = pd.read_csv(INPUT_CSV, comment='#')
    
    print("Filtrando y preparando la lista de trabajo...")
    df_filtered = df_koi[df_koi['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    df_filtered = df_filtered.dropna(subset=['kepid', 'koi_period', 'koi_duration', 'koi_time0bk'])

    df_confirmed = df_filtered[df_filtered['koi_disposition'] == 'CONFIRMED'].sample(n=N_SAMPLES_PER_CLASS, random_state=42)
    df_fp = df_filtered[df_filtered['koi_disposition'] == 'FALSE POSITIVE'].sample(n=N_SAMPLES_PER_CLASS, random_state=42)
    df_sample = pd.concat([df_confirmed, df_fp]).sample(frac=1, random_state=42)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        
        flux_cols = [f'flux_{i+1}' for i in range(201)]
        header = ['kepid', 'label'] + flux_cols
        writer.writerow(header)

        print(f"Muestra total de {len(df_sample)} KOIs. Iniciando descarga y procesamiento...")
        
        for index, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0]):
            kepid = int(row['kepid'])
            
            flux_vector = process_koi_lightcurve(
                kepid=kepid,
                period=row['koi_period'],
                duration=row['koi_duration'],
                transit_time=row['koi_time0bk']
            )
            
            if flux_vector is not None:
                label = 1 if row['koi_disposition'] == 'CONFIRMED' else 0
                writer.writerow([kepid, label] + list(flux_vector))
                f.flush()

    print(f"\n¡Proceso completado! Dataset disponible en '{OUTPUT_CSV}'")

if __name__ == '__main__':
    build_dataset_final_version()