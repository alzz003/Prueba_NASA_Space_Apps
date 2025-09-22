# utils.py (Versión Corregida y Robusta)
import lightkurve as lk
import numpy as np

N_BINS = 201

def process_koi_lightcurve(kepid, period, duration, transit_time):
    """
    Función robusta para buscar, descargar y procesar una curva de luz.
    El timeout está en el lugar correcto: en la descarga.
    """
    try:
        print(f"\nIniciando procesamiento para KIC {kepid}...")
        
        # --- CORRECCIÓN: Eliminado el 'timeout' de esta función ---
        # Esta búsqueda es rápida y no suele colgarse.
        search_result = lk.search_lightcurve(f'KIC {kepid}', author='Kepler')
        
        if not search_result:
            print(f"WARN: No se encontraron datos para KIC {kepid}")
            return None
        
        print(f"Búsqueda exitosa para KIC {kepid}. Descargando {len(search_result)} archivos...")

        # El timeout en la DESCARGA es el que nos protege de los cuelgues largos.
        lc_collection = search_result.download_all(download_dir='./fits_cache', timeout=120)
        
        print(f"Descarga completada para KIC {kepid}. Procesando...")
        
        lc = lc_collection.stitch().remove_outliers().normalize()
        folded_lc = lc.fold(period=period, epoch_time=transit_time)
        binned_lc = folded_lc.bin(bins=N_BINS)

        flux_vector = binned_lc.flux.value
        if np.isnan(flux_vector).any():
            flux_vector = np.nan_to_num(flux_vector, nan=1.0)
            
        print(f"Procesamiento de KIC {kepid} finalizado con éxito.")
        return flux_vector

    except Exception as e:
        print(f"ERROR: Fallo al procesar KIC {kepid}. Razón: {e}. Omitiendo este KOI.")
        return None