#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IA GEOLÓGICA - BATOLITO DE LA COSTA
Punto de entrada principal (CLI)
"""

import sys
import pandas as pd
from config.settings import PDF_DIR, OUTPUT_DIR, DEFAULT_EXCEL_NAME
from utils.logger import setup_logger
from core.pdf_engine import obtener_pdfs, extraer_texto_pdf, limpiar_texto
from core.nlp_engine import construir_modelo_nlp
from core.extractor import extraer_entidades_estructuradas, obtener_frecuencias
from core.classifier import clasificar_procesos
from core.trends import detectar_tendencias_mineralogicas, resumir_tendencias
from core.summarizer import generar_resumen_automatico, formatear_resumen
from core.graph_builder import construir_grafo_relaciones
from utils.export_manager import guardar_resultados_completos, exportar_json

# Inicializar logger
logger = setup_logger("IA_GEOLOGICA")

# Almacén global de resultados para la sesión
resultados_sesion = []

def procesar_todo():
    """Orquesta el procesamiento de todos los PDFs."""
    global resultados_sesion
    resultados_sesion = []
    
    archivos = obtener_pdfs(PDF_DIR)
    if not archivos:
        logger.warning(f"No se encontraron PDFs en {PDF_DIR}")
        return
        
    nlp = construir_modelo_nlp()
    
    for i, ruta in enumerate(archivos, 1):
        nombre = ruta.name
        logger.info(f"[{i}/{len(archivos)}] Analizando: {nombre}")
        
        # 1. Extracción y Limpieza
        texto_crudo = extraer_texto_pdf(str(ruta))
        texto_limpio = limpiar_texto(texto_crudo)
        
        # 2. Procesamiento NLP
        doc = nlp(texto_limpio[:1_000_000]) # Límite de seguridad
        
        # 3. Inteligencia Geológica
        entidades = extraer_entidades_estructuradas(doc)
        frecuencias = obtener_frecuencias(entidades)
        procesos = clasificar_procesos(entidades)
        tendencias = detectar_tendencias_mineralogicas(doc)
        resumen = generar_resumen_automatico(doc, entidades, procesos)
        
        # 4. Guardar datos del documento
        info_doc = {
            "archivo": nombre,
            "entidades": entidades,
            "frecuencias": frecuencias,
            "procesos": procesos,
            "tendencias": tendencias,
            "resumen": resumen
        }
        resultados_sesion.append(info_doc)
        
        # Mostrar pequeño avance en consola
        print(f"\n--- Resultados para {nombre} ---")
        print(f"Procesos Detectados: {', '.join(procesos.keys()) if procesos else 'Ninguno'}")
        print(f"Tendencias: {len(tendencias)} detectadas")
        
    logger.info("Procesamiento completado para todo el corpus.")

def mostrar_estadisticas():
    if not resultados_sesion:
        print("\n[!] Primero procese los PDFs (Opción 1).")
        return
        
    print("\n--- ESTADÍSTICAS GLOBALES DEL CORPUS ---")
    all_processes = []
    for r in resultados_sesion:
        all_processes.extend(r["procesos"].keys())
        
    if all_processes:
        from collections import Counter
        stats = Counter(all_processes)
        print("\nProcesos más frecuentes:")
        for proc, count in stats.most_common():
            print(f"- {proc}: {count} menciones/evidencias")
    else:
        print("No se detectaron procesos significativos.")

def generar_visualizaciones():
    if not resultados_sesion:
        print("\n[!] Primero procese los PDFs (Opción 1).")
        return
        
    logger.info("Generando redes de relaciones...")
    # Para el grafo usamos el primer documento como ejemplo o el corpus unido
    # Por simplicidad, tomaremos el primero con hallazgos
    for res in resultados_sesion:
        if res["entidades"]["MINERAL"]:
            # Necesitamos el 'doc' de spaCy, pero no lo guardamos en resultados_sesion por RAM.
            # Re-procesamos solo para el grafo del primero
            nlp = construir_modelo_nlp()
            texto = extraer_texto_pdf(f"pdfs/{res['archivo']}")
            doc = nlp(limpiar_texto(texto)[:500000]) # Solo una parte para el grafo
            construir_grafo_relaciones(doc, f"outputs/red_relaciones_{res['archivo']}.png")
            break

def mostrar_menu():
    print("\n" + "="*50)
    print("      IA GEOLÓGICA - BATOLITO DE LA COSTA")
    print("="*50)
    print(f"PDFs en cola: {len(obtener_pdfs(PDF_DIR))}")
    print(f"Documentos analizados en sesión: {len(resultados_sesion)}")
    print("-" * 50)
    print("[1] Procesar PDFs y Análisis Completo")
    print("[2] Ver Resúmenes de sesión")
    print("[3] Ver Estadísticas Científicas")
    print("[4] Generar Gráficos y Redes")
    print("[5] Exportar Resultados (Excel/JSON)")
    print("[0] Salir")
    print("="*50)

def main():
    while True:
        mostrar_menu()
        opcion = input("\nSeleccione una opción: ")

        if opcion == "1":
            procesar_todo()
        elif opcion == "2":
            if not resultados_sesion:
                print("\n[!] No hay datos en sesión.")
            for r in resultados_sesion:
                print(f"\nArchivo: {r['archivo']}")
                print(formatear_resumen(r["resumen"]))
        elif opcion == "3":
            mostrar_estadisticas()
        elif opcion == "4":
            generar_visualizaciones()
        elif opcion == "5":
            if not resultados_sesion:
                print("\n[!] No hay datos para exportar.")
            else:
                guardar_resultados_completos(resultados_sesion, str(OUTPUT_DIR / DEFAULT_EXCEL_NAME))
                exportar_json(resultados_sesion, str(OUTPUT_DIR / "datos_completos.json"))
        elif opcion == "0":
            print("\nSaliendo del sistema. ¡Hasta pronto!")
            sys.exit()
        else:
            print("\nOpción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()
