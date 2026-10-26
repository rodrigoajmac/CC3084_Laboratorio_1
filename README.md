# Observatorio de turismo de Guatemala

Tablero interactivo del Laboratorio 10 (CC3084). Permite explorar el flujo mensual de viajeros de 2009 a junio de 2026 y consultar los resultados de modelos desarrollados en el Laboratorio 1. Está dirigido a personas que toman decisiones en turismo; los datos son de uso académico.

**Entrega final:** [documento de diseño en PDF](output/pdf/Lab10_Diseno_Final.pdf) y [repositorio del proyecto](https://github.com/rodrigoajmac/CC3084_Laboratorio_1).

## Ejecutar localmente

Se necesita Python 3.10 o posterior. Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run Notebooks/resultados/app.py
```

En macOS o Linux, active el entorno con `source .venv/bin/activate` y ejecute los mismos comandos `python -m pip` y `python -m streamlit`. Streamlit mostrará la dirección local de la aplicación en la terminal.

## Usar el tablero

La barra lateral permite elegir la serie, el rango de años, el detalle mensual/trimestral/anual y el año que se quiere inspeccionar. La navegación tiene cuatro páginas:

1. **Panorama:** evolución temporal (V1), comparación de mercados y vías (V2), totales de años completos (V3) y detalle mensual (V4). Seleccionar un periodo en V1 actualiza las tarjetas y, si abarca al menos dos años, el perfil mensual de V6; pulsar una barra de V2 cambia la serie activa; pulsar una barra de V3 abre ese año en V4.
2. **Calendario y mercados:** mapa de calor por año y mes (V5), perfil de temporadas (V6) y nivel de 2025 frente a 2019 (V7).
3. **Pronósticos:** comparación del error de modelos (V8), observado frente a predicho (V9) y dispersión de ambos valores (V10). Se pueden elegir hasta tres modelos ya evaluados.
4. **Guía y límites:** significado de los colores, términos y advertencias de lectura.

Los datos observados se muestran en azul. El naranja destaca estimaciones o puntos que requieren atención; cada gráfica explica qué muestra y para qué puede servir.

## Archivos usados por la aplicación

La aplicación está en [`Notebooks/resultados/app.py`](Notebooks/resultados/app.py) y lee, desde la misma carpeta:

- `series_mensuales.csv`: siete series mensuales (total, tres países y tres vías de ingreso).
- `resultados.json`: indicadores y métricas de los modelos del Laboratorio 1.
- `predicciones.csv`: valores reales y predichos de las pruebas 70/30 y posterior a la pandemia.

No es necesario ejecutar los cuadernos ni entrenar modelos para abrir el tablero.

## Límites de interpretación

- Desde 2023, la fuente dejó de contar a los guatemaltecos que regresan. Para comparar el total de distintos años, seleccione **Visitantes no residentes**, que los excluye también de los años anteriores.
- La cobertura de cruceristas cambió desde 2023. La corrección de no residentes no resuelve la comparación de la vía marítima.
- La serie de guatemaltecos que regresan deja de medirse desde 2023; sus ceros posteriores no son una caída observada del turismo.
- Los datos de 2026 llegan hasta junio. Las barras de años completos excluyen 2026; su detalle mensual sí está disponible.
- Los pronósticos son estimaciones evaluadas con datos históricos. La pandemia y los cambios de medición limitan su interpretación; no demuestran causas ni garantizan resultados futuros.
