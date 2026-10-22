#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infografia.py
CC3084 - Data Science - Laboratorio 9: Visualización estática
Infografía: "El turismo de Guatemala ya superó la pandemia"

Genera UNA figura de 11 x 17 pulgadas y la guarda en dos formatos:
  - salidas/infografia.png  -> 2200 x 3400 px (200 dpi), versión digital
  - salidas/infografia.pdf  -> PDF vectorial de una página, fuentes incrustadas

Todas las cifras salen de los resultados del Laboratorio 1 (carpeta datos/):
  - series_mensuales.csv  (las 7 series mensuales)
  - resultados.json       (comparativo, métricas por modelo y partición de control)
Única cifra copiada a mano: el índice estacional (celda 51 del cuaderno 01).
Se recalcula desde el CSV y se imprime la diferencia, para que se vea si hay desfase.

Uso:
    python infografia.py
    python infografia.py --series datos/series_mensuales.csv --resultados datos/resultados.json --salida salidas
    python infografia.py --estricto      # falla si la reproducción del modelo se aleja más de 1 punto
"""
import argparse
import json
import unicodedata
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL
# ----------------------------------------------------------------------------------------
# Paleta: 2 colores temáticos + neutro + acento complementario (ver documento de explicación)
AZUL = '#1F4E79'        # temático 1: dato real, ganadores
AZUL_C = '#9DBBD9'      # temático 2: apoyo (mismo matiz, menor saturación)
GRIS = '#B9C0C8'        # neutro: contexto y referencia
NARANJA = '#D9661F'     # acento: pandemia, diciembre, marítima, ganador, estimación
NARANJA_T = '#A84A12'   # acento oscurecido para texto pequeño (contraste AA)
TEXTO = '#1B2733'       # casi negro azulado
TEXTO_2 = '#55606C'     # gris de texto secundario
TARJETA = '#F2F5F8'     # fondo muy claro de las tarjetas de KPI
REJILLA = '#E3E8ED'
BANDA_PRUEBA = '#EDF0F3'

W, H, DPI = 11.0, 17.0, 200
M = 0.60                 # margen lateral (pulgadas)
CW = W - 2 * M           # ancho útil

MESES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

# Índice estacional 2009-2019 (promedio del año = 100). Copiado de la celda 51 de
# 01_EDA_series_tiempo.ipynb, que es el valor citado en el documento de explicación.
INDICE_ESTACIONAL = dict(zip(MESES, [98.4, 81.2, 107.8, 105.7, 87.6, 95.5,
                                      107.4, 104.4, 82.7, 92.0, 103.5, 133.8]))
N_REGISTROS = 161_036    # registros de la base cruda (cuaderno 01, sección 0)


def elegir_fuente():
    """Segoe UI si existe (Windows); si no, la sans más parecida disponible."""
    disponibles = {f.name for f in fm.fontManager.ttflist}
    for nombre in ['Segoe UI', 'Open Sans', 'Noto Sans', 'Source Sans 3', 'Carlito',
                   'Liberation Sans', 'DejaVu Sans']:
        if nombre in disponibles:
            return nombre
    return 'DejaVu Sans'


FUENTE = elegir_fuente()
plt.rcParams.update({
    'font.family': [FUENTE, 'DejaVu Sans'],   # DejaVu como respaldo de glifos (≫, ▲, −)
    'pdf.fonttype': 42, 'ps.fonttype': 42,    # incrusta fuentes TrueType en el PDF
    'axes.unicode_minus': False,
    'mathtext.default': 'regular',
    'text.color': TEXTO, 'axes.labelcolor': TEXTO_2,
    'xtick.color': TEXTO_2, 'ytick.color': TEXTO_2,
    'axes.edgecolor': GRIS, 'axes.linewidth': 0.6,
})


# ----------------------------------------------------------------------------------------
# 2. CARGA DE DATOS
# ----------------------------------------------------------------------------------------
def _canon(col):
    """Normaliza el nombre de columna (los cuadernos 01 y 03 las nombran distinto)."""
    c = unicodedata.normalize('NFKD', str(col)).encode('ascii', 'ignore').decode().lower()
    if 'total' in c:
        return 'total'
    if 'salvador' in c:
        return 'es'
    if 'guatemala' in c:
        return 'gt'
    if 'estados' in c or 'usa' in c:
        return 'us'
    if 'aere' in c:
        return 'aerea'
    if 'terrest' in c:
        return 'terrestre'
    if 'maritim' in c:
        return 'maritima'
    return None


def cargar_series(ruta):
    df = pd.read_csv(ruta, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
    cols = {}
    for c in df.columns:
        k = _canon(c)
        if k and k not in cols:
            cols[k] = c
    faltan = {'total', 'gt', 'es', 'us', 'aerea', 'terrestre', 'maritima'} - set(cols)
    if faltan:
        raise SystemExit(f'Faltan columnas en {ruta}: {sorted(faltan)}. Columnas: {list(df.columns)}')
    S = pd.DataFrame({k: df[v].astype(float) for k, v in cols.items()}).asfreq('MS')
    S['nores'] = S['total'] - S['gt']      # visitantes no residentes (definición constante)
    return S


def cargar_resultados(ruta):
    with open(ruta, encoding='utf-8') as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------------------
# 3. CIFRAS DE LA INFOGRAFÍA (todas derivadas de los datos del Laboratorio 1)
# ----------------------------------------------------------------------------------------
def pronostico_sarima(serie, ini, fin_train, h, orden=(1, 0, 1), est=(1, 0, 1, 12)):
    """SARIMA sobre log(1+x), misma especificación del Laboratorio 1 (ajusta_sarima)."""
    y = np.log1p(serie.loc[ini:fin_train])
    res = SARIMAX(y, order=orden, seasonal_order=est,
                  enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=500)
    fc = res.get_forecast(steps=h)
    volver = lambda v: np.clip(np.expm1(np.asarray(v, float)), 0, None)
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq='MS')
    c80, c95 = fc.conf_int(alpha=0.20), fc.conf_int(alpha=0.05)
    return pd.DataFrame({'media': volver(fc.predicted_mean),
                         'lo80': volver(c80.iloc[:, 0]), 'hi80': volver(c80.iloc[:, 1]),
                         'lo95': volver(c95.iloc[:, 0]), 'hi95': volver(c95.iloc[:, 1])}, index=idx)


def calcular(S, R, estricto=False):
    D = {}
    anual = S.groupby(S.index.year).sum()
    D['anual'] = anual
    D['rec_hom'] = anual.loc[2025, 'total'] / (anual.loc[2019, 'total'] - anual.loc[2019, 'gt']) * 100
    D['tcac'] = ((anual.loc[2019, 'total'] / anual.loc[2009, 'total']) ** (1 / 10) - 1) * 100
    D['yoy_abr20'] = (S.loc['2020-04-01', 'total'] / S.loc['2019-04-01', 'total'] - 1) * 100
    D['min_total'] = S['total'].min()
    D['fecha_min'] = S['total'].idxmin()
    D['total_2025_M'] = anual.loc[2025, 'total'] / 1e6

    # Modelos: unión de las dos particiones (solo los que existen en ambas)
    ctrl = R['control_postcovid_detalle']['Total de viajeros']
    p7030 = R['series']['Total de viajeros']['comparacion']
    comunes = [m for m in ctrl if m in p7030]
    tabla = pd.DataFrame({m: {'mape_ctrl': ctrl[m]['MAPE'], 'rmse': ctrl[m]['RMSE'], 'mae': ctrl[m]['MAE'],
                              'mape_7030': p7030[m]['MAPE']} for m in comunes}).T.sort_values('rmse')
    D['modelos'] = tabla
    D['ganador'] = tabla.index[0]
    D['mape_ctrl'] = float(tabla.loc[D['ganador'], 'mape_ctrl'])

    # Rango de MAPE mejor-por-serie en la partición de control (para la nota del bloque 3)
    mejores = [v.get('MAPE_postcovid') for v in R['control_postcovid'].values()]
    mejores = [x for x in mejores if x is not None and np.isfinite(x)]
    D['rango_ctrl'] = (min(mejores), max(mejores))

    # Estimación del modelo ganador: (a) prueba post-pandemia, (b) futuro
    tr_fin, te_ini = '2024-12-01', '2025-01-01'
    D['fc_prueba'] = pronostico_sarima(S['total'], '2021-07-01', tr_fin, 18)
    real = S['total'].loc[te_ini:'2026-06-01']
    pred = D['fc_prueba']['media'].loc[real.index]
    ok = real > 0
    D['mape_repro'] = float(np.mean(np.abs(real[ok] - pred[ok]) / real[ok]) * 100)
    D['fc_futuro'] = pronostico_sarima(S['total'], '2023-01-01', '2026-06-01', 18)
    D['total_2027_M'] = D['fc_futuro'].loc['2027', 'media'].sum() / 1e6
    D['hi95_max_M'] = D['fc_prueba']['hi95'].max() / 1e6
    dif = abs(D['mape_repro'] - D['mape_ctrl'])
    msg = (f"MAPE reproducido {D['mape_repro']:.2f} % vs {D['mape_ctrl']:.2f} % del laboratorio "
           f"(diferencia {dif:.2f} pts)")
    if dif > 1:
        if estricto:
            raise AssertionError('REVISAR: ' + msg)
        print('  [REVISAR]', msg, '-> la reproducción se aleja más de 1 punto')
    else:
        print('  [OK]', msg)

    # Índice estacional recalculado desde el CSV (diagnóstico)
    pre = S['total'].loc[:'2019-12-01']
    idx_csv = (pre.groupby(pre.index.month).mean() / pre.mean() * 100)
    D['indice_csv'] = idx_csv
    D['dif_indice'] = float(np.max(np.abs(idx_csv.values - np.array(list(INDICE_ESTACIONAL.values())))))

    # Cuadrante
    cmpv = R['comparativo']
    D['cmp'] = cmpv
    return D


# ----------------------------------------------------------------------------------------
# 4. LIENZO Y UTILIDADES DE TEXTO
# ----------------------------------------------------------------------------------------
FIG = plt.figure(figsize=(W, H), dpi=DPI, facecolor='white')
RENDER = FIG.canvas.get_renderer()
# Capa de fondo con coordenadas en pulgadas (origen arriba-izquierda): 1 unidad = 1 pulgada
A = FIG.add_axes([0, 0, 1, 1], zorder=0)
A.set_xlim(0, W); A.set_ylim(H, 0); A.axis('off')


def ejes(x, y, w, h):
    """Ejes de datos ubicados con coordenadas en pulgadas desde arriba-izquierda."""
    ax = FIG.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.set_facecolor('none')
    return ax


def txt(x, y, s, size=9, weight='normal', color=TEXTO, ha='left', va='top', **kw):
    return A.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, **kw)


def ancho(s, size, weight='normal'):
    t = A.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(RENDER).width / FIG.dpi
    t.remove()
    return w


def ajustar(s, size, weight, max_w, min_size=8):
    """Reduce el tamaño hasta que el texto quepa en max_w (según la fuente disponible)."""
    while ancho(s, size, weight) > max_w and size > min_size:
        size -= 0.25
    return size


def envolver(s, size, weight, max_w):
    lineas, actual = [], ''
    for p in s.split():
        prueba = (actual + ' ' + p).strip()
        if ancho(prueba, size, weight) <= max_w or not actual:
            actual = prueba
        else:
            lineas.append(actual)
            actual = p
    lineas.append(actual)
    return lineas


def parrafo(x, y, s, size, weight='normal', color=TEXTO_2, max_w=3.0, inter=1.32, ha='left'):
    """Texto con salto de línea automático. Devuelve la coordenada y donde termina."""
    paso = size / 72 * inter
    for ln in envolver(s, size, weight, max_w):
        txt(x if ha == 'left' else x, y, ln, size, weight, color, ha=ha)
        y += paso
    return y


def encabezado(n, titulo, bajada, y):
    txt(M, y - 0.01, str(n), 16, 'bold', NARANJA)
    tam = ajustar(titulo, 16, 'bold', CW - 0.30, 12.5)
    txt(M + 0.30, y, titulo, tam, 'bold', TEXTO)
    tb = ajustar(bajada, 10.5, 'normal', CW - 0.30, 8.5)
    txt(M + 0.30, y + 0.29, bajada, tb, 'normal', TEXTO_2)


def fmt_pct(v, dec=0):
    return f'{v:.{dec}f} %'.replace('-', '−')


def estilo_ejes(ax, x_grid=False, y_grid=True):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.5, labelsize=7.4, pad=2)
    if y_grid:
        ax.yaxis.grid(True, color=REJILLA, lw=0.6)
    if x_grid:
        ax.xaxis.grid(True, color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------------------
# 5. BLOQUES DE LA INFOGRAFÍA
# ----------------------------------------------------------------------------------------
def bloque_titular(D):
    txt(M, 0.46, 'TURISMO RECEPTOR EN GUATEMALA  ·  FLUJO MIGRATORIO 2009–2026', 8.5, 'bold', NARANJA_T)
    titulo = 'El turismo de Guatemala ya superó la pandemia'
    tam = ajustar(titulo, 32, 'bold', CW, 22)
    txt(M, 0.66, titulo, tam, 'bold', TEXTO)
    sub = (f'Radiografía de 17 años de datos migratorios: un flujo que crece por tierra desde el vecindario, '
           f'que sigue un calendario estable y que hoy se puede pronosticar con un error cercano al '
           f'{D["mape_ctrl"]:.0f} %.')
    parrafo(M, 1.22, sub, 12.5, 'normal', TEXTO_2, CW * 0.86, 1.28)

    # Tarjetas de KPI: cada una responde una parte de la pregunta guía
    kpis = [
        (fmt_pct(D['rec_hom']), AZUL, 'visitantes extranjeros de 2025 frente a 2019, con la misma definición de medida'),
        (fmt_pct(D['tcac'], 1), AZUL, 'crecimiento anual compuesto del ingreso de viajeros entre 2009 y 2019'),
        (fmt_pct(D['yoy_abr20']), NARANJA,
         f'caída interanual en abril de 2020: el cierre de fronteras dejó ~{D["min_total"] / 1000:.0f} mil ingresos al mes'),
        (fmt_pct(D['mape_ctrl']), AZUL, 'error promedio (MAPE) del mejor modelo al pronosticar 18 meses hacia adelante'),
    ]
    y0, h = 1.86, 0.88
    gap = 0.14
    w = (CW - 3 * gap) / 4
    for i, (num, col, desc) in enumerate(kpis):
        x = M + i * (w + gap)
        A.add_patch(Rectangle((x, y0), w, h, facecolor=TARJETA, edgecolor='none'))
        A.add_patch(Rectangle((x, y0), 0.05, h, facecolor=col, edgecolor='none'))
        txt(x + 0.17, y0 + 0.05, num, 30, 'bold', col)
        parrafo(x + 0.17, y0 + 0.56, desc, 7.6, 'normal', TEXTO, w - 0.30, 1.22)


def bloque_panorama(D, S, y):
    encabezado(1, 'Diez años de crecimiento, un colapso y una recuperación completa',
               'Viajeros que ingresan al país cada mes. La línea azul excluye a los guatemaltecos residentes, '
               'que dejaron de contarse en 2023.', y)
    ax = ejes(M + 0.45, y + 0.62, CW - 0.45, 2.00)
    estilo_ejes(ax)
    T = pd.Timestamp
    ymax = float(np.ceil((S['total'].max() / 1e3 + 30) / 100)) * 100
    ax.plot(S.index, S['total'] / 1e3, color=GRIS, lw=1.0, zorder=2)
    ax.plot(S.index, S['nores'] / 1e3, color=AZUL, lw=1.5, zorder=3)
    ax.set_ylim(0, ymax)
    ax.set_xlim(T('2009-01-01'), T('2026-10-01'))
    ax.set_yticks(range(0, int(ymax), 100))
    ax.set_xticks([T(f'{a}-01-01') for a in range(2009, 2027)])
    ax.set_xticklabels([str(a) for a in range(2009, 2027)])
    ax.set_ylabel('miles de viajeros / mes', fontsize=7.4)

    # Pandemia: franja + etiqueta + reapertura
    ax.axvspan(T('2020-03-01'), T('2021-06-30'), color=NARANJA, alpha=0.14, lw=0, zorder=1)
    ax.text(T('2020-02-01'), ymax - 14, 'Pandemia: fronteras\ncerradas desde mar-2020', fontsize=7.4,
            color=NARANJA_T, fontweight='bold', ha='right', va='top', linespacing=1.15)
    ax.plot([T('2020-09-18')] * 2, [0, ymax * 0.80], color=NARANJA, lw=0.9, ls=':', zorder=2)
    ax.text(T('2020-10-10'), ymax * 0.80, '18-sep-2020:\nreapertura con\nprueba COVID', fontsize=7.0,
            color=NARANJA_T, ha='left', va='top', linespacing=1.15)
    f_min = D['fecha_min']
    ax.scatter([f_min], [S.loc[f_min, 'nores'] / 1e3], s=22, color=NARANJA, zorder=5, clip_on=False)
    ax.text(T('2020-02-01'), 20, f'Abr–may 2020: ~{D["min_total"] / 1000:.0f} mil al mes ({fmt_pct(D["yoy_abr20"])} interanual)',
            fontsize=7.4, color=NARANJA_T, fontweight='bold', ha='right', va='center')

    # Cambio de medición 2023
    ax.plot([T('2023-01-01')] * 2, [0, ymax], color=TEXTO, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(T('2023-03-01'), ymax - 14, 'Ene-2023: cambio de medición.\nSe dejan de contar residentes\n'
            'guatemaltecos y cruceristas.', fontsize=7.0, color=TEXTO, ha='left', va='top', linespacing=1.15)

    # Tendencia 2009-2019 sobre la serie total (TCAC)
    anual = D['anual']
    a_ = anual.loc[2009, 'total'] / 12 / 1e3 + 70
    b_ = anual.loc[2019, 'total'] / 12 / 1e3 + 55
    ax.annotate('', xy=(T('2019-03-01'), b_), xytext=(T('2009-07-01'), a_),
                arrowprops=dict(arrowstyle='-|>', color=TEXTO, lw=1.1, mutation_scale=9), zorder=6)
    ax.text(T('2009-04-01'), a_ + 40, f'+{D["tcac"]:.1f} % anual compuesto\n2009–2019', fontsize=7.6,
            color=AZUL, fontweight='bold', ha='left', va='bottom', linespacing=1.15)

    # Etiquetas directas de las dos líneas
    ax.text(T('2012-03-01'), 0.80 * ymax, 'Total registrado\n(incluye guatemaltecos residentes)',
            fontsize=7.2, color=TEXTO_2, ha='left', va='center', linespacing=1.15)
    ax.text(T('2013-06-01'), 55, 'Visitantes no residentes', fontsize=7.6, color=AZUL, fontweight='bold',
            ha='left', va='center')

    # Cifra de cierre
    f25 = pd.Timestamp('2025-07-01')
    ax.annotate(f'2025: {D["total_2025_M"]:.1f} M = {D["rec_hom"]:.0f} % de 2019',
                xy=(f25, S.loc[f25, 'nores'] / 1e3 + 6), xytext=(f25, 0.74 * ymax),
                fontsize=8.0, color=AZUL, fontweight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-', color=AZUL, lw=0.7))


def bloque_estacionalidad(D, y):
    ax = ejes(M, y, 4.45, 2.70)
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)
    dev = np.array([INDICE_ESTACIONAL[m] - 100 for m in MESES])
    cols = []
    for m, d in zip(MESES, dev):
        if m == 'Dic':
            cols.append(NARANJA)
        elif d >= 3:
            cols.append(AZUL)
        elif d <= -7:
            cols.append(GRIS)
        else:
            cols.append(AZUL_C)
    ax.bar(range(12), dev, color=cols, width=0.72, zorder=3)
    ax.axhline(0, color=TEXTO, lw=0.8, zorder=4)
    ax.set_ylim(-28, 56)
    ax.set_xlim(-0.6, 11.6)
    ax.yaxis.set_visible(False)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MESES)
    ax.tick_params(axis='x', length=0, labelsize=7.6, pad=2)
    ax.xaxis.set_ticks_position('bottom')
    for i, d in enumerate(dev):
        lab = f'{d:+.0f} %'.replace('-', '−')
        col = NARANJA_T if MESES[i] == 'Dic' else TEXTO
        if d >= 0:
            ax.text(i, d + 1.4, lab, ha='center', va='bottom', fontsize=7.6, fontweight='bold', color=col)
        else:
            ax.text(i, d - 1.4, lab, ha='center', va='top', fontsize=7.6, fontweight='bold', color=col)
    ax.text(2.5, 19, 'Semana\nSanta', ha='center', va='bottom', fontsize=7.4, fontweight='bold', color=TEXTO, linespacing=1.1)
    ax.text(6.5, 19, 'Vacaciones · Bono 14 ·\nFiestas Agostinas (SV)', ha='center', va='bottom', fontsize=7.4,
            fontweight='bold', color=TEXTO, linespacing=1.1)
    ax.text(11.5, 44, 'Aguinaldo, fiestas\ny diáspora', ha='right', va='bottom', fontsize=7.4,
            fontweight='bold', color=NARANJA_T, linespacing=1.1)
    bajos = (abs(dev[1]) + abs(dev[8])) / 2
    nota = (f'Diciembre recibe {dev[11]:.0f} % más viajeros que el mes promedio; febrero y septiembre '
            f'(temporada baja), ~{bajos:.0f} % menos. El patrón sobrevivió intacto a la pandemia: '
            f'cambió el volumen, no el calendario.')
    parrafo(M, y + 2.95, nota, 8.2, 'normal', TEXTO_2, 4.45, 1.28)


def bloque_cuadrante(D, y):
    x0, wtot = M + 4.80, CW - 4.80
    gx = 0.30
    wp = (wtot - gx) / 2
    cmpv = D['cmp']
    series = [('País: El Salvador', 'El Salvador'), ('País: Guatemala', 'Guatemala'),
              ('País: Estados Unidos de América', 'EE. UU.'),
              ('Vía: Aérea', 'Aérea'), ('Vía: Terrestre', 'Terrestre'), ('Vía: Marítima', 'Marítima')]
    pos = [0, 1, 2, 3.7, 4.7, 5.7]
    t_corte = D.get('_', None)
    ratio = cmpv['Vía: Marítima']['volatilidad_yoy'] / cmpv['Vía: Terrestre']['volatilidad_yoy']
    paneles = [
        dict(t='¿Mayor estacionalidad?', c='F_estacional', modo='max', xl='fuerza estacional (0 a 1)',
             fmt=lambda v: f'{v:.2f}', xlim=(0, 1.22), ticks=[0, 0.5, 1.0], tk=['0.0', '0.5', '1.0']),
        dict(t='¿Mayor crecimiento?', c='TCAC_09_19_%', modo='max', xl='% anual compuesto 2009–2019',
             fmt=lambda v: f'{v:.1f} %', xlim=(0, 16.5), ticks=[0, 5, 10, 15], tk=['0', '5', '10', '15']),
        dict(t='¿Mayor volatilidad?', c='volatilidad_yoy', modo='max', xl='desv. est. del cambio interanual (log)',
             fmt=lambda v: f'{v:.2f}', xlim=(0, 0.37), ticks=[0, 0.1, 0.2, 0.3], tk=['0.0', '0.1', '0.2', '0.3'],
             corte=0.245),
        dict(t='¿Más golpeada por la pandemia?', c='rec_homog_%', modo='min', xl='nivel 2025 vs. 2019 (%)',
             fmt=lambda v: f'{v:.0f} %', xlim=(0, 188), ticks=[0, 100], tk=['0', '100'], ref=100),
    ]
    alto_ax, paso = 0.96, 1.69
    for k, p in enumerate(paneles):
        r, c = divmod(k, 2)
        px = x0 + c * (wp + gx)
        py = y + r * paso
        vals = [cmpv[n][p['c']] for n, _ in series]
        v_arr = np.array([np.nan if v is None else float(v) for v in vals])
        grupos = [[0, 1, 2], [3, 4, 5]]
        ganadores = []
        for g in grupos:
            sub = v_arr[g]
            if np.all(np.isnan(sub)):
                continue
            ganadores.append(g[int(np.nanargmax(sub) if p['modo'] == 'max' else np.nanargmin(sub))])
        nombres_g = [series[i][1] for i in ganadores]
        # título del panel = la pregunta; subtítulo = la respuesta
        txt(px, py, p['t'], 8.8, 'bold', TEXTO)
        neg = p['modo'] == 'min' or p['c'] == 'volatilidad_yoy'
        txt(px, py + 0.19, '  ·  '.join(nombres_g), 8.4, 'bold', NARANJA_T if neg else AZUL)
        # el eje usa 0.66 in a la izquierda para las etiquetas de serie
        ax = ejes(px + 0.68, py + 0.42, wp - 0.68, alto_ax)
        estilo_ejes(ax, x_grid=True, y_grid=False)
        ax.spines['left'].set_visible(False)
        for i, v in enumerate(v_arr):
            es_gan = i in ganadores
            col = (NARANJA if series[i][1] == 'Marítima' else AZUL) if es_gan else AZUL_C
            if np.isnan(v):
                ax.text(0.01 * p['xlim'][1], pos[i], 'n/d: dejó de medirse', fontsize=7.0, style='italic',
                        color=TEXTO_2, va='center', ha='left')
                continue
            largo = min(v, p.get('corte', v))
            ax.barh(pos[i], largo, height=0.72, color=col, zorder=3)
            lab = p['fmt'](v)
            if v > largo:  # barra cortada en el borde y declarada
                xb = largo
                ax.plot([xb - 0.012, xb - 0.002], [pos[i] + 0.45, pos[i] - 0.45], color='white', lw=1.6, zorder=5)
                ax.plot([xb - 0.022, xb - 0.012], [pos[i] + 0.45, pos[i] - 0.45], color='white', lw=1.6, zorder=5)
                lab += ' ≫'
            ax.text(largo + 0.012 * p['xlim'][1], pos[i], lab, fontsize=7.2, va='center', ha='left',
                    fontweight='bold' if es_gan else 'normal', color=TEXTO)
        if 'ref' in p:
            ax.axvline(p['ref'], color=TEXTO_2, lw=0.8, ls=':', zorder=4)
        ax.set_xlim(*p['xlim'])
        ax.set_ylim(pos[-1] + 0.6, -0.6)
        ax.set_yticks(pos)
        ax.set_yticklabels([n for _, n in series], fontsize=7.4)
        for i, lbl in enumerate(ax.get_yticklabels()):
            if i in ganadores:
                lbl.set_fontweight('bold'); lbl.set_color(TEXTO)
        ax.set_xticks(p['ticks'])
        ax.set_xticklabels(p['tk'], fontsize=7.0)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel(p['xl'], fontsize=7.2, labelpad=2)
    nota = (f'Línea punteada = nivel de 2019. Volatilidad marítima fuera de escala: {int(ratio)}+ veces la terrestre '
            f'(barra cortada ≫). Guatemala no se compara en la pandemia: dejó de medirse en 2023.')
    parrafo(x0, y + 2 * paso - 0.04, nota, 7.2, 'normal', TEXTO_2, wtot, 1.25)


def bloque_modelos(D, S, y):
    n = D['modelos'].shape[0]
    mape = D['mape_ctrl']
    encabezado(3, f'Con datos post-pandemia el turismo es predecible: error de {mape:.0f} %',
               'Izquierda: error de cada algoritmo en las dos pruebas del laboratorio. '
               'Derecha: el modelo ganador frente a lo que realmente ocurrió.', y)
    top = y + 0.66
    lw_ = 5.05

    # leyenda directa (marcadores)
    A.scatter([M + 0.05], [top + 0.07], s=20, color=AZUL, zorder=5)
    txt(M + 0.14, top, 'prueba post-pandemia (ene-2025 a jun-2026)', 7.2, 'normal', TEXTO_2)
    A.scatter([M + 2.75], [top + 0.07], s=20, facecolor='white', edgecolor=AZUL, lw=1.0, zorder=5)
    txt(M + 2.84, top, 'prueba 70/30 con la pandemia', 7.2, 'normal', TEXTO_2)

    nombres = {'SARIMA(1,0,1)(1,0,1,12)': 'SARIMA (1,0,1)(1,0,1)$_{12}$',
               'SARIMA(0,0,1)(0,0,1,12)': 'SARIMA (0,0,1)(0,0,1)$_{12}$',
               'Seasonal naive': 'Ingenuo estacional', 'Holt-Winters': 'Holt-Winters',
               'Suav. exponencial simple': 'Suavizado exponencial', 'Prophet': 'Prophet'}
    ax = ejes(M + 1.78, top + 0.40, 2.05, 1.40)
    estilo_ejes(ax, x_grid=True, y_grid=False)
    ax.spines['left'].set_visible(False)
    tabla = D['modelos']
    ytab = np.arange(n)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.13, -0.95, 'RMSE', transform=tr, fontsize=7.4, fontweight='bold', ha='right', va='center')
    ax.text(1.43, -0.95, 'MAE', transform=tr, fontsize=7.4, fontweight='bold', ha='right', va='center')
    for i, (m, fila) in enumerate(tabla.iterrows()):
        gan = (i == 0)
        col = NARANJA if gan else AZUL
        a, b = fila['mape_ctrl'], fila['mape_7030']
        ax.plot([a, b], [i, i], color=GRIS, lw=1.6, zorder=2, solid_capstyle='round')
        ax.scatter([a], [i], s=26, color=col, zorder=4)
        ax.scatter([b], [i], s=26, facecolor='white', edgecolor=col if gan else AZUL, lw=1.0, zorder=4)
        lo, hi = min(a, b), max(a, b)
        ax.text(lo - 2.5, i, f'{a:.1f} %' if a <= b else f'{b:.1f} %', fontsize=7.0, ha='right', va='center',
                fontweight='bold' if gan else 'normal', color=NARANJA_T if gan else TEXTO)
        ax.text(hi + 2.5, i, f'{b:.0f} %' if a <= b else f'{a:.0f} %', fontsize=7.0, ha='left', va='center',
                color=TEXTO_2)
        ax.text(1.13, i, f'{fila["rmse"] / 1000:.1f}', transform=tr, fontsize=7.4, ha='right', va='center',
                fontweight='bold' if gan else 'normal', color=NARANJA_T if gan else TEXTO)
        ax.text(1.43, i, f'{fila["mae"] / 1000:.1f}', transform=tr, fontsize=7.4, ha='right', va='center',
                fontweight='bold' if gan else 'normal', color=NARANJA_T if gan else TEXTO)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(-17, 124)
    ax.set_yticks(ytab)
    ax.set_yticklabels([nombres.get(m, m) for m in tabla.index], fontsize=7.6)
    for i, lbl in enumerate(ax.get_yticklabels()):
        if i == 0:
            lbl.set_fontweight('bold'); lbl.set_color(NARANJA_T)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0 %', '25 %', '50 %', '75 %', '100 %'], fontsize=7.0)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('MAPE: error porcentual promedio', fontsize=7.2, labelpad=2)

    r1, r2 = D['rango_ctrl']
    m7 = tabla['mape_7030']
    nota = (f'RMSE y MAE en miles de viajeros por mes (prueba post-pandemia). En la prueba 70/30 todos fallan '
            f'({m7.min():.0f}–{m7.max():.0f} %): se entrenaron con la pandemia y no podían prever la reapertura. '
            f'Sin ese choque, el mejor modelo de cada serie baja a {r1:.0f}–{r2:.0f} %. '
            f'Gana el SARIMA por menor RMSE.')
    parrafo(M, top + 0.40 + 1.40 + 0.38, nota, 7.2, 'normal', TEXTO_2, lw_, 1.25)

    # --- Proyección del modelo ganador ---
    T = pd.Timestamp
    ax2 = ejes(M + lw_ + 0.30 + 0.55, top + 0.05, CW - lw_ - 0.30 - 0.55, 1.95)
    estilo_ejes(ax2)
    ymax = 600
    ax2.set_ylim(0, ymax)
    ax2.set_xlim(T('2021-07-01'), T('2027-12-31'))
    ax2.axvspan(T('2025-01-01'), T('2026-06-30'), color=BANDA_PRUEBA, lw=0, zorder=0)
    real = S['total'].loc['2021-07-01':]
    ax2.plot(real.index, real / 1e3, color=AZUL, lw=1.5, zorder=4)
    fp, ff = D['fc_prueba'], D['fc_futuro']
    for f in (fp, ff):
        ax2.fill_between(f.index, f['lo95'] / 1e3, f['hi95'] / 1e3, color=NARANJA, alpha=0.16, lw=0, zorder=2)
        ax2.fill_between(f.index, f['lo80'] / 1e3, f['hi80'] / 1e3, color=NARANJA, alpha=0.32, lw=0, zorder=3)
    est = pd.concat([fp['media'], ff['media']])
    ax2.plot(est.index, est / 1e3, color=NARANJA, lw=1.5, ls=(0, (4, 2)), zorder=5)
    ax2.plot([T('2023-01-01')] * 2, [0, ymax], color=TEXTO, lw=0.8, ls=':', zorder=1)
    ax2.set_xticks([T(f'{a}-01-01') for a in range(2022, 2029)])
    ax2.set_xticklabels([str(a) for a in range(2022, 2029)])
    ax2.set_yticks(range(0, ymax + 1, 100))
    ax2.set_ylabel('miles de viajeros / mes', fontsize=7.2)
    # etiquetas directas
    ax2.text(T('2021-10-01'), 395, 'Real', fontsize=8.0, fontweight='bold', color=AZUL, ha='left', va='center')
    ax2.text(T('2025-01-15'), ymax - 12, 'período de prueba', fontsize=7.0, color=TEXTO_2, ha='left', va='top')
    ax2.text(T('2021-09-01'), ymax - 12, 'oscura: IC 80 %\nclara: IC 95 %', fontsize=7.0,
             color=NARANJA_T, ha='left', va='top', linespacing=1.15)
    ax2.text(T('2025-02-01'), 62, f'Estimación (prueba)\nMAPE {mape:.1f} %', fontsize=7.2, fontweight='bold',
             color=NARANJA_T, ha='left', va='center', linespacing=1.15)
    ax2.text(T('2026-07-15'), 90, f'Estimación jul-2026\na dic-2027\n2027 ≈ {D["total_2027_M"]:.1f} M viajeros',
             fontsize=7.2, fontweight='bold', color=NARANJA_T, ha='left', va='center', linespacing=1.15)
    if D['hi95_max_M'] * 1e3 > ymax:
        ax2.text(T('2024-12-01'), ymax - 12, f'IC 95 % llega a {D["hi95_max_M"]:.1f} M ▲', fontsize=7.0,
                 color=NARANJA_T, ha='right', va='top')
    return top + 0.40 + 1.40 + 0.38 + 0.36


def bloque_recomendaciones(D, y):
    encabezado(4, 'Qué debe hacer el INGUAT', 'Cinco decisiones que se desprenden directamente de la evidencia.', y)
    c = D['cmp']
    es, te, us, ae, ma = (c['País: El Salvador'], c['Vía: Terrestre'], c['País: Estados Unidos de América'],
                          c['Vía: Aérea'], c['Vía: Marítima'])
    ind = INDICE_ESTACIONAL
    bajo = (abs(ind['Feb'] - 100) + abs(ind['Sep'] - 100)) / 2
    recs = [
        ('Invertir donde está el crecimiento: el vecindario y la frontera terrestre.',
         f'El Salvador (+{es["TCAC_09_19_%"]:.1f} % anual) y la vía terrestre (+{te["TCAC_09_19_%"]:.1f} %) crecen más del doble '
         f'que EE. UU. (+{us["TCAC_09_19_%"]:.1f} %) y la aérea (+{ae["TCAC_09_19_%"]:.1f} %). Promoción regional prioritaria.'),
        ('Planificar personal y capacidad con el calendario, no con el promedio.',
         f'Más personal y capacidad en diciembre (+{ind["Dic"] - 100:.0f} %), Semana Santa y julio–agosto; '
         f'febrero y septiembre (−{bajo:.0f} %) para ofertas de temporada baja.'),
        ('Tratar a EE. UU. como un mercado planificable y de temporada.',
         f'Es la serie más estacional ({us["F_estacional"]:.2f}) y la de mejor recuperación entre los países '
         f'({us["rec_homog_%"]:.0f} % de 2019): campañas anticipadas a verano y Navidad rinden más que esfuerzos '
         f'de volumen todo el año.'),
        ('Decidir explícitamente el futuro de los cruceros.',
         f'La vía marítima está en {ma["rec_homog_%"]:.0f} % de su nivel de 2019 y es la más volátil del estudio. '
         f'O se invierte en puertos y acuerdos con navieras, o se reasigna su presupuesto de promoción.'),
        ('Presupuestar con pronóstico y corregir la medición.',
         f'Un modelo simple anticipa 18 meses con ~{D["mape_ctrl"]:.0f} % de error. Publicar una serie con definición '
         f'constante y marcar los cierres como "no medido".'),
    ]
    yy = y + 0.62
    for i, (neg, ev) in enumerate(recs, 1):
        A.add_patch(Circle((M + 0.11, yy + 0.085), 0.095, facecolor=AZUL, edgecolor='none'))
        txt(M + 0.11, yy + 0.088, str(i), 7.6, 'bold', 'white', ha='center', va='center')
        txt(M + 0.30, yy, neg, 10.5, 'bold', TEXTO)
        fin = parrafo(M + 0.30, yy + 0.18, ev, 8.4, 'normal', TEXTO_2, CW - 0.30, 1.22)
        yy = fin + 0.07
    return yy


def bloque_ficha(y):
    A.plot([M, W - M], [y, y], color=GRIS, lw=0.6)
    cw = CW / 3
    fondo = 0
    cols = [
        ('FUENTE', f'Base de Migración 2009–2026 (jun), INGUAT / IGM. {N_REGISTROS:,} registros mensuales. '
                   f'Datos de uso académico.'),
        ('MÉTODO', '10 modelos por serie (SARIMA, Prophet, Holt-Winters, suavizamiento exponencial, ingenuo '
                   'estacional). Partición 70/30 y partición de control post-pandemia.'),
        ('AUTORES', 'Rodrigo Ajmac 22279 · Andrés Mazariegos 21749 · June Herrera 231038 · '
                    'CC3084 Data Science, UVG 2026'),
    ]
    for i, (t, s) in enumerate(cols):
        x = M + i * cw
        txt(x, y + 0.07, t, 7.6, 'bold', TEXTO_2)
        fondo = max(fondo, parrafo(x, y + 0.20, s, 7.6, 'normal', TEXTO_2, cw - 0.25, 1.25))
    return fondo


# ----------------------------------------------------------------------------------------
# 6. PRINCIPAL
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Genera la infografía del Laboratorio 9.')
    ap.add_argument('--series', default='series_mensuales.csv')
    ap.add_argument('--resultados', default='resultados.json')
    ap.add_argument('--salida', default='salidas')
    ap.add_argument('--estricto', action='store_true',
                    help='falla si el MAPE reproducido se aleja más de 1 punto del laboratorio')
    a = ap.parse_args()

    print(f'Fuente tipográfica: {FUENTE}')
    S = cargar_series(a.series)
    R = cargar_resultados(a.resultados)
    D = calcular(S, R, estricto=a.estricto)

    print('\nCifras que aparecen en la infografía (comparar con el documento de explicación):')
    print(f'  KPI 1  nivel 2025 vs 2019 (homogéneo) : {D["rec_hom"]:.1f} %   (documento: 124.1 %)')
    print(f'  KPI 2  crecimiento anual 2009-2019    : {D["tcac"]:.2f} %   (documento: 8.74 %)')
    print(f'  KPI 3  caída interanual abril 2020    : {D["yoy_abr20"]:.1f} %   (documento: -97.7 %; la tarjeta muestra {fmt_pct(D["yoy_abr20"])})')
    print(f'  KPI 4  MAPE del mejor modelo (control): {D["mape_ctrl"]:.1f} %   (documento: 7.2 %)')
    print(f'  Mínimo mensual del total              : {D["min_total"]:,.0f} ({D["fecha_min"]:%Y-%m})   (documento: 9,779 mayo 2020)')
    print(f'  Total 2025 / estimación 2027          : {D["total_2025_M"]:.2f} M / {D["total_2027_M"]:.2f} M   (documento: 3.6 M / ~3.6 M)')
    print(f'  Modelo ganador por RMSE (control)     : {D["ganador"]}')
    print(f'  Índice estacional: diferencia máx. contra recálculo desde el CSV = {D["dif_indice"]:.1f} puntos')

    bloque_titular(D)
    y1 = 2.94
    bloque_panorama(D, S, y1)
    y2 = 5.82
    encabezado(2, 'Un calendario que se repite cada año y seis series con personalidad propia',
               'Izquierda: cuánto se aparta cada mes del mes promedio (2009–2019). '
               'Derecha: las cuatro preguntas del laboratorio, con evidencia.', y2)
    bloque_estacionalidad(D, y2 + 0.70)
    bloque_cuadrante(D, y2 + 0.70)
    y3 = 10.22
    fin3 = bloque_modelos(D, S, y3)
    y4 = fin3 + 0.16
    fin4 = bloque_recomendaciones(D, y4)
    y_f = max(fin4 + 0.03, 16.28)
    fondo = bloque_ficha(y_f)
    if fondo > H - 0.08:
        print(f'  [REVISAR] la ficha termina en y={fondo:.2f} in y casi toca el borde de la página ({H} in)')
    print(f'\nLayout: bloque 3 termina en y={fin3:.2f} in, recomendaciones en y={fin4:.2f} in, '
          f'ficha desde y={y_f:.2f} in (página: {H} in)')

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    meta = {'Title': 'El turismo de Guatemala ya superó la pandemia',
            'Author': 'Rodrigo Ajmac, Andrés Mazariegos, June Herrera'}
    FIG.savefig(salida / 'infografia.png', dpi=DPI, facecolor='white')
    FIG.savefig(salida / 'infografia.pdf', facecolor='white', metadata=meta)
    from PIL import Image
    print('PNG:', Image.open(salida / 'infografia.png').size, '| PDF: 11 x 17 in, 1 página ->', salida.resolve())


if __name__ == '__main__':
    main()
