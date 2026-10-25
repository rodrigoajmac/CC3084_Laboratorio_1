# -*- coding: utf-8 -*-
"""
Observatorio de turismo · Guatemala — tablero interactivo (CC3084, Laboratorio 10)

Reutiliza los datos y modelos del Laboratorio 1 (no entrena modelos nuevos):
  series_mensuales.csv  -> 7 series mensuales (total, 3 países, 3 vías)
  resultados.json       -> comparativo de series y métricas de los modelos
  predicciones.csv      -> (opcional) predicciones de cada modelo; lo genera 03_modelado_con_export.py

Ejecutar:   streamlit run app.py
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Observatorio de turismo · Guatemala', page_icon='🧭',
                   layout='wide', initial_sidebar_state='expanded')

# ------------------------------------------------------------------ paleta (un significado por color)
AZUL, AZUL_C, GRIS = '#1F4E79', '#9DBBD9', '#B9C0C8'      # azul = dato observado; claro = apoyo; gris = contexto
NARANJA, NARANJA_T = '#D9661F', '#A84A12'                  # naranja = atención / estimación de un modelo
TEXTO, TEXTO_2, FONDO_T, REJILLA = '#1B2733', '#55606C', '#F2F5F8', '#E3E8ED'
COLORES_MODELO = ['#D9661F', '#6A4C93', '#2A8C82']         # solo para distinguir hasta 3 modelos (siempre con etiqueta)
MESES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
FUENTE = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

CATALOGO = {  # clave: (etiqueta, etiqueta corta, clave en resultados.json)
    'total': ('Total de viajeros', 'Total', 'Total de viajeros'),
    'es': ('El Salvador', 'El Salvador', 'País: El Salvador'),
    'gt': ('Guatemala (residentes que regresan)', 'Guatemala', 'País: Guatemala'),
    'us': ('Estados Unidos', 'EE. UU.', 'País: Estados Unidos de América'),
    'aerea': ('Vía aérea', 'Aérea', 'Vía: Aérea'),
    'terrestre': ('Vía terrestre', 'Terrestre', 'Vía: Terrestre'),
    'maritima': ('Vía marítima (cruceros)', 'Marítima', 'Vía: Marítima'),
}
CORTO_A_CLAVE = {v[1]: k for k, v in CATALOGO.items()}
MODELOS = {
    'SARIMA(1,0,1)(1,0,1,12)': ('SARIMA (1,0,1)(1,0,1)', 'Modelo estadístico que proyecta la tendencia y el calendario anual.'),
    'SARIMA(0,0,1)(0,0,1,12)': ('SARIMA (0,0,1)(0,0,1)', 'Variante más simple del SARIMA.'),
    'SARIMA(1,0,0)(1,0,0,12)': ('SARIMA (1,0,0)(1,0,0)', 'Variante del SARIMA basada en el pasado inmediato.'),
    'SARIMA(2,0,0)(1,0,0,12)': ('SARIMA (2,0,0)(1,0,0)', 'Variante del SARIMA basada en el pasado inmediato.'),
    'Seasonal naive': ('Ingenuo estacional', 'Repite lo que ocurrió en el mismo mes del año anterior.'),
    'Holt-Winters': ('Holt-Winters', 'Promedio ponderado que sigue la tendencia y el calendario.'),
    'Suav. exponencial simple': ('Suavizado exponencial', 'Promedio que da más peso a lo reciente; ignora el calendario.'),
    'Prophet': ('Prophet', 'Modelo de tendencia + calendario desarrollado por Meta.'),
}


def nombre_modelo(m):
    """Nombre legible; las variantes SARIMA se muestran como SARIMA (p,d,q)(P,D,Q)."""
    q = re.match(r'SARIMA\((\d),(\d),(\d)\)\((\d),(\d),(\d),12\)', m)
    if q:
        return 'SARIMA (%s,%s,%s)(%s,%s,%s)' % q.groups()
    return MODELOS.get(m, (m, ''))[0]


# ------------------------------------------------------------------ estilo
st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: {FUENTE}; }}
h1 {{ font-size: 1.8rem !important; color: {TEXTO}; margin-bottom: 0; }}
h3 {{ font-size: 1.15rem !important; color: {TEXTO}; margin-bottom: 0.1rem; }}
[data-testid="stMetric"] {{ background: {FONDO_T}; border-left: 4px solid {AZUL}; padding: 10px 14px; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.85rem; color: {TEXTO_2}; }}
.etq-obs {{ color: {AZUL}; font-weight: 600; }}
.etq-est {{ color: {NARANJA_T}; font-weight: 600; }}
.que {{ color: {TEXTO_2}; font-size: 0.9rem; margin: 0 0 0.3rem 0; }}
.nota {{ background: {FONDO_T}; border-left: 4px solid {NARANJA}; padding: 8px 12px; font-size: 0.9rem; color: {TEXTO}; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ datos
AQUI = Path(__file__).resolve().parent


def localizar(nombre, obligatorio=True):
    for base in (AQUI, AQUI / 'datos', AQUI / 'resultados', Path.cwd(), Path.cwd() / 'datos'):
        if (base / nombre).is_file():
            return base / nombre
    if obligatorio:
        st.error(f'No encuentro **{nombre}**. Colóquelo junto a `app.py` (o en `datos/`).')
        st.stop()
    return None


def _canon(col):
    c = unicodedata.normalize('NFKD', str(col)).encode('ascii', 'ignore').decode().lower()
    for k, clave in (('total', 'total'), ('salvador', 'es'), ('guatemala', 'gt'), ('estados', 'us'), ('usa', 'us'),
                     ('aere', 'aerea'), ('terrest', 'terrestre'), ('maritim', 'maritima')):
        if k in c:
            return clave
    return None


@st.cache_data(show_spinner=False)
def cargar():
    df = pd.read_csv(localizar('series_mensuales.csv'), index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
    cols = {}
    for c in df.columns:
        k = _canon(c)
        if k and k not in cols:
            cols[k] = c
    S = pd.DataFrame({k: df[v].astype(float) for k, v in cols.items()}).asfreq('MS')
    S['nores'] = S['total'] - S['gt']
    R = json.load(open(localizar('resultados.json'), encoding='utf-8'))
    ruta_p = localizar('predicciones.csv', obligatorio=False)
    P = pd.read_csv(ruta_p, parse_dates=['fecha']) if ruta_p else None
    return S, R, P


S, R, P = cargar()
CMP = R.get('comparativo', {})


def fmt_miles(v):
    return f'{v / 1e3:,.0f} mil'


# ------------------------------------------------------------------ utilidades de gráfica
def base(fig, alto=360):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=10, b=10), template='plotly_white',
                      font=dict(family=FUENTE, size=13, color=TEXTO), hoverlabel=dict(font_size=13),
                      legend=dict(orientation='h', y=1.12, x=0), paper_bgcolor='white', plot_bgcolor='white')
    fig.update_xaxes(gridcolor=REJILLA, linecolor=GRIS)
    fig.update_yaxes(gridcolor=REJILLA, linecolor=GRIS)
    return fig


def sin_atenuar(fig):
    fig.update_traces(unselected=dict(marker=dict(opacity=1)), selector=dict(type='bar'))
    return fig


def titulo(pregunta, que, para_que, tipo='obs'):
    etq = {'obs': '<span class="etq-obs">● Dato observado</span>',
           'est': '<span class="etq-est">◆ Estimación de un modelo</span>',
           'mix': '<span class="etq-obs">● Observado</span> · <span class="etq-est">◆ estimado</span>'}[tipo]
    st.markdown(f'### {pregunta}')
    st.markdown(f'<p class="que">{etq} — {que} <b>Sirve para:</b> {para_que}</p>', unsafe_allow_html=True)


def aviso(msg):
    st.markdown(f'<div class="nota">{msg}</div>', unsafe_allow_html=True)


def puntos(key):
    ev = st.session_state.get(key)
    try:
        return list(ev['selection']['points']) if ev else []
    except Exception:
        return []


def forma_rect(x0, x1, color, op):
    return dict(type='rect', xref='x', yref='paper', x0=x0, x1=x1, y0=0, y1=1, fillcolor=color, opacity=op,
                line_width=0, layer='below')


def forma_linea(x, color, dash):
    return dict(type='line', xref='x', yref='paper', x0=x, x1=x, y0=0, y1=1, line=dict(color=color, width=1, dash=dash))


# ------------------------------------------------------------------ estado y enlaces entre gráficas
ss = st.session_state
ss.setdefault('serie', 'total')
ss.setdefault('anio', 2025)
ss.setdefault('rango_sel', None)
ss.setdefault('n_serie', 0)

# (A) clic en una barra del comparativo -> cambia la serie activa de todo el tablero
pts = puntos('g_rank')
firma = tuple((p.get('y'), p.get('point_index')) for p in pts)
if firma and firma != ss.get('_firma_rank'):
    clave = CORTO_A_CLAVE.get(pts[0].get('y'))
    if clave and clave != 'total':
        ss['serie'] = clave
ss['_firma_rank'] = firma

# (B) clic en una barra anual -> cambia el año del detalle mensual
pts = puntos('g_anual')
firma = tuple((p.get('x'), p.get('point_index')) for p in pts)
if firma and firma != ss.get('_firma_anual'):
    try:
        ss['anio'] = int(round(float(pts[0]['x'])))
    except Exception:
        pass
ss['_firma_anual'] = firma

# ------------------------------------------------------------------ barra lateral: navegación y controles globales
with st.sidebar:
    st.markdown('## 🧭 Observatorio de turismo')
    st.caption('Flujo de viajeros a Guatemala · INGUAT/IGM · 2009 – jun 2026')
    pagina = st.radio('Ir a', ['1 · Panorama', '2 · Calendario y mercados', '3 · Pronósticos', '4 · Guía y límites'],
                      label_visibility='collapsed')
    st.divider()
    st.markdown('**Qué serie quiere mirar**')
    serie = st.selectbox('Serie', list(CATALOGO), key='serie', format_func=lambda k: CATALOGO[k][0],
                         help='También puede cambiarla haciendo clic en una barra del comparativo.')
    medida = 'nores'
    if serie == 'total':
        medida = st.radio('Definición de la medida', ['nores', 'total'], horizontal=False,
                          format_func=lambda m: {'nores': 'Visitantes no residentes (comparable 2009–2026)',
                                                 'total': 'Total registrado (incluye residentes hasta 2022)'}[m],
                          help='Desde 2023 la fuente dejó de contar a los guatemaltecos que regresan; '
                               'la primera opción evita comparar peras con manzanas.')
    anios = st.slider('Años a mostrar', 2009, 2026, (2009, 2026))
    nivel = st.radio('Nivel de detalle de la línea de tiempo', ['Mensual', 'Trimestral', 'Anual'], horizontal=True,
                     help='Más detalle = cada mes; menos detalle = totales por trimestre o por año.')
    opciones_anio = list(range(anios[1], anios[0] - 1, -1))
    if ss['anio'] not in opciones_anio:
        ss['anio'] = opciones_anio[0]
    anio_det = st.selectbox('Año para ver en detalle', opciones_anio, key='anio',
                            help='También puede elegirlo haciendo clic en una barra anual.')

contexto = (serie, medida, anios, nivel)
if contexto != ss.get('_contexto_panorama'):
    ss['rango_sel'] = None
    ss['n_serie'] += 1
ss['_contexto_panorama'] = contexto

s_full = S['nores'] if (serie == 'total' and medida == 'nores') else S[serie]
s = s_full.loc[f'{anios[0]}-01-01':f'{anios[1]}-12-31']
etq_serie = CATALOGO[serie][0] + (' — no residentes' if (serie == 'total' and medida == 'nores') else '')
clave_json = CATALOGO[serie][2]


def agregar(x, nivel):
    if nivel == 'Trimestral':
        y = x.resample('QS').sum()
        lab = [f'T{d.quarter} {d.year}' for d in y.index]
    elif nivel == 'Anual':
        y = x.resample('YS').sum()
        lab = [str(d.year) for d in y.index]
    else:
        y = x
        lab = [f'{MESES[d.month - 1]} {d.year}' for d in y.index]
    return y, lab


def rango_de_puntos(ps, nivel):
    """Convierte puntos agregados en el rango completo de meses que representan."""
    frecuencia = {'Mensual': 'M', 'Trimestral': 'Q', 'Anual': 'Y'}[nivel]
    periodos = pd.to_datetime([p['x'] for p in ps]).to_period(frecuencia)
    return periodos.min().start_time, periodos.max().end_time


def nota_medicion():
    if serie == 'gt':
        aviso('<b>Atención:</b> a partir de 2023 la fuente dejó de registrar a los guatemaltecos que regresan al país. '
              'Los ceros desde esa fecha son un cambio de medición, <b>no</b> una caída del turismo.')
    elif serie == 'total' and medida == 'total':
        aviso('<b>Atención:</b> la caída de 2023 en esta línea se debe a que se dejó de contar a los residentes '
              'guatemaltecos. Para comparar años use «Visitantes no residentes».')
    elif serie in ('aerea', 'terrestre', 'maritima'):
        aviso('Las vías incluyen a los residentes hasta 2022; la comparación con años posteriores debe leerse con cautela.')


# ================================================================== PÁGINA 1 · PANORAMA
if pagina.startswith('1'):
    st.title('¿Cómo ha evolucionado el turismo en Guatemala?')
    st.markdown(f'Serie seleccionada: **{etq_serie}**, {anios[0]}–{anios[1]}. '
                'Seleccione un periodo en V1, una serie en V2 o un año en V3 para ver más detalle.')
    nota_medicion()

    # V1 -------------------------------------------------------------------------------------------
    titulo('¿Cuánto flujo hay y cómo ha cambiado en el tiempo?',
           'Viajeros que ingresan al país en cada periodo.',
           'ver tendencia, estacionalidad y el choque de la pandemia. <i>Arrastre sobre la línea para elegir un periodo.</i>')
    y, lab = agregar(s, nivel)
    if y.empty or float(y.sum()) == 0:
        st.info('No hay datos para la selección actual. Amplíe el rango de años o cambie de serie.')
    else:
        fig = go.Figure(go.Scatter(x=y.index, y=y.values / 1e3, mode='lines+markers', customdata=lab,
                                   line=dict(color=AZUL, width=2), marker=dict(size=4 if nivel == 'Mensual' else 7, color=AZUL),
                                   hovertemplate='%{customdata}<br><b>%{y:,.0f} mil viajeros</b><extra></extra>'))
        fig.add_shape(forma_rect('2020-03-01', '2021-06-30', NARANJA, 0.13))
        fig.add_shape(forma_linea('2023-01-01', TEXTO, 'dash'))
        fig.add_annotation(x='2020-04-01', y=1, yref='paper', text='Pandemia', showarrow=False, xanchor='left',
                           yanchor='top', font=dict(color=NARANJA_T, size=12))
        fig.add_annotation(x='2023-02-01', y=1, yref='paper', text='Ene-2023: cambio de medición', showarrow=False,
                           xanchor='left', yanchor='top', font=dict(size=12))
        rs = ss.get('rango_sel')
        if rs:
            fig.add_shape(forma_rect(str(rs[0].date()), str(rs[1].date()), AZUL, 0.10))
        fig.update_yaxes(rangemode='tozero', title='miles de viajeros por periodo')
        fig.update_layout(dragmode='select', selectdirection='h')
        base(fig, 380)
        st.plotly_chart(fig, key=f"g_serie{ss['n_serie']}", on_select='rerun', selection_mode=('box', 'points'), width='stretch')
        if nivel == 'Anual' and 2026 in y.index.year:
            st.caption('El punto de 2026 incluye solo enero–junio; no representa un año completo.')
        ps = puntos(f"g_serie{ss['n_serie']}")
        if ps:
            ss['rango_sel'] = rango_de_puntos(ps, nivel)
        else:
            ss['rango_sel'] = None
    rs = ss.get('rango_sel')
    sp = s.loc[rs[0]:rs[1]] if rs else s
    c1, c2, c3, c4 = st.columns(4)
    per = f'{rs[0]:%b %Y} – {rs[1]:%b %Y}' if rs else f'{anios[0]}–{anios[1]}'
    c1.metric(f'Viajeros en el periodo', f'{sp.sum() / 1e6:,.2f} M', help=f'Periodo: {per}')
    c2.metric('Promedio mensual', fmt_miles(sp.mean()) if len(sp) else 'n/d')
    if len(sp) and sp.max() > 0:
        c3.metric('Mes de mayor flujo', f'{MESES[sp.idxmax().month - 1]} {sp.idxmax().year}', fmt_miles(sp.max()))
    else:
        c3.metric('Mes de mayor flujo', 'n/d')
    rec = CMP.get(clave_json, {}).get('rec_homog_%')
    c4.metric('Nivel 2025 vs. 2019', f'{rec:.0f} %' if rec is not None and np.isfinite(rec) else 'n/d',
              help='Indicador fijo de 2025 frente a 2019; no cambia con el periodo seleccionado. '
                   'Usa la misma definición en ambos años. 100 % = igual que antes de la pandemia.')
    if rs:
        st.caption(f'Periodo elegido en la línea de tiempo: **{per}**. Las tarjetas lo usan; '
                   'el perfil mensual de la página 2 lo usa si abarca al menos dos años.')
        if st.button('Quitar periodo seleccionado'):
            ss['rango_sel'] = None
            ss['n_serie'] += 1          # nueva clave = la gráfica vuelve a empezar sin selección
            st.rerun()

    col_a, col_b = st.columns([1, 1])
    # V2 -------------------------------------------------------------------------------------------
    with col_a:
        titulo('¿Qué mercados y vías destacan?',
               'Comparativo de las 6 series según la métrica elegida.',
               'decidir dónde enfocar la promoción. <i>Haga clic en una barra para cambiar de serie.</i>')
        metricas = {'TCAC_09_19_%': ('Crecimiento anual 2009–2019 (%)', '{:.1f} %'),
                    'F_estacional': ('Estacionalidad (0 a 1)', '{:.2f}'),
                    'volatilidad_yoy': ('Volatilidad (variación interanual)', '{:.2f}'),
                    'rec_homog_%': ('Nivel 2025 vs. 2019 (%)', '{:.0f} %')}
        m = st.selectbox('Métrica', list(metricas), format_func=lambda k: metricas[k][0], key='metrica_rank')
        filas = [(CATALOGO[k][1], CMP.get(CATALOGO[k][2], {}).get(m)) for k in CATALOGO if k != 'total']
        filas = [(n, (float(v) if v is not None and np.isfinite(v) else None)) for n, v in filas]
        filas.sort(key=lambda t: (t[1] is None, -(t[1] or 0)))
        tope = 0.3 if m == 'volatilidad_yoy' else None
        fig = go.Figure()
        nombres = [n for n, _ in filas]
        for n, v in filas:
            if v is None:
                fig.add_annotation(x=0, y=n, text='n/d: dejó de medirse', showarrow=False, xanchor='left',
                                   font=dict(color=TEXTO_2, size=12))
                continue
            largo = min(v, tope) if tope else v
            sel = CORTO_A_CLAVE[n] == serie
            fig.add_trace(go.Bar(x=[largo], y=[n], orientation='h', marker_color=AZUL if sel else AZUL_C,
                                 text=[metricas[m][1].format(v) + (' ≫' if tope and v > tope else '')],
                                 textposition='outside', cliponaxis=False,
                                 hovertemplate=f'{n}<br>{metricas[m][0]}: {metricas[m][1].format(v)}<extra></extra>',
                                 showlegend=False))
        if m == 'rec_homog_%':
            fig.add_shape(type='line', x0=100, x1=100, y0=0, y1=1, yref='paper', line=dict(color=TEXTO_2, dash='dot', width=1))
        fig.update_yaxes(categoryorder='array', categoryarray=nombres[::-1], title=None)
        fig.update_xaxes(rangemode='tozero', title=metricas[m][0])
        fig.update_layout(barmode='overlay')
        base(fig, 330)
        fig.update_layout(margin=dict(l=10, r=60, t=10, b=10))
        sin_atenuar(fig)
        st.plotly_chart(fig, key='g_rank', on_select='rerun', selection_mode=('points',), width='stretch')
        st.caption('Azul oscuro = serie activa. ' + ('Una barra fue recortada (≫) por estar fuera de escala.' if tope else ''))

    # V3 -------------------------------------------------------------------------------------------
    with col_b:
        titulo('¿Cómo les fue a los años completos?',
               'Total anual de viajeros de la serie activa.',
               'comparar años y fijar el año que quiere ver en detalle. <i>Haga clic en una barra.</i>')
        anual = s.resample('YS').sum()
        anual = anual[anual.index.year < 2026]
        if anual.empty or anual.sum() == 0:
            st.info('No hay años completos en esta selección. Use el detalle mensual para revisar 2026.')
        else:
            años = anual.index.year
            cols = [AZUL if a == anio_det else AZUL_C for a in años]
            fig = go.Figure(go.Bar(x=años, y=anual.values / 1e6, marker_color=cols,
                                   hovertemplate='%{x}<br><b>%{y:,.2f} millones</b><extra></extra>'))
            fig.update_xaxes(dtick=2, title=None)
            fig.update_yaxes(title='millones de viajeros', rangemode='tozero')
            base(fig, 330)
            sin_atenuar(fig)
            st.plotly_chart(fig, key='g_anual', on_select='rerun', selection_mode=('points',), width='stretch')
            if anio_det == 2026:
                st.caption('2026 se excluye porque solo incluye enero–junio; su detalle está abajo.')
            else:
                st.caption('Azul oscuro = año en detalle. 2026 se excluye porque solo incluye enero–junio.')

    # V4 -------------------------------------------------------------------------------------------
    titulo(f'¿Qué pasó mes a mes en {anio_det}?',
           'Detalle mensual del año elegido frente al año anterior y a 2019.',
           'ver en qué meses se ganó o perdió terreno. Es el nivel de máximo detalle.')
    ya = s_full[s_full.index.year == anio_det]
    if ya.empty or ya.sum() == 0:
        st.info(f'No hay datos de {anio_det} para esta serie. Elija otro año o serie.')
    else:
        if anio_det == 2026:
            st.caption('2026 contiene solo enero–junio. Las referencias se muestran para esos mismos meses.')
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[MESES[d.month - 1] for d in ya.index], y=ya.values / 1e3, name=str(anio_det),
                             marker_color=AZUL, hovertemplate='%{x}<br><b>%{y:,.0f} mil</b><extra></extra>'))
        prev = s_full[s_full.index.year == anio_det - 1]
        prev = prev[prev.index.month.isin(ya.index.month)]
        if len(prev):
            fig.add_trace(go.Scatter(x=[MESES[d.month - 1] for d in prev.index], y=prev.values / 1e3, mode='lines+markers',
                                     name=str(anio_det - 1), line=dict(color=GRIS, width=2),
                                     hovertemplate='%{x}<br>%{y:,.0f} mil<extra></extra>'))
        r19 = s_full[s_full.index.year == 2019]
        r19 = r19[r19.index.month.isin(ya.index.month)]
        if anio_det != 2019 and len(r19):
            fig.add_trace(go.Scatter(x=[MESES[d.month - 1] for d in r19.index], y=r19.values / 1e3, mode='lines',
                                     name='2019', line=dict(color=TEXTO_2, width=1.5, dash='dot'),
                                     hovertemplate='%{x}<br>%{y:,.0f} mil<extra></extra>'))
        fig.update_xaxes(categoryorder='array', categoryarray=MESES)
        fig.update_yaxes(title='miles de viajeros', rangemode='tozero')
        base(fig, 320)
        st.plotly_chart(fig, key='g_mes', width='stretch')

# ================================================================== PÁGINA 2 · CALENDARIO Y MERCADOS
elif pagina.startswith('2'):
    st.title('¿Cuándo llegan los viajeros y quiénes se recuperaron?')
    st.markdown(f'Serie seleccionada: **{etq_serie}**.')
    nota_medicion()

    # V5 heatmap -----------------------------------------------------------------------------------
    titulo('¿Qué meses concentran el flujo cada año?',
           'Viajeros por mes (columnas) y año (filas); más oscuro = más viajeros.',
           'ubicar temporadas altas y bajas y ver cómo cambió el patrón con la pandemia.')
    if s.empty or s.sum() == 0:
        st.info('No hay datos para esta selección.')
    else:
        piv = s.groupby([s.index.year, s.index.month]).sum().unstack()
        piv.columns = [MESES[c - 1] for c in piv.columns]
        z = piv.reindex(columns=MESES) / 1e3
        fig = go.Figure(go.Heatmap(z=z.values, x=MESES, y=z.index.astype(str), colorscale=[[0, FONDO_T], [1, AZUL]], zmin=0,
                                   colorbar=dict(title='miles'), xgap=1, ygap=1,
                                   hovertemplate='%{y} · %{x}<br><b>%{z:,.0f} mil viajeros</b><extra></extra>'))
        fig.update_yaxes(autorange='reversed', type='category', title=None)
        base(fig, 420)
        st.plotly_chart(fig, key='g_heat', width='stretch')

    c_a, c_b = st.columns(2)
    # V6 perfil estacional -------------------------------------------------------------------------
    with c_a:
        rs = ss.get('rango_sel')
        if rs and (rs[1] - rs[0]).days >= 700:
            base_p, desc = s_full.loc[rs[0]:rs[1]], f'periodo elegido en la página 1 ({rs[0]:%b %Y}–{rs[1]:%b %Y})'
        else:
            base_p, desc = s_full.loc['2009-01-01':'2019-12-31'], '2009–2019 (antes de la pandemia)'
        titulo('¿Qué tan por encima o por debajo del promedio está cada mes?',
               f'Diferencia de cada mes respecto de un mes promedio, base {desc}.',
               'planificar personal, capacidad y ofertas de temporada baja.')
        prom = base_p.groupby(base_p.index.month).mean()
        if base_p.empty or prom.mean() == 0 or len(prom) < 12:
            st.info('No hay suficientes datos para calcular el patrón mensual de esta selección.')
        else:
            dev = (prom / prom.mean() - 1) * 100
            cols = [NARANJA if d == dev.max() and d > 15 else (AZUL if d >= 3 else (GRIS if d <= -7 else AZUL_C)) for d in dev]
            fig = go.Figure(go.Bar(x=MESES, y=dev.values, marker_color=cols,
                                   text=[f'{d:+.0f} %'.replace('-', '−') for d in dev], textposition='outside', cliponaxis=False,
                                   hovertemplate='%{x}<br><b>%{y:+.1f} % vs. mes promedio</b><extra></extra>'))
            fig.add_shape(type='line', x0=-0.5, x1=11.5, y0=0, y1=0, line=dict(color=TEXTO, width=1))
            fig.update_yaxes(title='% frente al mes promedio', zeroline=False)
            base(fig, 340)
            st.plotly_chart(fig, key='g_perfil', width='stretch')
            st.caption('Naranja = el mes pico; azul = meses altos; gris = temporada baja; azul claro = cercanos al promedio. '
                       'Describe lo ocurrido; no explica por qué.')

    # V7 recuperación ------------------------------------------------------------------------------
    with c_b:
        titulo('¿Quién ya superó el nivel de 2019?',
               'Visitantes de 2025 como porcentaje de los de 2019, con la misma definición.',
               'identificar qué mercados y vías se recuperaron y cuáles no.')
        filas = [(CATALOGO[k][1], CMP.get(CATALOGO[k][2], {}).get('rec_homog_%')) for k in CATALOGO if k != 'total']
        filas = [(n, float(v)) for n, v in filas if v is not None and np.isfinite(v)]
        if not filas:
            st.info('No hay datos de recuperación disponibles.')
        else:
            filas.sort(key=lambda t: t[1])
            fig = go.Figure(go.Bar(x=[v for _, v in filas], y=[n for n, _ in filas], orientation='h',
                                   marker_color=[AZUL if v >= 100 else NARANJA for _, v in filas],
                                   text=[f'{v:.0f} %' for _, v in filas], textposition='outside', cliponaxis=False,
                                   hovertemplate='%{y}<br><b>%{x:.0f} % del nivel de 2019</b><extra></extra>'))
            fig.add_shape(type='line', x0=100, x1=100, y0=0, y1=1, yref='paper', line=dict(color=TEXTO_2, dash='dot', width=1))
            fig.update_xaxes(title='% del nivel de 2019 (100 = igual)', rangemode='tozero')
            base(fig, 340)
            fig.update_layout(margin=dict(l=10, r=50, t=10, b=10))
            st.plotly_chart(fig, key='g_rec', width='stretch')
            st.caption('Azul = igual o por encima de 2019; naranja = por debajo. Guatemala no se compara: dejó de medirse en 2023.')

# ================================================================== PÁGINA 3 · PRONÓSTICOS
elif pagina.startswith('3'):
    st.title('¿Qué tan confiable es el pronóstico?')
    st.markdown(f'Serie seleccionada: **{etq_serie}**. Aquí se comparan modelos ya entrenados en el Laboratorio 1; '
                'las líneas naranja/violeta/verde son **estimaciones**, no datos observados.')
    if serie == 'gt':
        aviso('Guatemala (residentes) dejó de medirse en 2023: sus pronósticos no son interpretables como turismo.')
    det = R.get('control_postcovid_detalle', {}).get(clave_json, {})
    p73 = R.get('series', {}).get(clave_json, {}).get('comparacion', {})
    if not det or not p73:
        st.info('No hay resultados de modelos para esta serie.')
        st.stop()
    comunes = [m for m in det if m in p73]

    # V8 dumbbell ----------------------------------------------------------------------------------
    titulo('¿Qué modelo se equivoca menos, y por qué cambia tanto la prueba?',
           'Error porcentual promedio (MAPE) de cada modelo en dos pruebas: con la pandemia (70/30) y después de ella.',
           'elegir un modelo confiable. Un error menor es mejor.', 'est')
    tabla = pd.DataFrame({m: {'mape_ctrl': det[m]['MAPE'], 'rmse': det[m]['RMSE'], 'mae': det[m]['MAE'],
                              'mape_7030': p73[m]['MAPE']} for m in comunes}).T.sort_values('rmse')
    fig = go.Figure()
    nombres = [nombre_modelo(m) for m in tabla.index]
    for i, (m, f) in enumerate(tabla.iterrows()):
        fig.add_trace(go.Scatter(x=[f['mape_ctrl'], f['mape_7030']], y=[nombre_modelo(m)] * 2, mode='lines',
                                 line=dict(color=GRIS, width=3), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=tabla['mape_7030'], y=nombres, mode='markers', name='Prueba 70/30 (con pandemia)',
                             marker=dict(size=11, color='white', line=dict(color=AZUL, width=2)),
                             hovertemplate='%{y}<br>Con pandemia: <b>%{x:.1f} %</b><extra></extra>'))
    fig.add_trace(go.Scatter(x=tabla['mape_ctrl'], y=nombres, mode='markers', name='Prueba post-pandemia',
                             marker=dict(size=11, color=AZUL),
                             hovertemplate='%{y}<br>Post-pandemia: <b>%{x:.1f} %</b><extra></extra>'))
    fig.update_yaxes(categoryorder='array', categoryarray=nombres[::-1], title=None)
    fig.update_xaxes(title='MAPE: error porcentual promedio (%)', rangemode='tozero')
    base(fig, 330)
    st.plotly_chart(fig, key='g_dumb', width='stretch')
    mej = tabla.index[0]
    aviso(f'<b>Lectura:</b> en la prueba que incluye la pandemia todos los modelos fallan ({tabla["mape_7030"].min():.0f}–'
          f'{tabla["mape_7030"].max():.0f} %) porque se entrenaron antes de una caída que no podían anticipar. '
          f'Después de la pandemia, el modelo con menor RMSE es <b>{nombre_modelo(mej)}</b> (MAPE {tabla.loc[mej, "mape_ctrl"]:.1f} %). '
          'Es una estimación sobre 18 meses de prueba: no garantiza el desempeño futuro.')

    # controles de modelo ---------------------------------------------------------------------------
    st.markdown('#### Compare modelos en detalle')
    cc1, cc2 = st.columns([1, 2])
    particion = cc1.radio('Prueba a mostrar', ['70/30', 'post-pandemia'],
                          format_func=lambda x: {'70/30': '70/30 (incluye la pandemia)', 'post-pandemia': 'Post-pandemia (ene-2025 a jun-2026)'}[x])
    metr = tabla.copy()
    if particion == '70/30':
        metr['mape'] = [p73[m]['MAPE'] for m in metr.index]
        metr['rmse'] = [p73[m]['RMSE'] for m in metr.index]
        metr['mae'] = [p73[m]['MAE'] for m in metr.index]
    else:
        metr['mape'] = metr['mape_ctrl']
    metr = metr.sort_values('rmse')
    por_defecto = list(metr.index[:3])
    elegidos = cc2.multiselect('Modelos (hasta 3)', list(metr.index), default=por_defecto, max_selections=3,
                               format_func=nombre_modelo,
                               help='Pase el cursor sobre un modelo en la guía para ver qué hace.')
    if not elegidos:
        st.info('Seleccione al menos un modelo para ver sus pronósticos.')
    else:
        col_l, col_r = st.columns([3, 2])
        pp = None
        if P is not None:
            pp = P[(P['serie'] == clave_json) & (P['particion'] == particion) & (P['modelo'].isin(elegidos))]
        with col_l:
            titulo('¿Cómo se comparan el pronóstico y la realidad?',
                   'Línea real frente al pronóstico de cada modelo elegido, en los meses de prueba.',
                   'ver en qué momentos el modelo acierta y en cuáles se desvía.', 'mix')
            if pp is None or pp.empty:
                st.info('Aún no hay predicciones guardadas para esta combinación. Ejecute `03_modelado_con_export.py` '
                        'para generar `predicciones.csv` (las métricas de arriba sí están disponibles).')
            else:
                fig = go.Figure()
                real = pp.drop_duplicates('fecha').sort_values('fecha')
                fig.add_trace(go.Scatter(x=real['fecha'], y=real['real'] / 1e3, mode='lines+markers', name='Real',
                                         line=dict(color=AZUL, width=2.5), marker=dict(size=5),
                                         hovertemplate='%{x|%Y-%m}<br>Real: <b>%{y:,.0f} mil</b><extra></extra>'))
                for i, m in enumerate(elegidos):
                    d = pp[pp['modelo'] == m].sort_values('fecha')
                    fig.add_trace(go.Scatter(x=d['fecha'], y=d['pred'] / 1e3, mode='lines', name=nombre_modelo(m),
                                             line=dict(color=COLORES_MODELO[i], width=2, dash=['dash', 'dot', 'dashdot'][i]),
                                             hovertemplate='%{x|%Y-%m}<br>' + nombre_modelo(m) + ': <b>%{y:,.0f} mil</b><extra></extra>'))
                fig.update_yaxes(title='miles de viajeros por mes', rangemode='tozero')
                base(fig, 360)
                st.plotly_chart(fig, key='g_pred', width='stretch')
        with col_r:
            titulo('¿Qué tan cerca queda cada predicción?',
                   'Cada punto es un mes: real (eje horizontal) contra pronóstico (vertical).',
                   'detectar si un modelo subestima o sobrestima de forma sistemática.', 'mix')
            if pp is None or pp.empty:
                st.info('Disponible cuando exista `predicciones.csv`.')
            else:
                fig = go.Figure()
                mx = float(max(pp['real'].max(), pp['pred'].max())) / 1e3
                fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode='lines', line=dict(color=GRIS, dash='dot'),
                                         name='Acierto perfecto', hoverinfo='skip'))
                for i, m in enumerate(elegidos):
                    d = pp[pp['modelo'] == m]
                    fig.add_trace(go.Scatter(x=d['real'] / 1e3, y=d['pred'] / 1e3, mode='markers', name=nombre_modelo(m),
                                             marker=dict(color=COLORES_MODELO[i], size=8, opacity=0.8),
                                             customdata=d['fecha'].dt.strftime('%Y-%m'),
                                             hovertemplate='%{customdata}<br>Real %{x:,.0f} mil · Pronóstico %{y:,.0f} mil<extra></extra>'))
                fig.update_xaxes(title='real (miles)', rangemode='tozero')
                fig.update_yaxes(title='pronóstico (miles)', rangemode='tozero')
                base(fig, 360)
                st.plotly_chart(fig, key='g_scatter', width='stretch')
        st.markdown('**Tabla comparativa** (complementa las gráficas)')
        tt = metr.loc[elegidos, ['mape', 'rmse', 'mae']].copy()
        tt.index = [nombre_modelo(m) for m in tt.index]
        tt.columns = ['MAPE (%)', 'RMSE (miles)', 'MAE (miles)']
        tt['RMSE (miles)'] = tt['RMSE (miles)'] / 1e3
        tt['MAE (miles)'] = tt['MAE (miles)'] / 1e3
        st.dataframe(tt.style.format({'MAPE (%)': '{:.1f}', 'RMSE (miles)': '{:,.1f}', 'MAE (miles)': '{:,.1f}'}),
                     width='stretch')

# ================================================================== PÁGINA 4 · GUÍA Y LÍMITES
else:
    st.title('Guía de lectura y límites')
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('### Cómo leer los colores')
        st.markdown(f'<span class="etq-obs">● Azul</span>: dato observado (oscuro = lo seleccionado, claro = apoyo).<br>'
                    f'<span style="color:{TEXTO_2}">● Gris</span>: contexto o referencia.<br>'
                    f'<span class="etq-est">◆ Naranja</span>: pide atención (pandemia, mes pico, nivel bajo 2019) '
                    'o es una <b>estimación</b> de un modelo.', unsafe_allow_html=True)
        st.markdown('### Observado, interpretado, estimado')
        st.markdown('- **Observado**: lo que registró la fuente (INGUAT/IGM).\n'
                    '- **Interpretación**: cómo leer el patrón; **no** explica causas.\n'
                    '- **Estimación**: lo que un modelo calcula; tiene error y no garantiza el futuro.')
        st.markdown('### Términos')
        st.markdown('- **MAPE**: error porcentual promedio. Un MAPE de 7 % significa que, en promedio, el pronóstico se aparta 7 % de lo real.\n'
                    '- **RMSE / MAE**: error medio en viajeros; el RMSE penaliza más los errores grandes.\n'
                    '- **Estacionalidad**: patrón que se repite cada año (0 = ninguno, 1 = muy marcado).\n'
                    '- **Volatilidad**: cuánto cambia una serie de un año al siguiente.')
    with g2:
        st.markdown('### Límites que conviene tener presentes')
        st.markdown('- Desde **enero de 2023** la fuente dejó de contar a los guatemaltecos que regresan. '
                    'La opción «no residentes» los excluye también de los años anteriores para permitir esa comparación.\n'
                    '- La cobertura de **cruceristas** también cambió desde 2023. La opción «no residentes» '
                    'no corrige la serie marítima; su comparación entre 2019 y 2025 requiere cautela.\n'
                    '- **2026 solo incluye enero–junio**; no se compara como año completo.\n'
                    '- La **pandemia** (mar 2020–jun 2021) es un choque que los modelos no podían prever.\n'
                    '- Los datos muestran **cuánto** y **cuándo**, no **por qué**: no se atribuyen causas.\n'
                    '- Los modelos son del Laboratorio 1 y se evaluaron sobre 18 meses; úselos como referencia, no como garantía.\n'
                    '- Datos de uso **académico**.')
        st.markdown('### Modelos incluidos')
        st.markdown('- **SARIMA** (varias versiones): modelo estadístico que proyecta la tendencia y el calendario anual; '
                    'las versiones difieren en cuánto del pasado usan.')
        for m in ('Seasonal naive', 'Holt-Winters', 'Suav. exponencial simple', 'Prophet'):
            st.markdown(f'- **{MODELOS[m][0]}**: {MODELOS[m][1]}')
