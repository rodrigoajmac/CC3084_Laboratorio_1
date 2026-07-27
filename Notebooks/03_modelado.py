# %% [markdown]
# # Laboratorio 1 — Series de Tiempo
# ## Parte 2: Análisis de las series, modelado y predicción
#
# **CC3084 — Data Science — Universidad del Valle de Guatemala — Semestre II 2026**
#
# *Rodrigo Ajmac — 22279 · Andres Mazariegos — 21749 · June Herrera — 231038*
#
# Este cuaderno cubre los puntos **2, 3, 4 y 5** del enunciado y continúa el trabajo iniciado
# en `02_Analysis_TimeSeries.ipynb`, respetando las series que allí se construyeron:
#
# - **Serie obligatoria:** total mensual de viajeros internacionales.
# - **Categoría i — Países de residencia (Top 3 acumulado):** El Salvador, Guatemala y
#   Estados Unidos de América.
# - **Categoría iii — Vías de ingreso:** Aérea, Terrestre y Marítima.
#
# En total se analizan **7 series mensuales**.
#
# **Nota sobre la definición de la medida.** Las series se construyen sobre el total de
# viajeros **sin filtrar por tipo**, igual que en el cuaderno 02. El enunciado sugiere usar
# `Turista + Excursionista` para comparar en todo el rango, pero bajo esa definición la
# **vía Marítima queda en cero desde 2017** (todo su volumen pasa a registrarse como
# `Cruceristas`), lo que dejaría sin serie a una de las tres vías que el propio enunciado
# exige construir. Se usa por tanto la medida completa y el quiebre metodológico de 2023 se
# documenta y se tiene en cuenta al interpretar los resultados.

# %% [markdown]
# ## 0. Configuración

# %%
import warnings, json, logging
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('prophet').setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from scipy import stats
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from prophet import Prophet

pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 60)

FIG = Path('figuras'); FIG.mkdir(exist_ok=True)
RES = Path('resultados'); RES.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
    'axes.spines.top': False, 'axes.spines.right': False,
})
AZUL, NARANJA, VERDE, ROJO, GRIS = '#1f4e79', '#e07b39', '#2e8b57', '#c0392b', '#7f8c8d'

def guardar(nombre):
    plt.savefig(FIG / f'{nombre}.png')

def miles(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f'{v/1000:,.0f}'))

# Todo lo que se imprime y se necesita en el informe se acumula aquí.
R = {}

# %% [markdown]
# ## 1. Construcción de las series (punto 3)
#
# Se reconstruyen las 7 series desde el archivo crudo, filtrando a las categorías
# consistentes (Turista + Excursionista) y agregando por mes con `groupby` + `sum`.

# %%
RUTA = '../Datos_Crudos/Base_Migracion_2009-2026jun.xlsx'
df = pd.read_excel(RUTA, sheet_name=0)
df['Viajero'] = df['Viajero'].astype(int)
df['fecha'] = pd.to_datetime(
    df['Año'].astype(str) + '-' + df['Mes cod'].astype(str).str.zfill(2) + '-01')

base = df   # medida completa, sin filtrar por tipo de viajero (igual que el cuaderno 02)

IDX = pd.date_range('2009-01-01', '2026-06-01', freq='MS')

def serie_de(mascara):
    return (base[mascara].groupby('fecha')['Viajero'].sum()
            .reindex(IDX).fillna(0).astype(float))

# Rankings sobre el acumulado del período completo (criterio del enunciado)
rank_pais = base.groupby('País')['Viajero'].sum().sort_values(ascending=False)
top3_pais = list(rank_pais.head(3).index)
VIAS = ['Aérea', 'Terrestre', 'Marítima']

print('Top 3 países por total acumulado del período completo (millones):')
print((rank_pais.head(5) / 1e6).round(2).to_string())
print(f'\nSeleccionados: {top3_pais}')
print('\nVías de ingreso (millones):')
print((base.groupby('Vía')['Viajero'].sum().sort_values(ascending=False) / 1e6).round(2).to_string())

TODO = pd.Series(True, index=base.index)
MASCARAS = {'Total de viajeros': TODO}
for p in top3_pais:
    MASCARAS[f'País: {p}'] = base['País'] == p
for v in VIAS:
    MASCARAS[f'Vía: {v}'] = base['Vía'] == v

SERIES = {n: serie_de(m) for n, m in MASCARAS.items()}

CLAVES = {n: n.replace('Total de viajeros', 'total')
             .replace('País: ', 'pais_').replace('Vía: ', 'via_')
             .replace('Estados Unidos de América', 'usa')
             .replace(' ', '_').replace('á', 'a').replace('é', 'e').replace('í', 'i')
             .replace('ó', 'o').replace('ú', 'u').lower() for n in SERIES}

pd.DataFrame(SERIES).to_csv(RES / 'series_mensuales.csv')
print(f'\n{len(SERIES)} series construidas.')

# %% [markdown]
# ### 4.a Inicio, fin y frecuencia de cada serie

# %%
ficha = pd.DataFrame({
    'inicio': [s.index.min().strftime('%Y-%m') for s in SERIES.values()],
    'fin': [s.index.max().strftime('%Y-%m') for s in SERIES.values()],
    'frecuencia': ['mensual (12/año)'] * len(SERIES),
    'n_obs': [len(s) for s in SERIES.values()],
    'meses_en_0': [int((s == 0).sum()) for s in SERIES.values()],
    'media': [s.mean() for s in SERIES.values()],
    'desv_est': [s.std() for s in SERIES.values()],
    'CV_%': [s.std() / s.mean() * 100 for s in SERIES.values()],
    'min': [s.min() for s in SERIES.values()],
    'max': [s.max() for s in SERIES.values()],
}, index=SERIES.keys())
print(ficha.round(1).to_string())
R['ficha'] = ficha.round(2).to_dict('index')

# %% [markdown]
# ## 2. Partición 70 / 30 (punto 2)
#
# El corte es **cronológico**: usar un corte aleatorio filtraría información del futuro
# hacia el entrenamiento y produciría métricas de error falsamente optimistas.

# %%
N = len(IDX)
CORTE = int(N * 0.70)
FECHA_CORTE = IDX[CORTE]

TRAIN = {n: s.iloc[:CORTE] for n, s in SERIES.items()}
TEST = {n: s.iloc[CORTE:] for n, s in SERIES.items()}
H = N - CORTE

print(f'Observaciones totales : {N}')
print(f'Entrenamiento : {IDX[0]:%Y-%m} a {IDX[CORTE-1]:%Y-%m}  ({CORTE} meses, {CORTE/N:.1%})')
print(f'Prueba        : {IDX[CORTE]:%Y-%m} a {IDX[-1]:%Y-%m}  ({H} meses, {H/N:.1%})')
R['split'] = {'n': N, 'n_train': CORTE, 'n_test': H,
              'train_ini': f'{IDX[0]:%Y-%m}', 'train_fin': f'{IDX[CORTE-1]:%Y-%m}',
              'test_ini': f'{IDX[CORTE]:%Y-%m}', 'test_fin': f'{IDX[-1]:%Y-%m}'}

fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
for ax, (n, s) in zip(axes.ravel(), SERIES.items()):
    ax.plot(TRAIN[n].index, TRAIN[n].values, lw=1, color=AZUL, label='Entrenamiento')
    ax.plot(TEST[n].index, TEST[n].values, lw=1, color=NARANJA, label='Prueba')
    ax.axvline(FECHA_CORTE, ls='--', c='k', lw=.8)
    ax.set_title(n, fontsize=9); miles(ax)
axes.ravel()[0].legend(fontsize=7)
axes.ravel()[-1].axis('off')
fig.suptitle('Las 7 series y la partición cronológica 70/30 (miles de viajeros)', y=1.0)
plt.tight_layout(); guardar('20_split_todas'); plt.show()

# %% [markdown]
# El corte cae en **abril de 2021**, dentro del hueco pandémico. Es una partición dura pero
# honesta: el entrenamiento termina en el piso del colapso y el conjunto de prueba contiene
# toda la recuperación. Ningún modelo entrenado hasta esa fecha puede anticipar la reapertura,
# así que los errores del test serán altos por construcción. Se respeta el 70/30 que pide el
# enunciado y al final se reporta una **partición alternativa post-pandemia** como control.

# %% [markdown]
# ## 3. Funciones de análisis y modelado
#
# Se define una vez el procedimiento completo y se aplica idénticamente a las 7 series,
# de modo que los resultados sean comparables entre sí.

# %%
MESES = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

def adf(x, nombre=''):
    """Dickey-Fuller aumentada. H0: hay raíz unitaria (NO estacionaria en media)."""
    x = pd.Series(x).dropna()
    st, p, lags, nobs, crit, _ = adfuller(x, autolag='AIC')
    return {'serie': nombre, 'estadistico': round(st, 3), 'p_valor': round(p, 4),
            'rezagos': int(lags), 'crit_5%': round(crit['5%'], 3),
            'estacionaria': 'sí' if p < 0.05 else 'no'}

def kpss_t(x, nombre=''):
    """KPSS. H0: la serie ES estacionaria. Complementa a la ADF."""
    x = pd.Series(x).dropna()
    st, p, lags, crit = kpss(x, regression='c', nlags='auto')
    return {'serie': nombre, 'estadistico': round(st, 3), 'p_valor': round(p, 4),
            'crit_5%': round(crit['5%'], 3), 'estacionaria': 'sí' if p > 0.05 else 'no'}

# Ventana pre-pandemia. Las medidas de fuerza de componentes y el diagnóstico de varianza
# se calculan aquí: el colapso de 2020-2021 infla la varianza del residuo de la STL en un
# orden de magnitud y hunde artificialmente F_S y F_T (una serie con estacionalidad
# evidente llega a marcar F_S < 0.10). Se reporta también el valor de muestra completa
# para dejar constancia del efecto.
PRE_INI, PRE_FIN = '2009-01', '2019-12'

def fuerza_componentes(s, period=12, pre=True):
    """Fuerza de tendencia y estacionalidad (Wang, Smith & Hyndman) vía STL."""
    x = s.loc[PRE_INI:PRE_FIN] if pre else s
    r = STL(np.log1p(x), period=period, robust=True).fit()
    var_r = np.var(r.resid)
    ft = max(0.0, 1 - var_r / np.var(r.trend + r.resid))
    fs = max(0.0, 1 - var_r / np.var(r.seasonal + r.resid))
    return float(ft), float(fs), r

def diag_varianza(s, pre=True):
    """¿Varianza constante? Se compara la desv. estándar móvil contra la media móvil."""
    x = s.loc[PRE_INI:PRE_FIN] if pre else s
    m = x.rolling(12).mean(); d = x.rolling(12).std()
    ok = m.notna() & d.notna() & (m > 0) & (d > 0)
    rho = float(np.corrcoef(m[ok], d[ok])[0, 1])
    # pendiente de log(sd) ~ log(media): 0 => varianza constante, 1 => sd proporcional al nivel
    pend = float(np.polyfit(np.log(m[ok]), np.log(d[ok]), 1)[0])
    pos = x[x > 0]
    lam = float(stats.boxcox_normmax(pos, method='mle')) if len(pos) > 20 else np.nan
    return {'corr_media_sd': round(rho, 3), 'pendiente_log': round(pend, 3),
            'lambda_boxcox': round(lam, 3)}

def n_diferencias(x, alpha=0.05, maxd=2):
    """Número de diferencias regulares necesarias según la ADF."""
    d, y = 0, pd.Series(x).dropna()
    while d < maxd and adfuller(y, autolag='AIC')[1] >= alpha:
        y = y.diff().dropna(); d += 1
    return d

def metricas(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    mae = mean_absolute_error(real, pred)
    rmse = float(np.sqrt(mean_squared_error(real, pred)))
    m = real > 0
    mape = float(np.mean(np.abs((real[m] - pred[m]) / real[m])) * 100) if m.any() else np.nan
    return mae, rmse, mape

def seasonal_naive(train, h, m=12):
    """Cada predicción es el valor del mismo mes del último ciclo observado."""
    ult = train.iloc[-m:].values
    return np.array([ult[i % m] for i in range(h)])

def a_conteo(x):
    """Devuelve la predicción como arreglo de numpy no negativo (viajeros)."""
    return np.clip(np.asarray(x, dtype=float), 0, None)

# %% [markdown]
# ### Modelos que se ajustan a cada serie
#
# Todos los modelos SARIMA se ajustan sobre **log(1+x)**: la transformación estabiliza la
# varianza (ver diagnóstico de cada serie) y el `+1` evita el `log(0)` de los meses de cierre
# total de fronteras en 2020. Las predicciones se devuelven a la escala original con `expm1`,
# de modo que **MAE y RMSE siempre están en viajeros**, comparables entre modelos.

# %%
def ajusta_sarima(tr_log, orden, orden_est):
    # enforce_stationarity/invertibility = True obliga a que las raíces de los polinomios
    # AR y MA queden fuera del círculo unitario. Sin esta restricción, algunas series
    # (las que tienen meses en cero por el cierre de fronteras de 2020) estiman raíces
    # explosivas y el pronóstico diverge: en la escala original llega a 10^21 viajeros.
    mod = SARIMAX(tr_log, order=orden, seasonal_order=orden_est,
                  enforce_stationarity=True, enforce_invertibility=True)
    return mod.fit(disp=False, maxiter=500)

def evalua_sarima(res, te, h):
    pred_log = res.get_forecast(steps=h).predicted_mean
    pred = a_conteo(np.expm1(np.asarray(pred_log)))
    mae, rmse, mape = metricas(te.values, pred)
    lb = acorr_ljungbox(res.resid[1:], lags=[12], return_df=True)
    # n_efec = observaciones que realmente entran en la verosimilitud tras diferenciar.
    # El AIC/BIC solo es comparable entre modelos que compartan este valor.
    d_, D_, m_ = res.model.k_diff, res.model.k_seasonal_diff, res.model.seasonal_periods
    return {'AIC': float(res.aic), 'BIC': float(res.bic), 'MAE': mae, 'RMSE': rmse,
            'MAPE': mape, 'LjungBox_p': float(lb['lb_pvalue'].iloc[0]),
            'n_efec': int(res.nobs - d_ - D_ * m_)}, pred

def modelos_no_arima(tr, te, h):
    """Holt-Winters, suavizamiento exponencial simple, seasonal naive y Prophet."""
    out, preds = {}, {}

    # Holt-Winters aditivo (sobre log1p para estabilizar varianza)
    try:
        hw = ExponentialSmoothing(np.log1p(tr), trend='add', seasonal='add',
                                  seasonal_periods=12,
                                  initialization_method='estimated').fit()
        p = a_conteo(np.expm1(np.asarray(hw.forecast(h))))
        mae, rmse, mape = metricas(te.values, p)
        out['Holt-Winters'] = {'AIC': float(hw.aic), 'BIC': float(hw.bic),
                               'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'LjungBox_p': np.nan,
                               'n_efec': len(tr)}
        preds['Holt-Winters'] = p
    except Exception as e:
        print('  HW falló:', e)

    # Suavizamiento exponencial simple (sin tendencia ni estacionalidad): línea base
    ses = SimpleExpSmoothing(np.log1p(tr), initialization_method='estimated').fit()
    p = a_conteo(np.expm1(np.asarray(ses.forecast(h))))
    mae, rmse, mape = metricas(te.values, p)
    out['Suav. exponencial simple'] = {'AIC': float(ses.aic), 'BIC': float(ses.bic),
                                       'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'LjungBox_p': np.nan,
                               'n_efec': len(tr)}
    preds['Suav. exponencial simple'] = p

    # Seasonal naive: referencia obligatoria, cualquier modelo debe superarla
    p = seasonal_naive(tr, h)
    mae, rmse, mape = metricas(te.values, p)
    out['Seasonal naive'] = {'AIC': np.nan, 'BIC': np.nan, 'MAE': mae, 'RMSE': rmse,
                             'MAPE': mape, 'LjungBox_p': np.nan, 'n_efec': len(tr)}
    preds['Seasonal naive'] = p

    # Prophet con estacionalidad anual multiplicativa
    dfp = pd.DataFrame({'ds': tr.index, 'y': tr.values})
    pr = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                 seasonality_mode='multiplicative', changepoint_prior_scale=0.5)
    pr.fit(dfp)
    fut = pd.DataFrame({'ds': te.index})
    p = a_conteo(pr.predict(fut)['yhat'])
    mae, rmse, mape = metricas(te.values, p)
    out['Prophet'] = {'AIC': np.nan, 'BIC': np.nan, 'MAE': mae, 'RMSE': rmse,
                      'MAPE': mape, 'LjungBox_p': np.nan, 'n_efec': len(tr)}
    preds['Prophet'] = p

    return out, preds

# %% [markdown]
# ## 4. Análisis completo de cada serie (punto 4)
#
# El bloque siguiente ejecuta, para **cada una de las 7 series**, los incisos 4.a a 4.k:
# gráfico, descomposición, diagnóstico de estacionariedad en varianza y en media
# (ACF + Dickey-Fuller aumentada + KPSS), selección de *p, d, q* por ACF/PACF y por
# `auto_arima`, ajuste de varios SARIMA, análisis de residuos, los cuatro modelos
# alternativos y la comparación final por MAE, RMSE, AIC y BIC.

# %%
def analiza(nombre, s, tr, te, clave):
    print('\n' + '=' * 100)
    print(f'SERIE: {nombre}')
    print('=' * 100)
    res_serie = {'nombre': nombre}

    # ---------- 4.a ficha ----------
    print(f'\n[4.a] Inicio {s.index.min():%Y-%m} | Fin {s.index.max():%Y-%m} | '
          f'Frecuencia mensual (12 obs/año) | n = {len(s)}')

    # ---------- 4.b + 4.c gráfico y descomposición ----------
    ft, fs, _ = fuerza_componentes(s, pre=True)      # caracterización limpia (2009-2019)
    ft_f, fs_f, stl = fuerza_componentes(s, pre=False)  # muestra completa (con pandemia)
    res_serie['fuerza_tendencia'] = round(ft, 3)
    res_serie['fuerza_estacional'] = round(fs, 3)
    res_serie['fuerza_tendencia_full'] = round(ft_f, 3)
    res_serie['fuerza_estacional_full'] = round(fs_f, 3)

    fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(s.index, s.values, lw=1, color=AZUL); ax[0].set_ylabel('serie'); miles(ax[0])
    ax[0].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'), color=ROJO, alpha=.10)
    ax[1].plot(stl.trend.index, stl.trend.values, lw=1.2, color=NARANJA); ax[1].set_ylabel('tendencia')
    ax[2].plot(stl.seasonal.index, stl.seasonal.values, lw=.9, color=VERDE); ax[2].set_ylabel('estacional')
    ax[3].plot(stl.resid.index, stl.resid.values, lw=.8, color=GRIS); ax[3].axhline(0, c='k', lw=.7)
    ax[3].set_ylabel('residuo')
    fig.suptitle(f'{nombre} — serie y descomposición STL sobre log(1+x)\n'
                 f'2009-2019: fuerza de tendencia = {ft:.2f} | fuerza estacional = {fs:.2f}', y=1.0)
    plt.tight_layout(); guardar(f'21_{clave}_descomp'); plt.show()

    print(f'\n[4.c] Fuerza de la tendencia    : {ft:.3f}  (pre-pandemia 2009-2019)')
    print(f'      Fuerza de la estacionalidad: {fs:.3f}  (pre-pandemia 2009-2019)')
    print(f'      Muestra completa 2009-2026 : tendencia {ft_f:.3f} | estacional {fs_f:.3f}')
    print(f'      La caída de ambas medidas al incluir 2020-2021 no significa que la serie pierda')
    print(f'      tendencia o estacionalidad: el residuo de la pandemia domina la varianza total.')

    # ---------- 4.d estacionariedad en varianza y transformación ----------
    dv = diag_varianza(s, pre=True)
    res_serie['varianza'] = dv
    print(f'\n[4.d] Estacionariedad en VARIANZA (diagnóstico sobre 2009-2019):')
    print(f'      corr(media móvil, desv. móvil) = {dv["corr_media_sd"]}')
    print(f'      pendiente log(sd)~log(media)   = {dv["pendiente_log"]}  (0 = varianza constante)')
    print(f'      lambda de Box-Cox              = {dv["lambda_boxcox"]}  (0 = log, 1 = sin transformar)')
    necesita_log = (dv['corr_media_sd'] > 0.4 or dv['pendiente_log'] > 0.4
                    or dv['lambda_boxcox'] < 0.5)
    res_serie['transformacion'] = 'log(1+x)' if necesita_log else 'ninguna'
    print(f'      -> ¿Requiere transformación por varianza? {"SÍ" if necesita_log else "no"}')
    print(f'      Se modela en log(1+x) en todos los casos: además de estabilizar la varianza,')
    print(f'      garantiza predicciones no negativas y hace comparables los AIC entre series.')

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
    ax[0].plot(s.index, s.rolling(12).mean(), color=AZUL, lw=1.1, label='media móvil 12m')
    ax[0].plot(s.index, s.rolling(12).std(), color=ROJO, lw=1.1, label='desv. estándar móvil 12m')
    ax[0].set_title('Escala original: la dispersión sigue al nivel'); ax[0].legend(fontsize=7); miles(ax[0])
    ls = np.log1p(s)
    ax[1].plot(s.index, ls.rolling(12).mean(), color=AZUL, lw=1.1, label='media móvil 12m')
    ax[1].plot(s.index, ls.rolling(12).std(), color=ROJO, lw=1.1, label='desv. estándar móvil 12m')
    ax[1].set_title('Escala log(1+x): la dispersión se aplana'); ax[1].legend(fontsize=7)
    fig.suptitle(f'{nombre} — diagnóstico de varianza', y=1.02)
    plt.tight_layout(); guardar(f'22_{clave}_varianza'); plt.show()

    # ---------- 4.e estacionariedad en media: ACF + ADF ----------
    y = np.log1p(s)
    pruebas = pd.DataFrame([
        adf(s, 'original'),
        adf(y, 'log(1+x)'),
        adf(y.diff(), 'log + 1 dif. regular'),
        adf(y.diff().diff(12), 'log + 1 reg. + 1 estacional'),
    ])
    pruebas_k = pd.DataFrame([
        kpss_t(y, 'log(1+x)'),
        kpss_t(y.diff(), 'log + 1 dif. regular'),
        kpss_t(y.diff().diff(12), 'log + 1 reg. + 1 estacional'),
    ])
    print('\n[4.e] Prueba de Dickey-Fuller aumentada (H0: NO estacionaria en media):')
    print(pruebas.to_string(index=False))
    print('\n      Prueba KPSS de contraste (H0: SÍ estacionaria):')
    print(pruebas_k.to_string(index=False))
    res_serie['adf'] = pruebas.to_dict('records')
    res_serie['kpss'] = pruebas_k.to_dict('records')

    # d por la ADF; D por la prueba OCSB de raíz unitaria estacional, no por un umbral ad-hoc.
    d = n_diferencias(y)
    try:
        D = int(pm.arima.nsdiffs(np.log1p(tr), m=12, max_D=1, test='ocsb'))
    except Exception:
        D = 1 if fs > 0.60 else 0
    # Con estacionalidad fuerte y sin diferencia estacional, se fuerza al menos un término
    # estacional AR/MA en los modelos (ver 4.f); si la prueba pide diferenciar, se diferencia.
    res_serie['d'], res_serie['D'] = int(d), int(D)
    print(f'\n      -> d = {d} diferencia(s) regular(es) [ADF]')
    print(f'      -> D = {D} diferencia(s) estacional(es) [prueba OCSB, m = 12]')
    print(f'         Fuerza estacional pre-pandemia = {fs:.2f}: la componente estacional es')
    print(f'         {"fuerte, por lo que todos los modelos llevan términos estacionales." if fs > 0.6 else "moderada."}')

    yd = y.copy()
    for _ in range(d):
        yd = yd.diff()
    if D:
        yd = yd.diff(12)
    yd = yd.dropna()

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    plot_acf(y.dropna(), lags=36, ax=ax[0, 0], title='ACF — log(1+x) sin diferenciar')
    plot_pacf(y.dropna(), lags=36, ax=ax[0, 1], method='ywm', title='PACF — log(1+x) sin diferenciar')
    plot_acf(yd, lags=36, ax=ax[1, 0], title=f'ACF — tras d={d}, D={D}')
    plot_pacf(yd, lags=36, ax=ax[1, 1], method='ywm', title=f'PACF — tras d={d}, D={D}')
    fig.suptitle(f'{nombre} — autocorrelación', y=1.01)
    plt.tight_layout(); guardar(f'23_{clave}_acf'); plt.show()

    # ---------- 4.f selección de p, q por ACF/PACF y por auto_arima ----------
    tr_log = np.log1p(tr)
    auto = pm.auto_arima(tr_log, seasonal=True, m=12, d=d, D=D,
                         start_p=0, start_q=0, max_p=3, max_q=3, max_P=2, max_Q=2,
                         information_criterion='aic', stepwise=True,
                         suppress_warnings=True, error_action='ignore', trace=False)
    o_auto, os_auto = auto.order, auto.seasonal_order
    print(f'\n[4.f] auto_arima propone: SARIMA{o_auto}{os_auto}  (AIC = {auto.aic():.2f})')
    res_serie['auto_arima'] = {'order': list(o_auto), 'seasonal_order': list(os_auto)}

    # ---------- 4.g varios modelos ARIMA ----------
    # Nota: la OCSB puede devolver D = 0 porque detecta *raíz unitaria* estacional, y una
    # estacionalidad determinista y estable (como la de estas series) no la tiene. Por eso,
    # además de los modelos con la D que indica la prueba, se incluye siempre el modelo
    # "aerolínea" clásico (0,1,1)(0,1,1,12) con diferenciación estacional forzada, para que
    # sea la comparación de AIC/BIC y error —y no un supuesto previo— la que decida.
    candidatos = {
        f'SARIMA{o_auto}{os_auto} (auto_arima)': (o_auto, os_auto),
        f'SARIMA(0,{d},1)(0,{D},1,12)': ((0, d, 1), (0, D, 1, 12)),
        f'SARIMA(1,{d},1)(1,{D},1,12)': ((1, d, 1), (1, D, 1, 12)),
        f'SARIMA(2,{d},0)(1,{D},0,12)': ((2, d, 0), (1, D, 0, 12)),
        'SARIMA(0,1,1)(0,1,1,12) (aerolínea)': ((0, 1, 1), (0, 1, 1, 12)),
        'SARIMA(1,1,1)(0,1,1,12)': ((1, 1, 1), (0, 1, 1, 12)),
    }
    tabla, predicciones, ajustados = {}, {}, {}
    tope = float(s.max()) * 10  # cota de cordura: 10 veces el máximo histórico
    for etq, (o, os_) in candidatos.items():
        try:
            fit = ajusta_sarima(tr_log, o, os_)
            m, p = evalua_sarima(fit, te, len(te))
            if not np.isfinite(p).all() or p.max() > tope:
                print(f'  {etq}: descartado, el pronóstico diverge '
                      f'(máx = {p.max():.3g} frente a un máximo histórico de {s.max():,.0f}).')
                continue
            tabla[etq] = m; predicciones[etq] = p; ajustados[etq] = fit
        except Exception as e:
            print(f'  {etq} falló: {e}')

    # ---------- 4.h modelos alternativos ----------
    otros, preds_otros = modelos_no_arima(tr, te, len(te))
    tabla.update(otros); predicciones.update(preds_otros)

    # ---------- 4.j comparación ----------
    comp = pd.DataFrame(tabla).T[['AIC', 'BIC', 'n_efec', 'MAE', 'RMSE', 'MAPE', 'LjungBox_p']]
    comp = comp.sort_values('RMSE')
    print('\n[4.g/h/j] Comparación de todos los modelos (MAE/RMSE en viajeros, orden por RMSE):')
    print(comp.round({'AIC': 1, 'BIC': 1, 'MAE': 0, 'RMSE': 0, 'MAPE': 1, 'LjungBox_p': 3}).to_string())
    print('      MAE, RMSE y MAPE son comparables entre TODOS los modelos (escala de viajeros,')
    print('      mismo conjunto de prueba). AIC y BIC solo son comparables entre modelos con el')
    print('      mismo n_efec: diferenciar cambia la muestra sobre la que se evalúa la verosimilitud.')

    # ---------- 4.k mejor modelo ----------
    mejor = comp.index[0]
    # El mejor SARIMA por AIC se busca solo dentro del grupo más numeroso de igual n_efec,
    # para que la comparación de verosimilitudes sea legítima.
    solo_sarima = comp[comp['AIC'].notna() & comp.index.str.startswith('SARIMA')]
    if len(solo_sarima):
        n_ref = solo_sarima['n_efec'].mode().iloc[0]
        grupo = solo_sarima[solo_sarima['n_efec'] == n_ref]
        mejor_arima_aic = grupo['AIC'].idxmin()
    else:
        n_ref, mejor_arima_aic = None, None
    print(f'\n[4.k] Mejor modelo por RMSE en el conjunto de prueba : {mejor}')
    print(f'      Mejor SARIMA por AIC (entre los de n_efec = {n_ref}): {mejor_arima_aic}')
    res_serie['comparacion'] = comp.round(3).to_dict('index')
    res_serie['mejor_rmse'] = mejor
    res_serie['mejor_aic'] = mejor_arima_aic

    # ---------- 4.i predicción con el mejor modelo ----------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tr.index, tr.values, lw=1, color=AZUL, label='Entrenamiento')
    ax.plot(te.index, te.values, lw=1.4, color='k', label='Prueba (real)')
    for etq, color in zip(comp.index[:3], [ROJO, NARANJA, VERDE]):
        ax.plot(te.index, predicciones[etq], lw=1.2, ls='--', color=color,
                label=f'{etq} (RMSE {comp.loc[etq, "RMSE"]:,.0f})')
    ax.axvline(FECHA_CORTE, ls=':', c='k', lw=.9)
    ax.set(title=f'{nombre} — predicción sobre el conjunto de prueba (3 mejores modelos)',
           ylabel='miles de viajeros'); miles(ax)
    ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout(); guardar(f'24_{clave}_pred'); plt.show()

    # ---------- residuos del mejor SARIMA ----------
    if mejor_arima_aic:
        fit = ajustados[mejor_arima_aic]
        resid = pd.Series(fit.resid[1:], index=tr.index[1:])
        lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
        jb_p = float(stats.jarque_bera(resid)[1])
        print(f'\n      Residuos de {mejor_arima_aic}:')
        print(f'        Ljung-Box(12) p = {lb["lb_pvalue"].iloc[0]:.4f} | '
              f'Ljung-Box(24) p = {lb["lb_pvalue"].iloc[1]:.4f}  (p>0.05 = ruido blanco)')
        print(f'        Jarque-Bera  p = {jb_p:.4f}  (p>0.05 = residuos normales)')
        res_serie['residuos'] = {'modelo': mejor_arima_aic,
                                 'ljungbox12_p': round(float(lb['lb_pvalue'].iloc[0]), 4),
                                 'ljungbox24_p': round(float(lb['lb_pvalue'].iloc[1]), 4),
                                 'jarquebera_p': round(jb_p, 4)}

        fig, ax = plt.subplots(2, 2, figsize=(12, 5.5))
        ax[0, 0].plot(resid.index, resid.values, lw=.8, color=GRIS); ax[0, 0].axhline(0, c='k', lw=.7)
        ax[0, 0].set_title('Residuos en el tiempo')
        ax[0, 1].hist(resid.values, bins=30, color=AZUL, alpha=.75)
        ax[0, 1].set_title('Histograma de residuos')
        plot_acf(resid, lags=36, ax=ax[1, 0], title='ACF de los residuos')
        stats.probplot(resid.values, dist='norm', plot=ax[1, 1]); ax[1, 1].set_title('Q-Q normal')
        fig.suptitle(f'{nombre} — diagnóstico de residuos, {mejor_arima_aic}', y=1.01)
        plt.tight_layout(); guardar(f'25_{clave}_residuos'); plt.show()

    return res_serie

# %%
R['series'] = {}
for nombre, s in SERIES.items():
    clave = CLAVES[nombre]
    R['series'][nombre] = analiza(nombre, s, TRAIN[nombre], TEST[nombre], clave)

# %% [markdown]
# ## 5. Análisis comparativo (punto 5)

# %% [markdown]
# ### 5.a Estacionalidad, tendencia, volatilidad e impacto de la pandemia
#
# Los cuatro criterios se miden con estadísticos explícitos para que la comparación sea
# reproducible y no dependa de la lectura visual de los gráficos:
#
# | Pregunta | Estadístico |
# |---|---|
# | ¿Mayor estacionalidad? | **Fuerza estacional** $F_S = 1 - \mathrm{Var}(R_t)/\mathrm{Var}(S_t+R_t)$ sobre la descomposición STL |
# | ¿Mayor tendencia de crecimiento? | **TCAC 2009–2019** (tasa de crecimiento anual compuesta pre-pandemia) |
# | ¿Mayor volatilidad? | **Desviación estándar de la tasa de cambio interanual** $\log(y_t/y_{t-12})$ |
# | ¿Más afectada por la pandemia? | **Caída de 2020 respecto de 2019** y **nivel de 2025 respecto de 2019** |

# %%
filas = []
for nombre, s in SERIES.items():
    ft, fs, _ = fuerza_componentes(s, pre=True)   # medido sobre 2009-2019
    anual = s.groupby(s.index.year).sum()
    tcac = (anual.loc[2019] / anual.loc[2009]) ** (1 / 10) - 1
    # Volatilidad: desviación estándar de la tasa de cambio interanual.
    # Se mide sobre 2009-2019 porque los meses en cero de 2020 (cierre total de fronteras)
    # hacen explotar log(y_t / y_{t-12}) y convertirían la medida en un termómetro de la
    # pandemia en lugar de una medida de volatilidad intrínseca. Se reporta también el
    # valor de muestra completa.
    pre = s.loc[PRE_INI:PRE_FIN]
    vol_yoy = float(np.log1p(pre).diff(12).dropna().std())
    vol_yoy_full = float(np.log1p(s).diff(12).replace([np.inf, -np.inf], np.nan).dropna().std())
    cv = pre.std() / pre.mean() * 100
    cv_full = s.std() / s.mean() * 100
    caida20 = (anual.loc[2020] / anual.loc[2019] - 1) * 100
    piso = s.loc['2020-01':'2021-06'].min()
    media_19 = s.loc['2019-01':'2019-12'].mean()
    caida_pico = (piso / media_19 - 1) * 100
    # Recuperación 2025 vs 2019. La versión CRUDA compara un 2019 que sí contabiliza a los
    # residentes de Guatemala contra un 2025 que ya no lo hace: mide el cambio de definición,
    # no la demanda. La versión HOMOGÉNEA excluye a esos residentes también de 2019, con lo
    # que ambos años quedan bajo la misma definición y la comparación es legítima.
    rec25 = anual.loc[2025] / anual.loc[2019] * 100
    sub_sgt = base[MASCARAS[nombre] & (base['País'] != 'Guatemala')]
    anual_sgt = sub_sgt.groupby('Año')['Viajero'].sum()
    base19_sgt = anual_sgt.get(2019, np.nan)
    rec25_h = anual.loc[2025] / base19_sgt * 100 if base19_sgt else np.nan
    filas.append({
        'serie': nombre, 'F_estacional': fs, 'F_tendencia': ft,
        'TCAC_09_19_%': tcac * 100, 'CV_pre_%': cv, 'CV_full_%': cv_full,
        'volatilidad_yoy': vol_yoy, 'volatilidad_yoy_full': vol_yoy_full,
        'caida_2020_%': caida20, 'piso_vs_media19_%': caida_pico,
        'rec_crudo_%': rec25, 'rec_homog_%': rec25_h,
    })
cmp5 = pd.DataFrame(filas).set_index('serie')
print(cmp5.round(3).to_string())
R['comparativo'] = cmp5.round(3).to_dict('index')

sin_total = cmp5.drop(index='Total de viajeros')
por_cat = {'Países de residencia': [i for i in sin_total.index if i.startswith('País')],
           'Vías de ingreso': [i for i in sin_total.index if i.startswith('Vía')]}

print('\n' + '=' * 90)
print('RESPUESTAS DEL PUNTO 5.a, POR CATEGORÍA')
print('=' * 90)
respuestas = {}
for cat, idx in por_cat.items():
    sub = cmp5.loc[idx]
    r = {
        'mayor_estacionalidad': (sub['F_estacional'].idxmax(), round(sub['F_estacional'].max(), 3)),
        'mayor_tendencia': (sub['TCAC_09_19_%'].idxmax(), round(sub['TCAC_09_19_%'].max(), 2)),
        'mayor_volatilidad': (sub['volatilidad_yoy'].idxmax(), round(sub['volatilidad_yoy'].max(), 3)),
        'mas_afectada_pandemia': (sub['caida_2020_%'].idxmin(), round(sub['caida_2020_%'].min(), 1)),
        'peor_recuperada': (sub['rec_homog_%'].idxmin(), round(sub['rec_homog_%'].min(), 1)),
        'peor_recuperada_crudo': (sub['rec_crudo_%'].idxmin(), round(sub['rec_crudo_%'].min(), 1)),
    }
    respuestas[cat] = r
    print(f'\n--- {cat} ---')
    print(f'  i.   Mayor estacionalidad : {r["mayor_estacionalidad"][0]}  (F_S = {r["mayor_estacionalidad"][1]})')
    print(f'  ii.  Mayor tendencia      : {r["mayor_tendencia"][0]}  (TCAC 09-19 = {r["mayor_tendencia"][1]}% anual)')
    print(f'  iii. Mayor volatilidad    : {r["mayor_volatilidad"][0]}  (sd log-yoy = {r["mayor_volatilidad"][1]})')
    print(f'  iv.  Más afectada 2020    : {r["mas_afectada_pandemia"][0]}  ({r["mas_afectada_pandemia"][1]}% vs 2019)')
    print(f'       Peor recuperada 2025 : {r["peor_recuperada"][0]}  ({r["peor_recuperada"][1]}% de 2019, comparación homogénea)')
    print(f'       [Si se usara la comparación cruda, contaminada por el cambio de definición,')
    print(f'        la respuesta sería {r["peor_recuperada_crudo"][0]} con {r["peor_recuperada_crudo"][1]}%.]')
R['respuestas_5a'] = respuestas

# %%
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
etq = [i.replace('País: ', 'País ').replace('Vía: ', 'Vía ')
        .replace('Estados Unidos de América', 'EE.UU.')
        .replace('Total de viajeros', 'TOTAL') for i in cmp5.index]
col = [GRIS if 'TOTAL' in e else (AZUL if e.startswith('País') else VERDE) for e in etq]
for a, (c, t) in zip(ax.ravel(), [
        ('F_estacional', 'Fuerza de la estacionalidad (0-1)'),
        ('TCAC_09_19_%', 'Crecimiento anual compuesto 2009-2019 (%)'),
        ('volatilidad_yoy', 'Volatilidad: sd de log(y_t / y_{t-12})'),
        ('caida_2020_%', 'Caída de 2020 respecto de 2019 (%)')]):
    a.barh(etq, cmp5[c].values, color=col)
    a.set_title(t, fontsize=9); a.grid(axis='y', alpha=0)
    a.tick_params(labelsize=7)
fig.suptitle('Comparación entre series — azul: países · verde: vías · gris: total', y=1.01)
plt.tight_layout(); guardar('30_comparativo'); plt.show()

# %% [markdown]
# ### Control: partición alternativa post-pandemia
#
# Como el corte del 70 % cae dentro del hueco pandémico, se repite la evaluación entrenando
# desde **julio 2021** (una vez reabiertas las fronteras) y dejando los últimos 18 meses como
# prueba. Sirve para separar el error atribuible al choque exógeno del error del modelo.

# %%
ctrl, detalle_ctrl = [], {}
for nombre, s in SERIES.items():
    sp = s.loc['2021-07':]
    tr2, te2 = sp.iloc[:-18], sp.iloc[-18:]
    tabla2, _preds = {}, {}

    # Mismo conjunto de modelos que en la partición principal, para que sea comparable
    d2 = n_diferencias(np.log1p(tr2))
    for etq, (o, os_) in {
            f'SARIMA(1,{d2},1)(1,0,1,12)': ((1, d2, 1), (1, 0, 1, 12)),
            f'SARIMA(0,{d2},1)(0,0,1,12)': ((0, d2, 1), (0, 0, 1, 12)),
            f'SARIMA(1,{d2},0)(1,0,0,12)': ((1, d2, 0), (1, 0, 0, 12))}.items():
        try:
            fit = ajusta_sarima(np.log1p(tr2), o, os_)
            m, p = evalua_sarima(fit, te2, len(te2))
            tabla2[etq] = m
        except Exception:
            pass
    try:
        otros2, _ = modelos_no_arima(tr2, te2, len(te2))
        tabla2.update(otros2)
    except Exception as e:
        print(f'  ({nombre}) modelos alternativos fallaron:', e)

    c2 = pd.DataFrame(tabla2).T.sort_values('RMSE')
    detalle_ctrl[nombre] = c2.round(2).to_dict('index')
    mejor2 = c2.index[0]

    fila = {'serie': nombre,
            'mejor_postcovid': mejor2,
            'RMSE_postcovid': round(c2.loc[mejor2, 'RMSE']),
            'MAPE_postcovid': round(c2.loc[mejor2, 'MAPE'], 1)}
    mj = R['series'][nombre]['mejor_rmse']
    fila['mejor_70_30'] = mj
    fila['MAPE_70_30'] = round(R['series'][nombre]['comparacion'][mj]['MAPE'], 1)
    ctrl.append(fila)

ctrl = pd.DataFrame(ctrl).set_index('serie')
print(ctrl.to_string())
R['control_postcovid'] = ctrl.to_dict('index')
R['control_postcovid_detalle'] = detalle_ctrl

# %% [markdown]
# ### 5.b Hallazgos para la toma de decisiones del INGUAT

# %%
resumen_final = pd.DataFrame({
    'mejor modelo (RMSE test)': {n: v['mejor_rmse'] for n, v in R['series'].items()},
    'MAPE %': {n: round(v['comparacion'][v['mejor_rmse']]['MAPE'], 1) for n, v in R['series'].items()},
    'RMSE': {n: round(v['comparacion'][v['mejor_rmse']]['RMSE']) for n, v in R['series'].items()},
    'mejor SARIMA (AIC)': {n: v['mejor_aic'] for n, v in R['series'].items()},
    'd': {n: v['d'] for n, v in R['series'].items()},
    'D': {n: v['D'] for n, v in R['series'].items()},
})
print(resumen_final.to_string())
R['resumen_final'] = resumen_final.to_dict('index')

with open(RES / 'resultados.json', 'w', encoding='utf-8') as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)
print('\nResultados guardados en resultados/resultados.json')
