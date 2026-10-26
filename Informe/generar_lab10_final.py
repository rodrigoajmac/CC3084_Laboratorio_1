"""Genera el documento de diseño final del Laboratorio 10.

Ejecutar desde la raíz: python Informe/generar_lab10_final.py
"""
from pathlib import Path
import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'pdf' / 'Lab10_Diseno_Final.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)

AZUL = '#1F4E79'
NARANJA = '#A84A12'
TEXTO = '#1B2733'
GRIS = '#55606C'
FONDO = '#F2F5F8'
plt.rcParams['font.family'] = 'DejaVu Sans'


class Pagina:
    def __init__(self, numero, titulo, subtitulo=None):
        self.fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_axis_off()
        self.y = 0.91
        self.ax.add_patch(plt.Rectangle((0, .962), 1, .038, color=AZUL, transform=self.ax.transAxes))
        self.ax.text(.08, .947, 'CC3084  |  LABORATORIO 10  |  DISEÑO FINAL', fontsize=8.5,
                     color=AZUL, fontweight='bold', transform=self.ax.transAxes)
        self.ax.text(.08, self.y, titulo, fontsize=18, color=TEXTO, fontweight='bold',
                     va='top', transform=self.ax.transAxes)
        self.y -= .047
        if subtitulo:
            self.texto(subtitulo, color=GRIS, size=9.5)
            self.y -= .008
        self.ax.plot([.08, .92], [self.y, self.y], color='#DDE5EB', lw=1, transform=self.ax.transAxes)
        self.y -= .026
        self.ax.text(.08, .047, 'Observatorio de turismo de Guatemala  |  25 de octubre de 2026',
                     fontsize=8, color=GRIS, transform=self.ax.transAxes)
        self.ax.text(.92, .047, str(numero), fontsize=8, color=GRIS,
                     ha='right', transform=self.ax.transAxes)

    def texto(self, texto, size=9.4, color=TEXTO, ancho=98, sangria=0, inter=.0165):
        for linea in textwrap.wrap(texto, width=ancho, break_long_words=False,
                                   break_on_hyphens=False) or ['']:
            self.ax.text(.08 + sangria, self.y, linea, fontsize=size, color=color,
                         va='top', transform=self.ax.transAxes)
            self.y -= inter
        self.y -= .010

    def seccion(self, titulo):
        self.y -= .007
        self.ax.text(.08, self.y, titulo, fontsize=12, color=AZUL, fontweight='bold',
                     va='top', transform=self.ax.transAxes)
        self.y -= .031

    def item(self, clave, texto, ancho=87):
        self.ax.text(.085, self.y, clave, fontsize=9.2, color=AZUL, fontweight='bold',
                     va='top', transform=self.ax.transAxes)
        self.y -= .019
        self.texto(texto, size=9, ancho=ancho, sangria=.012, inter=.015)

    def banda(self, titulo, texto):
        lineas = textwrap.wrap(texto, width=91, break_long_words=False)
        alto = .057 + len(lineas) * .016
        arriba = self.y - .012
        self.ax.add_patch(plt.Rectangle((.08, arriba - alto), .84, alto,
                                        facecolor=FONDO, edgecolor='#DDE5EB',
                                        transform=self.ax.transAxes))
        self.ax.text(.095, arriba - .013, titulo, fontsize=9.5, color=AZUL,
                     fontweight='bold', va='top', transform=self.ax.transAxes)
        for i, linea in enumerate(lineas):
            self.ax.text(.095, arriba - .037 - i * .016, linea, fontsize=8.8,
                         color=TEXTO, va='top', transform=self.ax.transAxes)
        self.y -= alto + .026

    def guardar(self, pdf):
        if self.y < .07:
            raise ValueError(f'Contenido fuera de página: {self.y:.3f}')
        if os.environ.get('LAB10_PREVIEW') == '1':
            previo = ROOT / 'tmp' / 'pdfs'
            previo.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(previo / f'lab10-final-{numero_actual[0]:02d}.png', dpi=120)
            numero_actual[0] += 1
        pdf.savefig(self.fig)
        plt.close(self.fig)


numero_actual = [1]
with PdfPages(OUT) as pdf:
    p = Pagina(1, 'Observatorio de turismo de Guatemala',
               'Documento de diseño final · Flujo de viajeros de 2009 a junio de 2026')
    p.texto('Equipo del proyecto: Rodrigo Ajmac, Andrés Mazariegos, June Herrera y Sebastián García. '
            'Este documento registra el tablero entregado y distingue las tareas planeadas de las comprobadas.')
    p.seccion('1. Propósito, audiencia y alcance')
    p.texto('El tablero ayuda a directivos del sector turístico a explorar patrones, comparar mercados y consultar '
            'pronósticos sin formación estadística. Utiliza siete series mensuales derivadas de la base de migración: '
            'total, tres países y tres vías de ingreso. Los datos son de uso académico.')
    p.banda('Incluye', 'Evolución mensual, trimestral y anual; temporadas; comparación de mercados y vías; '
            'resultados y errores de modelos existentes; advertencias de cobertura y guía de lectura.')
    p.banda('Fuera de alcance', 'No se entrenan modelos nuevos ni se estiman causas. No hay datos de gasto, '
            'ocupación hotelera, ambiente ni meses posteriores a junio de 2026.')
    p.seccion('2. Preguntas de gestión')
    for k, t in [
        ('P1', '¿Cómo evolucionó el flujo y qué nivel alcanzó tras la pandemia? Apoya la revisión del sector.'),
        ('P2', '¿Qué ocurrió en un año o mes concreto? Permite inspeccionar episodios y temporadas.'),
        ('P3', '¿Qué mercados o vías crecen, varían y se recuperan? Apoya la priorización de recursos.'),
        ('P4', '¿En qué meses se concentra el flujo? Ayuda a planificar capacidad y personal.'),
        ('P5', '¿Qué modelo se equivoca menos y cuándo falla? Orienta el uso prudente de pronósticos.'),
        ('P6', '¿Qué no se puede concluir con estos datos? Evita comparaciones y causas injustificadas.'),
    ]:
        p.item(k, t)
    p.guardar(pdf)

    p = Pagina(2, 'Navegación e interacción', 'Del panorama al detalle, con controles visibles en todas las páginas')
    p.seccion('3. Estructura de la interfaz')
    for k, t in [
        ('Barra lateral', 'Serie, definición de total, años, nivel de detalle y año seleccionado.'),
        ('Página 1 · Panorama', 'V1 evolución; V2 comparación; V3 años completos; V4 desglose mensual.'),
        ('Página 2 · Calendario', 'V5 mapa de calor; V6 perfil mensual; V7 nivel 2025 frente a 2019.'),
        ('Página 3 · Pronósticos', 'V8 errores de modelos; V9 real frente a predicho; V10 dispersión.'),
        ('Página 4 · Guía', 'Significado de colores, glosario y límites de las fuentes y los modelos.'),
    ]:
        p.item(k, t)
    p.seccion('Enlaces entre visualizaciones')
    p.banda('V2 → serie activa', 'Un clic en una barra cambia la serie de las vistas y de los pronósticos.')
    p.banda('V3 → V4', 'Un clic en un año completo fija el año cuyo detalle mensual aparece en V4.')
    p.banda('V1 → tarjetas y V6', 'Seleccionar puntos actualiza las tarjetas. V6 usa el periodo si abarca '
            'al menos dos años; si no, muestra la base prepandemia 2009–2019.')
    p.seccion('Control del nivel de detalle')
    p.texto('V1 cambia entre mes, trimestre y año. Al seleccionar un punto agregado, las tarjetas abarcan '
            'el periodo completo. V3→V4 permite bajar del total anual a sus meses. El usuario puede deshacer '
            'la selección y elegir las gráficas mediante la navegación y los controles.')
    p.guardar(pdf)

    p = Pagina(3, 'Bosquejo de la interfaz', 'Distribución de controles y gráficas en pantalla de escritorio')
    p.seccion('Panorama: vista general y detalle bajo demanda')

    def caja(x, y, w, h, titulo, detalle='', color=FONDO):
        p.ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                     edgecolor='#B9C0C8', transform=p.ax.transAxes))
        p.ax.text(x + .015, y + h - .025, titulo, fontsize=9.5, color=AZUL,
                  fontweight='bold', va='top', transform=p.ax.transAxes)
        if detalle:
            p.ax.text(x + .015, y + h - .051, detalle, fontsize=8.2, color=GRIS,
                      va='top', transform=p.ax.transAxes)

    caja(.08, .43, .19, .31, 'Barra lateral', 'Serie · años · detalle')
    caja(.30, .61, .62, .13, 'V1  Evolución temporal', 'Selección de periodo → tarjetas / V6')
    caja(.30, .53, .62, .075, 'Tarjetas de contexto', 'No cuentan como gráfica')
    caja(.30, .43, .295, .095, 'V2  Mercados y vías', 'Clic → serie activa')
    caja(.61, .43, .31, .095, 'V3  Totales anuales', 'Clic → V4 mensual')
    caja(.30, .34, .62, .075, 'V4  Detalle mensual', 'Año enfocado y referencias')
    p.ax.text(.08, .285, 'Otras páginas', fontsize=12, color=AZUL,
              fontweight='bold', transform=p.ax.transAxes)
    caja(.08, .16, .405, .10, 'Calendario y mercados', 'V5 calor · V6 perfil · V7 recuperación')
    caja(.515, .16, .405, .10, 'Pronósticos', 'V8 error · V9 líneas · V10 dispersión')
    p.y = .125
    p.texto('La cuarta página contiene la guía y los límites. La barra lateral permanece visible durante '
            'la navegación; los rótulos de serie y los avisos muestran el contexto de cada vista.',
            size=8.8, ancho=100)
    p.guardar(pdf)

    visuales = [
        ('V1 · Línea temporal', 'Flujo por mes, trimestre o año.', 'La continuidad facilita leer tendencia y choques; permite seleccionar un periodo.'),
        ('V2 · Barras horizontales', 'Crecimiento, estacionalidad, volatilidad o recuperación de seis categorías.', 'Las longitudes comparan valores y admiten nombres largos; clic cambia la serie.'),
        ('V3 · Barras anuales', 'Totales de 2009 a 2025.', 'Comparación cronológica con eje desde cero; clic abre un año en V4. Se excluye 2026 incompleto.'),
        ('V4 · Barras y referencias', 'Meses del año elegido frente al anterior y 2019.', 'Las barras enfocan el año; líneas muestran referencias sin saturar. 2026 usa solo enero–junio.'),
        ('V5 · Mapa de calor', 'Viajeros por año y mes.', 'La matriz revela temporadas y la ruptura de 2020; escala secuencial azul.'),
        ('V6 · Barras divergentes', 'Desviación mensual frente al promedio.', 'El cero significa un mes promedio; dirección y cifra muestran meses altos y bajos.'),
        ('V7 · Barras y umbral', 'Nivel de 2025 respecto a 2019 por mercado o vía.', 'La línea de 100 % facilita detectar recuperación; Guatemala se excluye por falta de medición.'),
        ('V8 · Puntos conectados', 'MAPE por modelo en dos pruebas.', 'La distancia entre puntos comunica cuánto cambia el error entre pruebas.'),
        ('V9 · Líneas', 'Valor real y hasta tres predicciones.', 'La coincidencia temporal permite ver aciertos y desvíos por mes.'),
        ('V10 · Dispersión', 'Valor real frente al pronosticado por mes.', 'La diagonal de acierto revela sesgo de sobrestimación o subestimación.'),
    ]
    for numero, bloque in [(4, visuales[:5]), (5, visuales[5:])]:
        p = Pagina(numero, f'Elección de gráficas · V{1 if numero == 4 else 6}–V{5 if numero == 4 else 10}',
                   'Diez visualizaciones interactivas distintas; tarjetas, controles y tabla no se cuentan')
        p.seccion('4. Visualizaciones y justificación')
        for nombre, dato, razon in bloque:
            p.banda(nombre, f'Dato: {dato} Elección e interacción: {razon}')
        p.texto('Cada gráfica lleva una pregunta, una descripción y un uso posible para la gestión. '
                'Los valores observados y las estimaciones se identifican por texto y estilo, además del color.',
                color=GRIS)
        p.guardar(pdf)

    p = Pagina(6, 'Sistema visual y experiencia de uso')
    p.seccion('5. Colores y tipografía')
    colores = [
        ('Azul profundo  #1F4E79', 'Dato observado y selección activa.'),
        ('Azul claro  #9DBBD9', 'Apoyo o elemento no seleccionado; nunca texto principal.'),
        ('Gris  #B9C0C8', 'Contexto y referencias.'),
        ('Naranja  #D9661F', 'Atención y estimaciones de modelos.'),
        ('Naranja oscuro  #A84A12', 'Texto de advertencia legible sobre fondo claro.'),
        ('Violeta / verde azulado', 'Segundo y tercer modelo, siempre con etiqueta y trazo distinto.'),
    ]
    for k, t in colores:
        p.item(k, t)
    p.texto('Azul y naranja ofrecen contraste de significado; las tonalidades claras quedan como relleno. '
            'La interfaz usa Segoe UI, Helvetica Neue y Arial; en gráficos, la fuente sans serif disponible. '
            'Títulos en forma de pregunta y ejes con unidades reducen la carga de interpretación.')
    p.seccion('Principios HCI y UX aplicados')
    for k, t in [
        ('Estado visible', 'La serie activa aparece en el encabezado y se resalta en V2 y V3.'),
        ('Lenguaje claro', 'MAPE, estacionalidad y volatilidad tienen explicación en la guía.'),
        ('Control', 'Selecciones reversibles, filtros explícitos y selector de año acotado al rango.'),
        ('Prevención de errores', 'Mensajes cuando no hay datos; 2026 se distingue como año incompleto.'),
        ('Accesibilidad', 'Se combinan color, etiquetas y trazos; no se depende solo del color.'),
    ]:
        p.item(k, t)
    p.guardar(pdf)

    p = Pagina(7, 'Herramientas, datos y límites')
    p.seccion('6. Framework, lenguaje y justificación')
    p.texto('Se eligió Python, Streamlit y Plotly. Streamlit ofrece navegación, estado compartido y caché con '
            'poco código; Plotly aporta acercamiento, selección e información al pasar el cursor. Python permite '
            'leer directamente los CSV y JSON de los laboratorios anteriores. Son herramientas gratuitas y de '
            'código abierto, adecuadas al plazo y a la experiencia del equipo.')
    p.texto('Dash también permite enlazar gráficas, pero exige más código de callbacks. R Shiny implicaría '
            'rehacer el flujo del proyecto en otro lenguaje. Panel y Voilà cubren partes de la necesidad, '
            'aunque ofrecen menos control inmediato para esta combinación de navegación y selecciones.', color=GRIS)
    p.seccion('Fuentes y resultados reutilizados')
    for k, t in [
        ('series_mensuales.csv', 'Siete series de 210 meses, enero de 2009 a junio de 2026.'),
        ('resultados.json', 'Comparativos y métricas de modelos existentes.'),
        ('predicciones.csv', '4,284 filas de valores reales y estimados, en dos pruebas.'),
    ]:
        p.item(k, t)
    p.seccion('Límites de interpretación')
    for k, t in [
        ('Medición 2023', 'Dejó de contarse a residentes guatemaltecos que regresan. La opción no residentes corrige el total histórico.'),
        ('Vía marítima', 'La cobertura de cruceristas también cambió; su cifra no separa caída real y pérdida de cobertura.'),
        ('Modelos', 'El total modelado es el total registrado, aunque en exploración se elija no residentes. La prueba 70/30 incluye el choque pandémico.'),
        ('Causalidad', 'Las series muestran cuánto y cuándo; no explican por qué.'),
    ]:
        p.item(k, t)
    p.guardar(pdf)

    p = Pagina(8, 'Plan, verificación y entrega')
    p.seccion('7. Planificación: tareas, fechas y responsables')
    for k, t in [
        ('22 oct · Rodrigo', 'Preguntas, alcance, paleta, bosquejos, tablero base y PDF de avance. Realizado.'),
        ('25 oct · Sebastián', 'Correcciones de Panorama, guía, README y revisión técnica de V5–V10. Realizado.'),
        ('25 oct · Sebastián', 'Documento de diseño final y comprobación de datos y rutas. Realizado.'),
        ('Plan original · Andrés', 'Validar predicciones y V8–V10. No se atribuye una entrega no confirmada.'),
        ('Plan original · June', 'Validar V5–V7 y prueba de uso externa. No se atribuye una entrega no confirmada.'),
    ]:
        p.item(k, t)
    p.seccion('Comprobaciones realizadas')
    p.texto('La aplicación abrió las cuatro páginas sin excepciones en Streamlit 1.45.1. V5–V7 y V8–V10 '
            'mostraron tres gráficas cada una para las siete series. Las métricas MAE y RMSE de las 98 '
            'combinaciones de serie, prueba y modelo coincidieron con predicciones.csv. Se verificó el '
            'rango completo de selecciones mensuales, trimestrales y anuales.')
    p.texto('La prueba de usabilidad con 2–3 personas ajenas al curso no se documentó; esta verificación '
            'técnica no la sustituye. La publicación en un servicio externo tampoco es requisito explícito '
            'de la entrega: la aplicación se ejecuta localmente con las instrucciones del README.', color=GRIS)
    p.seccion('Material a entregar')
    p.banda('PDF y repositorio', 'Este documento y https://github.com/rodrigoajmac/CC3084_Laboratorio_1. '
            'El repositorio contiene app.py, datos exportados, requisitos y guía de ejecución.')
    p.seccion('Referencias de diseño')
    p.texto('Shneiderman (1996), The Eyes Have It; Nielsen (1994), 10 Usability Heuristics; '
            'Few (2013), Information Dashboard Design; Munzner (2014), Visualization Analysis and Design; '
            'documentación oficial de Streamlit y Plotly.', size=8.8, color=GRIS)
    p.guardar(pdf)

print(OUT)
