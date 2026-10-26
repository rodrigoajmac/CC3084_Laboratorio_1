"""Compone el PDF final con el estilo y las páginas centrales del avance.

Las páginas 2–5 se conservan del PDF de avance. La portada y las páginas 6–7
se actualizan para reflejar el estado final sin dejar tareas antiguas marcadas
como pendientes. Ejecutar desde la raíz del repositorio:

    python Informe/generar_lab10_estilo_avance.py
"""
from pathlib import Path
import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from pypdf import PdfReader, PdfWriter


RAIZ = Path(__file__).resolve().parents[1]
AVANCE = RAIZ / 'Informe' / 'Lab10_Avances.pdf'
SALIDA = RAIZ / 'output' / 'pdf' / 'Lab10_Diseno_Final.pdf'
TEMP = RAIZ / 'tmp' / 'pdfs'
SALIDA.parent.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['pdf.fonttype'] = 42


class Pagina:
    def __init__(self, numero):
        self.numero = numero
        self.fig = plt.figure(figsize=(595.276 / 72, 841.89 / 72), facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 595.276)
        self.ax.set_ylim(0, 841.89)
        self.ax.set_axis_off()
        self.y = 760
        self.ax.text(297.6, 41, str(numero), fontsize=10.9, ha='center')

    def texto(self, contenido, tam=10.9, negrita=False, ancho=88, x=70.9,
              interlineado=13.55, espacio=5):
        estilo = 'bold' if negrita else 'normal'
        for linea in textwrap.wrap(contenido, width=ancho, break_long_words=False,
                                   break_on_hyphens=False) or ['']:
            self.ax.text(x, self.y, linea, fontsize=tam, fontweight=estilo, va='top')
            self.y -= interlineado
        self.y -= espacio

    def titulo(self, contenido):
        self.y -= 8
        self.texto(contenido, tam=14.3, negrita=True, ancho=75,
                   interlineado=18, espacio=7)

    def tabla(self, columnas, filas, anchos, tam=8.7, alto=28):
        izquierda = 70.9
        derecha = izquierda + sum(anchos)
        if derecha > 524:
            raise ValueError('La tabla supera el margen derecho')

        def ajustar(texto, ancho, negrita=False):
            fuente = FontProperties(family='Times New Roman', size=tam,
                                    weight='bold' if negrita else 'normal')
            resultado = []
            linea = ''
            for palabra in texto.split():
                candidata = f'{linea} {palabra}'.strip()
                if linea and TextPath((0, 0), candidata, prop=fuente).get_extents().width > ancho - 12:
                    resultado.append(linea)
                    linea = palabra
                else:
                    linea = candidata
            if linea:
                resultado.append(linea)
            return resultado

        self.ax.plot([izquierda, derecha], [self.y + 5, self.y + 5], color='black', lw=.8)
        x = izquierda + 4
        cabeceras = [ajustar(cab, w, negrita=True) for cab, w in zip(columnas, anchos)]
        for lineas, w in zip(cabeceras, anchos):
            for j, linea in enumerate(lineas):
                self.ax.text(x, self.y - 3 - j * 10.5, linea, fontsize=tam,
                             fontweight='bold', va='top')
            x += w
        self.y -= max(20, 10 + max(map(len, cabeceras)) * 10.5)
        self.ax.plot([izquierda, derecha], [self.y + 4, self.y + 4], color='black', lw=.45)
        for fila in filas:
            x = izquierda + 4
            celdas = [ajustar(valor, w) for valor, w in zip(fila, anchos)]
            for lineas, w in zip(celdas, anchos):
                for j, linea in enumerate(lineas):
                    self.ax.text(x, self.y - j * 10.5, linea, fontsize=tam, va='top')
                x += w
            self.y -= max(alto, 8 + max(map(len, celdas)) * 10.5)
        self.ax.plot([izquierda, derecha], [self.y + 7, self.y + 7], color='black', lw=.8)
        self.y -= 10

    def guardar(self, pdf):
        if self.y < 54:
            raise ValueError(f'La página {self.numero} llega hasta y={self.y:.1f}')
        if os.environ.get('LAB10_PREVIEW') == '1':
            self.fig.savefig(TEMP / f'lab10-estilo-{self.numero:02d}.png', dpi=150)
        pdf.savefig(self.fig)
        plt.close(self.fig)


def portada(pdf):
    p = Pagina(1)
    p.y = 760
    for texto, tam, espacio in [
        ('Universidad del Valle de Guatemala', 12, 15.5),
        ('Facultad de Ingeniería', 10.9, 15.5),
        ('Departamento de Ciencias de la Computación', 10.9, 15.5),
        ('CC3084 Data Science — Semestre II, 2026', 10.9, 23.5),
        ('Laboratorio 10: Visualización interactiva', 12, 17.5),
        ('Documento de diseño — Entrega final', 12, 25.5),
    ]:
        p.ax.text(297.6, p.y, texto, fontsize=tam, ha='center')
        p.y -= espacio
    p.ax.text(297.6, p.y, 'Observatorio de turismo: tablero interactivo del flujo de viajeros',
              fontsize=14.3, fontweight='bold', ha='center')
    p.y -= 18
    p.ax.text(297.6, p.y, 'a Guatemala (2009–junio 2026)', fontsize=14.3,
              fontweight='bold', ha='center')
    p.y -= 27
    p.texto('Integrantes: Rodrigo Ajmac · Andrés Mazariegos · June Herrera · Sebastián García',
            tam=10.9, ancho=100)
    p.texto('Guatemala, 25 de octubre de 2026', tam=10.9)
    p.titulo('1. Propósito, audiencia y alcance del tablero')
    p.texto('El tablero está dirigido a directivos y tomadores de decisiones del sector turístico '
            'sin formación estadística. Permite explorar el flujo de viajeros que ingresa a Guatemala, '
            'identificar patrones, consultar pronósticos y comprender sus limitaciones. Es el primer '
            'módulo del observatorio de turismo sostenible.')
    p.texto('Fuente de datos. Base de Migración 2009–2026 (junio), usada en el Laboratorio 1. '
            'Siete series mensuales: total, tres países (El Salvador, Guatemala y Estados Unidos) '
            'y tres vías (aérea, terrestre y marítima). Datos de uso académico.')
    p.y -= 8
    p.tabla(('Incluye', 'No incluye (y por qué)'), [
        ('Exploración mensual, trimestral y anual de siete series.',
         'Modelos nuevos: se reutilizan resultados del Laboratorio 1.'),
        ('Calendario de temporadas y comparaciones entre mercados y vías.',
         'Causas: la migración muestra cuánto y cuándo, no por qué.'),
        ('Resultados y errores de modelos, guía y advertencias.',
         'Gasto, hoteles, ambiente ni meses posteriores a junio de 2026.'),
    ], (225, 225), tam=9.0, alto=41)
    p.texto('Distinción entre observado, interpretado y estimado. Cada gráfica indica el origen de '
            'sus cifras y su utilidad para la gestión. Las lecturas sugeridas describen patrones; '
            'los pronósticos tienen error y no constituyen garantías.')
    p.guardar(pdf)


def pagina_seis(pdf):
    p = Pagina(6)
    p.texto('Comparación de herramientas (continuación de la sección 6)', tam=10.9,
            negrita=True, espacio=13)
    p.tabla(('Herramienta', 'Gráficos enlazados', 'Encaje con el equipo', 'Decisión'), [
        ('Streamlit + Plotly', 'Sí, selección y estado', 'Python; publicación gratuita', 'Elegida'),
        ('Dash', 'Sí, callbacks', 'Python; más código', 'Alternativa'),
        ('R Shiny', 'Sí, reactividad', 'Requiere cambiar lenguaje', 'Descartada'),
        ('Panel / Voilà', 'Parcial o limitado', 'Menos control inmediato', 'Descartadas'),
    ], (110, 112, 150, 78), tam=8.4, alto=35)
    p.texto('Streamlit permite navegación, controles y estado compartido con poco código; Plotly '
            'aporta selección, acercamiento e información al pasar el cursor. Los CSV y JSON de los '
            'laboratorios anteriores se leen directamente en Python. La aplicación se puede ejecutar '
            'sin volver a entrenar modelos.')
    p.titulo('7. Planificación de tareas y estado de cierre')
    p.tabla(('Tarea', 'Responsable / fecha', 'Estado comprobado'), [
        ('Preguntas, alcance, paleta y bosquejos', 'Rodrigo · 22 oct', 'Avance entregado'),
        ('Tablero base y navegación', 'Rodrigo · 22 oct', 'Código en repositorio'),
        ('Panorama, guía y README', 'Sebastián · 25 oct', 'Commit 1ef0351'),
        ('V5–V10 y métricas: revisión técnica', 'Sebastián · 25 oct', 'Verificado'),
        ('Corrección de pronósticos y límites', 'Sebastián · 25 oct', 'Commit 3f9ed5a'),
        ('Documento de diseño final', 'Sebastián · 25 oct', 'PDF generado'),
        ('Prueba con 2–3 usuarios externos', 'Plan original: June y Sebastián', 'No documentada'),
    ], (188, 145, 117), tam=8.25, alto=28)
    p.texto('Las tareas originalmente previstas para Andrés y June no se atribuyen como aportes '
            'finales sin una confirmación o un commit propio. El tablero ya contenía V5–V10 antes '
            'de la revisión de cierre.', tam=10.3, ancho=90)
    p.guardar(pdf)


def pagina_siete(pdf):
    p = Pagina(7)
    p.titulo('Conclusiones y verificación del diseño')
    p.texto('El tablero contiene diez visualizaciones interactivas, tres enlaces entre gráficas y '
            'resultados de modelos previos. Las tarjetas, controles y tabla comparativa son elementos '
            'complementarios y no se cuentan entre las diez.')
    p.texto('La aplicación abrió las cuatro páginas sin excepciones con Streamlit 1.45.1. Las siete '
            'series mostraron V5–V7 y V8–V10. Para las 98 combinaciones de serie, prueba y modelo, '
            'MAE y RMSE de predicciones.csv coincidieron con resultados.json. También se verificó '
            'el periodo completo de las selecciones mensuales, trimestrales y anuales.')
    p.texto('Limitaciones. Desde 2023 se dejó de registrar a residentes guatemaltecos que regresan; '
            'el tablero ofrece una serie de no residentes para comparar el total. La cobertura de '
            'cruceristas cambió y la vía marítima requiere cautela. Los modelos del total usan el '
            'total registrado, aun cuando la exploración esté en no residentes. El año 2026 llega '
            'solo hasta junio. La prueba externa de usabilidad aún no está documentada.')
    p.texto('El diseño prioriza preguntas de gestión y permite pasar de una vista general al detalle. '
            'La separación entre datos observados, estimaciones y advertencias ayuda a interpretar '
            'los resultados sin atribuir causas que la fuente no puede demostrar.')
    p.titulo('Repositorio del proyecto')
    enlace = 'https://github.com/rodrigoajmac/CC3084_Laboratorio_1'
    p.ax.text(70.9, p.y, enlace, fontsize=10.9, va='top', url=enlace)
    p.y -= 21
    p.titulo('Referencias')
    for ref in [
        'Shneiderman, B. (1996). The eyes have it. IEEE Symposium on Visual Languages.',
        'Nielsen, J. (1994). 10 Usability Heuristics for User Interface Design.',
        'Few, S. (2013). Information Dashboard Design (2.ª ed.). Analytics Press.',
        'Munzner, T. (2014). Visualization Analysis and Design. CRC Press.',
        'Knaflic, C. N. (2015). Storytelling with Data. Wiley.',
        'Itten, J. (1961). The Art of Color. Van Nostrand Reinhold.',
        'W3C (2018). Web Content Accessibility Guidelines (WCAG) 2.1.',
        'Streamlit y Plotly: documentación oficial (docs.streamlit.io; plotly.com/python).',
    ]:
        p.texto('– ' + ref, tam=10.4, ancho=90, espacio=3)
    p.guardar(pdf)


partes = TEMP / 'lab10-paginas-actualizadas.pdf'
with PdfPages(partes) as pdf:
    portada(pdf)
    pagina_seis(pdf)
    pagina_siete(pdf)

original = PdfReader(AVANCE)
nuevas = PdfReader(partes)
escritor = PdfWriter()
escritor.add_page(nuevas.pages[0])
for indice in range(1, 5):
    escritor.add_page(original.pages[indice])
escritor.add_page(nuevas.pages[1])
escritor.add_page(nuevas.pages[2])
with SALIDA.open('wb') as archivo:
    escritor.write(archivo)
print(SALIDA)
