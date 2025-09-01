'''
Archivo: visualization.py
Descripción: Archivo para la visualización de gráficos
    utilizados en el reporte.
Autora: Andrea Medina Rico
'''

from visualization.statistic import Statistic

'''
Función: visualization
Descripción: Función para visualizar gráficos a partir de los datos de entrada.
    Gráficos utilizados para el análisis y reporte.
Params:
    - data: DataFrame con los datos a visualizar.
Returns:
    - None
'''
def visualization(data):
    stat = Statistic()

    # Matriz de correlación
    stat.correlation_matrix(data)

    # Subplots de sctatterplot matplot
    scatter_cols = ['flipper_length_cm', 'culmen_length_cm', 'culmen_depth_cm',
                    'species_Gentoo', 'species_Adelie', 'island_Biscoe', 'island_Dream', 'sex']
    stat.scatter_subplots(data, scatter_cols, 'body_mass_kg')

    # Kdeplot
    kdeplot_cols = ['body_mass_kg', 'flipper_length_cm', 'culmen_length_cm', 'culmen_depth_cm']
    stat.kdeplot_subplots(data, kdeplot_cols)
