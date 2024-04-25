# Module imports
# Data Managing
import numpy as np
from numpy import linalg
import pandas as pd
# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from typing import Optional, Callable, List, Dict, Tuple,Sequence
from Paquetes.optimization import execution_time
from sklearn.exceptions import NotFittedError
from Paquetes.utils import print_loading_bar


##################################################################################################################
######################################### Customize classes for Eda  ############################################
##################################################################################################################

class Eda:

  """
    Clase Eda para llevar a cabo un analisis exploratorio del dataframe deseado que se le pase como argumento

    Atributos
    ---------
      - self.data : Objeto DataFrame completo
      - self.num_data : Objeto DataFrame numerico (columnas con datos tipo : entero y flotante)
      - self.cat_data : Objeto DataFrame categorico (columnas con datos tipo : object y bool)
      - self.columns : Objeto lista con el nombre (tipo str) de todas las columnas del DataFrame
      - self.num_cols : Objeto lista con el nombre (tipo str) de las columnas numericas del DataFrame
      - self.cat_cols : Objeto lista con el nombre (tipo str) de las columnas categoricas del DataFrame
      - self.date_cols
      - self.dtypes
      - self.auto_eda : bool 


    Propiedades (decorador @property)
    ---------------------------------
      - infocat: Muestra informacion de las columnas categoricas y tipo bool del df
      - infonum: Muestra informacion de las columnas numericas y devuelve u df con valores: MAX,MIN,STD y MEAN de cada columna
       * @property: Es un decorador incorporado (built-in decorator) en Python que nos permite definir propiedades en una clase.
        Las propiedades son metodos especiales a los que se puede acceder como atributos, y pueden tener getters, setters y deleters.

    Metodos
    -------
      - __init__ [built-in method]
      - __str__ [built-in method]
      - plot_hist : Grafica el histograma y la funcion de densidad de probabilidad de las columnas deseadas
      - plot_corr_heatmap : Grafica la matriz de correlacion calculada a partir de las columnas numericas del df en forma de heatmap
      - plot_bar : Grafico de barras para las columnas categoricas especificadas
      - plot_combined_bar : Grafico de barras combinado para las columnas categoricas especificadas
      - plot_boxplot : Grafico de caja
      - plot_scatter : Grafico de puntos para dos columnas numericas [mirar docstring del metodo para mas informacion]

  """

  def __init__(self, data: pd.DataFrame , auto_eda : bool = False, target_var :str = "") -> None:

    """
    Metodo que se ejecuta siempre al instanciar la clase y que inicializa los atributos del objeto creado al instaciar esta clase. Estos atributos se pueden crear in situ a raiz
    de los argumentos que se le pasen a la clase al instanciarla o definirse dentro de esta propia funcion o simplemente ser argumentos que se hayan pasado duramte la instancia
    de la clase para crear el objeto.

    Parametros
    ----------
      - data: Objeto DataFrame completo
      - auto_eda: bool
      - target_var: str (for auto eda only)

    Retorna
    -------
      None

    """
    # Atribute initialization
    self.data = data
    self.num_data = self.data.select_dtypes(["Float64","Float32","Int64","Int32"])
    self.cat_data = self.data.select_dtypes(["object","bool","string"])
    self.columns = list(self.data.columns)
    self.num_cols = list(self.data.select_dtypes(["Float64","Float32","Int64","Int32"]).columns)
    self.cat_cols = list(self.data.select_dtypes(["object","bool","string"]).columns)
    self.date_cols = list(self.data.select_dtypes(["datetime64[ns]"]).columns)
    self.dtypes = self.data.dtypes
    self.plotly_cmaps =  [
                              "aggrnyl", "agsunset", "blackbody", "bluered", "blues", "blugrn", "bluyl", "brwnyl",
                              "bugn", "bupu", "burg", "burgyl", "cividis", "darkmint", "electric", "emrld",
                              "gnbu", "greens", "greys", "hot", "inferno", "jet", "magenta", "magma",
                              "mint", "orrd", "oranges", "oryel", "peach", "pinkyl", "plasma", "plotly3",
                              "pubu", "pubugn", "purd", "purp", "purples", "purpor", "rainbow", "rdbu",
                              "rdpu", "redor", "reds", "sunset", "sunsetdark", "teal", "tealgrn", "turbo",
                              "viridis", "ylgn", "ylgnbu", "ylorbr", "ylorrd", "algae", "amp", "deep",
                              "dense", "gray", "haline", "ice", "matter", "solar", "speed", "tempo",
                              "thermal", "turbid", "armyrose", "brbg", "earth", "fall", "geyser", "prgn",
                              "piyg", "picnic", "portland", "puor", "rdgy", "rdylbu", "rdylgn", "spectral",
                              "tealrose", "temps", "tropic", "balance", "curl", "delta", "oxy", "edge",
                              "hsv", "icefire", "phase", "twilight", "mrybm", "mygbm"
                            ] 
    
    # Call Auto eda internal method
    if auto_eda:
      if target_var in self.columns:
        self._auto_eda( depVar = target_var)
      else:
        print(f"Error - target var :{target_var} for automatic Eda, not in columns of the dataframe ")
        
    
  # dunder magic special methods:
  def __str__(self) -> str:

    """
    Modificacion del dunder method ( built in or magic mnethod) que devuelve una representación en forma de cadena (str) de un objeto de clase. Puede ser llamado con las funciones incorporadas
    str () y print ().

    Parametros
    ----------
      None

    Retorna
    -------
      - (str)
    """
    return f'Eda_object'
  
  
  def _auto_eda(self, depVar : str)->None:
    """Auto EDA library // url of documentation: https://pypi.org/project/autoviz/ """
    #importing Autoviz class
    from autoviz.AutoViz_Class import AutoViz_Class
    import os
    
    # Saving the figure: make the directory to save the images. If it does not exist create it
    if not os.path.exists(f"images"):
      os.mkdir("images")
    if not os.path.exists(f"images\\AutoEda"):
      os.mkdir(f"images\\AutoEda")
      
    #Instantiate the AutoViz class
    AV = AutoViz_Class()
    self.auto_eda = AV.AutoViz(
                              filename = "", 
                              depVar= depVar,
                              dfte = self.data,
                              header = 0,
                              chart_format = 'png',
                              verbose = 2, 
                              save_plot_dir = f'images\\AutoEda\\{self}'
                              )
    

  @property
  def infocat(self) -> None:

    """
    Propiedad que printea informacion de las columnas categoricas

    Parametros
    ----------
      - None

    Retorna
    -------
      - None

    """
    #Recorre todas las columnas comprueba que es una feature categorica y si es asi extrae la informacion
    for col in self.cat_cols:
      print("__________________________________________________________________")
      print(f"Categorias para la columna == {col}: ", list(self.cat_data[f"{col}"].value_counts().index) )
      print(f"Frecuencia de las categorias : \n", (self.cat_data[f"{col}"].value_counts()))
      print(f"Numero de categorias en la columna == {col} --- ", self.cat_data[f"{col}"].nunique() )
      print(f"Numero de valores nulos en la columna == {col} --- ", self.cat_data[f"{col}"].isnull().sum() )
      print(f"Informacion generica == {col} : \n ", self.cat_data[f"{col}"].describe(include= ["object","bool"]))
      print("__________________________________________________________________")

  @property
  def infonum(self) -> pd.DataFrame :

    """
    Propiedad que printea informacion de las columnas numericas y devuelve df con informacion de cada columna.

    Parametros
    ----------
      None

    Retorna
    -------
      - df_num : (pd.Dataframe)

    """
    #Recorre todas las columnas comprueba que es una feature categorica y si es asi extrae la informacion
    df_num = pd.DataFrame()
    ceros = []
    nulos = []
    for col_index in range(len(self.num_cols)):

      col = self.num_cols[col_index]
      print("__________________________________________________________________")
      print(f"Numero de valores diferentes en la columna == {col} --- ",len(self.num_data[f"{col}"].value_counts()))
      print(f"Tamaño de la columna == {col} --- ",self.num_data[f"{col}"].shape[0]) # muestra el tamaño o forma de la serie
      print(f"% de valores unicos frente al numero de muestras de la columna == {col} --- ", round(100*(1- ((- self.num_data[f"{col}"].nunique() + self.num_data.shape[0] )/ self.num_data.shape[0])),3),"%")
      print(f"Media de la columna == {col} --- ",round(self.num_data[f"{col}"].mean(), 4)) # la media
      print("------------------------------------------------------------------")
      print(f"Maximo valor de la columna == {col} --- ", self.num_data[f"{col}"].max()) # El maximo valor de la columna
      print(f"Informacion del maximo valor en la columna : {col}:\n", self.num_data.iloc[self.data[f"{col}"].idxmax()]) # Demas features asociadas al valor maximo de esa columna
      print("------------------------------------------------------------------")
      print(f"Numero de valores nulos en la columna == {col} ---  ",self.num_data[f"{col}"].isnull().sum())
      nulos.append(self.num_data[f"{col}"].isnull().sum())

      num_ceros_col = (self.num_data[f"{col}"][(self.num_data[f"{col}"] == 0) | (self.num_data[f"{col}"] == 0.0)]).count()
      porcentaje_ceros = ( num_ceros_col/self.num_data[f"{col}"].count() ) *100
      ceros.append(num_ceros_col)
      print(f"Numero de valores '0' en la columna == {col} ---  ",num_ceros_col)
      print(f"Porcentaje de valores '0' en la columna == {col} ---  ",porcentaje_ceros.round(3))
      # Rango de cada numerical feature (para posible proceso futuro de normalizacion/estandarizacion)
      df_num.loc["COUNT",f"{col}"] = self.num_data.describe().loc["count",f"{col}"]
      # for future upgrades of pandas, to manage error of write str in numeirc (float64) column use : pd.to_numeric('-----', errors='coerce')
      df_num.loc["MIN",f"{col}"] = round(self.num_data[f"{col}"].min(),2)
      df_num.loc["25%",f"{col}"] = round(self.num_data.describe().loc["25%",f"{col}"],2)
      df_num.loc["50%",f"{col}"] = round(self.num_data.describe().loc["50%",f"{col}"],2)
      df_num.loc["75%",f"{col}"] = round(self.num_data.describe().loc["75%",f"{col}"],2)
      df_num.loc["MAX",f"{col}"] = round(self.num_data[f"{col}"].max(),2)
      df_num.loc["MEAN",f"{col}"] = round(self.num_data[f"{col}"].mean(),2)
      df_num.loc["STD",f"{col}"] = round(self.num_data[f"{col}"].std(),2)
      df_num.loc["NA",f"{col}"] = round(self.num_data[f"{col}"].isnull().sum(),2)
      df_num.loc["NA %",f"{col}"] = round(self.num_data[f"{col}"].isnull().sum()/(self.num_data[f"{col}"].isnull().count())*100,2)
      df_num.loc["CEROS",f"{col}"] = round(num_ceros_col,0)
      df_num.loc["CEROS %",f"{col}"] = round(porcentaje_ceros,0)
      print("__________________________________________________________________")

    # Bar plotting: Plot con valores nulos y con valores "0"
    figure = plt.figure(figsize=(15,15), layout = 'constrained')
    axes_nul = plt.subplot(2,1,1)
    axes_cero = plt.subplot(2,1,2)
    axes_nul.bar(x = self.num_cols, height = nulos , label = self.num_cols, align = 'center' , edgecolor = 'black',color = 'lightgreen')
    axes_cero.bar(x = self.num_cols, height = ceros , label = self.num_cols, align = 'center', edgecolor = 'black',color = 'cyan')
    # Set x and y labels for each axes on the grid:
    axes_nul.set_title(label = "NA Values")
    axes_nul.set_ylabel("Frequency")
    # Fijan etiquetas de las categorias del eje x
    axes_nul.set_xticks(ticks = range(len(self.num_cols)), labels = self.num_cols,  minor=False)
    axes_cero.set_xticks(ticks = range(len(self.num_cols)), labels = self.num_cols,  minor=False)
    axes_nul.set_xticklabels(labels = self.num_cols, rotation = 30)
    axes_cero.set_title(label = "Cero Values")
    axes_cero.set_ylabel("Frequency")
    axes_cero.set_xticklabels(labels = self.num_cols, rotation = 30)
    # Grid on axes
    axes_nul.grid(linewidth = 0.5, alpha=1)
    axes_cero.grid(linewidth = 0.5, alpha=1)
    #plt.show()

    return df_num

  def plot_hist(self,**kwargs) -> None:

    """
    Grafica el histograma de cada columna numerica utilizando matplot lib

    Parametros
    ----------
      Key word arguments:
        - fig_x_size : (int) Tamaño de cada figura en x
        - fig_y_size : (int) Tamaño de cada figura en y
        - fig_rows :(int)  Numero de filas del grid de figura
        - fig_cols : (int) Numero de columnas del grid de figura
        - linewidth : (float) El ancho de linea de la figura (el marco de la figura)
        - layout : (str) ['constrained','compressed', 'tight', None] Motor de ajuste (algoritmo) de los ejes para evitar solapamiento
        - bins : (int) Numero de bins de cada histograma
        - density : (bool) Sobreescribe en el histograma la funcion de densidad de probabilidad de la distribucion
        - stacked : (bool) Si es True, los datos múltiples se apilan uno encima del otro Si es False, los datos múltiples se disponen uno al lado del otro.
        - divide_feature : (str) Nombre de la columna categorica para dividir la distribucion ploteda (se recomienda que esta columna tenga pocas categorias, mejor si es binaria)

    Retorna
    -------
      None

    """

    ## Inicializacion del diccionario: kwargs [key-value no definidos como argumento de la funcion]
    # Diccionario para comparar (predefined kw values):
    possible_key_value = {

                          # Pyplot class .subplots() method keyword arguments:
                          'fig_x_size': 10, # Tamaño de cada figura en x
                          'fig_y_size': 8, # Tamaño de cada figura en y
                          'fig_rows': 1, # Numero de filas del grid de figura
                          'fig_cols': 3, # Numero de columnas del grid de figura
                          'linewidth': 0.5,  # El ancho de linea de la figura (el marco de la figura)
                          'layout': None , # Motor de ajuste de los ejes para evitar solapamiento

                          # Axes class .hist() method keyword arguments:
                          'bins': 100, # Bins de cada histograma
                          'density': False, # Sobreescribe la funcion de densidad de probabilidad de la distribucion
                          'stacked': False,

                          # Para dividr una distribucion en funcion de otra columna [DEBE SER CATEGORICA Y CON POCAS CATEGORIAS]
                          'divide_feature': None,

                          }

    # Comparacion y completado del diccionario: kwargs
    for k,v in possible_key_value.items():
      if k in kwargs:
        pass
      else:
        kwargs[k] = v

    ## Resto de variables necesarias:
    # Numero de figuras necesarias en funcion del grid y el numero de columnas numericas de data:
    total_fig = int(np.ceil(len(self.num_cols)/(kwargs["fig_rows"]*kwargs["fig_cols"]))) # Redondeo al entero mayor para asegurar que se plotean todas las features
    column_data_index = 0 # Contador para acceder a cada columna numerica

    matplot_lib_colors = [
                            'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white',
                            'cyan', 'magenta', 'lime', 'teal', 'olive', 'maroon', 'navy', 'coral', 'indigo', 'gold',
                            "teal", "lawngreen", "turquoise", "r", "darkmagenta", "skyblue", "darkgray",
                            "navy", "forestgreen","b", "g", "r", "c", "m", "y", "k", "firebrick", "darkslategray", "darkorchid", "olive", "dimgray",
                            "royalblue", "mediumseagreen", "indianred", "teal", "purple", "goldenrod", "dimgrey",
                            "cornflowerblue", "limegreen", "darkgoldenrod", "mediumaquamarine", "mediumvioletred", "yellowgreen", "grey",
                            "steelblue", "mediumspringgreen", "brown", "mediumturquoise", "blueviolet", "khaki", "silver",
                            "skyblue", "springgreen", "saddlebrown", "turquoise", "darkviolet", "darkkhaki", "lightgray",
                            "dodgerblue", "chartreuse", "sienna", "paleturquoise", "mediumorchid", "darkolivegreen", "gainsboro",
                            "deepskyblue", "lawngreen", "peru", "lightcyan", "thistle", "darkseagreen"

                          ]

    ## Plotting
    column_data_index = 0
    
    # Comprobacion que la columna especificada en: divide_feature es categorica:
    if kwargs['divide_feature'] != None:
      if self.data[kwargs['divide_feature']].nunique() >= 6 :
          print(f"KEY WORD ERRROR -- THE FEATURE {kwargs['divide_feature']} HAS TOO MANY UNIQUE VALUES")
    for _ in range(total_fig):
      figure,axes = plt.subplots(nrows=kwargs["fig_rows"], ncols=kwargs["fig_cols"], linewidth= kwargs["linewidth"], layout = kwargs["layout"], label='Histograms', \
                                 figsize=(kwargs["fig_x_size"],kwargs["fig_y_size"]) )

      if len(axes.shape) > 1: # Evitar error si fig_rows = 1
        for i in range(axes.shape[0]):
          for j in range(axes.shape[1]):

            try: # Uso excepciones: evitar error por out of range

              # Set x and y labels for each axes on the grid:
              axes[i,j].set_xlabel(f"{self.num_cols[column_data_index]}")
              axes[i,j].set_ylabel("Frequency")

              # Grid on axes
              axes[i,j].grid(linewidth=kwargs["linewidth"], alpha=1)

              # Plotting divided Histogram using another categorical existin feature overlay if exists
              if kwargs['divide_feature'] != None and self.data[kwargs['divide_feature']].nunique() < 6:
                # Paleta de colores aleatorios para cada axes y cada barra
                np.random.seed(column_data_index)
                color = [matplot_lib_colors[c] for c in np.random.randint(0,len(matplot_lib_colors),len(self.data[kwargs['divide_feature']].value_counts().index))]
                try:
                    for cat_index,cat_divide in enumerate(self.data[kwargs['divide_feature']].unique()):
                      print(cat_divide)

                      # Plotea el histograma de todas las columna numericas divididas en funcion de su frecuencia para todas las categorias de la variable especificada categorica
                      x =  self.data[self.num_cols[column_data_index]][self.data[kwargs['divide_feature']] == cat_divide]
                      axes[i,j].hist(x, bins=kwargs["bins"], color = color[cat_index] , label = f"{cat_divide}" )


                except:
                  print(f"ERRROR PLOTTING HISTOGRAM USING THE FEATURE: {kwargs['divide_feature']}")

              else:

                # Normal Histogram plotting
                axes[i,j].hist(self.num_data[self.num_cols[column_data_index]], bins=kwargs["bins"], density = kwargs["density"], stacked = kwargs["stacked"], label = str(self.num_cols[column_data_index]))

                # Density function plotting using seaborn
                if  kwargs['density'] == True:
                  sns.kdeplot(x=self.num_data[self.num_cols[column_data_index]], ax =  axes[i,j], label = 'Probability Density Function')

              # Legend on axes
              axes[i,j].legend(loc='best')

            except:
              pass
            column_data_index += 1

      else: # Evitar error si fig_rows = 1
        for i in range(axes.shape[0]):

          try: # Uso excepciones: evitar error por out of range

            # Set x and y labels for each axes on the grid:
            axes[i].set_xlabel(f"{self.num_cols[column_data_index]}")
            axes[i].set_ylabel("Frequency")

            # Grid on axes
            axes[i].grid(linewidth=kwargs["linewidth"], alpha=1)


            # Plotting divided Histogram using another categorical existin feature overlay if exists
            if kwargs['divide_feature'] != None and self.data[kwargs['divide_feature']].nunique() < 6:

              # Paleta de colores aleatorios para cada axes y cada barra
              np.random.seed(column_data_index)
              color = [matplot_lib_colors[c] for c in np.random.randint(0,len(matplot_lib_colors),len(self.data[kwargs['divide_feature']].value_counts().index))]

              try:

                  for cat_index,cat_divide in enumerate(self.data[kwargs['divide_feature']].unique()):

                    # Plotea el histograma de todas las columna numericas divididas en funcion de su frecuencia para todas las categorias de la variable especificada categorica
                    x =  self.data[self.num_cols[column_data_index]][self.data[kwargs['divide_feature']] == cat_divide]
                    axes[i].hist(x, bins=kwargs["bins"], color = color[cat_index] ,label = f"{cat_divide}", alpha = 0.5)


              except:
                print(f"ERRROR PLOTTING HISTOGRAM USING THE FEATURE: {kwargs['divide_feature']}")

            else:

              # Histogram plotting
              axes[i].hist(self.num_data[self.num_cols[column_data_index]], bins=kwargs["bins"], density = kwargs["density"] , stacked = kwargs["stacked"], label = str(self.num_cols[column_data_index]) )

              # Density function plotting using seaborn
              if  kwargs['density'] == True:
                sns.kdeplot(x=self.num_data[self.num_cols[column_data_index]], ax =  axes[i], label = 'Probability Density Function')

            # Legend on axes
            axes[i].legend(loc='best')

          except:
            pass
          column_data_index += 1

      # Adjust the padding between and around subplots
      if kwargs['layout'] == None:
        figure.tight_layout()

     # plt.show() # Figure object tiene atributo show [para plotear solo una figura especifica]

  def plot_corr_heatmap(self,**kwargs) -> None:

    """
    Calcula la matriz de correlacion (coeficiente de pearson) entre las columnas numericas del dataframe y grafica el mapa de calor de esta mediante la libreria seaborn

    Parametros
    ----------
      key word arguments:
        - fig_x_size : (int) Tamaño de cada figura en x
        - fig_y_size : (int) Tamaño de cada figura en y
        - linewidth : (float) El ancho de linea de la figura (el marco de la figura)
        - layout : (str) ['constrained','compressed', 'tight', None] Motor de ajuste (algoritmo) de los ejes para evitar solapamiento
        - cmap : (str) [valores posibles en la variable: seaborn_palettes] Nombre del palette (conjunto de colores) para realizar el mapeo de la matriz de correlacion
        - annot: (bool) para ploter o no el valor numerico de corrrelacion (entre variables) en la celda que corresponda del mapa de calor
        - fmt : (str) especificacion del formato numerico en el caso de que : annot = True

    Retorna
    -------
      None

    """

    ## Inicializacion del diccionario: kwargs [key-value no definidos como argumento de la funcion]
    # Diccionario para comparar (predefined kw values):
    possible_key_value = {

                          # Pyplot class .subplots() method keyword arguments:
                          'fig_x_size': 15, # Tamaño de cada figura en x
                          'fig_y_size': 15, # Tamaño de cada figura en y
                          'layout': None , # Motor de ajuste de los ejes para evitar solapamiento
                          'linewidth': 0.5,  # El ancho de linea de la figura (el marco de la figura)

                          # Seaborn class .heatmap() method keyword arguments:
                          'cmap': 'coolwarm', # Mapeo de los valores numericos de la matriz con el espacio de colores en el seaborn palette elegido
                          'annot': True, # If True, plot el valor numerico de corrrelacion (entre variables) en la celda del mapa de calor
                          'fmt': '.2f' # Especificacion del formato numerico

                          }

    # Comparacion y completado del diccionario: kwargs
    for k,v in possible_key_value.items():
      if k in kwargs:
        pass
      else:
        kwargs[k] = v

    # Calculo matriz correlacion de las columnas numericas
    A_corr = self.num_data.corr(method = 'pearson')

    # Plotting del mapa de calor de la matriz de correlacion
    figure,axes = plt.subplots( nrows = 1, ncols=1, linewidth= kwargs["linewidth"], layout = kwargs["layout"], figsize=(kwargs["fig_x_size"], kwargs["fig_y_size"]))

    # Tittle setting on axes
    axes.set_title("Correlation Matrix Heatmap")

    # Drawing on specify axes de correlation heatmap
    try:
      sns.heatmap(A_corr, ax = axes , annot = kwargs["annot"], cmap = kwargs["cmap"] , fmt = kwargs["fmt"])
    except: # Por si el cmap no existe
      print(f"COLOR MAP {kwargs['cmap']} DOES NOT EXIST [matplotlib colormap : ,'coolwarm' has been applied]")
      sns.heatmap(A_corr, ax = axes , annot = kwargs["annot"], cmap = 'coolwarm', fmt = kwargs["fmt"])


    # Adjust the padding between and around subplots
    if kwargs['layout'] == None:
      figure.tight_layout()

    # Showing
    # plt.show()

  def plot_bar(self,**kwargs) -> None:

    """
    Plot de grafico de barras de las variables categoricas

    Parametros
    ----------
      Key word arguments:
        - fig_x_size : (int) Tamaño de cada figura en x
        - fig_y_size : (int) Tamaño de cada figura en y
        - fig_rows :(int)  Numero de filas del grid de figura
        - fig_cols : (int) Numero de columnas del grid de figura
        - linewidth : (float) El ancho de linea de la figura (el marco de la figura)
        - layout : (str) ['constrained','compressed', 'tight', None] Motor de ajuste (algoritmo) de los ejes para evitar solapamiento
        - width : (float) Ancho de las barras graficadas
        - align : (str) ['center', 'edge'] 'center' alinea la base de cada barra con el eje x y 'edge' alinea los bordes izquierdos de las barras con las posiciones x
        - plot_limit_categories : (int) Limite de categorias para graficar en las columnas categoricas
        - rotation : (float) Grados de rotacion de los x ticks labels


    Retorna
    -------
      None

    """

    ## Inicializacion del diccionario: kwargs [key-value no definidos como argumento de la funcion]
    # Diccionario para comparar (predefined kw values):
    possible_key_value = {

                          # Pyplot class .subplots() method keyword arguments:
                          'fig_x_size': 10, # Tamaño de cada figura en x
                          'fig_y_size': 10, # Tamaño de cada figura en y
                          'fig_rows': 1, # Numero de filas del grid de figura
                          'fig_cols': 3, # Numero de columnas del grid de figura
                          'linewidth': 0.5,  # El ancho de linea de la figura (el marco de la figura) y de cada axes tambien
                          'layout': 'constrained' , # Motor de ajuste de los ejes para evitar solapamiento

                          # Axes class .bar() method keyword arguments:
                          'width': 0.8, # Ancho de las barras de cada barplot
                          'align': 'center', # Parametro tipo keyword: alinear la base de la barra con el eje x o 'edge' para que este en el limite

                          # Own keyword arguments:
                          'plot_limit_categories': 20, # Establece un limite de categorias para plotear en todas las columnas categoricas

                          # Axes object .set_xticklabels() method keyword arguments
                          'rotation': 0 ,


                          }

    # Comparacion y completado del diccionario: kwargs
    for k,v in possible_key_value.items():
      if k in kwargs:
        pass
      else:
        kwargs[k] = v

    ## Resto de variables necesarias:
    # Numero de figuras necesarias en funcion del grid y el numero de columnas numericas de data:
    total_fig = int(np.ceil(len(self.cat_cols)/(kwargs["fig_rows"]*kwargs["fig_cols"]))) # Redondeo al entero mayor para asegurar que se plotean todas las features
    column_data_index = 0 # Contador para acceder a cada columna numerica
    colors = [
                "teal", "lawngreen", "turquoise", "r", "darkmagenta", "skyblue", "darkgray",
                "navy", "forestgreen","b", "g", "r", "c", "m", "y", "k", "firebrick", "darkslategray", "darkorchid", "olive", "dimgray",
                "royalblue", "mediumseagreen", "indianred", "teal", "purple", "goldenrod", "dimgrey",
                "cornflowerblue", "limegreen", "darkgoldenrod", "mediumaquamarine", "mediumvioletred", "yellowgreen", "grey",
                "steelblue", "mediumspringgreen", "brown", "mediumturquoise", "blueviolet", "khaki", "silver",
                "skyblue", "springgreen", "saddlebrown", "turquoise", "darkviolet", "darkkhaki", "lightgray",
                "dodgerblue", "chartreuse", "sienna", "paleturquoise", "mediumorchid", "darkolivegreen", "gainsboro",
                "deepskyblue", "lawngreen", "peru", "lightcyan", "thistle", "darkseagreen"
              ]


    ## Plotting
    column_data_index = 0
    if kwargs["fig_rows"] == 1 and kwargs["fig_cols"] == 1:
      kwargs["fig_rows"] = 2
      kwargs["fig_cols"] = 1

    for _ in range(total_fig):
      figure,axes = plt.subplots( nrows=kwargs["fig_rows"], ncols=kwargs["fig_cols"], linewidth= kwargs["linewidth"], layout = kwargs["layout"], \
                                 label='Bar plot', figsize=(kwargs["fig_x_size"],kwargs["fig_y_size"]) )
      if len(axes.shape) > 1: # Evitar error si fig_rows = 1
        for i in range(axes.shape[0]):
          for j in range(axes.shape[1]):

            # Set x and y labels for each axes on the grid:
            try: # Uso excepciones: evitar error por out of range

              # Set x and y labels for each axes on the grid:
              axes[i,j].set_xlabel(f"{self.cat_cols[column_data_index]}")
              axes[i,j].set_ylabel("Frequency")

              # Grid on axes
              axes[i,j].grid(linewidth=kwargs["linewidth"], alpha=1)

              # Calculo de la frecuencia de cada categoria en cada columna
              categories,values = self.cat_data[self.cat_cols[column_data_index]].value_counts().index, self.cat_data[self.cat_cols[column_data_index]].value_counts().values

              # Establece una paleta de colores (lista con nombre de colores de variable colors) aleatoria por grafica (cada axes) y barra (cada categoria por columna)
              np.random.seed(column_data_index)
              color = [colors[i] for i in np.random.randint(0,len(colors),len(categories))]

              # Bar plotting
              if len(categories) < int(kwargs["plot_limit_categories"]): # Si hay muchas categorias no ploteamos

                # Fijan etiquetas de las categorias del eje x
                axes[i,j].set_xticks(ticks = range(len(categories)), labels = categories,  minor=False)

                # Rotation of x axis ticks or labels
                axes[i,j].set_xticklabels(categories,  rotation = int(kwargs["rotation"]), ha='right')


                # Bar plotting
                axes[i,j].bar(categories, values , label = categories, color = color)

                # Legend
                axes[i,j].legend(title=f'{self.cat_cols[column_data_index]}',loc='best')

            except:
              pass

            column_data_index += 1

      else: # Evitar error si fig_rows = 1
        for i in range(axes.shape[0]):

          try:

            # Set x and y labels for each axes on the grid
            axes[i].set_xlabel(f"{self.cat_cols[column_data_index]}")
            axes[i].set_ylabel("Frequency",fontsize=8)

            # Grid on axes
            axes[i].grid(linewidth=kwargs["linewidth"], alpha=1)

            # Calculo de la frecuencia de cada categoria en cada columna
            categories,values = self.cat_data[self.cat_cols[column_data_index]].value_counts().index, self.cat_data[self.cat_cols[column_data_index]].value_counts().values

            # Establece una paleta de colores (lista con nombre de colores de variable colors) aleatoria por grafica (cada axes) y barra (cada categoria por columna)
            np.random.seed(column_data_index) # Diferenet semilla para cada axes/grafica
            color = [colors[c] for c in np.random.randint(0,len(colors),len(categories))] # list comprehension

            if len(categories) < int(kwargs["plot_limit_categories"]): # Si hay muchas categorias no ploteamos

              # Fijan etiquetas de las categorias del eje x
              axes[i].set_xticks(ticks = range(len(categories)), labels = categories,  minor=False)

              # Rotation of x axis ticks or labels
              axes[i].set_xticklabels(categories, rotation = int(kwargs["rotation"]), ha='right')


              # Bar plotting
              axes[i].bar(categories, values, label = categories, color = color)

              # Plot legend
              axes[i].legend(title=f'{self.cat_cols[column_data_index]}',loc='best')


          except:
            pass

          column_data_index += 1

      # Adjust the padding between and around subplots
      if kwargs['layout'] == None:
        figure.tight_layout()

      # figure.show() # Figure object tiene atributo show [para plotear solo una figura especifica]

  def plot_combined_bar(self,**kwargs) -> None:

      """
      Plot de grafico de barras dividiendo cada barra ploteada de la variable deseada (target_feature) teniendo en cuenta las categorias de otra columna indicada (divide_feature).
      La altura de cada barra viene determinada otra variable numerica indicada a traves del argumento: value_feature.

      Parametros
      ----------
        Key word arguments:
          - fig_x_size : (int) Tamaño de cada figura en x
          - fig_cols : (int) Numero de columnas del grid de figura
          - linewidth : (float) El ancho de linea de la figura (el marco de la figura)
          - layout : (str) ['constrained','compressed', 'tight', None] Motor de ajuste (algoritmo) de los ejes para evitar solapamiento
          - estimator : (str or callable that maps vector -> scalar) Función estadística para estimar dentro de cada bin categórico.
            * Ejemplos de funciones que mapeen un vector a un escalar pueden ser: np.mean, np.sum, ... funciones de numpy que se apliquen a un vector y calculan la altura de esas barras
              por ejemplo: al usar np.mean se calcula la media de todo el vector y te la plotea en el eje y
                           al usar np.sum o 'sum' se suma todo el vector y te lo plotea en el eje y
          - errorbar : (str) Nombre del método de la barra de error (ya sea "ci", "pi", "se" o "sd"), o None para ocultar la barra de error.
          - orient : (str) ['v','h','x','y'] Orientacion de las barras ploteadas
          - target_feature: (str) Nombre de la columna principal cuyas categorias aparecen sobre el eje x
          - divide_feature: (str) Nombre de la columna que determina como se van a dividir las categorias de la columna target_feature. Si no se especifica ninguna
                                  hace la division de target_feature con todas las coloumnas categoricas que no superen: max_categories.
          - value_feature: (str) Nombre de la columna que determina la altura de cada barra
          - max_categories : (int) Numero maximo de categorias de la/las columnas en divide_feature [ es decir, maximo numero de categorias en las que dividir la target feature]
          - color : (int) Indice de la lista seaborn_palettes (sirve para cambiar el color de la paleta de colores)


      Retorna
      -------
        None

      """

      ## Inicializacion del diccionario: kwargs [key-value no definidos como argumento de la funcion]
      # Diccionario para comparar (predefined kw values):
      possible_key_value = {

                            # Pyplot class .figure() method keyword arguments:
                            'fig_x_size': 14, # Tamaño de cada figura en x
                            'fig_cols': 2, # Numero de columnas del grid de figura
                            'linewidth': 0.5,  # El ancho de linea de la figura (el marco de la figura) y de cada axes tambien
                            'layout': None , # Motor de ajuste de los ejes para evitar solapamiento

                            # Axes class .barplot() method keyword arguments:
                            'estimator': None, # Estimator puede ser un callable (es decir funcion) np.mean, 'sum', ... funciones de numpy que se apliquen a un vector y calculan la altura de esas barras
                                               # np.mean te hace la media de todo el vestor y te la plotea en el eje y
                                               # np.sum te suma todo el vector y te lo plotea en el eje y
                            'errorbar': 'sd',
                            'orient':'v', # orientacion de las barras

                            # Own keyword arguments:
                            'target_feature': None, # Feature ppal a plotear
                            'divide_feature': None, # Nombre de otra feature categorica para subdividir cada barra ploteada (es decir, en funcion de cada categoria de esa columna)
                            'value_feature': None, # Nombre de  feature NUMERICA para la altura de cada barra
                            'max_categories': 6, # Numero maximo de categorias de la/las columnas en divide_feature [ es decir, para dividir la target feature]
                            'color':0, # index del palette en seaborn_palettes
                            'verbose':0

                            }

      # Comparacion y completado del diccionario: kwargs
      for k,v in possible_key_value.items():
        if k in kwargs:
          pass
        else:
          kwargs[k] = v

      ## Resto de variables necesarias:
      seaborn_palettes = [
                            'deep', 'muted', 'bright', 'pastel', 'dark',
                            'colorblind', 'Set1', 'Set2', 'Set3', 'Paired',
                            'Accent', 'Gist_earth', 'Prism', 'Ocean', 'Dark2',
                            'Paired', 'coolwarm', 'husl', 'viridis', 'cubehelix'
                        ]

      ## Plotting

      # Avoiding error plotting
      if kwargs["value_feature"] not in self.num_cols:

        print("ERROR: key word argument: 'value_feature' must be a numeric column of the dataframe")

      elif kwargs["value_feature"] == None:

        print("ERROR: you must define the key word argument: 'value_feature' ")

      elif kwargs["target_feature"] not in self.cat_cols and kwargs["target_feature"] != None:

        print("ERROR: key word argument: 'target_feature' must be a categorical column of the dataframe")

      elif kwargs["target_feature"] == None:

        print("ERROR: you must define the key word argument: 'target_feature'")


      else:

        # Calculo del numero de filas y columnas en funcion del numero de axes (graficas) necesarias
        ncols = kwargs['fig_cols']

        # Filtrado de las categorical columns con excesivas categorias
        categorias_efectivas = ([cat for cat in self.cat_cols if self.data[cat].nunique() < kwargs["max_categories"]])
        num_categorias_efectivas = len(categorias_efectivas)
        
        if kwargs["verbose"] == 1:
          print("Columnas categoricas a comparar: ",categorias_efectivas)
        
        nrows = ((num_categorias_efectivas - 1) // ncols ) + 1 # si el numero de graficas no multiplo del numero de columnas se coge el entero menor y se le suma uno
        
        # nrows nunca puede ser cero
        if num_categorias_efectivas == 1:
          nrows = 1
        # Obtain the figure object with .figure() method of class pyplot
        figure = plt.figure(figsize=(kwargs["fig_x_size"],nrows * 5), layout = kwargs["layout"])

        # Contador
        column_data_index = 0

        if kwargs["divide_feature"] == None:

          for _ , col_divide in enumerate(categorias_efectivas):
            if kwargs["verbose"] == 1:
              print(col_divide,kwargs["target_feature"])
              
            if kwargs["target_feature"] != col_divide:
   
              # Obtain axes object with .subplot() method
              column_data_index += 1
              
              axes = plt.subplot(nrows, ncols, column_data_index)
              
              if kwargs["verbose"] == 1:
                print("(nrows, ncols) : ",(nrows, ncols))
                
              new_dataframe = self.data.loc[:,[str(kwargs["target_feature"]),str(kwargs["value_feature"]),col_divide]] # Df auxiliar con features de interes
              
              if kwargs["verbose"] == 1:
                print(new_dataframe.head())
                
              sns.barplot( 
                          data = new_dataframe, 
                          x = str(kwargs["target_feature"]) , 
                          y = str(kwargs["value_feature"]) , 
                          hue = col_divide , 
                          orient = kwargs['orient'] , 
                          palette = seaborn_palettes[int(kwargs["color"])] ,
                          saturation = 1.0 ,
                          estimator = kwargs['estimator'], 
                          errorbar  = kwargs['errorbar'], 
                          ax = axes
                          )

              # Grid on axes
              axes.grid(linewidth=kwargs["linewidth"], alpha=1)
            else:
              
              # Obtain axes object with .subplot() method
              column_data_index += 1
              
              axes = plt.subplot(nrows, ncols, column_data_index)
              
              if kwargs["verbose"] == 1:
                print("(nrows, ncols) : ",(nrows, ncols))
                

              new_dataframe = self.data.loc[:,[str(kwargs["target_feature"]),str(kwargs["value_feature"])]] # Df auxiliar con features de interes
              
              if kwargs["verbose"] == 1:
                print(new_dataframe.head())
                
              sns.barplot(  
                          data = new_dataframe, 
                          x = str(kwargs["target_feature"]) , 
                          y = str(kwargs["value_feature"]) ,  
                          orient = kwargs['orient'] ,
                          saturation = 1.0 ,
                          estimator = kwargs['estimator'], 
                          errorbar = kwargs['errorbar'], 
                          ax = axes
                          )

              # Grid on axes
              axes.grid(linewidth=kwargs["linewidth"], alpha=1)

        elif kwargs["divide_feature"] != None:

          # Obtain axes object with .subplot() method
          column_data_index += 1
          axes = plt.subplot(nrows, ncols, column_data_index)
          
          if kwargs["verbose"] == 1:
            print("(nrows, ncols",(nrows, ncols))
            
          # Obtain the drew axes object with .barplot() method
          if str(kwargs["value_feature"]) != str(kwargs["divide_feature"]):
            new_dataframe = self.data.loc[:,[str(kwargs["target_feature"]),str(kwargs["value_feature"]),str(kwargs["divide_feature"])]]
          else:
            new_dataframe = self.data.loc[:,[str(kwargs["target_feature"]),str(kwargs["value_feature"])]]
            
          if kwargs["verbose"] == 1:
            print(new_dataframe.head())
            
          sns.barplot(
                        data = new_dataframe, 
                        x = str(kwargs["target_feature"]) , 
                        y = str(kwargs["value_feature"]) , 
                        hue = str(kwargs["divide_feature"]) , 
                        orient = kwargs['orient'] ,
                        palette = seaborn_palettes[int(kwargs["color"])] ,
                        saturation = 1.0 ,
                        estimator = kwargs['estimator'], 
                        errorbar  = kwargs['errorbar'] ,
                        ax = axes
                        )

          # Grid on axes
          axes.grid(linewidth=kwargs["linewidth"], alpha=1)

        # Adjust the padding between and around subplots
        if kwargs["layout"] == None:
          figure.tight_layout()

        # Showing the figure (plotting)
        # plt.show()



  def plot_boxplot(self,**kwargs) -> None:

        """
        Grafica un diagrama de cajas para mostrar las distribuciones con respecto a las variables numericas. Un diagrama de caja (o diagrama de caja y bigotes) muestra
        la distribución de datos cuantitativos. La caja muestra los cuartiles del conjunto de datos, mientras que los bigotes se extienden para mostrar el resto de
        la distribución, excepto los puntos que se determinan como "atípicos" mediante un método que es función del rango intercuartílico.

        Parametros
        ----------
          key word arguments:
            - fig_x_size : (int) Tamaño de cada figura en x
            - fig_y_size : (int) Tamaño de cada figura en y
            - linewidth : (float) El ancho de linea de la figura (el marco de la figura)
            - layout : (str) ['constrained','compressed', 'tight', None] Motor de ajuste (algoritmo) de los ejes para evitar solapamiento
            - color : (int), Index del palette en seaborn_palettes

        Retorna
        -------
          None

        """

        ## Inicializacion del diccionario: kwargs [key-value no definidos como argumento de la funcion]
        # Diccionario para comparar (predefined kw values):
        possible_key_value = {

                              # Pyplot class .figure() method keyword arguments:
                              'fig_x_size': 16, # Tamaño de cada figura en x
                              'fig_y_size': 20, # Numero de columnas del grid de figura
                              'linewidth': 0.5,  # El ancho de linea de la figura (el marco de la figura) y de cada axes tambien
                              'layout': None , # Motor de ajuste de los ejes para evitar solapamiento

                              # Own keyword arguments:
                              'color':0, # index del palette en seaborn_palettes

                              }

        # Comparacion y completado del diccionario: kwargs
        for k,v in possible_key_value.items():
          if k in kwargs:
            pass
          else:
            kwargs[k] = v

        ## Resto de variables necesarias:
        colors = [
                    "teal", "lawngreen", "turquoise", "r", "darkmagenta", "skyblue", "darkgray",
                    "navy", "forestgreen","b", "g", "r", "c", "m", "y", "k", "firebrick", "darkslategray", "darkorchid", "olive", "dimgray",
                    "royalblue", "mediumseagreen", "indianred", "teal", "purple", "goldenrod", "dimgrey",
                    "cornflowerblue", "limegreen", "darkgoldenrod", "mediumaquamarine", "mediumvioletred", "yellowgreen", "grey",
                    "steelblue", "mediumspringgreen", "brown", "mediumturquoise", "blueviolet", "khaki", "silver",
                    "skyblue", "springgreen", "saddlebrown", "turquoise", "darkviolet", "darkkhaki", "lightgray",
                    "dodgerblue", "chartreuse", "sienna", "paleturquoise", "mediumorchid", "darkolivegreen", "gainsboro",
                    "deepskyblue", "lawngreen", "peru", "lightcyan", "thistle", "darkseagreen"
                  ]

        # Calcular el número de filas y columnas necesarias
        num_rows = (len(self.num_cols) - 1) // 4 + 1  # Asegurar al menos 1 fila
        num_cols = 4

        # Configurar el tamaño de la figura
        figure = plt.figure(figsize=(kwargs["fig_x_size"], kwargs["fig_y_size"]), layout = kwargs["layout"])

        # Crear subgráficos para cada variable numérica
        for i, column in enumerate(self.num_cols):

            # Creacion de las axes (grafica)
            axes = plt.subplot(num_rows, num_cols, i + 1)

            sns.boxplot(x=self.num_data[column], saturation = 1, color = colors[kwargs["color"]] , ax = axes)  # Mostrar outliers
            # Titulo
            axes.set_title(f'Boxplot de {column}')

            # Grid on axes
            axes.grid(linewidth=kwargs["linewidth"], alpha=1)

        # Ajustar el diseño y mostrar la figura
        if kwargs["layout"] == None:
          figure.tight_layout()

      # plt.show()
          

  def plot_violin(self,**kwargs) -> None:

        """
        Digrama de violin
        Parametros
        ----------
          key word arguments:
            - kde_feature : List[str]
                            Nombre columna a estimar su funcion de densidad de probabilidad
            - diving_feature : str
                               Nombre columna para filtrar y dividir el eje x 
            - color_feature : str
                              Nombre columna para, en funcion de sus valores unicos (no deben ser excesivos), 
                              filtrar por colores
            - title : str

        Retorna
        -------
           - None

        """

        ## Inicializacion del diccionario: kwargs [key-value no definidos como argumento de la funcion]
        # Diccionario para comparar (predefined kw values):
        possible_key_value = {
                              # Key word arguments for violin method
                              "kde_feature " : [],
                              "diving_feature": "",
                              "color_feature": "",
                              # Key word arguments for update_layout method
                              "title" : "",
                  

                              }

        # Comparacion y completado del diccionario: kwargs
        for k,v in possible_key_value.items():
          if k in kwargs:
            pass
          else:
            kwargs[k] = v

        # Creting the violin plot using express module of plotly
        for _ , feature in enumerate(kwargs["kde_feature"]):
          fig = px.violin(
                            self.data, 
                            y= feature , 
                            x= kwargs["diving_feature"], 
                            color = kwargs["color_feature"], 
                            box=True, 
                            points="all",

                        )
          
          # Layout for the figure
          fig.update_layout(
                  title = kwargs["title"] if kwargs["title"] is not None else "",
                  xaxis = dict(
                              title = kwargs["diving_feature"],
                              autorange = True,
                              showline = True,
                              showgrid = True,
                              gridcolor = None,
                              showticklabels = True,
                              zeroline = False,
                              linecolor = None,
                              linewidth = 0.5,
                              ticks = 'outside',
                              tickfont = dict(
                                              family = 'Arial',
                                              color = 'rgb(82,82,82)',
                                              size = 12
                                            )
              
                  ),
                  yaxis = dict(
                              title = feature,
                              autorange = True,
                              showline = True,
                              showgrid = True,
                              showticklabels = True,
                              gridcolor = None,
                              zeroline = True,
                              linecolor = None,
                              linewidth = 0.5,
                              ticks = 'outside',
                              tickfont =dict(
                                              family = 'Arial',
                                              color = 'rgb(82,82,82)',
                                              size = 12
                                            )
              
                  ),
                  autosize = False,
                  width=1000,
                  height = 600,
                  margin = dict(
                                autoexpand = True,
                                l= 80,
                                r = 90,
                                t = 60,
                                b = 60
                                ),
                  showlegend = True,
                  plot_bgcolor = None,
                  legend = dict(
                                bgcolor = 'white',
                                bordercolor = 'black',
                                borderwidth = 0.5,
                                title = dict(
                                              font = dict(
                                                          family = 'Arial',
                                                          color = 'black',
                                                          size = 16
                                                        ),
                                              text = kwargs["color_feature"],
                                              side = 'top'
                                              ),
                                font = dict(
                                              family = 'Arial',
                                              color = 'rgb(82,82,82)',
                                              size = 12
                                            )

                                )
                  )
          
          fig.show()

  def plot_scatter_3d(self,**kwargs) -> None:

        """
        Digrama de violin
        Parametros
        ----------
          key word arguments:
            - x_feature : List[str]
            - y_feature : List[str]
            - z_feature : str
            - color_feature : str
                              Nombre columna filtrar por colores
            - title : str

        Retorna
        -------
           - None

        """
        ## Inicializacion del diccionario: kwargs [key-value no definidos como argumento de la funcion]
        # Diccionario para comparar (predefined kw values):
        possible_key_value = {
                              # Key word arguments for scatter_3d method
                              "x_feature " : [],
                              "y_feature": [],
                              "z_feature": "",
                              "color_feature": "",
                               "opacity" : 1,

                              # Key word arguments for update_layout method
                              "title" : "",
                              "cmap" : 0,

                              }

        # Comparacion y completado del diccionario: kwargs
        for k,v in possible_key_value.items():
          if k in kwargs:
            pass
          else:
            kwargs[k] = v

        # Creting the scatter 3d plot using express module of plotly
        for _ , x_feature in enumerate(kwargs["x_feature"]):

          for _ , y_feature in enumerate(kwargs["y_feature"]):

            if x_feature != y_feature:

              # Creating the figure with the 3D scatter
              fig = px.scatter_3d(
                                              self.data, 
                                              y= y_feature , 
                                              x= x_feature, 
                                              z = kwargs["z_feature"],
                                              color = kwargs["color_feature"], 
                                              opacity = kwargs["opacity"],
                                              color_continuous_scale  = self.plotly_cmaps[kwargs["cmap"]]

                                            )

              # Layout for the figure
              fig.update_layout(
                      title = kwargs["title"] if kwargs["title"] is not None else "",
                      xaxis = dict(
                                  title = x_feature,
                                  autorange = True,
                                  showline = True,
                                  showgrid = True,
                                  gridcolor = None,
                                  showticklabels = True,
                                  zeroline = False,
                                  linecolor = None,
                                  linewidth = 0.5,
                                  ticks = 'outside',
                                  tickfont = dict(
                                                  family = 'Arial',
                                                  color = 'rgb(82,82,82)',
                                                  size = 12
                                                )
                  
                      ),
                      yaxis = dict(
                                  title = y_feature,
                                  autorange = True,
                                  showline = True,
                                  showgrid = True,
                                  showticklabels = True,
                                  gridcolor = None,
                                  zeroline = True,
                                  linecolor = None,
                                  linewidth = 0.5,
                                  ticks = 'outside',
                                  tickfont =dict(
                                                  family = 'Arial',
                                                  color = 'rgb(82,82,82)',
                                                  size = 12
                                                )
                  
                      ),
                      autosize = False,
                      width=1000,
                      height = 600,
                      margin = dict(
                                    autoexpand = True,
                                    l= 80,
                                    r = 90,
                                    t = 60,
                                    b = 60
                                    ),
                      showlegend = True,
                      plot_bgcolor = None ,
                      legend = dict(
                                    bgcolor = 'white',
                                    bordercolor = 'black',
                                    borderwidth = 0.5,
                                    title = dict(
                                                  font = dict(
                                                              family = 'Arial',
                                                              color = 'black',
                                                              size = 16
                                                            ),
                                                  text = kwargs["z_feature"],
                                                  side = 'top'
                                                  ),
                                    font = dict(
                                                  family = 'Arial',
                                                  color = 'rgb(82,82,82)',
                                                  size = 12
                                                )

                                    )
                      )
              
              fig.show()
  def plot_joint_plot(
                      self, 
                      height : int = 10, 
                      x : str = "",
                      y : str = "",
                      kind : str = "scatter",
                      hue : Optional[str] = None,
                      alpha : float = 1,
                      ) -> None:
    """
    Docstring
    Parametros
    ----------
      key word arguments:
        - height : int = 10 | Tamaño de cada figura en 
        - x : str = "" | Nombre variables numericas a plotear en el eje x
        - y : str = "" | Nombre variables numericas a plotear en el eje x
        - kind" : str = "scatter" | posible values: { “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }
        - hue : str = None 
        - alpha : float = 1 | Opacity

    Retorna
    -------
        - None

    """
    if x != "" and y != "":
      sns.jointplot(
                      x = x,
                      y = y,
                      data = self.data,
                      kind = kind,
                      hue = hue,
                      height=height,
                      alpha = alpha,
                    )
    

  def plot_bar_polar(self,**kwargs) -> None:

        """
        Digrama de violin
        Parametros
        ----------
          key word arguments:
            - x_feature : List[str]
            - y_feature : List[str]
            - z_feature : str
            - color_feature : str
                              Nombre columna filtrar por colores
            - title : str

        Retorna
        -------
           - None

        """
        ## Inicializacion del diccionario: kwargs [key-value no definidos como argumento de la funcion]
        # Diccionario para comparar (predefined kw values):
        possible_key_value = {
                              # Key word arguments for scatter_3d method
                              "r_feature " : [],
                              "theta_feature": [],
                              "color_feature": "",
                              "size_feature": "",
                              # Key word arguments for update_layout method
                              "title" : "",
                              "cmap":0,

                              }

        # Comparacion y completado del diccionario: kwargs
        for k,v in possible_key_value.items():
          if k in kwargs:
            pass
          else:
            kwargs[k] = v

        # Creting the scatter 3d plot using express module of plotly
        for _ , r_feature in enumerate(kwargs["r_feature"]):

          for _ , theta_feature in enumerate(kwargs["theta_feature"]):

            if r_feature != theta_feature:

              # Creating the figure with the 3D scatter

              fig = px.scatter_polar(
                                      self.data, 
                                      r=r_feature, 
                                      theta=theta_feature,
                                      color=kwargs["color_feature"], 
                                      size=kwargs["size_feature"],
                                      color_continuous_scale = self.plotly_cmaps[kwargs["cmap"]],
                                      width=1000,
                                      height = 600
                                    )

              # Layout for the figure
              fig.update_layout(
                                    title = kwargs["title"] if kwargs["title"] is not None else "",
                                    autosize = False,
                                    margin = dict(
                                                  autoexpand = True,
                                                  l= 80,
                                                  r = 90,
                                                  t = 60,
                                                  b = 60
                                                  ),
                                    showlegend = True,
                                    plot_bgcolor = None ,
                                    legend = dict(
                                                  bgcolor = 'white',
                                                  bordercolor = 'black',
                                                  borderwidth = 0.5,
                                                  title = dict(
                                                                font = dict(
                                                                            family = 'Arial',
                                                                            color = 'black',
                                                                            size = 16
                                                                          ),
                                                                text = kwargs["color_feature"] if kwargs["color_feature"] is not None else "",
                                                                side = 'top'
                                                                ),
                                                  font = dict(
                                                                family = 'Arial',
                                                                color = 'rgb(82,82,82)',
                                                                size = 12
                                                              )
                                                  )
                                  )
              fig.show()


  def plot_scatter(self,**kwargs) -> None:

          """
          Grafica de puntos. Enfrenta dos variables numericas en cada eje. Realiza el filtrado de cada punto por tamaño y color segun otras dos variables (deben ser numericas)
          que se quieran del dataframe. Es capaz de plotear y localizar posibles outliers segun cierta formula [valor outlier si > (mean +(std*threshold)] o quitarlos de la grafica para mejor visualizacion de la relacion
          entre ambas variables numericas.

          Parametros
          ----------
            key word arguments:
              - fig_x_size : int = 16
                             Tamaño de cada figura en x
              - fig_cols : int = 3
                           Numero de columnas del grid de figura
              - linewidth : float = 0.5
                            El ancho de linea de la figura (el marco de la figura)
              - layout : str = 'constrained'
                         ['constrained','compressed', 'tight', None] Motor de ajuste (algoritmo) de los ejes para evitar solapamiento
              - x :  List[str] = []
                     Lista con las variables numericas a plotear en el eje x
              - y : List[str] =[]
                    Lista con las variables numericas a plotear en el eje y
              - size : Optional[str] = None 
                       Nombre de la variable para agrupar y producir puntos de diferentes tamaños
              - hue : Optional[str] = None  
                      Nombre de la variable de agrupación que producirá puntos con diferentes colores
              - color : int = 0
                        Index del palette en seaborn_palettes
              - plotting_lib : str = 'seaborn'
                               ['seaborn','matplot','plotly'] Nombre de la biblioteca utilizada para graficar
              - umbral : int = 3
                         Umbral para filtrar los outliers
              - method : str | Metodo para el calculo de outliers | "own" [default] , "RIC" 
              - show_outliers : bool = False
                                Flag para mostrar en la grafica los outliers o no. False : no muestra outliers
              - opacity : float = 1
              - plotly_colorscale : int = 0
              - plotly_bgcolor : Optional[str] = None
              - save_figure : Optional[str] = None
                              "jpeg","png","WebP"
              - name_figure : str = 'no_name_figure'
              - title : str


          Retorna
          -------
            - None

          """

          ## Inicializacion del diccionario: kwargs [key-value no definidos como argumento de la funcion]
          # Diccionario para comparar (predefined kw values):
          possible_key_value = {

                                # Pyplot class .figure() method keyword arguments:
                                'fig_x_size': 16, # Tamaño de cada figura en x
                                'fig_cols': 3, # Numero de columnas del grid de figura
                                'linewidth': 0.5,  # El ancho de linea de la figura (el marco de la figura) y de cada axes tambien
                                'layout': 'constrained' , # Motor de ajuste de los ejes para evitar solapamiento

                                # sns class .scatterplot() [seaborn] method keyword arguments and Axes class .scatter() [matplot] method keyword arguments:
                                'x': [], # Lista con las variables numericas a plotear en el eje x
                                'y': [], # Lista con las variables numericas a plotear en el eje y
                                'size' : None, # Grouping variable that will produce points with different sizes
                                'hue': None, # Grouping variable that will produce points with different colors

                                # Own keyword arguments:
                                'color': 0, # index del palette en seaborn_palettes
                                'plotting_lib':'seaborn', # Biblioteca de graficacion
                                'umbral': 3, # Umbral o threshold para outliers (valor limite del outlier es = umbral*(media + desviacion tipica))
                                'method': "own" ,# metodo para el calculo de outliers
                                'show_outliers': False, # True: dibuja y muestra los outliers / False: plotea filtrando los outliers, pero devuelve Dataframe con info de los outliers
                                'opacity': 1,
                                'plotly_colorscale' : 0,
                                'plotly_bgcolor' : None,
                                'save_figure' : None, # ["jpeg","png","WebP"]
                                'name_figure' : 'no_name_figure',
                                'title': ''
                              
                                }

          # Comparacion y completado del diccionario: kwargs
          for k,v in possible_key_value.items():
            if k in kwargs:
              pass
            else:
              kwargs[k] = v

          ## Resto de variables necesarias:
          colors = [
                      "teal", "lawngreen", "turquoise", "r", "darkmagenta", "skyblue", "darkgray",
                      "navy", "forestgreen","b", "g", "r", "c", "m", "y", "k", "firebrick", "darkslategray", "darkorchid", "olive", "dimgray",
                      "royalblue", "mediumseagreen", "indianred", "teal", "purple", "goldenrod", "dimgrey",
                      "cornflowerblue", "limegreen", "darkgoldenrod", "mediumaquamarine", "mediumvioletred", "yellowgreen", "grey",
                      "steelblue", "mediumspringgreen", "brown", "mediumturquoise", "blueviolet", "khaki", "silver",
                      "skyblue", "springgreen", "saddlebrown", "turquoise", "darkviolet", "darkkhaki", "lightgray",
                      "dodgerblue", "chartreuse", "sienna", "paleturquoise", "mediumorchid", "darkolivegreen", "gainsboro",
                      "deepskyblue", "lawngreen", "peru", "lightcyan", "thistle", "darkseagreen"
                    ]

          seaborn_palettes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'rocket', 'mako']

          matplotlib_colormaps = [
                                'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                                'twilight', 'twilight_shifted', 'hsv', 'cubehelix', 'gist_earth',
                                'ocean', 'gist_ncar', 'terrain', 'flag', 'prism',
                                'nipy_spectral', 'jet', 'rainbow', 'turbo', 'mako'
                              ]

          # Manejo de errores en la introduccion de los keyword arguments: las variables hue,size deben ser numericas
          if kwargs["size"] != None and  kwargs["hue"] != None:
            if kwargs["size"] not in self.num_cols or kwargs["hue"] not in self.num_cols:
              print("ERROR: key word argument 'size' or/and 'hue' must be numeric features")

          # Calculo del numero de filas en funcion del numero de columnas definidas para la figura
          if kwargs["x"] != [] and kwargs["y"] != []:

            fig_rows = ( (len(kwargs["x"])*len(kwargs["y"])) // int(kwargs["fig_cols"]) ) + 1  # Asegurar al menos 1 fila
            print("Filas de la figura: ",fig_rows)
            print("Columnas de la figura: ",int(kwargs["fig_cols"]))


            # Calculo de lista con nombre de las columnmas a enfrentar (evitar repeticion)
            plotting_vars = []

            if type(kwargs["x"]) == list and type(kwargs["y"]) == list:

              plotting_vars = (kwargs["x"]+kwargs["y"])
              print("plotting_vars = ",plotting_vars)
            else:
              print('Error : "x" and "y" should be List[str] arguments')

          # Filtramos el tipo de libreria a utilizar
          if kwargs["plotting_lib"] != 'plotly':

            # Obtencion del objeto figure y defincion de sus atributos mediante el metodo de pyplot .figure()
            figure = plt.figure(figsize=(kwargs["fig_x_size"],fig_rows*5), layout = kwargs["layout"])

          # Plotting
          graph_index = 0
          if kwargs["x"] != [] and kwargs["y"] != []:

            for x_fture in kwargs["x"]:

              # Manejo de errores en la introduccion de los keyword arguments: las variables 'x' deben ser numericas
              if x_fture not in self.num_cols:
                print(f"ERROR: key word argument '{x_fture}' must be a numeric feature")
                return None

              for y_fture in kwargs["y"]:

                # Manejo de errores en la introduccion de los keyword arguments: las variables 'y' deben ser numericas
                if y_fture not in self.num_cols:
                  print(f"ERROR: key word argument '{y_fture}' must be a numeric feature")
                  return None

                # Para no graficar en ambos ejes la misma feature
                if x_fture != y_fture:

                  # Internal method to extract outliers and clean dataframe of them
                  df_outliers , out_df_col, df_clean = self._get_outliers(columnas = [x_fture,y_fture], threshold = kwargs["umbral"], method = kwargs["method"])

                  # Filtramos el tipo de libreria a utilizar
                  if kwargs["plotting_lib"] == 'plotly':

                    # Llamada al metodo interno para plotear el grafico de dispersion
                    if kwargs["show_outliers"] == False:
                      self._scatter_plotly( 
                                              x_feature = x_fture , 
                                              y_feature = y_fture, 
                                              title = kwargs["title"], 
                                              color_feature = kwargs["hue"],
                                              size_feature = kwargs["size"], 
                                              opacity = kwargs["opacity"],
                                              colorscale  = kwargs["plotly_colorscale"],
                                              bgcolor =  kwargs["plotly_bgcolor"],
                                              save_figure = kwargs["save_figure"],
                                              name_figure = kwargs["name_figure"]
                                          )
                    else:
                      self._scatter_plotly( 
                                              x_feature = x_fture , 
                                              y_feature = y_fture, 
                                              title = kwargs["title"], 
                                              color_feature = kwargs["hue"],
                                              size_feature = kwargs["size"], 
                                              outlier_df = df_outliers ,
                                              outlier_df_column_name = out_df_col,
                                              opacity = kwargs["opacity"],
                                              colorscale  = kwargs["plotly_colorscale"],
                                              bgcolor =  kwargs["plotly_bgcolor"],
                                              save_figure = kwargs["save_figure"],
                                              name_figure = kwargs["name_figure"]
                          
                                           )

                  if kwargs["plotting_lib"] != 'plotly':
                    
                    graph_index +=1 # Indice de cada axes (grafica tipo scatter)

                    # Creacion objeto axes mediante el metodo de pyplot .subplot()
                    axes = plt.subplot(fig_rows,int(kwargs["fig_cols"]) , graph_index )

                    # Label of axes and tittle
                    axes.set_xlabel(x_fture)
                    axes.set_ylabel(y_fture)

                    if kwargs["show_outliers"] == False:

                      # Plotting sin outliers
                      if kwargs['plotting_lib'] == 'seaborn':

                        axes = sns.scatterplot(
                                                data = df_clean ,  
                                                x= x_fture ,
                                                y = y_fture, 
                                                hue = kwargs["hue"] ,
                                                size = kwargs["size"],
                                                palette = seaborn_palettes[kwargs["color"]], 
                                                legend = 'auto',
                                                #color = colors[kwargs["color"]],
                                                alpha = kwargs["opacity"]
                                                )
                        
                        # Tittle 
                        axes.set_title(f'{kwargs["title"]} -- without outliers')

                      elif kwargs['plotting_lib'] == 'matplot':

                        scatter_ = axes.scatter(
                                                data = df_clean , 
                                                x= x_fture , 
                                                y= y_fture, 
                                                c =  kwargs["hue"],
                                                s =  kwargs["size"] , 
                                                cmap = matplotlib_colormaps[kwargs["color"]],
                                                alpha = kwargs["opacity"]
                                                )
                        
                        # Extract legend handles and labels
                        handles_, labels_ = scatter_.legend_elements()

                        # Add legend to the current axes
                        plt.gca().add_artist(plt.legend(handles_, labels_, title='Class'))  

                        # Tittle 
                        axes.set_title(f'{kwargs["title"]} -- without outliers')

                    else: # Show_outliers == True

                      # Plotting con outliers
                      if kwargs['plotting_lib'] == 'seaborn':

                      
                        # Scatter of samples
                        axes = sns.scatterplot(
                                               data = self.num_data , 
                                               x= x_fture ,
                                               y = y_fture, 
                                               hue = kwargs["hue"] , 
                                               size = kwargs["size"], 
                                               palette = seaborn_palettes[kwargs["color"]] ,
                                               #color = colors[kwargs["color"]], 
                                               legend = 'auto',
                                               ax = axes,
                                               alpha = kwargs["opacity"]
                                               )

                        # Scatter showing outliers
                        axes = sns.scatterplot(
                                                data = df_outliers , 
                                                x= out_df_col[0] ,
                                                y = out_df_col[1], 
                                                label ="Outliers",
                                                s=50, 
                                                color='none', 
                                                edgecolor='black',
                                                linewidths=1.5, 
                                                marker='o', 
                                                ax = axes)
                      
                        # Tittle 
                        axes.set_title(f'{kwargs["title"]} -- with marked outliers')

                      elif kwargs['plotting_lib'] == 'matplot':

                        # Scatter of samples
                        scatter_1 = axes.scatter(
                                                  data = self.num_data, 
                                                  x= x_fture , 
                                                  y= y_fture, 
                                                  c =  kwargs["hue"],
                                                  s =  kwargs["size"] , 
                                                  cmap = matplotlib_colormaps[kwargs["color"]],
                                                  alpha = kwargs["opacity"]
                                                  )

                        # Extract legend handles and labels
                        handles, labels = scatter_1.legend_elements()

                        # Scatter showing outliers
                        axes.scatter(data = df_outliers , x= out_df_col[0] , y= out_df_col[1],label="Outliers",marker='o', facecolors='none', edgecolors='black', s=50)
                        
                        # Legend for the classes 
                        axes.add_artist(axes.legend(handles, labels, title='Class'))  # Add legend to the current axes

                        # Create a legend for the outliers
                        axes.add_artist(plt.legend(loc = 'lower right'))  # Add legend to the current axes 

                        # Tittle 
                        axes.set_title(f'{kwargs["title"]} -- with marked outliers')


                    # Grid on axes
                    axes.grid( linewidth = kwargs["linewidth"], alpha= 0.5)

                    # Ajustar el diseño y mostrar la figura
                    if kwargs["layout"] == None:
                      figure.tight_layout()
                    #figure.show() # En el caso de llamar a este metodo desde un notebook no necesraio ".show()" en el notebook se llama de manera automatica ha dicho metodo

          elif kwargs["x"] == [] or kwargs["y"] == []:

            print("ERROR KEY WORD ARGUMENT: x OR/AND y MUST BE DEFINED")


  # Internal Eda class methods:
  def _get_outliers(self, columnas :List[str], threshold : float = 1, method :str = "own") -> Tuple[pd.DataFrame,List[str],pd.DataFrame]:
    """
    Metodo interno que obtiene los outliers de dos columnas dadas y devuelve un array con en cada fila las coordenadas de ese punto
    Parameters
    ----------
      - columnas : List[str]
      - treshold: positive float number to filter outliers
      - method : str | Metodo para el calculo de outliers | "own" [default] , "RIC" 
    Return
    ------
      - Tuple[pd.DataFrame,List[str],pd.DataFrame]
        One tuple with two dataframes: outliers dataframe and clean dataframe and a list with the names of the outlier df columns
    """
    # Filtering not numeric columns
    columnas = [i for i in columnas if i in self.num_cols]

    # Copy of original data frame
    clean_data = self.num_data.copy()

    # List of outliers arrays
    outliers = []

    # Definition of the name of the outliers dataframe columns 
    df_outliers_column_name = [col+"_outlier" for col in columnas]

    for _ , column_name in enumerate(columnas):

      if method == "own":
        # Valores limitantes superiores e inferiores
        upper_limit = self.num_data[column_name].mean() + (threshold *self.num_data[column_name].std())
        lower_limit = self.num_data[column_name].mean() - (threshold *self.num_data[column_name].std())
      elif method == "RIC":
        # Calculo de priemer, segundo y tercer cuartiles
        Q1,Q2,Q3 = np.quantile(
                                    a = self.num_data[column_name].values,
                                    q = [0.25,0.5,0.75]
                                  )
        RIC = Q3 - Q1 # Rango intercuartilico
        upper_limit = Q3 + (threshold *RIC)
        lower_limit = Q3 - (threshold *RIC)
      else:
        raise NameError(f"name {method} is not a valid argument")
      
      # Valores outliers
      outliers.append(np.array(clean_data[columnas][clean_data[column_name] <= lower_limit].values))
      outliers.append(np.array(clean_data[columnas][clean_data[column_name] >= upper_limit].values))
                                
      # Clean of outliers dataframe
      clean_data = clean_data[clean_data[column_name] < upper_limit]
      clean_data = clean_data[clean_data[column_name]  > lower_limit]
    print(f"Detected outliers in {df_outliers_column_name[0],df_outliers_column_name[1]} : ", pd.DataFrame(data=np.concatenate(outliers), columns = df_outliers_column_name).shape[0])
    return pd.DataFrame(data=np.concatenate(outliers), columns = df_outliers_column_name), df_outliers_column_name, clean_data
  
  # Internal Eda class methods:
  def _scatter_plotly(
                        self, 
                        x_feature :str, 
                        y_feature : str,
                        title : Optional[str] = None,  
                        color_feature : Optional[str] = None,
                        size_feature : Optional[str] = None , 
                        outlier_df : pd.DataFrame = pd.DataFrame(data=[]) ,
                        outlier_df_column_name : Optional[List[str]] = None,
                        opacity : Optional[float] = 1,
                        colorscale : Optional[int] = 1,
                        bgcolor : Optional[int] = None,
                        save_figure : Optional[str] = None,
                        name_figure : Optional[str] = "no_name"

                        ) -> None:
      """
      Metodo interno que que crea graficos de dispersion utilizando libreria plotly, 
      Parameters
      ----------
        -  x_feature : str
                    Nombre de la columna del df a plotear en el eje x
        - y_feature : str
                      Nombre de la columna del df a plotear en el eje y
        - title : Optional[str]
                  titulo de la grafica
        - color_feature : Optional[str]
                          Nombre de la columna del df a utilizar para subdividir cada sample por color
        - size_feature : Optional[str] 
                        Nombre de la columna del df a utilizar para subdividir cada sample por tamaño
        - outlier_df : Optional[pd.DataFrame]
                      df con outliers para graficarlos 
        - outlier_df_column_name : Optional[List[str]] 
                                   nombre de las columnas del df de outliers con el objetivo de indexar
        - opacity: float 
        - colorscale : int
        - bgcolor : str
                    Name for bg color
        - save_figure: Optional[str]
        - name_figure : Optional[str]
      Return
      ------
        - None
      """
      # Libs for saving figures and manage making dirs
      import os
      import plotly.io as pio
      if save_figure !=None:
        # Saving the figure: make the directory to save the images. If it does not exist create it
        if not os.path.exists("images"):
          os.mkdir("images")
      
      # Setup marker types and colors for pyplot
      markers = ['x', 'circle','square', 'triangle-up', 'diamond']
      colors = ['blue', 'red', 'lightgreen','gray', 'cyan']

      # Normalize the size column for plotting betwwen [0,1]* cte values
      if size_feature != None:
        size_feature_norm = 20 * (self.data[size_feature].values-np.min(self.data[size_feature].values))/(np.max(self.data[size_feature].values)-np.min(self.data[size_feature].values))

      # Scatter plots
      scatters = []
      show_title = True
      
      # Filtracion por si color_feature no es discreta o tiene muchas categorias (siendo categorica)
      if self.data[color_feature].nunique() < 11:
        show_title = True
        for idx,target_class in enumerate(self.data[color_feature].unique()):
          scatters.append(
                          go.Scatter(
                                        x = np.array(self.data[x_feature][self.data[color_feature] == target_class]),
                                        y = np.array(self.data[y_feature][self.data[color_feature] == target_class]), 
                                        mode = 'markers',
                                        textfont =   dict(
                                                            family = 'Arial',
                                                            color = 'black',
                                                            size = 12
                                                          ),
                                        name = f'{target_class}',
                                        marker = dict(
                                                      color = colors[idx], 
                                                      symbol = markers[idx],
                                                      size = size_feature_norm if size_feature is not None else 10,
                                                      line =  dict(
                                                                    color = colors[idx], 
                                                                    width = 0.5,
                                                                  ),
                                                      
                                                      opacity = opacity
                                                      ),
                                      )
                            )
      elif self.data[color_feature].nunique() >= 11:
        show_title = False
        scatters.append(
                          go.Scatter(
                                        x = np.array(self.data[x_feature]),
                                        y = np.array(self.data[y_feature]), 
                                        mode = 'markers',
                                        textfont =   dict(
                                                            family = 'Arial',
                                                            color = 'black',
                                                            size = 12
                                                          ),
                                        name = f'Marker size : {size_feature}', # for plotting in the legend the size feature name
                                        marker = dict(
                                                      color = self.data[color_feature].values,
                                                      autocolorscale = False, 
                                                      colorscale = self.plotly_cmaps[colorscale],
                                                      showscale = True,
                                                      colorbar = dict(
                                                                        title=f'{color_feature}', # for plotting in the colorbar the color feature name
                                                                        len = 0.5
                                                                        
                                                                        ),
                                                      size = size_feature_norm if size_feature is not None else 10,
                                                      opacity = opacity
                                                      ),
                                      )
                            )

      # Outliers plot
      if not outlier_df.empty and outlier_df_column_name != None :
        show_title = True
        scatters.append(
                    go.Scatter(
                                  x = np.array(outlier_df[outlier_df_column_name[0]]),
                                  y = np.array(outlier_df[outlier_df_column_name[1]]), 
                                  mode = 'markers',
                                  textfont =   dict(
                                                      family = 'Arial',
                                                      color = 'rgb(82,82,82)',
                                                      size = 12
                                                    ),
                                  name = 'Outlier',
                                  marker = dict(
                                                color = 'black', 
                                                symbol = 'circle-open',
                                                size = 10,
                                                line =  dict(
                                                              color = 'black', 
                                                              width = 1.5,
                                                            ),
                                                opacity = opacity
                                                ),
                                )
                      )
        
      
      # Add data to the figure
      fig = go.Figure(data = scatters)

      # Figure layout 
      fig.update_layout(
                        title = title if title is not None else f"{y_feature} vs. {x_feature}",
                        xaxis = dict(
                                    title = x_feature,
                                    autorange = True,
                                    showline = True,
                                    showgrid = True,
                                    gridcolor =None,
                                    showticklabels = True,
                                    zeroline = False,
                                    linecolor = None,
                                    linewidth = 0.2,
                                    ticks = 'outside',
                                    tickfont = dict(
                                                    family = 'Arial',
                                                    color = 'rgb(82,82,82)',
                                                    size = 12
                                                  )
                    
                        ),
                        yaxis = dict(
                                    title =y_feature,
                                    autorange = True,
                                    showline = True,
                                    showgrid = True,
                                    showticklabels = True,
                                    gridcolor = None,
                                    zeroline = True,
                                    linecolor = None,
                                    linewidth = 0.5,
                                    ticks = 'outside',
                                    tickfont =dict(
                                                    family = 'Arial',
                                                    color = 'rgb(82,82,82)',
                                                    size = 12
                                                  )
                    
                        ),
                        autosize = False,
                        width=1000,
                        height = 600,
                        margin = dict(
                                      autoexpand = True,
                                      l= 70,
                                      r = 120,
                                      t = 60,
                                      b = 60
                                      ),
                        showlegend = True,
                        plot_bgcolor = None if bgcolor is None else bgcolor,
                        legend = dict(
                                      bgcolor = 'white',
                                      bordercolor = 'black',
                                      borderwidth = 0.5,
                                      title = dict(
                                                    font = dict(
                                                                family = 'Arial',
                                                                color = 'black',
                                                                size = 16
                                                              ),
                                                    text = 'Class' if show_title else "",
                                                    side = 'top'
                                                    ),
                                      font = dict(
                                                    family = 'Arial',
                                                    color = 'rgb(82,82,82)',
                                                    size = 12
                                                  )

                                      )

                        )
      #fig.show()
      
      # Saving the figure
      if save_figure != None:
        #fig.savefig(f'images/{name_figure}.{save_figure}', bbox_inches='tight')
        fig.write_image("images/f.png")
        
        
      else:
        fig.show()

  def plot_decision_regions(    self,
                                classifier : object,
                                resolution : float = 0.02,
                                columns_to_pred : List[str] = [],
                                class_legend: str = "",
                                xaxis_title : Optional[str] = None,
                                yaxis_title : Optional[str] = None,
                                title: Optional[str] = None,
                                library :str = "plotly"
                            ) -> None:
      """
      Plot the decision regions computed by a given classifier.

      Parameters
      ----------
          - columns_to_pred: list
              column names of dataframe to be classified.

          - column_target : str
              Target name of dataframe

          - classifier : object
              Classifier already fitted.

          - library: str -> "plotly", "matplot", "seaborn"
                library name plotting

      """

      # Setup marker types and colors for matplot
      colors_matplot = ["b", "r", "g", "c", "m", "y", "k", "firebrick", "darkslategray", "darkorchid", "olive", "dimgray"]
      marker_matplot = ['X','o','v','1','2']
      matplotlib_colormaps = ['viridis', 'coolwarm','plasma', 'inferno', 'magma', 'cividis']

      # Setup marker types and colors for pyplot
      markers = ['x', 'circle','square', 'triangle-up', 'diamond']
      colors = ['blue', 'red', 'lightgreen','gray', 'cyan']

      # Filtracion por columnas solo numericas del dataframe
      columns_to_pred = [col if col in self.num_cols else print(f"{col} is not numeric") for col in columns_to_pred]
      print("columns to predict",columns_to_pred)
      X = self.data[columns_to_pred].values

      # Evitar errores con argumentos de entrada
      if class_legend == "":
        print("Error: you must specify a target column to clasify")
        return None
      elif class_legend in self.num_cols:
          y_pred = self.data[class_legend].values
          print("y_pred.shape : ",y_pred.shape)
      else:
          print(f"Error: {class_legend} not found \n You must specify a valid/existing target column predicted")
          return None

      print("X shape : ", X.shape)

      # Plot the decision surface
      x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
      x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
      xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
      z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
      z = z.reshape(xx1.shape)

      # Unique classes
      unique_y = np.unique(y_pred)

      # Define the colorscale based on unique classes
      colorscale = [[i / (len(unique_y) - 1), color] for i, color in enumerate(colors[:len(unique_y)])]

      # Create a contour plot
      if library == "plotly":
        contour = go.Contour(
                              x=np.arange(x1_min, x1_max, resolution),
                              y=np.arange(x2_min, x2_max, resolution),
                              z=z,
                              showscale=False,
                              colorscale=colorscale,  # Use a custom colorscale
                              opacity=0.5
                            )

        # Create scatter plots for predicted samples
        scatter = []
        for idx, cl in enumerate(np.unique(y_pred)):
            scatter.append(go.Scatter(
                                  x=X[y_pred == cl, 0],
                                  y=X[y_pred == cl, 1],
                                  mode='markers',
                                  marker=dict(color=colors[idx], symbol=markers[idx]),
                                  name=f'Class : {cl}'
                                ))

        # Combine contour and scatter plots
        data = [contour] + scatter

        # Setup the layout
        layout = go.Layout(
                            xaxis=dict(title=xaxis_title if xaxis_title is not None else ""),
                            yaxis=dict(title=yaxis_title if yaxis_title is not None else ""),
                            title=title if title is not None else "",
                            margin=dict(l=40, r=40, b=40, t=40),
                            width=900,
                            height = 700,

                          )

        # Create the figure and plot it
        fig = go.Figure(data=data, layout=layout)
        fig.show()

      if library == "matplot" or library == "seaborn":

        figure= plt.figure(figsize=(10,8), dpi=110)
        axes  = figure.add_subplot(1,1,1)

        # Contourf method (regions are colur)
        contour = axes.contourf(
                                xx1,
                                xx2,
                                z,
                                cmap = matplotlib_colormaps[1],
                                alpha = 0.2
                              )
        # Scatter using matplot
        if library == "matplot":

          for idx, cl in enumerate(np.unique(y_pred)):

            scatter = axes.scatter(
                                  x = X[y_pred == cl,0],
                                  y = X[y_pred == cl,1],
                                  c = colors_matplot[idx],
                                  marker = marker_matplot[idx],
                                  s=50
                                  )

        # Scatter using seaborn
        if library == "seaborn":

          sns.scatterplot(
                              data = pd.DataFrame(X),
                              x = X[:,0],
                              y = X[:,1],
                              hue = y_pred,
                              style= y_pred,
                              s = 50,
                              palette = matplotlib_colormaps[1],
                              markers = marker_matplot,
                              ax = axes,
                              legend = 'auto'
                          )
        axes.set(
                      xlim=(x1_min, x1_max),
                      ylim=(x2_min, x2_max),
                      xlabel= xaxis_title if xaxis_title is not None else 'X',
                      ylabel= yaxis_title if yaxis_title is not None else 'Y',
                      title = title if title is not None else "Classifier contour plot",

              )
        axes.grid(linewidth = 0.5 , alpha=1)
        fig.show()

      elif library != "matplot" and library != "seaborn" and library != "plotly" :
        print(f"Error: {library} is not a valid library name ")




##################################################################################################################
########################## Customize functions for Eda an danalisis of predictions of models #####################
##################################################################################################################
        
def plt_decison_boundaries (
                                y_test : np.array ,
                                x_test : np.array ,
                                y_train : np.array ,
                                x_train : np.array ,
                                classifier : object,
                                resolution : float = 0.02,
                                xaxis_title : Optional[str] = None,
                                yaxis_title : Optional[str] = None,
                                title: Optional[str] = None,
                            ) -> None:
      """
      Plot the decision regions computed by a given classifier.
      Function that fits a clasifier o train data, predict the test data and plot de decision boundaries distingin into test and train data

      Parameters
      ----------

          - classifier : object
                         Classifier without been fitted
          - x_test, x_train, y_test, y_train : np.arrays with the data
          - resolution: float for the meshgrid
          - xaxis_title, yaxis_title, title: str with the tittles for the axis

      Return
      ------
           - None

      """

      # Setup marker types and colors for pyplot
      markers = ['x', 'circle','square', 'triangle-up', 'diamond']
      colors = ['blue', 'red', 'lightgreen','gray', 'cyan']
      
      # Check if was already fitted
      try: 
          classifier.predict(np.array(x_train)) # check if it was fitted using predict method
          print(f'Error for {classifier.__class__.__name__} : model already fitted')
          return None
        
      except NotFittedError as e:
          try:
              print(f'Note for {classifier.__class__.__name__} : {e}')
          except :
              print(e)
              
          print_loading_bar(iterations = 11,message= f'Training {classifier.__class__.__name__} model', efect_time = 0.5)
          # fit the clasifier on the x_train and y_train data
          classifier.fit(X = x_train, y = y_train)
              

      # Complete x array set to create the meshgrid
      X = np.concatenate([x_train, x_test])

      # Plot the decision surface
      x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
      x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
      xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))

      # Predict of all points at meshgrid
      z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
      z = z.reshape(xx1.shape)

      # Predict of test set
      y_pred = classifier.predict(x_test)

      # Create "y" vectors with real anda predicted targets
      y_real_pred = np.concatenate([y_train, y_pred])
      y_real =  np.concatenate([y_train, y_test])
      print("\n")
      print("X shape : ", X.shape)
      print("y_real_pred shape : ", y_real_pred.shape)
      print("y_real shape : ", y_real.shape)

      # Unique classes
      unique_y = np.unique(y_real)

      # Define the colorscale based on unique classes
      colorscale = [[i / (len(unique_y) - 1), color] for i, color in enumerate(colors[:len(unique_y)])]

      # Create a contour plot
      contour = go.Contour(
                              x=np.arange(x1_min, x1_max, resolution),
                              y=np.arange(x2_min, x2_max, resolution),
                              z=z,
                              showscale=False,
                              colorscale = colorscale,  # Use a custom colorscale
                              opacity=0.3
                            )

      # Create scatter plots showing real class for each sample
      scatter = []
      # y real scatter
      for idx, cl in enumerate(np.unique(y_real)):
          scatter.append(go.Scatter(
                                x=X[y_real == cl, 0],
                                y=X[y_real == cl, 1],
                                mode='markers',
                                marker=dict(color = colors[idx], symbol = markers[idx]),
                                name=f'Real class : {cl}'
                              ))
      # Show the test sample set
      scatter.append(go.Scatter(
                                  x=x_test[:, 0],
                                  y=x_test[:, 1],
                                  mode='markers',
                                  marker=dict(
                                                color='rgba(135, 206, 250, 0.0)',
                                                size=10,
                                                line=dict(
                                                          color='black',
                                                          width=2,
                                                      ),
                                               ),
                                  name='Test set'
                                )
                      )

      # Combine contour and scatter plots
      data = [contour] + scatter

      # Setup the layout
      layout = go.Layout(
                          xaxis=dict(title=xaxis_title if xaxis_title is not None else ""),
                          yaxis=dict(title=yaxis_title if yaxis_title is not None else ""),
                          title=title if title is not None else "",
                          margin=dict(l=40, r=40, b=40, t=40),
                          width=900,
                          height = 700,

                        )

      # Create the figure and plot it
      fig = go.Figure(data=data, layout=layout)
      fig.show()

def plot_surface(
                      z : np.ndarray,
                      dy : float = 0.1,
                      dx : float = 0.1,
                      x_range : Tuple[int,int] = (-10,10),
                      y_range : Tuple[int,int] = (-10,10),
                      xaxis_title : Optional[str] = None,
                      yaxis_title : Optional[str] = None,
                      title: Optional[str] = None,
                    ) -> None:
    """
    Ploting of a surface according to a mathematical functionwith two independent variables x and y 

    Parameters
    ----------
        - z : str
              explicit mathematical expression of the function, must be str data type
        - dx,dy: float 
                 Steps for grid in the x and y axis
        - x_range,y_range: int 
                        Min and Max values for the x and y axis
        - xaxis_title, yaxis_title, title: str 

    Return
    ------
        - None

    """
    # Creating x and y arrays (grid for evaluating function "z")
    x_values = np.arange(x_range[0],x_range[1],dx)
    y_values = np.arange(y_range[0],y_range[1],dy)

    fig = go.Figure(data=[go.Surface(x=x_values,y = y_values,z=z)])

    # Projection of the contours in the z direction using a heatmap
     
    fig.update_traces(
                        contours_z=dict(
                                          show = True, 
                                          usecolormap = True,
                                          highlightcolor = "limegreen", 
                                          project_z = True
                                                      
                                          )
                        )
  
     # Figure layout
    fig.update_layout(
                      title= title if title is not None else '',
                      scene_camera_eye = dict(x=1.87, y=0.88, z=-0.64),
                      xaxis = dict(
                                  title = xaxis_title,
                                  autorange = True,
                                  showline = True,
                                  showgrid = True,
                                  gridcolor = None,
                                  showticklabels = True,
                                  zeroline = False,
                                  linecolor = None,
                                  linewidth = 0.5,
                                  ticks = 'outside',
                                  tickfont = dict(
                                                  family = 'Arial',
                                                  color = 'rgb(82,82,82)',
                                                  size = 12
                                                )
                  
                      ),
                      yaxis = dict(
                                  title =yaxis_title,
                                  autorange = True,
                                  showline = True,
                                  showgrid = True,
                                  showticklabels = True,
                                  gridcolor = None,
                                  zeroline = True,
                                  linecolor = None,
                                  linewidth = 0.5,
                                  ticks = 'outside',
                                  tickfont =dict(
                                                  family = 'Arial',
                                                  color = 'rgb(82,82,82)',
                                                  size = 12
                                                )
                  
                      ),
                      autosize = False,
                      width=1000,
                      height = 600,
                      margin = dict(
                                    autoexpand = True,
                                    l= 70,
                                    r = 120,
                                    t = 60,
                                    b = 60
                                    ),
                      showlegend = True,
                      plot_bgcolor = None,
                      legend = dict(
                                    bgcolor = 'white',
                                    bordercolor = 'black',
                                    borderwidth = 0.5,
                                    title = dict(
                                                  font = dict(
                                                              family = 'Arial',
                                                              color = 'black',
                                                              size = 16
                                                            ),
                                                  text = 'Z',
                                                  side = 'top'
                                                  ),
                                    font = dict(
                                                  family = 'Arial',
                                                  color = 'rgb(82,82,82)',
                                                  size = 12
                                                )

                                    )
                      )
    
    fig.show()


def visual_decision_tree(class_names : List[str] , feature_names : List[str] , tree , relative_path : str, tree_name : str) -> None:

  """
  Visual plot of a decision tree predictor clasifier object (must be fitted) 
  Parameters
  ----------
  ...
  Return
  ------
  None
  """
  # Libraries import 
  from pydotplus import graph_from_dot_data
  from sklearn.tree import export_graphviz
  import os

  # Saving the figure: make the directory to save the images. If it does not exist create it
  if not os.path.exists(f"images"):
    os.mkdir("images")
  if not os.path.exists(f"images\\{relative_path}"):
    os.mkdir(f"images\\{relative_path}")
  

  dot_data = export_graphviz(
                            tree,
                            filled = True,
                            rounded = True,
                            class_names = class_names,
                            feature_names = feature_names,
                            out_file=None)
  graph = graph_from_dot_data(dot_data)
  graph.write_png(f'images\\{relative_path}\\{tree_name}.png')
