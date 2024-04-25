import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import  Callable
from itertools import product
from sklearn.base import BaseEstimator, TransformerMixin,OneToOneFeatureMixin
from sklearn.preprocessing import FunctionTransformer
from Paquetes.optimization import execution_time
from typing import  Callable, Optional, List, Dict, Tuple

##########################################################################################
############################## CUSTOM TRANSFORMERS  ######################################
##########################################################################################

### TRATAMIENTO DE FECHAS ----------------------------------------------------------------
@execution_time
def combine_date_columns(df: pd.DataFrame = None) -> pd.DataFrame:

  """
  Docstring ...

  Parametros
  ----------
    - df: (pd.DataFrame) ...

  Retorna
  -------
    pd.DataFrame

  """
  new_df = pd.DataFrame(df)
  print(f"Df input type en {combine_date_columns.__name__} : ",type(df))

  # Combina las columnas de fecha en una nueva columna 'compact_date'
  new_df['compact_date'] = pd.to_datetime(
                                            new_df['arrival_date_year'].astype(str) + '-' +
                                            new_df['arrival_date_month'].astype(str) + '-' +
                                            new_df['arrival_date_day_of_month'].astype(str),
                                            errors='ignore'
                                          )  # Configura errors='coerce' para manejar fechas incorrectas, que no deberian haber

  new_df = new_df.sort_values('compact_date')

  # new_df["reservation_status_date"] = pd.to_datetime(df['reservation_status_date'], format = "%Y-%m-%d" ,  errors='raise')

  print("Viejas columnas del df antes de la transformacion: \n ", new_df.select_dtypes(["object","int64","datetime64[ns]"]).columns)
  print("Numero de viejas columnas del df antes de la transformacion: ",len(new_df.select_dtypes(["object","int64","datetime64[ns]"]).columns))

  new_df.drop(labels = ['compact_date'] , axis = 1, inplace = True)

  print("Nuevas columnas del df despues de la transformacion: \n ", new_df.select_dtypes(["object","int64","datetime64[ns]"]).columns)
  print("Numero de nuevas columnas del df despues de la transformacion: ",len(new_df.select_dtypes(["object","int64","datetime64[ns]"]).columns))

  # print("Data shape after date transformation : ", new_df.shape)
  # print("New datetime columns after date transformation : ", new_df.select_dtypes(["datetime64[ns]"]).columns)
  # for col in new_df.columns:
  #   print(f"Hay en {col} NA  = {new_df[f'{col}'] .isnull().sum()} ")
  # print(f"Termina {combine_date_columns.__name__} ")

  print("-----------------------------------------------------------------")


  return new_df



### TRATAMIENTO DE OUTLIERS ----------------------------------------------------------------------------
@execution_time
def tratamiento_de_outliers(df: pd.DataFrame = None, columnas_numericas: list[str] = []) -> pd.DataFrame:

  """
  Dado un Dataset con sus correspondientes columnas numericas y bajo la tecnica de Winzorizacion se tratan los datos.
  Basicamente se sustituyen los valores mas atipicos por los del percentil 95 o 5 mas cercano

  Parámetros
  ----------
    - df: Dataset a modificar.
    - columnasNumericas: Lista de columnas numericas.

  Retorna
  -------
    - df_winsorized : (pd.DataFrame)
  """
  from scipy.stats.mstats import winsorize

  dfWinsorized = df.copy()
  #print(f"Df input type en {tratamiento_de_outliers.__name__} : ",type(df))
  print(f"Columnas a tratar outliers en {tratamiento_de_outliers.__name__} : " , columnas_numericas)

  # Recorre una a una las columnas numericas
  for columna in columnas_numericas:

      #print(f"Nombre de columna a tratar por sus outliers : {columna}")

      # Recorro cada columna del dataset
      quantil95 = df[columna].quantile(0.95)
      quantil05 = df[columna].quantile(0.05)

      #print('quantil95: ' + str(quantil95))
      #print('quantil05: ' + str(quantil05))

      # Obtener el IQR
      iqr = quantil95 - quantil05

      # Percentil superior e inferior
      superior = quantil95 + (1.5 * iqr)
      inferior = quantil05 - (1.5 * iqr)

      # Obtengo los outliers iniciales
      outliers = df[(df[columna] < inferior) | (df[columna] > superior)]
      outliers.head()
      #print(f'Numero de outliers previo: {len(outliers)}')

      dfWinsorized[columna] = winsorize(dfWinsorized[columna], limits = [0.05, 0.05], inplace = False)

      # Obtengo los outliers finales
      outliers = dfWinsorized[(dfWinsorized[columna] < inferior) | (dfWinsorized[columna] > superior)]
      outliers.head()
      #print(f'Numero de outliers finales: {len(outliers)}')
  #print(f"Cantidad de filas {tratamiento_de_outliers.__name__} " , dfWinsorized.shape)
  print(f"Termina {tratamiento_de_outliers.__name__} ")

  print("-----------------------------------------------------------------")

  return dfWinsorized

@execution_time
def tratamiento_outliers_2(df: pd.DataFrame = pd.DataFrame() , columnas_numericas: list[str] = []) -> pd.DataFrame:
    """
    Dado un Dataset con sus correspondientes columnas numericas elimina los outliers segun la formula especificada

    Parámetros
    ----------
    - df: Dataset a modificar.
    - columnasNumericas: Lista de columnas numericas que se quireen filtrar sus outliers.

    Retorna
    -------
    - df_aux : (pd.DataFrame)
    """

    # Manejo de errores
    print(f"Df input type in {tratamiento_outliers_2.__name__} :", type(df))

    if df.empty:
        raise ValueError(f"EMPTY DATAFRAME PASSED AS ARGUMENT TO THE FUNCTION: {tratamiento_outliers_2.__name__}")

    if len(columnas_numericas) == 0:
        print(f"NUMERIC COLUMN TO TRANSFORM SHOULD BE PASSED AS ARGUMENT TO THE FUNCTION: {tratamiento_outliers_2.__name__}")
        return df



    df_aux = pd.DataFrame(df)
    # umbral de filtrado [para considerar si es o no un outlier]
    threshold = 2

    for col_index , col in enumerate(list(columnas_numericas)):

        # Calculo de media, std de cada columna
        x_media = df_aux[f"{col}"].mean()
        x_std = df_aux[f"{col}"].std()

        # Mask
        df_aux = df_aux[df_aux[f"{col}"] < (x_media + (threshold*x_std))] # Primer filtrado (valores x) de filas con valores outliers
        df_aux = df_aux[df_aux[f"{col}"] > (x_media - (threshold*x_std))]
        print("df_aux", df_aux)

    print(f"df_aux shape en  {tratamiento_outliers_2.__name__} :", df_aux.shape)
    print(f"Termina {tratamiento_outliers_2.__name__} ")
    print("-----------------------------------------------------------------")

    return df_aux


### Funcion utilizad con .apply() de pandas en otrso transformers
# Funcion a aplicar a cada componente de una columna
def digit_numeric(x = None, categoriasTop: list = [] ) -> str:
  """
  Funcion que detecta si el str de input son solo digitos o solo letras y devuelve la nueva categoria que se aplica
  Parameters
  ----------
    - x (str)
    - categoriasTop
  Return
  ------
      - (str)
  """
  # Convertir elementos lista a str
  categoriasTop_str = [str(element) for element in categoriasTop]

  if type(x) == float or type(x) == int:
      x = str(x)
      if x not in categoriasTop_str:
          # Comprueba si es solo si primer caracter es un digito o una letra por error de comas
          if x[0].isdigit():
              return '999'
          elif x[0].isalpha():
              return 'other'
      return x
  else:
      if x not in categoriasTop_str:
          if x[0].isdigit():
              return '999'
          elif x[0].isalpha():
              return 'other'
      else:
          return str(x)

### Funcion utilizad con .apply() de pandas en otrso transformers
def cat_imputer( x = None, categoriasTop: list = [], ncat = None) -> str:
  """
  Funcion que detecta si el str de input son solo digitos o solo letras y devuelve la nueva categoria que se aplica
  Parameters
  ----------
    - x (str)
    - categoriasTop
    - ncat
  Return
  ------
      - (str)
  """
  # Convertir elementos lista a str
  categoriasTop_str = [str(element) for element in categoriasTop]

  if type(x) == float or type(x) == int:
      x_old = x
      x_str = str(x)
      if x_str not in categoriasTop_str:
        return ncat
      else:
        return x_old
  else:
      if x not in categoriasTop_str:
         return str(ncat)
      else:
          return str(x)

### TRATAMIENTO DE FEATURES CON UN EXCESIVO NUMERO DE CATEGORIAS (FEATURE ENGINEERING)
@execution_time
def reducir_categorias_columna(df: pd.DataFrame = None, columna : list[str] = [], nueva_cat : list[str] = [] ,cantCategorias : int =  10) -> pd.DataFrame:

  """
  Dado una columna de un Dataset genera otra nueva columna con la cantidad de categorias que se quiera.

  Parámetros
  ----------
    - df : ( pd.DataFrame) Dataset a modificar.
    - columna : (list) Columna a modificar.
    - nueva_cat: (list)  Nombre de la nueva categoria a generar
    - cantCategorias : (int) Cantidad de nuevas categorias a generar , todas las categorias restantes quedaran bajo la columna 'Otros'

  Retorna
  -------
    - new_df ( pd.DataFrame): ...

  """

  new_df = pd.DataFrame(df)
  print(f"Df input type en {reducir_categorias_columna.__name__} : ",type(df))
  print(f"Columnas a reducir categorias en {reducir_categorias_columna.__name__} : " , columna)

  for col , cat in zip(columna,nueva_cat):
    print(f"columna reduce num categorias a  : {cantCategorias} :",col)
    print("Nombre nueva cat : ",cat)

    # Obtener las categorias mas usuales
    categoriasTop = new_df[col].value_counts().nlargest(cantCategorias).index


    # Crear una nueva columna con las categorias que se quiere y categoriza en 'Otros' al resto
    new_df[f"{col}"] = new_df[f"{col}"].apply(cat_imputer, categoriasTop = categoriasTop, ncat = cat)

    #new_df['columnaAux'] = new_df[col].apply(lambda x: str(x) if x in categoriasTop else 'other')

    print(f"Nuevas columnas en {col} : ",new_df[str(str(col))].value_counts().index)

  print(f"Columnas categoricas del output new_df en {reducir_categorias_columna.__name__} : ", new_df.select_dtypes(["object"]).columns)
  print(f"Tipo del new_df output en {reducir_categorias_columna.__name__}: ", type(new_df))
  for col in new_df.columns:
    print(f"Hay en {col} NA  = {new_df[f'{col}'] .isnull().sum()} ")
  print(f"Termina {reducir_categorias_columna.__name__} ")
  print("-----------------------------------------------------------------")

  return new_df

@execution_time
def reducir_categorias_columna_2(df: pd.DataFrame = None, columna : list[str] = [], nueva_cat : list[str] = [] ,cantCategorias : int =  10) -> pd.DataFrame:

  """
  Dado una columna de un Dataset genera otra nueva columna con la cantidad de categorias que se quiera.

  Parámetros
  ----------
    - df : ( pd.DataFrame) Dataset a modificar.
    - columna : (list) Columna a modificar.
    - nueva_cat: (list)  Nombre de la nueva categoria a generar
    - cantCategorias : (int) Cantidad de nuevas categorias a generar , todas las categorias restantes quedaran bajo la columna 'Otros'

  Retorna
  -------
    - new_df ( pd.DataFrame): ...

  """

  new_df = pd.DataFrame(df)
  print(f"Df input type en {reducir_categorias_columna.__name__} : ",type(df))
  print(f"Columnas a reducir categorias en {reducir_categorias_columna.__name__} : " , columna)

  for col , cat in zip(columna,nueva_cat):
    print(f"columna reduce num categorias a  : {cantCategorias} :",col)
    print("Nombre nueva cat : ",cat)

    # Obtener las categorias mas usuales
    categoriasTop = df[str(col)].value_counts().index[0:cantCategorias]


    # Crear una nueva columna con las categorias que se quiere y categoriza en 'Otros' al resto
    new_df[f"{col}"] = new_df[f"{col}"].apply(cat_imputer, categoriasTop = categoriasTop, ncat = cat)

    #new_df['columnaAux'] = new_df[col].apply(lambda x: str(x) if x in categoriasTop else 'other')

    print(f"Nuevas columnas en {col} : ",new_df[str(str(col))].value_counts().index)

  print(f"Columnas categoricas del output new_df en {reducir_categorias_columna.__name__} : ", new_df.select_dtypes(["object"]).columns)
  print(f"Tipo del new_df output en {reducir_categorias_columna.__name__}: ", type(new_df))
  for col in new_df.columns:
    print(f"Hay en {col} NA  = {new_df[f'{col}'] .isnull().sum()} ")
  print(f"Termina {reducir_categorias_columna.__name__} ")
  print("-----------------------------------------------------------------")

  return new_df


### CONVERSION DE COLUMNAS NUMERICAS EN CATEGORICAS ------------------------------
def convertir_columna_categorica(df: pd.DataFrame = None, columna : list[str] = []) -> pd.DataFrame:
  """
  Dado una columna de un Dataset la convierte en categorica.

  Parámetros
  ----------
    - df : ( pd.DataFrame) Dataset a modificar.
    - columna : (list[str]) Columnas a modificar.

  Retorna
  -------
    - new_df ( pd.DataFrame): ...

  """

  new_df = pd.DataFrame(df)
  print(f"Df input type en {convertir_columna_categorica.__name__} : ",type(df))
  print(f"Columnas ea convertir a categoricas en {convertir_columna_categorica.__name__} : " , columna)

  for col in columna:
    # Convierte la columna a categorica
    new_df[col] = new_df[col].astype(dtype = 'object', errors = 'raise')
    print(f'Tipo de datos de la columna {col} tras cambiar el tipo de datos a object : {new_df[col].dtypes}')

  print(f"Df output type en {convertir_columna_categorica.__name__} : ", type(new_df))
  for col in new_df.columns:
    print(f"Hay en {col} NA  = {new_df[f'{col}'] .isnull().sum()} ")

  print(f"Termina {convertir_columna_categorica.__name__} ")
  print("-----------------------------------------------------------------")


  return new_df




### TRATAMIENTO DE VALORES NA -------------------------------------------------
def tratamiento_datos_faltantes(df: pd.DataFrame = None) -> pd.DataFrame:

  """
  ...

  Parámetros
  ----------
    - df : ( pd.DataFrame) Dataset a modificar.

  Retorna
  -------
    - new_df ( pd.DataFrame): ...
  """
  new_df = pd.DataFrame(df)
  print(f"Df input type en {tratamiento_datos_faltantes.__name__} : ",type(df))

  for columna in df.columns:

    # Deteccion de existencia de valores NA:
    if df[f"{columna}"].isnull().sum() != 0:

      # Filtramos por tipo de columna si numerica o categorica o bool o datetime
      if columna in list(df.select_dtypes(["object"]).columns):

        print(f"columna {columna} es categorica ")
        new_df[columna] = df[columna].fillna(str(999), inplace = False) # Igual a tratar a los datos vacios como una categoria nueva
        # new_df[columna].astype(str)
        print(f'Tipo de datos de la columna {columna} tras rellenar NA values es : {new_df[columna].dtypes}')

      elif columna in list(df.select_dtypes(["int64","float64"]).columns):

        print(f"columna {columna} es numerica ")
        new_df[columna] = df[columna].fillna(int(999), inplace = False) # Igual a tratar a los datos vacios como una categoria nueva
        print(f'Tipo de datos de la columna {columna} tras rellenar NA values es : {new_df[columna].dtypes}')

      elif columna in list(df.select_dtypes(["boolean"]).columns):
        print(f"columna {columna} es boleana ")
        new_df[columna] = df[columna].fillna(False, inplace = False) # Igual a tratar a los datos vacios como una categoria nueva
        print(f'Tipo de datos de la columna {columna} tras rellenar NA values es : {new_df[columna].dtypes}')

      elif columna in list(df.select_dtypes(["datetime64"]).columns):
        print(f"columna {columna} es datetime64, no se ha implementado metodo para dropear NA en este tipo de columnas ")
        pass

      else:
        print(f"No se reconoce el tipo de dato de la columna {columna}")

  print(f"Df output type en {tratamiento_datos_faltantes.__name__} : ", type(new_df))
  print(f"Termina {tratamiento_datos_faltantes.__name__} ")
  print("-----------------------------------------------------------------")


  return new_df


### TRATAMIENTO DE VALORES NA MEDIANTE UN CUSTOM DROPPER TRANSFORMER CLASS ---------------
class CustomDropper(BaseEstimator, TransformerMixin):
  """
    Clase customizada de tipo transformer (clase de sklearn para transformar columnas del dataframe).
    Hereda de las clases:
    - BaseEstimator:
    - TransformerMixin: Clase incorpora el metodos fit_transform y set_output

    Atributos
    ---------
      - self.column_transform : (list[str]) Lista con los nombres de las columnas que se quieren dropear


    Propiedades (decorador @property)
    ---------------------------------
      None

    Metodos
    -------
      - __init__ [built-in method]
      - fit
      - transform


  """
  def __init__(self, column_transform :list[str] = [str]) -> None:
    """
    Metodo que inicializa la clase.

    Parametros de la funcion (se pasan como argumentos en la instancia de la clase)
      - column_transform: (list[str]) or str) (el tipo de dato del argumento se cambia internamente como parametro de la funcion
        osea, como argumento puede ser str o list[str]; internamente siempre se convierte a list).
        Lista con los nombres de las columnas que se quieren codificar [deben ser categoricas]


    Retorna:
      None

    """
    self.column_transform = column_transform

  def fit(self, X, y=None):
      return self

  def transform(self, X, y=None) -> pd.DataFrame:
    """
    Funcion que dropea las columnas especificadas

    Parametros:
      - X: df a transformar
      - y

    Retorna:
      X (pd.DataFrame)

    """
    x_aux = X.copy()

    if len(self.column_transform) == 0:
        print(f"COLUMNS TO DROP SHOULD BE SPECIFIED")
        return x_aux

    if len(self.column_transform) != 0:

      for col in self.column_transform:
          x_aux.drop(f"{col}",axis =1, inplace =True)

      return x_aux


### ORDINAL ENCODER TRANSFORMER CUSTOMIZE CLASS -------------------------------------------------
class LabelCustomEnc(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
  """
  Clase customizada de tipo transformer (clase de sklearn para transformar columnas del dataframe).
  Hereda de las clases:
  - BaseEstimator:
  - TransformerMixin: Clase incorpora el metodos fit_transform y set_output
  - OneToOneFeatureMixin: Clase incorpora el metodo get_feature_names_out que permite obtener las features out

  Atributos
  ---------
    - column_transform: (list[str]) or str) nombre de las columnas del df a transformar

  Propiedades (decorador @property)
  ---------------------------------
    None

  Metodos
  -------
    - __init__ [built-in method]. Constructor
    - fit
    - transform

  """

  def __init__(self, column_transform :list[str] = []) -> None:

    """
    Metodo (built-in) que inicializa la clase. Constructor

    Parametros de la funcion (se pasan como argumentos en la instancia de la clase)
    ----------
      - column_transform: (list[str]) or str) (el tipo de dato del argumento se cambia internamente como parametro de la funcion
        osea, como argumento puede ser str o list[str]; internamente siempre se convierte a list).
        Lista con los nombres de las columnas que se quieren codificar [deben ser categoricas]

    Retorna
    -------
      None

    """
    self.column_transform = column_transform

  def fit(self, X, y=None):

    return self

  def transform(self, X, y=None) -> pd.DataFrame:
    """
    Funcion que codifica mediante numero enteros las categorias de las columnas pasadas como argumento
    al crear el objeto LabelCustomEnc. La categoria mas frecuente en la columna tendra asociado numero entero mayor
    e igual al numero de categorias en dicha columna. Y la categoria menos frecuente el numero 0.

    Parametros
    ----------
      - X: columnas del df a transformar
      - y

    Retorna
    -------
      X (pd.DataFrame)

    """
    self.column_transform = list(self.column_transform)
    for col in self.column_transform:
      cat_list = list(X[f"{col}"].value_counts().index)
      num_int_cod = np.arange(len(cat_list)-1,-1,-1) # Invierte el orden vector creado para que el mayor entero se asocie a la categoria mas frecuente
                                                      # (porque value_counts() devuelve como primer indice de la serie la mas frecuente y
                                                      # y al iterar (por logica) queremnos que se asocie/cambie el mayor entero por la categoria +freq
      for num,cat in zip(num_int_cod,cat_list):
        X[f"{col}"].replace(f'{cat}', num,inplace=True)


    return X
    

class CatTransf(BaseEstimator, TransformerMixin):
  """
  Clase customizada de tipo transformer (clase de sklearn para transformar columnas del dataframe).
  Hereda de las clases:
  - BaseEstimator:
  - TransformerMixin: Clase incorpora el metodos fit_transform y set_output

  Atributos
  ---------
    - compare_column: (list[str]) or str) Nombre de la columna con la que se quiere comparar
    - fill_columns : (list[str]) or str) Lista con los nombres de las columnas que se quieren rellenar
    - target_name : (str o np.nan / pd.NA) Nombre de la categoria en las columnas fill_columns que se quiere cambiar comparando con el valor en compare_column

  Propiedades (decorador @property)
  ---------------------------------
    None

  Metodos
  -------
    - __init__ [built-in method]
    - fit
    - transform
    """
  def __init__(self , compare_column :list[str] = [], fill_columns :list[str] = []  ,target_name = np.nan) -> None:
      
    self.compare_column = compare_column
    self.fill_columns = fill_columns
    self.target_name = target_name

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None) -> pd.DataFrame:
    """
    Docstring

    Parametros
    ----------
      - X: columnas del df a transformar
      - y

    Retorna
    -------
      - X (pd.DataFrame)

    """
    compare_name = ""
    rows,columns = X.shape
    for col in self.fill_columns:

      df_aux = X.loc[X[f"{col}"] == self.target_name]
      compare_cat = df_aux [f"{self.compare_column[0]}"].values
      compare_index =  df_aux [f"{self.compare_column[0]}"].index

      for cat,index in zip(compare_cat,compare_index):

        df_mask = X[(X[f"{self.compare_column[0]}"] == cat) & (X[f"{col}"] != self.target_name)]
        if df_mask[f"{col}"].iloc[0] != None:
          X.at[index,f"{col}"] =  df_mask[f"{col}"].iloc[0]
    return X

class CatImpMFrequent(BaseEstimator, TransformerMixin):
  """
  Clase customizada de tipo transformer (clase de sklearn para transformar columnas del dataframe).
  Hereda de las clases:
  - BaseEstimator:
  - TransformerMixin: Clase incorpora el metodos fit_transform y set_output

  Atributos
  ---------
    - column_transform: (list[str]) or str) Lista con los nombres de las columnas que se quieren rellenar
    - replace_values : list[str o np.nan / pd.NA]) list --- Nombre de la categoria que se quiere cambiar por las mas frecuente de la columna

  Propiedades (decorador @property)
  ---------------------------------
    None

  Metodos
  -------
    - __init__ [built-in method]
    - fit
    - transform
  """
  def __init__(self, column_transform :list[str] =  [], replace_values : list[str] =  []) -> None:
      
      self.replace_values = replace_values
      self.column_transform = column_transform

  def fit(self, X, y=None):
      return self

  def transform(self, X, y=None) -> pd.DataFrame:
    """
    Docstring

    Parametros
    ----------
      - X: columnas del df a transformar
      - y

    Retorna
    -------
      - X (pd.DataFrame)

    """
    if self.column_transform != []:
      for col in self.column_transform:
        mas_freq = str(X[f"{col}"].value_counts().index[0])
        if self.replace_values != []:
          for r in self.replace_values:
            mascara = X[f"{col}"] == r
            X.loc[mascara, f"{col}"]= mas_freq
    return X

## Custom categorical transformer class:
## Input parameters:
#__________________________________________________________________________________
# column_transform --- str list --- Lista con los nombres de las columnas que se quieren rellenar
# replace_values --- tuple list --- lista de tuplas con categorias que se quieren sustituir por el valor deseado
# new_value --- str --- Nuevo nombre de la categoria
# nota: cada valor en la posicion lista: column_transform va a asociado al mismo inidice/posicion en las listas: new_value y replace_values
#__________________________________________________________________________________
class CatChanger(BaseEstimator, TransformerMixin):
  """
  Clase customizada de tipo transformer (clase de sklearn para transformar columnas del dataframe).
  Hereda de las clases:
  - BaseEstimator:
  - TransformerMixin: Clase incorpora el metodos fit_transform y set_output

  Atributos
  ---------
    - column_transform: (list[str]) or str) Lista con los nombres de las columnas que se quieren rellenar
    - replace_values : (list[tuple]) lista de tuplas con las categorias de esas columnas en (column_transform) que se quieren sustituir por el valor deseado en new_value
    - new_value : (list[str]) Nuevo nombre de la categoria

  Propiedades (decorador @property)
  ---------------------------------
    None

  Metodos
  -------
    - __init__ [built-in method]
    - fit
    - transform
  """
  def __init__(self, column_transform :list[str] =  [], replace_values : list[tuple] =  [()], new_value:  list[str] = [])-> None:
      self.replace_values = replace_values
      self.column_transform = column_transform
      self.new_value = new_value
  def fit(self, X, y=None):
      return self

  def transform(self, X, y=None) -> pd.DataFrame:
    """
    Docstring

    Parametros
    ----------
      - X: columnas del df a transformar
      - y

    Retorna
    -------
      - x_new (pd.DataFrame)

    """
    x_new = pd.DataFrame(X)
    if self.column_transform != [] and self.new_value != [] and len(self.new_value) == len(self.column_transform) :
      for col,newval in zip(self.column_transform,self.new_value):
          if self.replace_values != [] and len(self.replace_values) > 0:
              for old_index_vals, oldvals in enumerate(self.replace_values):
                      try: 
                          for i in range(len(oldvals)):
                              x_new[f"{col}"].replace(to_replace = oldvals[i], value = {newval},inplace=True)
                      except TypeError:
                          x_new[f"{col}"].replace(to_replace = oldvals, value = newval,inplace=True)
    return x_new
  

@execution_time
def col_for_stratified(x :np.ndarray ,rangos : np.ndarray ,categories : List[str], info : Optional[bool] = False) -> None:
    """
    Funcion que, dado una secuencia de valores numericos representados en un array 1D "rangos", asocia cada par de valores y crea una lista con
    tuplas de cada par de valores numericos. Esto son los rangos numericos. Despues, mapea cada valor de otro array "x" en funcion de si se encuentra o no 
    dentro de alguno de estos rangos o intervalos y le otorga el valor asociado (mismo valor del indice) a ese valor numerico en otra variable categorica "categories". 
    Es decir, sirve para crear una nueva columna categorica que encapsula en clases (en función de su valor numérico) a otra feature. 
    Tipicamente se aplica previamnete a aplicar stratified splits. 
    Parametros
    ----------
        - x : np.ndarray 
              Array de la columna numerica la cual se debe mapear
        - rangos : np.ndarray
                   Vector con valores "limite" para crear los rangos numericos (e.g :rangos = np.array([0,1,3] -> funcion internamente : rangos_tr = [(0,1),(1,2)]) 
        - categories : List[str]
                       Nombre con las categorias que se quieren crear, el numero de categorias debe ser igual a rangos.shape[0]-1
        - info : Optional[bool]
                 Variable booleana para mostrar informacion de parametros internos de la funcion
    Retorna
    -------
        - np.ndarray 

    """
    # Creacion de los intervalos o rangos a aplicar sobre x
    rangos_tr = [(rangos[idx],rangos[idx+1]) for idx,_ in enumerate(rangos) if idx < rangos.shape[0]-1]

    if info:
        print("Intervalos / Rangos numericos : ", rangos_tr)
        print("Numero de intervalos / Rangos numericos : ",len(rangos_tr))

    result = []
    for x_i in x:
        for idx , tuple_ranges in enumerate(rangos_tr):

            if np.logical_and(tuple_ranges[0] < x_i, x_i <= tuple_ranges[1]):
                result.append(categories[idx])
                
            # Para valores x_i que superen el valor superior del ultimo intervalo, se añaden a este ultimo intervalo y mapean el nombre de la categoria asociada:
            if idx == len(rangos)-1: 
                if x_i > tuple_ranges[1]:
                    result.append(categories[idx])

    return np.array(result)

@execution_time
def check_test_train_proportions(cat :str , x_train :pd.DataFrame ,x_test :pd.DataFrame, x_original :pd.DataFrame ) -> pd.DataFrame:
    """
    Crea un dataframe donde se refleja la proporcion de las categorias de una cierta columna de los dataframes de test y train tras un split del dataset original
    y se clacula el error como resta de proporciones
    Parameters
    ----------
        - cat : str
                Nombre de la columna a comprobar sus proporciones, debe ser un nombre existente para el test set y para el train set
        - x_train : pd.DataFrame
        - x_test : pd.DataFrame
        - x_original : pd.DataFrame
        
        
    Retorna
    -------
        - pd.DataFrame
     
    """
    # Calculate the proportions and errors
    og = x_original[cat].value_counts().values/ x_original.shape[0]
    train = x_train[cat].value_counts().values/ x_train.shape[0]
    test = x_test[cat].value_counts().values/ x_test.shape[0]

    # Creating the Dataframe
    check_df_proportions = pd.DataFrame(
                                        data = np.array([og ,train, train - og, test, test-og, train-test]).T, 
                                        columns = ["%Original", "%Train","Train-Og-Error", "%Test", "Test-Og-Error", "Train-Test-Error"] , 
                                        index = (x_train[f"{cat}"].value_counts().index.values )
                                        )
    
    return check_df_proportions