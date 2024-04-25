### Feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import  Callable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from itertools import product



def select_kBest(data: pd.DataFrame = None, target_feature :str = None, encodings: list[Callable] = [], num_transformers : list[Callable] = [], \
                 score_func: list[Callable] = [], threshold_score : float = 0.1, kbest : int = 3) -> None:

  """
  Funcion que aplica SelectKbest tras codificar las columnas categoricas y aplicar una transformacion (o no) sobre todas las columnas numericas del df (tras la codificacion todas seran numericas).
  Despues crea graficas que describen la puntuacion (score) dada a cada nueva feature generada por la funcion estadistica elegida para cada caso.

  Parametros
  ----------
    key word arguments:
      - data : (pd.DataFrame) Dataframe sobre el que se quiere llevar acabo el analisis de dimensionalidad
      - target_feature: (str) Variable a predecir
      - score_func : (list[Callable]) Lista con las funciones que realizan los test estadisticos entre las features en "X" e "y" y devuelven un par de matrices (scores y pvalores o solo scoores)
      - encodings : (list[Callable]) Lista con los codificadores de las columnas categoricas que se quieren aplicar
      - num_transformers : (list[Callable])  Lista con los numerical transformers que se quieren aplicar sobre todo el df tras la codificacion de las columnas categoricas
      - threshold_score : (float) valor para filtra el score minimo que debe de tener cierta feature para ser ploteada y ser considereda [su objetivo es cuando hay excesivas features, dropear las de menor score
                          la formula utilizada es: [valor mininmo de score para plotear] = threshold_score * [valor maximo de todo el array de scores]
                          y asi visualizar mejor las graficas]
      - kbest: (int) numero de mejores features a seleccionar segun el test estadistico empleado (score_func)

  Retorna
  -------
    None

  """
  # mutual_info_regression: para sparse data
  # f_regression: no sparse data

  # Manejo de de valores faltantes, SelectKbest no maneja NA values es necesario dropearlos del dataframe
  col_with_NA = [(col,data[f"{col}"].isnull().sum()) for col in data.columns if data[f"{col}"].isnull().sum() > 0] # Lista con tuplas = (columna, numero de NA)
  print("col_with_NA: ",col_with_NA)

  # Drop de esos NA, el usuario por pantalla elige si dropear la columna entera o las filas. [si hay muchos NA en esa columna drop de columna si no de fila]
  data_drop = data.copy()
  for col_name,num_NA in col_with_NA:

    print(f"En la columna '{col_name}' con {num_NA} valores NA, hay: {data[f'{col_name}'].shape[0]} filas y el {(num_NA/data[f'{col_name}'].shape[0]) *100} % son valores NA")
    n = int(input(f"insertar: '1' para borrar la columna {col_name} o '0' para borrar sus filas con valores NA -- "))
    print("-----------------------------------------------------------------")
    while n != 1 and n != 0:
      n = input("Input error: no existe esa opcion, vuelva a introducir '1' o '0' : ")
      print("-----------------------------------------------------------------")
    if n == 1:
      data_drop.drop(labels = f'{col_name}', axis = 1, inplace = True)
    if n == 0:
      data_drop.dropna( subset = [f"{col_name}"], inplace = True)
      print(f"Actualizacion del dataframe tras el drop -- numero filas :  {data_drop[f'{col_name}'].shape[0]} ")
      print("-----------------------------------------------------------------")
    print(f"Actualizacion del dataframe tras el drop -- numero columnas :  {len(data_drop.columns)} ")
    print("-----------------------------------------------------------------")


  # Division: target y features
  x = data_drop.drop(labels = [f"{target_feature}"], axis = 1, inplace = False)
  y = data_drop[f"{target_feature}"]

  # Nombre de las col cat de x
  cat_features_x = list(x.select_dtypes(["object","bool"]).columns)

  print("Number of Numeric columns", len(x.select_dtypes(["int64","float64"]).columns))
  print("Number of Categoric columns", len(x.select_dtypes(["object"]).columns))

  # lista con tuplas de las posibles combinaciones
  combinations = list(product(set(encodings), set(num_transformers), set(score_func))) # lista de tuplas de todas posibles combinaciones no repetidas: [(ecoder,transformer,n_pca),(...)]
  print('combinations',combinations)

  # Calculo de filas y columnas de la figura para graficar
  figure_cols = 2
  figure_rows = int(np.ceil(len(combinations) / figure_cols))

  # defincion de la figura
  figure = plt.figure(figsize = (14,figure_rows*6), layout = 'constrained', )

  for index_comb, comb in enumerate(combinations):

    # Instancia de la clase SelectKBest
    selector = SelectKBest(score_func = comb[2], k= kbest)

    # Instancia de la clase ColumnTransformer con encoder de las categorical features
    preprocessor = ColumnTransformer( transformers=[

                                                    ('encoder', comb[0], cat_features_x),
                                                      ],
                                        remainder='passthrough'
                                      )
    preprocessor.set_output(transform ="pandas")

    # Instancia de la clase Pipeline y definicion de sus steps (entre ellos el numerical transformer si tiene)
    pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('transformer', comb[1]),
                            ("Select k best",selector),
                            ])

    # Fit pipeline
    pipeline.fit(x,y)

    # Obtencion de las nuevas features output tras cada step del pipeline
    feature_after_coded = pipeline['preprocessor'].get_feature_names_out(input_features = x.columns)
    feature_after_num_transform = pipeline['transformer'].get_feature_names_out(input_features = feature_after_coded)
    feature_selected = pipeline['Select k best'].get_feature_names_out(input_features = feature_after_num_transform) # Features elgidas por select k best

    # Obtencion de scores y p_values otorgados por select k best
    scores = pipeline["Select k best"].scores_
    print("Scores shape = ",scores.shape[0])

    # Almacenamiento de los scores de cada feature en dataframe
    df_info = pd.DataFrame(data = {"scores":scores , "features":feature_after_num_transform})
    print("Df Sin sort \n", df_info.head(15))
    print(df_info.dtypes)
    print("Features elegidas de mayor score: ",feature_selected)

    # Ordenamos los scores en orden descendente en un nuevo df
    df_sorted = df_info.sort_values(by=['scores'], inplace = False, ascending = False)
    print("Df antes de aplicar threshold score \n",df_sorted.head(15))

    #Filtro del df por threshold_score: se filtra por umbral * valor_max_score
    df_sorted = df_sorted.loc[df_sorted["scores"] >= threshold_score*np.max(scores) ]
    print("Df despues de aplicar threshold score \n ",df_sorted.head(15))

    # Plotting scores vs features
    selector_axes = plt.subplot(figure_rows,figure_cols,index_comb+1)
    selector_axes.grid(linewidth = 0.5,  alpha=1)
    selector_axes.set_title(label = f'Pipeline used: ({comb[0]},{comb[1]},{comb[2].__name__}) \n Minimum score applied: {round(threshold_score*np.max(scores),3)}')
    sns.barplot(df_sorted, x="scores", y="features",orient ='h', ax = selector_axes, palette = 'coolwarm')

  figure.show()


"""
# EJEMPLO DE USO DE LA FUNCION:

# Encoders (transformer objects):

onehot = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
ordinal = OrdinalEncoder()


# Transformers (transformer objects):

estandar = StandardScaler() # Estandarizacion
norm = MinMaxScaler(feature_range=(0, 1)) # Normalizacion
max_abs = MaxAbsScaler()
robust = RobustScaler(quantile_range=(25.0,75.0))

### llamada a la funcion:
df_informativo_pca = select_kBest(
                                  data = data, 
                                  target_feature = "adr",
                                  encodings = [onehot], 
                                  num_transformers = [estandar],  
                                  score_func =  [f_regression], 
                                  threshold_score = 0.3
                                  )
"""
