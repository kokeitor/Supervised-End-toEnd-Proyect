import pandas as pd
import numpy as np

def drop_na(data: pd.DataFrame = None) -> pd.DataFrame:
  """
  Dropper NA function

  Parametros
  ----------
    - df: (pd.DataFrame) : Dataframe to drop na values

  Retorna
  -------
    - pd.DataFrame

  """
  # Crea una lista de tuplas con nombre columna y numero de NA de esa columna
  col_with_NA = [(col,data[f"{col}"].isnull().sum()) for col in data.columns if data[f"{col}"].isnull().sum() > 0] # Lista con tuplas = (columna, numero de NA)
  print("col_with_NA: ",col_with_NA)

  # Drop de esos NA, el usuario por pantalla elige si dropear la columna entera o las filas. [si hay muchos NA en esa columna drop de columna si no de fila]
  data_drop = data.copy()
  for col_name , num_NA in col_with_NA:

    print(f"En la columna '{col_name}' con {num_NA} valores NA, hay: {data[f'{col_name}'].shape[0]} filas y el {(num_NA/data[f'{col_name}'].shape[0]) *100} % son valores NA")
    n = str(input(f"insertar: '1' para borrar la columna {col_name} - '0' para borrar sus filas con valores NA - 'pass' para no modificarla "))
    print("-----------------------------------------------------------------")
    while n != "1" and n != "0" and n != "pass":
      n = input("Input error: no existe esa opcion, vuelva a introducir '1' o '0' : ")
      print("-----------------------------------------------------------------")
    if n == "1":
      data_drop.drop(labels = f'{col_name}', axis = 1, inplace = True)
    if n == "0":
      data_drop.dropna( subset = [f"{col_name}"], inplace = True)
      print(f"Actualizacion del dataframe tras el drop -- numero filas :  {data_drop[f'{col_name}'].shape[0]} ")
      print("-----------------------------------------------------------------")
    if n == "pass":
      print(f"No se ha modificado la columna : {col_name}")
      print("-----------------------------------------------------------------")
    print(f"Actualizacion del dataframe tras el drop -- numero columnas :  {len(data_drop.columns)} ")
    print("-----------------------------------------------------------------")

  return data_drop

"""
# EJEMPLO DE USO DE LA FUNCION
data_for_drop = data.copy()
data_for_drop = drop_na(data_for_drop)

"""