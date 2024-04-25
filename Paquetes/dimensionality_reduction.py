import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import  Callable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from itertools import product
from typing import  Callable, Optional, List, Dict, Tuple
import seaborn as sns
# UMAP: Uniform manifold approximation and projection (non-linear dimensionality reduction)
import umap
import umap.plot
from Paquetes.visualization import Eda


class PCAStudy:
  
  @staticmethod
  def combined_study(*pca_stduy) -> pd.DataFrame:
    """combinar dataframes (self.results) de los objetos PCA en un unico datfarame conjunto usando concatenate

    Returns:
        pd.DataFrame: _description_
    """
    if len(pca_stduy) > 0:
      for idx,pca_study_i in enumerate(pca_stduy):
        
        if idx == 0:
          df_combined = pca_study_i.results
        else:
          df_combined = pd.concat([df_combined, pca_study_i.results])
          
      return df_combined
    else:
      print("Note : Only one PCA study has been provided")
      return pca_stduy.results
  
  def __init__(self, data: pd.DataFrame, encoder : object, pca_components: int , num_transformer : Optional[object] = None,verbose : int = 0) -> None:
    self.data = data
    self._encoder = encoder
    self._num_transformer = num_transformer
    self.pca_components = pca_components
    self.verbose = verbose
    self._is_fitted = False
    
    if self.pca_components <= 0:
      raise ValueError("The number of PCA components must be > 0")
    
    if self._num_transformer == None:
      self.is_numerical_transformed = False
    else:
      self.is_numerical_transformed = True
    
      
  @property
  def encoder(self) -> List[object]:
    return self._encoder
  
  @property
  def num_transformer(self) -> List[object]:
    if self._num_transformer == None:
      print("Warning : No numerical transformer has been provided")
    return self._num_transformer
  
  @encoder.setter
  def encoder(self, value: List[object]) -> None:
    self._encoder= value
    
  @num_transformer.setter
  def num_transformer(self, value: List[object]) -> None:
    self.is_numerical_transformed = True
    self._num_transformer = value

    
  def _initialize_info_df(self) -> None:
    """_summary_
    """

      # Cuando se quiere aplicar una transformacion numerica (Estandarizacion, normalizacion ...)
    if self.is_numerical_transformed:
      self.case_to_study = [(self._encoder,self._num_transformer,self.pca_components)] # lista de tupla  [(ecoder,transformer,n_pca)]
      if self.verbose ==1:
        print('combinations',self.case_to_study)
    else: # Cuando no se quiere aplicar una transformacion numerica
      self.case_to_study = [(self._encoder,self.pca_components)]
      if self.verbose ==1:
        print('combinations',self.case_to_study)
    self.results = pd.DataFrame(
                                columns = ['ENCODER','ORIGINAL_FEATURES','ENCODED_FEATURES','NUMERICAL_TRANSFORMER','PCA_COMPONENTS','ORIGINAL_VARIANCE_RETAINED'], 
                                index = range(len(self.case_to_study))
                                )
  @property
  def study_case(self) -> None:
    self._initialize_info_df()
    self._user_drop_NA()
    self._pipeline_factory()
    self._fit()
    self._transform()
    
  def _pipeline_factory(self) -> None:
    # Pca object
    self.pca = PCA(n_components = self.pca_components)
    # Df fill
    self.results.loc[0, 'PCA_COMPONENTS'] = self.pca_components
    self.results.loc[0, 'ENCODER'] = self._encoder
    self.preprocessor = ColumnTransformer( 
                                            transformers=[

                                                            ('encoder', self._encoder, list(self.data.select_dtypes(["object","bool"]).columns)),
                                                          ],
                                              remainder='passthrough' 
                                          )
    if self.is_numerical_transformed:
      self.results.loc[0, 'NUMERICAL_TRANSFORMER'] = self._num_transformer
      self.pipeline = Pipeline(
                                steps=[
                                        ('preprocessor', self.preprocessor),
                                        ('num_transf', self._num_transformer ),
                                        ("pca",self.pca),
                                      ]
                              )
    else:
      self.results.loc[0, 'NUMERICAL_TRANSFORMER'] = 'None'
      self.pipeline = Pipeline(
                                steps=[
                                        ('preprocessor', self.preprocessor),
                                        ("pca",self.pca),
                                      ]
                              )
  def _fit(self) -> Pipeline:
    _fitted_pipe = self.pipeline.fit(self.data)
    self.eigenvectors = self.pipeline["pca"].components_
    
    self._is_fitted = True
    self.results.loc[0, 'ORIGINAL_VARIANCE_RETAINED'] = np.sum(self.get_variance)
    self._encoder_name = self.pipeline.named_steps['preprocessor'].transformers[0][1]
    
    # Obtencion de las features tras cada step del pipeline
    self.encoded_columns = self.pipeline['preprocessor'].get_feature_names_out(input_features = self.data.columns) #obtener nombre de las features tras un preprocesamiento en cierto step del pipeline
    if self.is_numerical_transformed:
      self.num_encoded_columns = self.pipeline['num_transf'].get_feature_names_out(input_features = self.encoded_columns) #obtener nombre de las features en el step 2 del pipeline
    
    if self.verbose == 1:
      print(f"Numero de features tras step {str(self.pipeline['preprocessor'])}: ",len(self.encoded_columns))
      if self.is_numerical_transformed:
        print(f"Numero de features tras step {str(self.pipeline['num_transf'])}: ",len(self.num_encoded_columns))
      print("-----------------------------------------------------------------")

    # Df fill
    self.results.loc[0, 'ORIGINAL_FEATURES'] = len(self.data.columns)
    self.results.loc[0, 'ENCODED_FEATURES'] = len(self.encoded_columns)
    
    return _fitted_pipe
  
  def _transform(self) -> None:
    self.data_transformed = self.pipeline.transform(self.data)
  
  @property
  def get_variance(self) -> np.ndarray:
    if self._is_fitted:
      return self.pca.explained_variance_ratio_
    else:
      print(f"Error in property get_variance : PCA not fitted, call study_case method first")
  @property
  def plot_variance_retained(self) -> None:
      """Plot del porcentaje de varianza retenida por cada PCA creada"""
      
      if self._is_fitted:
        
        if self.verbose == 1:
          print(f'Number of PCA components choosen after using {self._encoder_name}: {(len(self.get_variance))}') 
          print(f'Explained variance ratio: \n {(self.get_variance)}') 
          print(f'Fraction of original variance (or information) kept by each principal component axis (or image vector) after apllying {self._encoder_name} : {round(np.sum(self.get_variance),3)}') # image vector == vector proyectado
          print("-----------------------------------------------------------------")
          
        # Plot del porcentaje de varianza retenida por cada PCA creada
        plt.figure(figsize=(12, 9), layout ='constrained',linewidth = 0.1)
        plt.bar(range(1,len(self.get_variance) +1 ), self.get_variance, alpha=1, align='center', label=f'Individual explained variance',color = 'cyan', edgecolor = 'black')
        plt.ylabel('Explained variance ratio')
        plt.xlabel(f'Principal components using {self._encoder_name}')
        plt.xticks(range( 1,len(self.get_variance) +1))
        plt.grid()
        plt.legend(loc='best')
        plt.show()
        
      else:
        print(f"Error in method plot_variance_retained : PCA not fitted, call study_case method first")
  @property
  def plot_feature_weights(self) -> None:
    """_summary_
    """
    if self._is_fitted:
      if self.verbose ==1:
        print(f"Eigenvectors : \n {self.eigenvectors}")
        
      _plot_cols = self.encoded_columns 
      if self.is_numerical_transformed:
        _plot_cols = self.num_encoded_columns 

      for indx,eigenvector in enumerate(self.eigenvectors[0:self.pca_components,:]):
        plt.figure(figsize=(15, 10), layout= 'constrained')
        plt.bar( _plot_cols, eigenvector, color='blue', edgecolor = 'black')
        plt.xlabel('Features')
        plt.ylabel('weights [lineal combinations]')
        plt.xticks(rotation=40)
        plt.title(f'{indx+1} Principal Component Weights for Each Feature using {self._encoder_name}')
        plt.grid()
        plt.show()
    else:
      print(f"Error in method plo_feature_weights : PCA not fitted, call study_case method first")
    """    
    plt.figure(figsize=(15, 10), layout= 'constrained')
    plt.bar(feature_namesfor_plotting, first_principal_component_plotting, color='blue', edgecolor = 'black')
    plt.xlabel('Features')
    plt.ylabel('Value')
    plt.xticks(rotation=40)
    plt.title(f'First Principal Component Weights for Each Feature using {codifier_name}')
    plt.grid()
    plt.show()
    """
  
    
  def _user_drop_NA(self)-> None:
    """User NA interactive dropper"""
    # Manejo de de valores faltantes, porque PCA no maneja NA values es necesario dropearlos del dataframe
    no_NA = False
    col_with_NA = []
    for col in self.data.columns:
      if self.data[f"{col}"].isnull().sum() > 0:
        col_with_NA.append((col,self.data[f"{col}"].isnull().sum()))
      else:
        no_NA = True
        
    if self.verbose == 1:
      print("Columns with NA ",col_with_NA)
      
    if no_NA:
      if self.verbose == 1:
        print("No NA values in the dataframe")
    else:
      # Drop de esos NA, el usuario por pantalla elige si dropear la columna entera o las filas. [si hay muchos NA en esa columna drop de columna si no de fila]
      data_drop = self.data.copy()
      for col_name,num_NA in col_with_NA:

        print(f"En la columna '{col_name}' con {num_NA} valores NA, hay: {self.data[f'{col_name}'].shape[0]} filas y el {(num_NA/self.data[f'{col_name}'].shape[0]) *100} % son valores NA")
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
        self.data = data_drop
    
    
  
def pca(
          data: pd.DataFrame, 
          encodings: List[object] = [], 
          num_transformers : List[object] = [], 
          pca_components: List[int] = [],
          verbose : int = 0,
          
          ) -> pd.DataFrame:
    
  """
  Funcion que aplica PCA tras codificar las columnas categoricas y aplicar una transformacion (o no) sobre todas las columnas numericas del df (tras la codificacion todas seran numericas).
  Despues retorna un dataframe con informacion del PCA realizado y crea graficas que tambien lo describen

  Parametros
  ----------
    key word arguments:
      - data : (pd.DataFrame) Dataframe sobre el que se quiere llevar acabo el analisis de dimensionalidad
      - encodings : (list[object]) Lista con los codificadores de las columnas categoricas que se quieren aplicar
      - num_transformers : (List[object])  Lista con los numerical transformers que se quieren aplicar sobre todo el df tras la codificacion de las columnas categoricas
      - pca_components: (List[int]) lista con el numero de pca components que se quieren aplicar en el estudio de reduccion de dimensionalidad
                                    Nota: pca_components = None, se calculara --> pca_components = n_features
      - verbose : (int) 

  Retorna
  -------
    pd.DataFrame

  """

  # Manejo de errores en los key word arguments
  if encodings == []:
    raise ValueError("ERROR: key word argument encodings it's empty and minimum one encoder must be defined")

  # Inicializacion del df informativo
  df_columns = ['ENCODER','ORIGINAL_FEATURES','NEW_FEATURES','NUMERICAL_TRANSFORMER','PCA_COMPONENTS','ORIGINAL_VARIANCE_RETAINED']
  
  # Cuando se quiere aplicar una transformacion numerica (Estandarizacion, normalizacion ...)
  if num_transformers != []:
    combinations = list(product(set(encodings), set(num_transformers), set(pca_components))) # lista de tuplas de todas posibles combinaciones no repetidas: [(ecoder,transformer,n_pca),(...)]
    df_rows = len(combinations)
    
    if verbose ==1:
      print('combinations',combinations)

  else: # Cuando no se quiere aplicar una transformacion numerica

    combinations = list(product(set(encodings), set(pca_components)))
    df_rows = len(combinations)
    
    if verbose ==1:
      print('combinations',combinations)

  df_pca = pd.DataFrame(columns = df_columns, index = range(df_rows))

  data_drop = _user_drop_NA(data = data , verbose  = verbose)
  
  def _user_drop_NA(data : pd.DataFrame, verbose: int = verbose)-> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        verbose (int, optional): _description_. Defaults to verbose.

    Returns:
        pd.DataFrame: _description_
    """
    
    # Manejo de de valores faltantes, porque PCA no maneja NA values es necesario dropearlos del dataframe
    no_NA = False
    col_with_NA = []
    for col in data.columns:
      if data[f"{col}"].isnull().sum() > 0:
        col_with_NA.append((col,data[f"{col}"].isnull().sum()))
      else:
        no_NA = True
        
    if verbose == 1:
      print("col_with_NA: ",col_with_NA)
      
    if no_NA:
      if verbose == 1:
        print("No NA values in the dataframe")
      return data
    else:
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
        return data_drop


  # Inicio de bucles
  for comb_index , combination in enumerate(combinations):

    if len(combination) == 3: # == num_transformers != []:

      # Manejo de errores en los key word arguments
      if combination[2] <= 0:
        raise ValueError("ERROR: The number of PCA components must be > 0")

      else:

        # Pca object
        pca = PCA(n_components = combination[2])

        # Df fill
        df_pca.loc[comb_index, 'PCA_COMPONENTS'] = combination[2]

        # Df fill
        df_pca.loc[comb_index, 'NUMERICAL_TRANSFORMER'] = combination[1]

        # Instancia de la clase ColumnTransformer y definicion de su encoder
        preprocessor = ColumnTransformer( transformers=[

                                                        ('encoder', combination[0], list(data_drop.select_dtypes(["object","bool"]).columns)),
                                                        ],
                                          remainder='passthrough' # IMPORTANTE: columnas del df no "procesadas" en el ColumnTransformer las conserva;
                                                                  # con "drop" las dropea del df que le pasa al siguiente step del pipeline
                                        )

        # Df fill
        df_pca.loc[comb_index, 'ENCODER'] = combination[0]

        # Instancia de la clase Pipeline y definicion de sus steps (entre ellos el numerical transformer si tiene)
        pipeline = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('num_transf', combination[1]),
                                ("pca",pca),
                                ])
    elif len(combination) == 2: # num_transformers == []:

      # Manejo de errores en los key word arguments
      if combination[1] <= 0:
        raise ValueError("ERROR: The number of PCA components must be > 0")

      else:

        # Pca object
        pca = PCA(n_components = combination[1])

        # Df fill
        df_pca.loc[comb_index, 'PCA_COMPONENTS'] = combination[1]

        # Instancia de la clase ColumnTransformer y definicion de su encoder
        preprocessor = ColumnTransformer( transformers=[

                                                        ('encoder', combination[0], list(data_drop.select_dtypes(["object","bool"]).columns)),
                                                        ],
                                          remainder='passthrough' # IMPORTANTE: columnas del df no "procesadas" en el ColumnTransformer las conserva;
                                                                  # con "drop" las dropea del df que le pasa al siguiente step del pipeline
                                        )
        # Df fill
        df_pca.loc[comb_index, 'ENCODER'] = combination[0]

        # Instancia de la clase Pipeline y definicion de sus steps (entre ellos el numerical transformer si tiene)
        pipeline = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ("pca",pca),
                                ])

    data_transformed = pipeline.fit_transform(data_drop)
    variance = pca.explained_variance_ratio_
    codifier_name = pipeline.named_steps['preprocessor'].transformers[0][1]
    # Df fill
    df_pca.loc[comb_index, 'ORIGINAL_VARIANCE_RETAINED'] = np.sum(variance)
    
    _plot_variance_retained(variance = variance, codifier_name = codifier_name)
    if verbose == 1:
      print(f'Number of PCA components choosen after using {codifier_name}: {(len(variance))}') 
      print(f'Explained variance ratio: \n {(variance)}') 
      print(f'Fraction of original variance (or information) kept by each principal component axis (or image vector) after apllying {codifier_name}:{(np.sum(variance))}') # image vector == vector proyectado
      print("-----------------------------------------------------------------")
      
      
      def _plot_variance_retained(variance : np.ndarray, codifier_name : str) -> None:
        """Plot del porcentaje de varianza retenida por cada PCA creada
        Args:
            variance (np.ndarray): vector con autovalori / sum(autovalori) [i es cada componente ppal PCA]
            codifier_name (str): nombre del codificador categorico utilizado
        Returns:
          None
        """
        # Plot del porcentaje de varianza retenida por cada PCA creada
        plt.figure(figsize=(12, 9), layout ='constrained',linewidth = 0.1)
        plt.bar(range(1,len(variance) +1 ), variance, alpha=1, align='center', label=f'Individual explained variance',color = 'cyan', edgecolor = 'black')
        plt.ylabel('Explained variance ratio')
        plt.xlabel(f'Principal components using {codifier_name}')
        plt.xticks(range( 1,len(variance) +1))
        plt.grid()
        plt.legend(loc='best') # Matplotlib intentará elegir la ubicación óptima de la leyenda para evitar que se solape con los datos trazados
        plt.show()

    if len(combination) == 3:
      # Plot del peso de cada feature (tras preprocesamiento, antes de PCA) en el PCA 1:
      feature_names_after_step = pipeline['preprocessor'].get_feature_names_out(input_features = data_drop.columns) #obtener nombre de las features tras un preprocesamiento en cierto step del pipeline
      feature_names_after_step_2 = pipeline['num_transf'].get_feature_names_out(input_features = feature_names_after_step) #obtener nombre de las features en el step 2 del pipeline
      print(f"Numero de features tras step {str(pipeline['preprocessor'])}: ",len(feature_names_after_step))
      print(f"Numero de features tras step {str(pipeline['num_transf'])}: ",len(feature_names_after_step_2))
      print("-----------------------------------------------------------------")

      # Df fill
      df_pca.loc[comb_index, 'ORIGINAL_FEATURES'] = len(data_drop.columns)
      df_pca.loc[comb_index, 'NEW_FEATURES'] = len(feature_names_after_step_2)

      first_principal_component = pipeline["pca"].components_[0]

      # Ploteann solo features que tenga un cierto peso/relevancia (poostivo o negativo) sobre PCA 1
      umbral = 0.1
      feature_namesfor_plotting = [i  for i,j in (zip(feature_names_after_step_2 , first_principal_component)) if abs(j) > umbral]
      first_principal_component_plotting = [j  for j in first_principal_component if abs(j) > umbral]

      plt.figure(figsize=(15, 10),layout= 'constrained')
      plt.bar(feature_namesfor_plotting, first_principal_component_plotting, color='blue', edgecolor = 'black')
      plt.xlabel('Features')
      plt.ylabel('Value')
      plt.xticks(rotation=40)
      plt.title(f'First Principal Component Weights for Each Feature using {codifier_name}')
      plt.grid()
      plt.show()

    if len(combination) == 2:

      # Plot del peso de cada feature (tras preprocesamiento, antes de PCA) en el PCA 1:
      feature_names_after_step = pipeline['preprocessor'].get_feature_names_out(input_features = data_drop.columns) #obtener nombre de las features tras un preprocesamiento en cierto step del pipeline
      print(f"Numero de features tras step {str(pipeline['preprocessor'])}: ",len(feature_names_after_step))

      # Df fill
      df_pca.loc[comb_index, 'ORIGINAL_FEATURES'] = len(data_drop.columns)
      df_pca.loc[comb_index, 'NEW_FEATURES'] = len(feature_names_after_step)

      first_principal_component = pipeline["pca"].components_[0]

      # Ploteann solo features que tenga un cierto peso/relevancia (poostivo o negativo) sobre PCA 1
      umbral = 0.1
      feature_namesfor_plotting = [i  for i,j in (zip(feature_names_after_step , first_principal_component)) if abs(j) > umbral]
      first_principal_component_plotting = [j  for j in first_principal_component if abs(j) > umbral]

      plt.figure(figsize=(15, 10), layout= 'constrained')
      plt.bar(feature_namesfor_plotting, first_principal_component_plotting, color='blue', edgecolor = 'black')
      plt.xlabel('Features')
      plt.ylabel('Value')
      plt.xticks(rotation=40)
      plt.title(f'First Principal Component Weights for Each Feature using {codifier_name}')
      plt.grid()
      plt.show()


  return df_pca

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
df_informativo_pca = pca(data = data, encodings = [onehot], num_transformers = [estandar,norm,max_abs,robust], pca_components =  [20,1250])

"""

def seq_feature_selector(
                          X : np.ndarray,
                          y: np.ndarray,
                          estimator : object,
                          n_features_to_select : int = 1,
                          tol : Optional[float] = None,
                          direction : str = 'backward',
  
                            ) -> Tuple[np.ndarray,np.ndarray]:
  """Seq feture selector"""
  # Sklearn class import
  from sklearn.feature_selection import SequentialFeatureSelector
  
  # Error in the parameters
  if direction not in ("forward", 'backward'):
      raise ValueError(f"direction must be 'forward' or 'backward'; got {direction}")
    
  if type(n_features_to_select) != int and n_features_to_select != "auto":
      raise ValueError(f"n_features_to_select must be 'auto' or int; got {n_features_to_select}")
    
  elif type(n_features_to_select) != int and n_features_to_select == "auto" and  tol ==None:
    raise ValueError(f"tol must be defined as a criteria to select number the number of features, must be a float")
    
  # Fit object of the instance class on X,y train sets
  sfs = SequentialFeatureSelector(
                              estimator = estimator,  
                              n_features_to_select=n_features_to_select , 
                              direction = direction
                              )
  sfs.fit(X,y)
  return sfs.transform(X)


def UMAPStudy(
                X_train :np.ndarray ,
                y_train: np.ndarray , 
                X_test: Optional[np.ndarray]= None ,
                y_test: Optional[np.ndarray] = None , 
                n_neighbors : int = 15, 
                min_dist : float = 0.1, 
                n_components : int = 2, 
                random_state : int = 42,
                metric : str = 'euclidean', 
                title : str = ''
                )-> Tuple[np.ndarray,np.ndarray,umap.umap_.UMAP]:
  """_summary_

  Args:
      X (np.ndarray): _description_
      y (np.ndarray): _description_
      n_neighbors (int, optional): _description_. Defaults to 15.
      min_dist (float, optional): _description_. Defaults to 0.1.
      n_components (int, optional): _description_. Defaults to 2.
      metric (str, optional): _description_. Defaults to 'euclidean'.
      title (str, optional): _description_. Defaults to ''.

  Returns:
      np.ndarray: _description_
  """
  mapper  = umap.UMAP(
                      n_neighbors=n_neighbors,
                      min_dist=min_dist,
                      n_components=n_components,
                      metric=metric,
                      random_state=random_state,
                  )
  
  data_train = np.hstack((X_train,y_train.reshape(-1,1)))
  data_test= np.hstack((X_test,y_test.reshape(-1,1)))
  
  mapper.fit(data_train)
  u_train = mapper.transform(data_train)
  u_test = mapper.transform(data_test)
  
  umap.plot.points(mapper, labels=y_train, theme='fire')
  
  if n_components == 2:
      eda_data_train = Eda(data =pd.DataFrame(data = np.hstack((u_train,y_train.reshape(-1,1))), columns = ['0','1','target']) )
      eda_data_test = Eda(data =pd.DataFrame(data = np.hstack((u_test,y_test.reshape(-1,1))), columns = ['0','1','target']) )
      eda_data_train.plot_scatter(
                          fig_x_size = 17, 
                          fig_cols = 2,
                          linewidth =  1,  
                          layout = 'constrained' , 
                          x =  ['0'], 
                          y = ['1'],  
                          size = None, 
                          hue = "target", 
                          color = 0, 
                          plotting_lib = 'seaborn', 
                          umbral = 999999999, 
                          show_outliers = True, 
                          method = "RIC",
                          opacity = 0.7,
                          plotly_colorscale = 22, # [19,20,21,22,23,24]
                          plotly_bgcolor = "white",# str or none
                          save_figure = False,
                          name_figure = ""  ,
                          title = title  + ' (train)',                          
                      )
      eda_data_test.plot_scatter(
                    fig_x_size = 17, 
                    fig_cols = 2,
                    linewidth =  1,  
                    layout = 'constrained' , 
                    x =  ['0'], 
                    y = ['1'],  
                    size = None, 
                    hue = "target", 
                    color = 0, 
                    plotting_lib = 'seaborn', 
                    umbral = 999999999, 
                    show_outliers = True, 
                    method = "RIC",
                    opacity = 0.7,
                    plotly_colorscale = 22, # [19,20,21,22,23,24]
                    plotly_bgcolor = "white",# str or none
                    save_figure = False,
                    name_figure = ""  ,
                    title = title + ' (test)',                          
                )


  if n_components == 3:
      eda_data_train = Eda(data =pd.DataFrame(data = np.hstack((u_train,y_train.reshape(-1,1))), columns = ['0','1','2','target']) )
      eda_data_test = Eda(data =pd.DataFrame(data = np.hstack((u_test,y_test.reshape(-1,1))), columns = ['0','1','2','target']) )
      eda_data_train.plot_scatter_3d(
                                  x_feature  = '0',
                                  y_feature = '1',
                                  z_feature = '2',
                                  color_feature  = 'target',
                                  title =  title + '  (train)'
                              )
      eda_data_test.plot_scatter_3d(
                            x_feature  = '0',
                            y_feature = '1',
                            z_feature = '2',
                            color_feature  = 'target',
                            title =  title +' (test)'
                        )


  return u_train,u_test,mapper


  """
  UMAP USING FUNCTION
  for n_neighbor in [60]:
     for n_c in [1,2]:
          for metric_i in ['euclidean','manhattan','hamming','cosine']:
               _ = draw_umap(
                              X= resampled_train_sets["SVMSMOTE"][0], 
                              y=resampled_train_sets["SVMSMOTE"][1], 
                              n_neighbors = n_neighbor,
                              n_components = n_c, 
                              min_dist= 0.1,
                              metric = metric_i,
                              title = f'n_neighbors: {n_neighbor} n_components: {n_c}  metric: {metric_i}' 
                              )
  """

def testing():
  import os
  train = pd.read_csv(os.path.abspath("C:\\Users\\Koke\\Desktop\\MASTER_IA\\TareaFinalSupervised\\data\\train.csv"))
  pca1  = PCAStudy(data =train , encoder = OneHotEncoder(), pca_components = 2 , num_transformer = StandardScaler(),verbose  = 1)
  pca1.plot_variance_retained
  pca1.study_case
  pca1.plot_variance_retained
  print(pca1.results.head())
  
if __name__ == '__main__':
  testing()