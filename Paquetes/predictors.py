import pandas as pd
import numpy as np
from typing import  Callable, Optional, List, Dict, Tuple
from sklearn.base import BaseEstimator, TransformerMixin,OneToOneFeatureMixin
from sklearn.preprocessing import FunctionTransformer
from Paquetes.optimization import execution_time
import plotly.graph_objs as go

"""
USO DE LA LIBRERIA TYPING :

Ejemplo 1 :
    from typing import  Sequence
    def square(elems: Sequence[float]) -> List[float]:

    Explicacion:
    En este caso "elems" es una secuencia de algun tipo de numeros flotantes. Una secuencia puede ser un iterable de tipo List, una Tuple o cualquier estructura que soprte el metdo len() y .__getitem__() independientemente
    de su tipo

Ejemplo 2 :
    from typing import  Tuple, List
    elems_list: List[float]  = [1.0, 2.0]
    elems_tuple: Tuple[float,str] = (1.0, "hola")

    Explicacion:
    En este caso "elems_list" es una lista de flotantes donde por ser una lista (iterable o secuencia mutable) normalmente estan constituidas por varios elementos del mismo tipo y por ello
    no hace falta especificar cada tipo de cada elemento interno de la lista.
    En este caso "elems_tuple" es una tupla (iterable o secuencia inmutable) que, en general, esta constituida por una cantidad de elementos fija y de diferente tipo; por ello
    si necesita que se especifique el tipo de cada elemento interno.
    de su tipo

Ejemplo 3 :
    from typing import  Optional
    def square(elems: Optional[str] = None) -> List[float]:

    Explicacion:
    The Optional type simply says that a variable either has the type specified or is None

FALTA MIRAR :
@classmethod
typing: https://realpython.com/python-type-checking/

"""

class Perceptron:
    """
    Clase propia que establece el algoritmo del perceptron

    Atributos
    ---------
        - alpha: float
        - n_iter: int
        - seed : int
        - weights: np.ndarray
        - errors : list
        - plot_info ( plotear numero de actulaizaciones frente a numero de iteraciones)

    Propiedades (decorador @property)
    ---------------------------------
        - None

    Metodos
    -------
        - __init__ [built-in method]. Constructor
        - fit
        - _itera : metodo interno
        - predict
        - _plot_errors


    """
    def __init__(self, alpha : float = 0.1, n_iter : int = 50, seed : int = 1, plot_info : bool = True, sigmoid : bool = False) -> None:

        """
        Metodo (built-in) que inicializa la clase. Constructor

        """
        self.alpha = alpha
        self.n_iter = n_iter
        self.seed = seed
        self.plot_info = plot_info
        self.sigmoid = sigmoid

    @execution_time
    def fit(self, X : np.ndarray, y : np.ndarray) -> object:
        """
        Docstring

        Parametros
        ----------
            - X
            - y
        Retorna
        -------
            - self

        """
         # Inicializacion de los pesos a floats muy pequeños (cuyos valores se extraen de una distibucion gaussiana o normal)
        self._initialize_weights( m = X.shape[1]) # Extrae muestras aleatorias de una distribución normal (gaussiana)

        self.errors = []
        for _ in range(self.n_iter) :

          # Llamada al metodo que incorpora el algoritmo del perceptron simple
          self.weights = self._itera( X= X, y = y , w = self.weights)

        # plot errors vs iters
        if self.plot_info:
            self._plot_errors(n_iter = self.n_iter, errors = self.errors)

        return self


    def _initialize_weights(self, m : int) -> None:
        """
        Initialize weights to small random numbers (from a gaussian probabilistic ditribution)

        Parametros
        ----------
            - m: int
                tamaño del vector de pesos (sin tener en cuenta el termino "bias")
        Retorna
        -------
            - float o np.ndarray (si X es un matriz con todos los samples): W.T * X + b ó W*.T * X*
        """
        self.rgen = np.random.RandomState(seed = self.seed) # Generador de numeros aleatorios: https://numpy.org/doc/1.14/reference/generated/numpy.random.RandomState.html
        self.weights = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1 + m) # Extrae muestras aleatorias de una distribución normal (gaussiana)
        self.w_initialized = True

    def _itera(self,X : np.ndarray, y : np.ndarray, w : np.ndarray ) -> np.ndarray:
        """
        metodo interno que para cada sample point hace el producto escalar con los pesos, evalua si supera cierto umbral, calcula la  y_pred
        y actualiza los pesos y sesgo si esta erronea. Tambien lleva un conteo de los errores por iteracion

        Parametros
        ----------
            - X
            - y
            - w (se inicializan en el metodo fit antes de la primera iteracion)
        Retorna
        -------
            - w (actualizados despues de un "pasada" por todos los sample points)
        """
        errors = 0
        for i in range(X.shape[0]):

                y_pred = self.predict(X= X[i,:])
                #print("y_pred : ",y_pred)

                # actualizacion de los pesos solo si la prediccion es erronea:
                if y[i] - y_pred != 0:

                    # Conteo de errores / actualizacion pesos
                    errors += 1

                    # Termino comun para actualizacion de pesos y sesgo:
                    delta = self.alpha * (y[i] - y_pred)

                    # Actualizacion de sesgo
                    #print("b_old = ",w[0])
                    w[0] = w[0] + delta
                    #print("b new = ",w[0])

                    # Actualizacion de pesos
                    #print("w_old = ",w[1:])
                    w[1:] = w[1:] + ( delta * X[i,:] )
                    #print("w_old = ",w[1:])

                    # Recursividad (por cada actualizacion de pesos se debe volver a iterar desde el primer punto mediante la recursividad)
                    # Activar la recursividad ralentiza considerablemente el algoritmo (aun desconoxco si mejora su precision)
                    #self.itera(X = X, y = y,   w  = w )
                    #print("w tras recursividad = ",w)
        self.errors.append(errors)
        return w

    def net_input(self,X : np.ndarray) -> float:
        """
        """
        return  np.dot(X ,self.weights[1:]) + self.weights[0] # z = w0 (o b) + w1 * x1 + w2 *x2 ..

    def _escalon_activation(self, z : np.ndarray) -> np.ndarray:
        """
        Funcion umbral o funcion escalon modificada
        """
        return np.where(z >= 0.0, 1, -1)

    def _sigmoid_activation(self, z : np.ndarray) -> np.ndarray:
        """
        """
        return 1 / ( 1+ np.exp(-z) )

    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        Predice una serie de sample points

        Parametros
        ----------
            - X (np.ndarray de samples a prdecir)
        Retorna
        -------
            - y_pred (np.ndarray valor de la prediccion)
        """
        if self.sigmoid == True:

          return self._sigmoid_activation(z = self.net_input(X))

        else:

          return self._escalon_activation(z = self.net_input(X))


    def _plot_errors(self, n_iter: int = None, errors : List[int] = []) -> None:
        """
        Plotting de los errores en cada iteracion

        Parametros
        ----------
            - X (np.ndarray de samples a prdecir)
        Retorna
        -------
            - y_pred (np.ndarray valor de la prediccion
        """
        # Representamos visualmente los errores en cada época de entrenamiento
        fig = go.Figure()
        fig.add_trace(
                        go.Scatter(
                                        x=list(range(1, n_iter + 1)),
                                        y = errors,
                                        mode="lines+markers",
                                        marker = dict(size=8),
                                        line = dict(color="red")
                                    )
                     )
        fig.update_layout(
                            xaxis_title = "Época",
                            yaxis_title = "Número de actualizaciones",
                            margin=dict(l=40, r=40, b=40, t=40),
                            title = f"Perceptron Simple - learning rate : {self.alpha}",
                            width=900,
                            height = 700,

        )
        fig.show()


class PerceptronClase:
    """
    Clase propia que establece el algoritmo de Adaline

    Atributos
    ---------
        - eta : learning rate
        - n_iter : numero de iteraciones
        - seed : numero de semilla inicial para generar los pesos aleatorios
        - weights

    Propiedades (decorador @property)
    ---------------------------------
        None

    Metodos
    -------
        - __init__ [built-in method]. Constructor
        - fit

    """
    def __init__(self, eta : float = 0.1, n_iter : int = 50, seed : int = 1, plot_info : bool = True ) -> None:

        """
        Metodo (built-in) que inicializa la clase. Constructor

        Parametros de la funcion (se pasan como argumentos en la instancia de la clase)
        ----------
            - eta : learning rate
            - n_iter : numero de iteraciones
            - seed : numero de semilla inicial para generar los pesos aleatorios

        Retorna
        -------
            None

        """
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed
        self.plot_info = plot_info

    @execution_time
    def fit(self, X : np.ndarray, y : np.ndarray) -> object:
        """
        Docstring

        Parametros
        ----------
            - X
            - y
        Retorna
        -------
            - self

        """

         # Inicializacion de los pesos a floats muy pequeños (cuyos valores se extraen de una distibucion gaussiana o normal)
        self._initialize_weights( m = X.shape[1]) # Extrae muestras aleatorias de una distribución normal (gaussiana)

        self.errors = []

        for _ in range(self.n_iter): # si en python no se utiliza una variable se pone como caracter barra baja "_"
            errors = 0
            for xi,yi in zip(X,y):
                updated = self.eta * (yi - self.predict(xi))
                if updated != 0:
                  self.weights[0] = self.weights[0] + updated # actualizacion de b
                  self.weights[1:] = self.weights[1:] + updated * xi  # actualizacion de weights
                errors += int(updated != 0.0)
            self.errors.append(errors) # numero de errores por iteracion

        # plot errors vs iters
        if self.plot_info:
            self.plot_errors(n_iter = self.n_iter, errors = self.errors)

        return self

    def _initialize_weights(self, m : int) -> None:
        """
        Initialize weights to small random numbers (from a gaussian probabilistic ditribution)

        Parametros
        ----------
            - m: int
                tamaño del vector de pesos (sin tener en cuenta el termino "bias")
        Retorna
        -------
            - float o np.ndarray (si X es un matriz con todos los samples): W.T * X + b ó W*.T * X*
        """
        self.rgen = np.random.RandomState(seed = self.seed) # Generador de numeros aleatorios: https://numpy.org/doc/1.14/reference/generated/numpy.random.RandomState.html
        self.weights = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1 + m) # Extrae muestras aleatorias de una distribución normal (gaussiana)
        self.w_initialized = True

    def net_input(self,X : np.ndarray) -> float:
        """
        """
        return  np.dot(X ,self.weights[1:]) + self.weights[0] # z = w0 (o b) + w1 * x1 + w2 *x2 ...

    def predict(self,X : np.ndarray)-> float:
        """
        """
        return np.where(self.net_input(X) >= 0 , 1, -1) # Funcion umbral o funcion escalon modificada

    def plot_errors(self, n_iter: int = None, errors : List[int] = []) -> None:
        """
        Plotting de los errores en cada iteracion

        Parametros
        ----------
            - X (np.ndarray de samples a prdecir)
        Retorna
        -------
            - y_pred (np.ndarray valor de la prediccion
        """
        # Representamos visualmente los errores en cada época de entrenamiento
        fig = go.Figure()
        fig.add_trace(
                        go.Scatter(
                                        x=list(range(1, n_iter + 1)),
                                        y = errors,
                                        mode="lines+markers",
                                        marker = dict(size=8),
                                        line = dict(color="red")
                                    )
                     )
        fig.update_layout(
                            xaxis_title = "Época",
                            yaxis_title = "Número de actualizaciones",
                            margin=dict(l=40, r=40, b=40, t=40),
                            width=900,
                            height = 700,

        )
        fig.show()


class Adaline:
    """
    Clase propia que establece el algoritmo del adaline

    Atributos
    ---------
        - eta : float
                learning rate
        - n_iter : int
                numero de iteraciones
        - seed : int
                numero de semilla inicial para generar los pesos aleatorios
        - weights: np.ndarray
        - cost : list
                lista de los valores para cada iteracion de la funcionm de coste
        - plot_info : Bool
                 plotear numero de actulaizaciones frente a numero de iteraciones)

    Propiedades (decorador @property)
    ---------------------------------
        None

    Metodos
    -------
        - __init__ [built-in method]. Constructor
        - fit

    """
    def __init__(self, eta : float = 0.1, n_iter : int = 50, seed : int = 1 ,plot_info : bool = True) -> None:

        """
        Metodo (built-in) que inicializa la clase. Constructor

        Parametros de la funcion (se pasan como argumentos en la instancia de la clase)
        ----------
            - eta : learning rate
            - n_iter : numero de iteraciones
            - seed : numero de semilla inicial para generar los pesos aleatorios

        Retorna
        -------
            None

        """
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed
        self.plot_info = plot_info

    def __str__(self) -> str:
        return "Adaline"


    @execution_time
    def fit(self, X : np.ndarray, y : np.ndarray) -> np.ndarray:
        """
        Docstring

        Parametros
        ----------
            - X
            - y
        Retorna
        -------
            - self

        """

        rgen = np.random.RandomState(seed = self.seed) # Generador de numeros aleatorios: https://numpy.org/doc/1.14/reference/generated/numpy.random.RandomState.html
        self.weights = rgen.normal(loc = 0.0 , scale = 0.01, size = X.shape[1] + 1 ) # Extrae muestras aleatorias de una distribución normal (gaussiana)

        print("X shape: ",X.shape)
        """
        print("y : ",y,y.shape)
        print("w : ",self.weights[1:],self.weights.shape)
        print("b : ",self.weights[0])
        """
        # Cost function values in each iteration (with diferent updated weights)
        self.cost = []

        for _ in range(self.n_iter): # si en python no se utiliza una variable se pone como caracter barra baja "_"

            # Gradient descent implementation
            net_input = self.net_input(X)
            #print("net_input shape : ", net_input.shape)
            activation_output = self.linear_activation(z = net_input)

            errors = y - activation_output
            #print("errors shape : ", errors.shape)

            # computing cost function value
            self.cost.append(self.cost_function(X,y))

            # clase:

            """
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
            """

            # Update weights
            self.weights[1:] = self.weights[1:] + self.eta * np.dot(X.T,errors)
            self.weights[0] = self.weights[0] + self.eta * np.sum(errors)

            # computing cost function value
            # self.cost.append(self.cost_function(X,y))



        if self.plot_info == True:
            self.plot_errors( n_iter = self.n_iter , errors = np.log10(self.cost))

        return self


    def net_input(self,X : np.ndarray) -> np.ndarray:
        """
        Metodo que calcula el valor de entrada de red (W.T * X + b ó W*.T * X*)

        Parametros
        ----------
            - X
        Retorna
        -------
            - float o np.ndarray (si X es un matriz con todos los samples): W.T * X + b ó W*.T * X*

        """
        return  np.dot(X ,self.weights[1:]) + self.weights[0] # z = w0 (o b) + w1 * x1 + w2 *x2 ...

    def cost_function(self, X : np.ndarray, y : np.ndarray) -> float:

        """
        Metodo que calcula el valor de la funciuon de coste: SSE

        Parametros
        ----------
            - X : features
            - y : target
        Retorna
        -------
            - float : valor de la funcion de coste SSE

        """
        return 0.5 * np.sum([( y[i] - self.linear_activation(self.net_input(X[i,:])) )**2 for i in range(X.shape[0])])

    def linear_activation(self, z : float = None) -> float:
        """
        Metodo que representa la funcion lineal de activacion : f(z) = z

        Parametros
        ----------
            - z : argumento de la funcion de activacion lineal
        Retorna
        -------
            - z : salida de la funcion de activacion lineal

        """
        return z

    def predict(self,X : np.ndarray)-> np.ndarray:
        """
        """
        return np.where(self.linear_activation(self.net_input(X)) >= 0 , 1, -1) # Funcion umbral o funcion escalon modificada

    def plot_errors(self, n_iter: int = None, errors : List[int] = []) -> None:
        """
        Plotting de los errores en cada iteracion

        Parametros
        ----------
            - X (np.ndarray de samples a prdecir)
        Retorna
        -------
            - y_pred (np.ndarray valor de la prediccion
        """
        # Representamos visualmente los errores en cada época de entrenamiento
        fig = go.Figure()
        fig.add_trace(
                        go.Scatter(
                                        x=list(range(1, n_iter + 1)),
                                        y = errors,
                                        mode="lines+markers",
                                        marker = dict(size=8),
                                        line = dict(color="red")
                                    )
                        )
        fig.update_layout(
                            xaxis_title = "Época",
                            yaxis_title = "Cost function",
                            margin=dict(l=30, r=30, b=30, t=30),
                            title = f"{self} - learning rate : {self.eta}",
                            width=900,
                            height = 700,

        )
        fig.show()


class AdalineSGD:
    """
    Clase propia que establece el algoritmo del adaline

    Atributos
    ---------
        - eta : float
                learning rate
        - n_iter : int
                numero de iteraciones
        - seed : int
                numero de semilla inicial para generar los pesos aleatorios
        - weights: np.ndarray
        - cost : list
                lista de los valores para cada iteracion de la funcionm de coste
        - plot_info : Bool
                 plotear numero de actulaizaciones frente a numero de iteraciones)
        - shuffle
        - w_initialized

    Propiedades (decorador @property)
    ---------------------------------
        None

    Metodos
    -------
        - __init__ [built-in method]. Constructor
        - fit

    """
    def __init__(self, eta : float = 0.1, n_iter : int = 50, seed : int = 1 ,plot_info : bool = True, shuffle : bool = True) -> None:

        """
        Metodo (built-in) que inicializa la clase. Constructor

        Parametros de la funcion (se pasan como argumentos en la instancia de la clase)
        ----------
            - eta : learning rate
            - n_iter : numero de iteraciones
            - seed : numero de semilla inicial para generar los pesos aleatorios

        Retorna
        -------
            None

        """
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed
        self.plot_info = plot_info
        self.shuffle = shuffle
        self.w_initialized = False # Al instanciar el objeto, no tiene inicializados los pesos

    def __str__(self) -> str:
        return "AdalineSGD"

    @execution_time
    def fit(self, X : np.ndarray, y : np.ndarray) -> np.ndarray:
        """
        Docstring

        Parametros
        ----------
            - X
            - y
        Retorna
        -------
            - None

        """

        # Inicializacion de los pesos a floats muy pequeños (cuyos valores se extraen de una distibucion gaussiana o normal)
        self._initialize_weights( m = X.shape[1]) # Extrae muestras aleatorias de una distribución normal (gaussiana)

        print("X shape: ",X.shape)

        # Cost function values in each iteration (with diferent updated weights)
        self.cost = []

        for _ in range(self.n_iter): # si en python no se utiliza una variable se pone como caracter barra baja "_"

            # Shuffle in each iteration of the training dataset (avoid cycles)
            if self.shuffle:
                X , y = self._shuffle(X,y)

            # cost initialization
            cost = 0

            # Stochastic gradient descent
            for xi,yi in zip(X,y):

                # Updating weights | sumatorio de funcion coste de cada instancia/sample point (despues se hace la media como funcionn de coste global por iter)
                cost += self._update_weights(xi = xi, yi = yi)

            # computing cost function value for each iteration
            self.cost.append(cost / (X.shape[0]))

        if self.plot_info == True:
            self.plot_errors( n_iter = self.n_iter , errors = self.cost)
        return self

    def _initialize_weights(self, m : int) -> None:
        """
        Initialize weights to small random numbers (from a gaussian probabilitic ditribution)

        Parametros
        ----------
            - m: int
                tamaño del vector de pesos (sin tener en cuenta el termino "bias")
        Retorna
        -------
            - float o np.ndarray (si X es un matriz con todos los samples): W.T * X + b ó W*.T * X*
        """
        self.rgen = np.random.RandomState(seed = self.seed) # Generador de numeros aleatorios: https://numpy.org/doc/1.14/reference/generated/numpy.random.RandomState.html
        self.weights = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1 + m) # Extrae muestras aleatorias de una distribución normal (gaussiana)
        self.w_initialized = True

    def _shuffle(self, X, y) -> np.ndarray:
        """Shuffle training data"""
        r = self.rgen.permutation(len(y)) # permutacion del training set para evitar ciclos durante el fit y para cada iteracion
        return X[r], y[r]

    def _update_weights(self, xi : np.array , yi : int) -> None:

        """
        Call linear activation function(net input) method and update weights for each instance or sample (stochastic gradient descent)

        Parametros
        ----------
            - xi: np.array
                sample or instance data point
            - yi: int
                sample or instance target class real value

        Retorna
        -------
            - cost : float
                funcion de coste "para cada instancia"
        """

        # Output of activation function
        activation_output = self.linear_activation(z =  self.net_input(xi))

        # Error
        error = yi - activation_output

        # Updating weights
        self.weights[1:] = self.weights[1:] + self.eta * np.dot(xi,error)
        self.weights[0] = self.weights[0] + self.eta * error

        # computing cost for each instance weight update
        cost = 0.5 * error**2

        return cost

    def net_input(self,X : np.ndarray) -> np.ndarray:
        """
        Metodo que calcula el valor de entrada de red (W.T * X + b ó W*.T * X*)

        Parametros
        ----------
            - X
        Retorna
        -------
            - float o np.ndarray (si X es un matriz con todos los samples): W.T * X + b ó W*.T * X*

        """
        return  np.dot(X ,self.weights[1:]) + self.weights[0] # z = w0 (o b) + w1 * x1 + w2 *x2 ...

    def linear_activation(self, z : float = None) -> float:
        """
        Metodo que representa la funcion lineal de activacion : f(z) = z

        Parametros
        ----------
            - z : argumento de la funcion de activacion lineal
        Retorna
        -------
            - z : salida de la funcion de activacion lineal

        """
        return z

    def cost_function(self, X : np.ndarray, y : np.ndarray) -> float:

        """
        Metodo que calcula el valor de la funciuon de coste: SSE

        Parametros
        ----------
            - X : features
            - y : target
        Retorna
        -------
            - float : valor de la funcion de coste SSE

        """
        return 0.5 * np.sum([( y[i] - self.linear_activation(self.net_input(X[i,:])) )**2 for i in range(X.shape[0])])

    def predict(self,X : np.ndarray)-> np.ndarray:
        """Return class label after unit step"""
        return np.where(self.linear_activation(self.net_input(X)) >= 0 , 1, -1) # Funcion umbral o funcion escalon modificada

    def plot_errors(self, n_iter: int = None, errors : List[int] = []) -> None:
        """
        Plotting de los errores en cada iteracion

        Parametros
        ----------
            - X (np.ndarray de samples a prdecir)
        Retorna
        -------
            - y_pred (np.ndarray valor de la prediccion
        """
        # Representamos visualmente los errores en cada época de entrenamiento
        fig = go.Figure()
        fig.add_trace(
                        go.Scatter(
                                        x=list(range(1, n_iter + 1)),
                                        y = errors,
                                        mode="lines+markers",
                                        marker = dict(size=8),
                                        line = dict(color="red")
                                    )
                        )
        fig.update_layout(
                            xaxis_title = "Época",
                            yaxis_title = "Cost function",
                            margin=dict(l=30, r=30, b=30, t=30),
                            title = f"{self} - - learning rate : {self.eta}",
                            width=900,
                            height = 700,

        )
        fig.show()



class PerceptronOvA:
    """
    Clase propia que establece el algoritmo del perceptron implementando la tecnica one versus all (clasificacion multiclase)

    Atributos
    ---------


    Propiedades (decorador @property)
    ---------------------------------
      - None

    Metodos
    -------
      - __init__ [built-in method]. Constructor
      - fit
      - _itera : metodo interno
      - preict
      - _plot_errors

    """
    def __init__(self, alpha : float = 0.1, n_iter : int = 50, seed : int = 1, plot_info : bool = True , sigmoid : bool = False) -> None:

        """
        Metodo (built-in) que inicializa la clase. Constructor

        """
        self.alpha = alpha
        self.n_iter = n_iter
        self.seed = seed
        self.plot_info = plot_info
        self.sigmoid = sigmoid

    @execution_time
    def fit(self, X : np.ndarray, y : np.ndarray) -> object:
        """
        Instancia tantos objetos perceptron simple como clases a predecir en y, los entrena y ...

        Parametros
        ----------
            - X
            - y
        Retorna
        -------
            - self

        """
        self._classes = np.unique(y)
        # Create new "y" targets (One versus all)
        y_news = [np.where(y == classi, 1 , -1) for classi in self._classes]

        # Create and train: Perceptrons fitted with new "y" targets
        self.perceptrons = [Perceptron(alpha = self.alpha, n_iter = self.n_iter, seed = self.seed , plot_info = self.plot_info).fit(X,y_i) for y_i in y_news]

        return self

    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        Predice una serie de sample points

        Parametros
        ----------
            - X (np.ndarray de samples a prdecir)
        Retorna
        -------
            - y_pred (np.ndarray valor de la prediccion)
        """

        # Prediccion clase
        predictions = np.array([perceptron_i.predict(X) for perceptron_i in self.perceptrons]).T

        # Asignamos la clase de mayor probabilidad como resultado
        y_pred = self._classes[np.argmax(predictions, axis=1)]

        return y_pred


class LogisticRegressionGD:
    """
    Clase propia que establece el algoritmo del LogisticRegression

    Atributos
    ---------
        - eta : float
                learning rate
        - n_iter : int
                   numero de iteraciones
        - seed : int
                 numero de semilla inicial para generar los pesos aleatorios
        - weights: np.ndarray
        - cost : list
                 valores de la funcionm de coste para cada iteracion
        - threshold: float
                     Probabilidad minima de que yi = 1 sabiendo que x = xi (en la practica: es el valor minimo de la distribucion de bernoulli)
        - w_initialized: bool
                         Indica si los pesos se han inicializado o no
        - plot_info : Bool
                      plotear numero de actulaizaciones frente a numero de iteraciones)
        - log_cost: Bool
                    Inidaca si se quiere plotear el logaritmo de la funcion de coste o no
        - lammbda : float
                    Controls the severity of regularization


    Propiedades (decorador @property)
    ---------------------------------
        - None

    Metodos
    -------
        - __init__ [built-in method]. Constructor
        - __str__[built-in method]
        - fit
        - _initialize_weights
        - net_input
        - _escalon_activation
        - _sigmoid_activation
        - cost_function


    """
    def __init__(self, eta : float = 0.1, n_iter : int = 50, seed : int = 1 ,plot_info : bool = True, threshold : float = 0.5, log_cost : bool = False, lammbda : float = 0.0) -> None:

        """
        Metodo (built-in) que inicializa la clase. Constructor

        Parametros de la funcion (se pasan como argumentos en la instancia de la clase)
        ----------
            - eta : learning rate
            - n_iter : numero de iteraciones
            - seed : numero de semilla inicial para generar los pesos aleatorios

        Retorna
        -------
            None

        """
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed
        self.plot_info = plot_info
        self.threshold = threshold
        self.log_cost = log_cost
        self.lammbda = lammbda
        self.w_initialized = False

    def __str__(self) -> str:
        return "Logistic Regression"


    @execution_time
    def fit(self, X : np.ndarray, y : np.ndarray) -> object:
        """
        Docstring

        Parametros
        ----------
            - X
            - y
        Retorna
        -------
            - self

        """
        # Inicializacion de pesos
        self._initialize_weights ( m = X.shape[1])

        # Cost function values in each iteration (with diferent updated weights)
        self.cost = []

        for _ in range(self.n_iter): # si en python no se utiliza una variable se pone como caracter barra baja "_"

            # Gradient descent implementation
            activation_output = self._sigmoid_activation(z = self.net_input(X))

            # Calculo del vector de errores
            errors = y - activation_output

            # computing Cost function value
            self.cost.append(self.cost_function(X,y))

            # Update weights (with bregularization L2)
            # Duda de forma 1 (creo que es la forma 1 dividiendo entre m) o forma 2
            # 1:
            self.weights[1:] = (self.weights[1:] * (1-((self.lammbda*self.eta)/X.shape[0]))) + self.eta * np.dot(X.T,errors)
            # 2:
            #self.weights[1:] = (self.weights[1:] * (1-(self.lammbda*self.eta)))+ self.eta * np.dot(X.T,errors)
            self.weights[0] = self.weights[0] + self.eta * np.sum(errors)

        if self.plot_info:
          if self.log_cost:
            self.plot_errors(errors = np.log10(self.cost))
          else:
            self.plot_errors( errors = self.cost)

        return self

    def _initialize_weights(self, m : int) -> None:
        """
        Initialize weights to small random numbers (from a gaussian probabilistic ditribution)

        Parametros
        ----------
            - m: int
                tamaño del vector de pesos (sin tener en cuenta el termino "bias")
        Retorna
        -------
            - float o np.ndarray (si X es un matriz con todos los samples): W.T * X + b ó W*.T * X*
        """
        self.rgen = np.random.RandomState(seed = self.seed) # Generador de numeros aleatorios: https://numpy.org/doc/1.14/reference/generated/numpy.random.RandomState.html
        self.weights = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1 + m) # Extrae muestras aleatorias de una distribución normal (gaussiana)
        self.w_initialized = True


    def net_input(self,X : np.ndarray) -> np.ndarray:
        """
        """
        return  np.dot(X ,self.weights[1:]) + self.weights[0] # z = w0 (o b) + w1 * x1 + w2 *x2 ..

    def _escalon_activation(self, z : np.ndarray) -> np.ndarray:
        """
        Funcion umbral o funcion escalon modificada.
        Convierte el problema a un problema de clasificacion binario (transforma el output de la sigmoid function [numero real [0,1] (una probabilidad) en 1 o 0)
        """
        return np.where(z >= self.threshold, 1, 0)

    def _sigmoid_activation(self, z : np.ndarray) -> np.ndarray:
        """
        """
        return 1. / ( 1. + np.exp(-z) )

    def cost_function(self, X : np.ndarray, y : np.ndarray) -> float:

        """
        Metodo que calcula el valor de la funciuon de coste para la regresion logistica

        Parametros
        ----------
            - X : features
            - y : target
        Retorna
        -------
            - float : valor de la funcion de coste

        """
        cost = 0
        for xi,yi in zip(X,y):
          sigmoid_output = self._sigmoid_activation(self.net_input(xi))
          cost += - ( (yi * np.log(sigmoid_output)) + ((1-yi) * np.log(1 - sigmoid_output)) )
        cost = cost / X.shape[0]
        
        # Añade termino de regularizacion a la funcion de coste
        cost += (self.lammbda *0.5* np.sum(self.weights**2))/X.shape[0]
        return cost

    def predict(self,X : np.ndarray)-> np.ndarray:
        """
        """
        return self._escalon_activation(self._sigmoid_activation(z = self.net_input(X = X)))

    def plot_errors(self, errors : np.array) -> None:
        """
        Plotting de los errores en cada iteracion

        Parametros
        ----------
            - X (np.ndarray de samples a prdecir)
        Retorna
        -------
            - y_pred (np.ndarray valor de la prediccion
        """
        # Representamos visualmente los errores en cada época de entrenamiento
        fig = go.Figure()
        fig.add_trace(
                        go.Scatter(
                                        x=list(range(1, self.n_iter + 1)),
                                        y = errors,
                                        mode="lines+markers",
                                        marker = dict(size=8),
                                        line = dict(color="red")
                                    )
                        )
        fig.update_layout(
                            xaxis_title = "Época",
                            yaxis_title = "Cost function",
                            margin=dict(l=30, r=30, b=30, t=50),
                            title = f"{self} - learning rate : {self.eta}",
                            width=900,
                            height = 700,

        )
        fig.show()
