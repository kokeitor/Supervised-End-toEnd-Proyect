import pandas as pd
import numpy as np
import plotly.graph_objs as go
from typing import  Callable, Optional, List, Dict, Tuple



def weights_vs_c(
                    X : np.ndarray , 
                    y: np.ndarray,
                    clasifier : object , 
                    C_values : np.ndarray = np.arange(0,1,0.1), 
                    features_names : Optional[List[str]] = None

                    )-> None:
    """
    Funcion que plotea los weights asociados a cada feature para cada valor del hiperparametro de 
    regularizacion inverso (C) de entre un rango de valores que se quiera. La funcion requiere un objeto predictor
    clasificador que admita el hiperparametro de regularizacion inversa C.

    Prameters
    ---------
        - X : np.ndarray
            features matrix

        - y: np.ndarray
            array with target classes
        - clasifier : object
                    predictor clasifier object of sklearn which implement fit method and C inverse hyperparameter of regularization
        - features_names: Optional[List[str]]
                        List of str to name the features weights in the plot
    ...

    Return 
    ------
        - None
    ...

    """
    weights, params_c = [], []

    for c in C_values:
        lr = clasifier(C=c, random_state=1)
        lr.fit(X, y)
        weights.append(lr.coef_[1])
        params_c.append(c)

    # Plotea la grafica weights_feature vs C hyperparameter
    weights = np.array(weights)
    fig = go.Figure()
    for c_i_weights in weights:
        for idx , w_i in enumerate(c_i_weights):
            if features_names != None:
                fig.add_trace(go.Scatter(
                                            x=params_c,
                                            y=weights[:, idx], 
                                            name=f'{features_names[idx]}', 
                                            mode = 'lines+markers', 
                                            line=dict(dash="dash")
                                        ))

            else:
                fig.add_trace(go.Scatter(x=params_c, y=weights[:, idx], line=dict(dash="dash")))


    fig.update_layout(
                        xaxis_type="log",
                        title = " Weights vs. C inverse regularization hyperparameter ",
                        xaxis = dict(
                                    title = "C",
                                    autorange = True,
                                    showline = True,
                                    showgrid = True,
                                    gridcolor = None,
                                    gridwidth = 0.05,
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
                                    title ="Peso (w)",
                                    autorange = True,
                                    showline = True,
                                    showgrid = True,
                                    showticklabels = True,
                                    gridcolor = None,
                                    gridwidth = 0.05,
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
                        showlegend = False,
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

"""
Example of use:
weights_vs_c(
                    X = X_combined_std, 
                    y = y_combined,
                    clasifier = LogisticRegression, 
                    C_values = np.exp((np.arange(-5, 5))), 
                    features_names = ["feature 1","feature 2"]
                    )
"""
