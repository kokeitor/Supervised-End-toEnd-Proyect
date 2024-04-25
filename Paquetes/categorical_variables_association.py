from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd


def identify_associations(df: pd.DataFrame, categorical_variables: list):
    # Crear una matriz para almacenar p-valores y Cramér's V
    matrix_size = len(categorical_variables)
    result_matrix = np.zeros((matrix_size, matrix_size, 2))

    # Llenar la matriz con los p-valores y Cramér's V
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j:
                contingency_table = pd.crosstab(
                    df[categorical_variables[i]], df[categorical_variables[j]]
                )
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                num_obs = np.sum(contingency_table.values)
                cramers_v = np.sqrt(
                    chi2 / (num_obs * (min(contingency_table.shape) - 1))
                )
                result_matrix[i, j, 0] = p
                result_matrix[i, j, 1] = cramers_v

    # Crear un DataFrame a partir de la matriz
    p_values = pd.DataFrame(
        (result_matrix[:, :, 0]),
        columns=categorical_variables,
        index=categorical_variables,
    )
    cramers_values = pd.DataFrame(
        (result_matrix[:, :, 1]),
        columns=categorical_variables,
        index=categorical_variables,
    )

    # Mostrar el DataFrame
    print(f"p_values: \n{p_values}\n\n")
    print(f"cramers_values: \n{cramers_values}")
