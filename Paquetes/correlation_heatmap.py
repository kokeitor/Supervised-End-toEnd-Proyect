import pandas as pd
import matplotlib.pyplot as plt


# Finds correlation between variables and creates a correlation heatmap.
# Returns a correlation num of all variables in DataFrame
# Takes a DataFrame as a parameter
def corr_heatmap(df_numeric: pd.DataFrame):
    corr_num = df_numeric.corr()
    # Create heatmap
    plt.figure(figsize=(22, 10))
    sns.heatmap(corr_num, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()
