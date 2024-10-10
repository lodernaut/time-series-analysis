# Previsão de séries temporais

import datetime
import os

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# ajusta o tamanho padrão das figuras(gráficos) gerados.
mpl.rcParams["figure.figsize"] = (8, 6)
# desativa a exibição da grade (grid) nos eixos dos gráficos.
mpl.rcParams["axes.grid"] = False

# previsões por hora, subamostrar os dados de intervalos de 10 minutos para intervalos de uma hora:
df = pd.read_csv("csv_path")
