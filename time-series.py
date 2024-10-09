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
