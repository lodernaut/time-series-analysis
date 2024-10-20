# Previsão de séries temporais
import datetime
import os
import webbrowser

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as snsv

# Suprimindo avisos sobre GPU do TensorFlow "não recomendável"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desativa o uso de GPUs pelo TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0 = Todos logs, 1 = Remover INFO, 2 = Remover avisos, 3 = Remover todos, exceto erros
)
import tensorflow as tf

# Garante que a pasta "figures/" exista
os.makedirs("figures", exist_ok=True)

# ajusta o tamanho padrão das figuras(gráficos) gerados.
mpl.rcParams["figure.figsize"] = (8, 6)
# desativa a exibição da grade (grid) nos eixos dos gráficos.
mpl.rcParams["axes.grid"] = False

df = pd.read_csv("jena_climate_2009_2016.csv")
# previsões por hora, submete os dados de intervalos de 10 minutos para intervalos de uma hora(60):
# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")

# print(df.head())

plot_cols = ["T (degC)", "p (mbar)", "rho (g/m**3)"]
plot_features = df[plot_cols]
# Define a coluna de datas e horas como índice do DataFrame
plot_features = plot_features.set_index(date_time)
_ = plot_features.plot(subplots=True)

plt.savefig("figures/graphic1.png")
# webbrowser.open("figures/graphic1.png")
plt.close()
print("Completed")

# Para o segundo gráfico, com os primeiros 480 dados:
plot_features = df[plot_cols][:480]
plot_features = plot_features.set_index(
    date_time[:480]
)  # Define o índice para os primeiros 480
_ = plot_features.plot(subplots=True)

plt.savefig("figures/graphic2.png")
# webbrowser.open("figures/graphic2.png")
plt.close()
print("Completed")

# Estatísticas do conjunto de dados:
static_data = df.describe().transpose()
# print(static_data)

wv = df["wv (m/s)"]
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df["max. wv (m/s)"]
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame.
df["wv (m/s)"].min()  # retorna o menor valor presente na coluna "wv (m/s)"


static_data = df.describe().transpose()
print(static_data)

# histograma bidimensional para visualizar a distribuição conjunta entre as duas variáveis
# bins=(50, 50) indica que o gráfico será dividido em uma grade de 50 x 50 células.
# vmax=400 define o valor máximo de contagem nas células para limitar a intensidade das cores no gráfico.
# plt.colorbar(): Adiciona uma barra de cores ao lado do gráfico para indicar a escala de contagem de valores (quantas vezes uma determinada combinação de direção e velocidade do vento ocorre).
plt.hist2d(df["wd (deg)"], df["wv (m/s)"], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel("Wind Direction (deg)")
plt.ylabel("Wind Velocity (m/s)")

plt.savefig("figures/graphic3.png")
# webbrowser.open("figures/graphic3.png")
plt.close()
print("Completed")

# Convertendo as colunas de direção e velocidade do vento em um 'vetor' de vento.
wv = df["wv (m/s)"]
max_wv = df.pop("max. wv (m/s)")

# Convert to radians. // conversão é necessária para usar funções trigonométricas abaixo, seno e cosseno.
wd_rad = df.pop("wd (deg)") * np.pi / 180

# Calculate the wind x and y components.
# Wx componente horizontal do vetor de vento
# Wy componente vertical do vetor de vento
# Wx e Wy representam como a velocidade do vento se distribui nos eixos X e Y
df["Wx"] = wv * np.cos(wd_rad)
df["Wy"] = wv * np.sin(wd_rad)

# Calculate the max wind x and y components.
plt.hist2d(df["Wx"], df["Wy"], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel("Wind X (m/s)")
plt.ylabel("Wind Y (m/s)")
ax = plt.gca()
ax.axis("tight")

plt.savefig("figures/graphic4.png")
# webbrowser.open("figures/graphic4.png")
plt.close()
print("Completed")

# Trecho abaixo dá ao modelo acesso aos recursos de frequência mais importantes
# Poderá determinar quais frequências são importantes extraindo recursos com Fast Fourier Transform.
timestamp_s = date_time.map(pd.Timestamp.timestamp)  # Conversão do tempo em timestamps

day = 24 * 60 * 60
year = (365.2425) * day

# Criação de colunas para capturar ciclos diários e anuais (transformações trigonométricas)

# representar o tempo como sinais periódicos usando funções trigonométricas (seno e cosseno), o que facilita a captura de padrões cíclicos nos dados de séries temporais.
df["Day sin"] = np.sin(
    timestamp_s * (2 * np.pi / day)
)  # Representa a variação cíclica do tempo durante um dia, usando a função seno. Isso transforma o tempo em um ciclo que se repete a cada 24 horas.
df["Day cos"] = np.cos(
    timestamp_s * (2 * np.pi / day)
)  # Representa a mesma variação cíclica, mas com um deslocamento de fase de 90 graus, usando a função cosseno.
df["Year sin"] = np.sin(
    timestamp_s * (2 * np.pi / year)
)  # Representa a variação cíclica durante o ano, capturando padrões sazonais (como mudanças climáticas ao longo das estações) com a função seno.
df["Year cos"] = np.cos(
    timestamp_s * (2 * np.pi / year)
)  # A mesma variação cíclica, mas com a função cosseno.

# Essas transformações são úteis porque o tempo, em vez de ser tratado como uma sequência linear (o que não capturaria bem padrões sazonais ou diários), agora é representado como um ciclo contínuo.
# Ajuda o modelo a entender que o tempo "se repete" de forma previsível, tanto diariamente quanto anualmente.

# Visualização dos sinais do ciclo diário:
plt.plot(np.array(df["Day sin"])[:25])  # Plotam os primeiros 25 valores das colunas
plt.plot(np.array(df["Day cos"])[:25])
plt.xlabel("Time [h]")
plt.title("Time of day signal")

plt.savefig("figures/graphic5.png")
webbrowser.open("figures/graphic5.png")
plt.close()
print("Completed")
