# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:13:38 2020

@author: laguila
"""

#https://python-graph-gallery.com

import joypy
from pywaffle import Waffle
#import calmap
import os
import numpy as np

import pandas as pd
from pandas.plotting import andrews_curves
from pandas.plotting import parallel_coordinates

from sklearn.cluster import AgglomerativeClustering

import seaborn as sns

#matplotlib and related imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Patch
import matplotlib.patches as patches

from scipy.spatial import ConvexHull
from scipy.signal import find_peaks
from scipy.stats import sem
import scipy.cluster.hierarchy as shc

import squarify
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.seasonal import seasonal_decompose

from dateutil.parser import parse

from IPython.display import Image

#import geopandas
#import folium
#from folium.plugins import TimeSliderChoropleth
#from branca.element import Template, MacroElement

os.chdir("D:/MachineLearning/Basics/Python")
def print_files():        
    for dirname, _, filenames in os.walk('datos'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            
            
#%% RESUMEN

#Grafica dividida
fig = plt.figure(figsize = (10, 10), dpi= 80)
gs = fig.add_gridspec(5, 5)  #dividir figura en 5x5
ax1 = fig.add_subplot(gs[:4, :-1]) #plotear en 0-4 x 0-4
ax1.plot([1,2,3,])
ax2 = fig.add_subplot(gs[4:5, -1:]) #plotear en 0-4 x 0-4
ax2.plot([1,2,3])

#Grafica normal
fig = plt.figure(figsize = (15, 5))  #instanciar la figura
ax = fig.add_subplot()  #una sola grafica
ax.plot([1,2,3,])

#sub graficas
fig = plt.figure(figsize = (15, 5))  #instanciar la figura
(ax1, ax2), (ax3, ax4) = fig.subplots(2,2) #varias graficas

#subplots abreviado
fig, axes = plt.subplots(2, 1) #figura y subplots a la vez
axes[0].plot(np.arange(10))
axes[1].plot(np.arange(10))

#Figura y 3 graficas
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12, 6))

#grafica actual (al usar con otras librerias)
ax = plt.gca() #acceder a la grafica actual

#Añadir grafica
fig = plt.figure(figsize = (15, 5))  #instanciar la figura
ax = fig.add_subplot()  #una sola grafica
ax.plot([1,2,3,])
ax2 = fig.add_axes([0.2, 0.2, 0.5, 0.5])
ax2.plot([3,2,1,])

#Empezar a añadir subplots
plt.figure(figsize=(16, 35))
plt.subplot(6, 2, 1)

#En Seaborn
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time")
g.map(sns.histplot, "tip")

#%% SEABORN
'''
relplot: 
    scatterplot, lineplot
catplot: 
    categorical scatterplots: stripplot, swarmplot
    categorical distributions: boxplot, violinplot, boxenplot
    categorical estimate: pointplot, barplot, countplot
'''

#%% Plots-----------------------------------------------------------------------
#%%Linea
fig = plt.figure(figsize = (15, 5))  #instanciar la figura
ax = fig.add_subplot()  #una sola grafica
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
S = np.cos(X)
ax.plot(X,S, color = "red", alpha = 0.5, lw = 3, label = "Sine") #representar puntos en X,S


flights = sns.load_dataset("flights")
sns.relplot(data=flights, x="year", y="passengers", hue="month", kind="line")

flights_wide = flights.pivot(index="year", columns="month", values="passengers")
sns.relplot(data=flights_wide, kind="line")

#%%Lineas  discontinuas
x = np.linspace(0, 10, 500)
y = np.sin(x)

fig, ax = plt.subplots()
line1, = ax.plot(x, y, label='Using set_dashes()')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

#%% Linea de colores
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
x = np.linspace(0, 3 * np.pi, 500)
y = np.sin(x)
dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig = plt.figure(figsize = (15, 5))  #instanciar la figura
ax = fig.add_subplot()  #una sola grafica

# Create a continuous norm to map from data points to colors
lc = LineCollection(segments, cmap='viridis')
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)



#%%Mapa de calor waffle
values = {'suv': 62,
 'compact': 47,
 'midsize': 41,
 'subcompact': 35}
plt.figure(
    FigureClass = Waffle,
    rows = 7,
    columns = 34,
    values = values,
    legend = {'loc': 'upper left', 'bbox_to_anchor': (1, 1), "fontsize": "12"},
    figsize = (20, 7)
)

#%%Mapa de calor
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
X, y = load_iris(return_X_y=True)
X=pd.DataFrame(X)
cors=X.corr(method='pearson')

plt.matshow(cors, cmap=plt.cm.RdYlGn)
plt.colorbar()

import seaborn as sns
ax = sns.heatmap(cors, vmin=0, cmap = sns.cm.rocket_r)

#%% stackplot
import numpy as np
import matplotlib.pyplot as plt

year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
population_by_continent = {
    'africa': [228, 284, 365, 477, 631, 814, 1044, 1275],
    'americas': [340, 425, 519, 619, 727, 840, 943, 1006],
    'asia': [1394, 1686, 2120, 2625, 3202, 3714, 4169, 4560],
    'europe': [220, 253, 276, 295, 310, 303, 294, 293],
    'oceania': [12, 15, 19, 22, 26, 31, 36, 39],
}

fig, ax = plt.subplots()
ax.stackplot(year, population_by_continent.values(),
             labels=population_by_continent.keys())
plt.show()

#%%Parallel coordinates
PATH = "datos/diamonds_filter.csv"
df = pd.read_csv(PATH)
df.head()
fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot()
ax = parallel_coordinates(df, 'cut', colormap = "Dark2")

#%%Puntos
fig = plt.figure(figsize = (15, 5))  #instanciar la figura
ax = fig.add_subplot()  #una sola grafica
tips = sns.load_dataset("tips")
plt.scatter(tips.total_bill, tips.tip)
sns.relplot(tips.total_bill, tips.tip, hue = tips.smoker)
sns.scatterplot(tips.total_bill, tips.tip, hue = tips.smoker)

#----categorical
tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", data=tips)
sns.catplot(x="day", y="total_bill", kind="swarm", data=tips)

#%% puntos personalizados 
x = np.random.rand(10)
y = np.random.rand(10)
z = np.sqrt(x**2 + y**2)

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)

# marker symbol
axs[0, 0].scatter(x, y, s=80, c=z, marker=">")
axs[0, 0].set_title("marker='>'")

# marker from TeX
axs[0, 1].scatter(x, y, s=80, c=z, marker=r'$\alpha$')
axs[0, 1].set_title(r"marker=r'\$\alpha\$'")

# marker from path
verts = [[-1, -1], [1, -1], [1, 1], [-1, -1]]
axs[0, 2].scatter(x, y, s=80, c=z, marker=verts)
axs[0, 2].set_title("marker=verts")

# regular polygon marker
axs[1, 0].scatter(x, y, s=80, c=z, marker=(5, 0))
axs[1, 0].set_title("marker=(5, 0)")

# regular star marker
axs[1, 1].scatter(x, y, s=80, c=z, marker=(5, 1))
axs[1, 1].set_title("marker=(5, 1)")

# regular asterisk marker
axs[1, 2].scatter(x, y, s=80, c=z, marker=r'$\clubsuit$')
axs[1, 2].set_title("marker=(5, 2)")

plt.tight_layout()
plt.show()

#%%Event collection
from matplotlib.collections import EventCollection
xdata = np.random.random([2, 10])

# split the data into two parts
xdata1 = xdata[0, :]
xdata2 = xdata[1, :]

# sort the data so it makes clean curves
xdata1.sort()
xdata2.sort()

# create some y data points
ydata1 = xdata1 ** 2
ydata2 = 1 - xdata2 ** 3

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xdata1, ydata1, color='tab:blue')

# create the events marking the x data points
xevents1 = EventCollection(xdata1, color='tab:blue', linelength=0.05)

# create the events marking the y data points
yevents1 = EventCollection(ydata1, color='tab:blue', linelength=0.05,
                           orientation='vertical')

ax.add_collection(xevents1)
ax.add_collection(yevents1)

#%% Polygon
fig, ax = plt.subplots()
ax.fill("time", "signal",
        data={"time": [0, 1, 2], "signal": [0, 1, 0]})

#%% Errorbar

fig = plt.figure()
x = np.arange(10)
y = 2.5 * np.sin(x / 20 * np.pi)
yerr = np.linspace(0.05, 0.2, 10)

plt.errorbar(x, y + 3, yerr=yerr, uplims=True, lolims=False, label='both limits (default)')

#%%Histograma
PATH = 'datos/mpg_ggplot2.csv'
df = pd.read_csv(PATH)
df.head()
fig = plt.figure(figsize = (15, 5))  #instanciar la figura
ax = fig.add_subplot()  #una sola grafica
ax.hist(df["displ"], 40, orientation = 'vertical')

sns.histplot(x=df["displ"], hue=df["cyl"], multiple="stack")
sns.displot(x=df["displ"], hue=df["cyl"], multiple="stack") #histplot por detras

#%% Bivariate distributions
penguins = sns.load_dataset("penguins")
sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind='kde', hue = "species")
sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm") #variable continua
sns.displot(penguins, x="bill_length_mm", y="island") #variable continua

#%%Grafico de barras
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)
d = df["manufacturer"].value_counts() #tiene que ser un diccionario o serie temporal, con su indice
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot()
ax.bar(d.index, d.values, color = "red") #color = "red" if y_ < y.mean() else "green"

sns.catplot(x="cyl", y = "manufacturer", data = df, kind = 'bar')
sns.catplot(x="cyl", y = "manufacturer", data = df, kind = 'point')
sns.catplot(y = "manufacturer", data = df, kind='count')

labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 35, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
men_std = [2, 3, 4, 1, 2]
women_std = [3, 5, 2, 3, 3]
width = 0.35       # the width of the bars: can also be len(x) sequence

#Apiladas
fig, ax = plt.subplots()
ax.bar(labels, men_means, width, yerr=men_std, label='Men')
ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,label='Women')

#Contiguas
x = np.arange(len(labels))
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

#%%Barras hortizontales
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)
y = df.manufacturer.value_counts()
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot()
ax.barh(y.index, y.values)

#%% Barras horizontales rotas
fig, ax = plt.subplots()
ax.broken_barh([(110, 30), (150, 10)], (10, 9), facecolors='tab:blue')
ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9),
               facecolors=('tab:orange', 'tab:green', 'tab:red'))

#%%BoxPlot
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot()
ax.boxplot(df.hwy, vert = True, whis = 0.75) # make the boxplot lines shorter


plt.figure(figsize = (10, 10), dpi = 80)
ax = sns.boxplot(x = "class", y = "hwy", data = df) #catplot del tipo boxplot
ax = sns.catplot(x = "class", y = "hwy", data = df, kind='boxen')



#%%Pie chart
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)
d = df["manufacturer"].value_counts()
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot()
ax.pie(d.values, # pass the values from our dictionary
       labels = d.index, # pass the labels from our dictonary
       autopct = '%1.1f%%', # specify the format to be plotted
       textprops = {'fontsize': 10, 'color' : "white"} # change the font size and the color of the numbers inside the pie
      )

#%%Andrews curves
PATH = "datos/diamonds_filter.csv"
df = pd.read_csv(PATH)
andrews_curves(df, 'cut', colormap = 'Set1')

#%%Correlaciones
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot()
ax.imshow(df[["displ", "cty", "hwy"]].corr(), cmap = 'viridis', interpolation = 'nearest') 
   
corr=df[["displ", "cty", "hwy"]].corr()
corr.style.background_gradient(cmap='viridis')

#%%Horizozontal lines
PATH = "datos/mtcars.csv"
df = pd.read_csv(PATH)
df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()
# df.sort_values("x_plot", inplace = True)
# df.reset_index(inplace = True)
colors = ["red" if x < 0 else "green" for x in df["x_plot"]]
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot()
ax.hlines(y = df.index, xmin = 0 , xmax = df["x_plot"],  color = colors, linewidth = 5)

#%%Anotaciones
fig = plt.figure(figsize = (15, 5))  #instanciar la figura
ax = fig.add_subplot()  #una sola grafica
X = np.linspace(-np.pi, np.pi, 25, endpoint=True)
S = np.cos(X)
plt.scatter(X, S)
ax.text(0, #posicion en x
        0.5, #posicion en y
        "Annotation1", #texto
        color = "red",  
        horizontalalignment='right', 
        size = 10)

ax.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
             xy = (0, 0), 
             xycoords = 'data',
             fontsize = 16)

ax2 = fig.add_axes([0.15, 0.15, 0.15, 0.15])
ax2.plot([1,2,1], color = "pink") 

#%%Regresiones
PATH = 'datos/mpg_ggplot2.csv'
df = pd.read_csv(PATH)
df.head()
sns.lmplot("displ", "hwy", df, hue = "cyl") #robust = True para outliers, logistic = True, para logistica
sns.lmplot("displ", "hwy", df, hue = "cyl", col = "cyl") # by specifying the col, seaborn creates several axes for each group

sns.lmplot("displ", "hwy", df, order=2)

#%%Densidad
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)
fig = plt.figure(figsize = (10, 8)) 
sns.kdeplot(df["cty"], shade=True)
sns.displot(df["cty"], kind = "kde") #histplot por detras

#%%Jittering
PATH = 'datos/mpg_ggplot2.csv'
df = pd.read_csv(PATH)
plt.figure(figsize = (10, 7))
ax = sns.stripplot(df["cty"], df["hwy"])

#%%PairPlot
df = sns.load_dataset('iris')
sns.pairplot(df, 
             kind = "reg", # optional: make a regression line for eac hue and each variables
             hue = "species"
            );
#---
enguins = sns.load_dataset("penguins")
g = sns.PairGrid(penguins)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)



#%% Jointplot
penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species", kind="hist")

g = sns.JointGrid(data=penguins, x="bill_length_mm", y="bill_depth_mm")
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot)

#%%Violin
tips = sns.load_dataset("tips")
g = sns.catplot(x="day", y="total_bill", kind="violin", inner=None, data=tips, split=True)
sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax)

#%%square plot
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)
label_value = df["class"].value_counts()
plt.figure(figsize = (20, 10))
squarify.plot(sizes = label_value.values, label = label_value.index, alpha = 0.8)


#%%Dendograma
PATH = "datos/USArrests.csv"
df = pd.read_csv(PATH)
fig = plt.figure(figsize = (10, 7))
dend = shc.dendrogram(shc.linkage(df[['Murder', 'Assault', 'UrbanPop', 'Rape']], method = 'ward'), 
                      labels = df["State"].values, 
                      color_threshold = 100)


#%%Grafico acumulado
PATH = "datos/nightvisitors.csv"
df = pd.read_csv(PATH)
df.set_index("yearmon", inplace = True)
y = [df[col].values for col in df.columns]
fig = plt.figure(figsize = (14, 10))
ax = fig.add_subplot()
ax.stackplot(df.index, y, labels = df.columns)

#%%Rellenar entre lineas
x=[1, 2, 3, 4, 5, 6, 7]
y=[-2, -1, 0, 1, 2, 3, 4]
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x, y)
ax.fill_between(x, 0, y, facecolor='green', interpolate = True, alpha = 0.3) #fill para todo x entre 0 e y


#%%Time series
PATH = 'datos/AirPassengers.csv'
df = pd.read_csv(PATH)
fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)
plot_acf(df["value"], ax = ax1, lags = 50)
plot_pacf(df["value"], ax = ax2, lags = 15)

df["date"] = pd.to_datetime(df["date"]) # convert to datetime
df.set_index("date", inplace = True)
df["date"] = df.index
df["month_name"] = df["date"].dt.month_name() # extracts month_name
df["month_name"] = df["month_name"].apply(lambda x: x[:3]) # passes from January to Jan
df["year"] = df["date"].dt.year # extracts year
df["new_date"] = df["month_name"].astype(str) + "-" +df["year"].astype(str) # Concatenaes Jan and year --> Jan-1949  
result = seasonal_decompose(df)
fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex = True, figsize = (12,10))
result.observed.plot(ax = axes[0], legend = False)
result.trend.plot(ax = axes[1], legend = False)
result.seasonal.plot(ax = axes[2], legend = False)
result.resid.plot(ax = axes[3], legend = False)

#%% contour
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)







#%%
#%%
#%%

#%%tema permanente
plt.rcParams.update({'font.size':25})
sns.set_theme()
sns.set_style("whitegrid") #darkgrid, whitegrid, dark, white, and ticks
print(plt.style.available)
plt.style.use('fivethirtyeight')

#tema temporal
with sns.axes_style("darkgrid"):
    print('plot')

#Customizar
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

#%%ticks
ax.xaxis.set_ticks_position('bottom')
ax.get_xticklabels().set_fontsize(16)
ax.get_xticklabels().set_bbox(dict(facecolor = 'white', edgecolor = 'None', alpha = 0.65 )) #ticks
ax.set_yticks(df.index)
ax.set_yticklabels(df.cars)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["a", "b", "c"], rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$']) # r before each string means raw string
ax.tick_params(axis = 'y', labelsize = 12, labelrotation = 90)

#%%Linea vertical
ax.axvline(x=-1.1, color='red')

#%% marcadores
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]

#%%ejes
ax.spines['bottom'].set_position(('data',0))
ax.spines['bottom'].set_color('black') # this helps change the color
ax.spines['bottom'].set_alpha(.3) # and adds some transparency to the spines
ax.invert_yaxis() #invertir ejes
ax.axison = False #quitar ejes
ax2 = ax.twinx() #eje secundario ax2, en la figura de ax1 (no se añade otra figura)
ax.set_facecolor("red") #color de fondo 
ax.grid(linestyle='--', alpha=0.5) #grid

#%%limites
axes.set(xlim = (0.5, 7.5), ylim = (0, 50)) #limite de los ejes
plt.xlim(X.min()*1.5, X.max()*1.5) #limites eje x


#%%colores

cmap = plt.get_cmap("tab20c")
inner_colors = cmap(np.arange(10)) #6 colores de esa paleta
plt.bar(np.arange(10), np.random.randint(0, 10, 10), color=inner_colors)

#--
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
#from colorspacious import cspace_converter
from collections import OrderedDict
cmaps = OrderedDict()
cmaps['Perceptually Uniform Sequential'] = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
cmaps['Sequential'] = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps['Sequential (2)'] = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink','spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia','hot', 'afmhot', 'gist_heat', 'copper']
cmaps['Diverging'] = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern','gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg','gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral','gist_ncar']

nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps.items())
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps.items():
    plot_color_gradients(cmap_category, cmap_list, nrows)

plt.show()

#%%Colores seaborn
sns.set_palette()
#CATEGORICAL: Colores por defecto de matplotlib. Variaciones:deep, muted, pastel, bright, dark, and colorblind
sns.color_palette('tab10')
sns.color_palette("hls", 8) #8 colores
sns.color_palette("husl", 8)
sns.color_palette("Set2")
sns.color_palette("Paired")
#SEQUENTIAL: "rocket", "mako", "flare", and "crest". Also "magma", "viridis". Añadir "_r" para reversed. Blues, "YlOrBr", "vlag", "icefire", "Spectral", "coolwarm"
sns.color_palette("rocket", as_cmap=True)

sns.color_palette("cubehelix", as_cmap=True)
sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True) #ch for cubehelix, pasando los parametros start y rot

sns.light_palette("seagreen", as_cmap=True)
sns.dark_palette("#69d", reverse=True, as_cmap=True)
sns.color_palette("light:b", as_cmap=True)
sns.color_palette("dark:salmon_r", as_cmap=True)
sns.color_palette("Blues", as_cmap=True)
sns.color_palette("YlOrBr", as_cmap=True)

sns.color_palette("vlag", as_cmap=True)
sns.color_palette("icefire", as_cmap=True)
sns.diverging_palette(145, 300, s=60, as_cmap=True)
sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

#%% Otros
plt.tight_layout()  #margenes entre graficas
plt.title("Sine and Cosine functions (Original by N. Rougier)") #titulo de la grafica
plt.suptitle("Scatter plot with regression lines on different axes", fontsize = 10)
plt.colorbar()
plt.legend(loc = "upper left", fontsize = 12) #leyenda

#%%






























     
#%% 
# ----------------------------------------------------------------------------------------------------
# get the data
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

#----------------------------------      FIGURE      -------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (15, 5))
ax = fig.add_subplot()

# ----------------------------------     AXES       -------------------------------------------------
# plot the data
ax.plot(X,S, color = "red", alpha = 0.5, lw = 3, label = "Sine")
ax.plot(X,C, color = "green", alpha = 0.5, lw = 3, label = "Cosine")

# removes the right and top spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# changes the position of the other spines
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.spines['bottom'].set_color('black') # this helps change the color
ax.spines['bottom'].set_alpha(.3) # and adds some transparency to the spines

ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.spines['left'].set_color('black')
ax.spines['left'].set_alpha(.3)

# adjust the x and y ticks
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor = 'white', edgecolor = 'None', alpha = 0.65 ))
    
# -----------------------------------       PLOT      --------------------------------------------------
# change the x and y limit
plt.xlim(X.min()*1.5, X.max()*1.5)
plt.ylim(C.min()*1.5, C.max()*1.5)

# change the ticks
plt.xticks(
    [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'] # r before each string means raw string
)

# annotate different values
t = 2*np.pi/3
# plot a straight line to connect different points
ax.plot([t, t], [0, np.sin(t)], color ='red', linewidth = 1.5, linestyle = "--", alpha = 0.5)
ax.scatter(t, np.sin(t), 50, color ='red', alpha = 0.5)
ax.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$' + "\nNotice the the arrows are optional",
             xy = (t, np.sin(t)), 
             xycoords = 'data',
             fontsize = 16)

# do the same for cosine
plt.plot([t, t], [0, np.cos(t)], color = 'green', linewidth = 1.5, linestyle = "--", alpha = 0.5)
plt.scatter(t, np.cos(t), 50, color = 'green', alpha = 0.5)
plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
             xy = (t, np.cos(t)), 
             xycoords = 'data',
             xytext = (t/2, -1), 
             fontsize = 16,
             arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3,rad=.2"))

# add the title adn a legend
plt.title("Sine and Cosine functions (Original by N. Rougier)")
plt.legend(loc = "upper left", fontsize = 12);

#%% METHODOLOGY

#USAR MEJOR LA FORMA EXPLICITA: FIGURA + ADD SUBPLOT + PLOT

# create the figure
fig = plt.figure()

# add a subplot to the figure (the explicit way)
# Passing the numbers is optional and you can pass 111 but I will stick with this way. 
# That's why I call it explicit way.
# 1, 1, 1 means: 1 axes in a 1 row 1 column grid. More on this later.
ax1 = fig.add_subplot(1, 1, 1)

# some data
x = [1, 2, 3, 4, 5]
y = [3, 2, 1, 4, 5]

# plot basic things
ax1.plot(x, y);

############################### EQUIVALENTE A ESTO: FIGURA + SUBPLOTS + PLOT
fig = plt.figure()
ax1 = fig.subplots()
ax1.plot(x, y);
###############################EQUIVALENTE A ESTO: PLOT
plt.plot(x, y);
################################ EQUIVALE A ESTO: EJES + PLOT
ax1 = plt.axes()
ax1.plot(x, y);
############################## EQUIVALENTE A: SUBPLOT+PLOT
ax1 = plt.subplot()
ax1.plot(x, y);
############################## EQUIVALENTE: FIGURA +  ADD_AXES+GCA+PLOT
fig = plt.figure()
# add axes
fig.add_axes()
# gca is get current axes, since matplotlib always plots on the current axes.
ax1 = plt.gca()
# plot
ax1.plot(x, y);


#%% VARIAS GRAFICAS

####################################### UNA FORMA
fig = plt.figure()
# create a 4 plots and use tuple unpacking to name everyplot
(ax1, ax2), (ax3, ax4) = fig.subplots(2,2)
ax1.plot([1,2,3], color = "red")
ax2.plot([3,2,1], color = "blue")
ax3.plot([4,4,4], color = "orange")
ax4.plot([5,4,5], color = "black")
plt.tight_layout()

###################################### USANDO UN FOR
nrows = 2
ncolumns = 2
fig, axes = plt.subplots(nrows, ncolumns)

# axes is just a tuple as we saw before
# since se specified 
for row in range(nrows):
    for column in range(ncolumns):
        ax = axes[row, column]
        ax.plot(np.arange(10))

######################################## USANDO GRIDSPEC. METODO POR DEFECTO QUE VAMOS A USAR

fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot([1,2,3,])
ax2 = fig.add_subplot(gs[0,1])
ax2.plot([1,3,1,])
ax3 = fig.add_subplot(gs[1,0])
ax3.plot([3,2,1,])
ax4 = fig.add_subplot(gs[1,1])
ax4.plot([3,1,3,])
plt.tight_layout()
        
##################################### GRAFICA DENTRO DE UNA GRAFICA

fig = plt.figure(figsize = (20, 10))
(ax1, ax2), (ax3, ax4) = fig.subplots(2,2)
ax1.plot([1,2,3], color = "red")
ax2.plot([3,2,1], color = "blue")
ax3.plot([4,4,4], color = "orange")
ax3_bis = fig.add_axes([0.15, 0.15, 0.15, 0.15])
ax3_bis.plot([1,2,1], color = "pink") # you add it to the figure!
ax3_bis.annotate(" ",
                xy = (0.5, 0.5),
                xycoords = "axes fraction",
                va = "center",
                ha = "center")

# ax4.plot([5,4,5], color = "black")
# ax4.annotate("Just to demonstrate the power of matplotlib", 
#              xy = (0.5, 0.5), # fraction of the ax4. In the center.
#              xycoords = "axes fraction", # you can also specify data and pass the values of the x and y axis.
#              va = "center",
#              ha = "center")

plt.tight_layout()


#%% SCATTER

# get the data
PATH = 'datos/midwest_filter.csv' 
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(1,1,1,)

# ----------------------------------------------------------------------------------------------------
# iterate over each category and plot the data. This way, every group has it's own color. Otherwise everything would be blue
for cat in sorted(list(df["category"].unique())):
    # filter x and the y for each category
    ar = df[df["category"] == cat]["area"]
    pop = df[df["category"] == cat]["poptotal"]
    
    # plot the data
    ax.scatter(ar, pop, label = cat, s = 10)
    
# ----------------------------------------------------------------------------------------------------
# prettify the plot

# eliminate 2/4 spines (lines that make the box/axes) to make it more pleasant
ax.spines["top"].set_color("None") 
ax.spines["right"].set_color("None")

# set a specific label for each axis
ax.set_xlabel("Area") 
ax.set_ylabel("Population")

# change the lower limit of the plot, this will allow us to see the legend on the left
ax.set_xlim(-0.01) 
ax.set_title("Scatter plot of population vs area.")
ax.legend(loc = "upper left", fontsize = 10);

#%% Bubble plot with encircling

# Useful for:
# Visualize the relationship between data but also helps us encircle a specific group we might want to draw the attention to.

# get the data
PATH = 'datos/midwest_filter.csv' 
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(1,1,1,)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
size_total = df["poptotal"].sum()
# we want every group to have a different marker
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"] 

# ----------------------------------------------------------------------------------------------------
# iterate over each category and plot the data. This way, every group has it's own color and marker.
for cat, marker in zip(sorted(list(df["category"].unique())), markers):
    # filter x and the y for each category
    ar = df[df["category"] == cat]["area"]
    pop = df[df["category"] == cat]["poptotal"]
    
    # this will allow us to set a specific size for each group.
    size = pop/size_total
    
    # plot the data
    ax.scatter(ar, pop, label = cat, s = size*10000, marker = marker)

# ----------------------------------------------------------------------------------------------------
# create an encircle
# based on this solution
# https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot

# steps to take:

# filter a specific group
encircle_data = df[df["state"] == "IN"]

# separete x and y
encircle_x = encircle_data["area"]
encircle_y = encircle_data["poptotal"]

# np.c_ concatenates over the second axis
p = np.c_[encircle_x,encircle_y]

# uing ConvexHull (we imported it before) to calculate the limits of the polygon
hull = ConvexHull(p)

# create the polygon with a specific color based on the vertices of our data/hull
poly = plt.Polygon(p[hull.vertices,:], ec = "orange", fc = "none")

# add the patch to the axes/plot)
ax.add_patch(poly)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# eliminate 2/4 spines (lines that make the box/axes) to make it more pleasant
ax.spines["top"].set_color("None")
ax.spines["right"].set_color("None")

# set a specific label for each axis
ax.set_xlabel("Area")
ax.set_ylabel("Population")

# change the lower limit of the plot, this will allow us to see the legend on the left
ax.set_xlim(-0.01) 
ax.set_title("Bubble plot with encircling")
ax.legend(loc = "upper left", fontsize = 10);

#%% Buble plot with encircling 2

# get the data
PATH = 'datos/midwest_filter.csv' 
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(1,1,1,)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
size_total = df["poptotal"].sum()
# we want every group to have a different marker
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"] 

# ----------------------------------------------------------------------------------------------------
# create an encircle
# based on this solution
# https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot
def encircle(x,y, ax = None, **kw):
    '''
    Takes an axes and the x and y and draws a polygon on the axes.
    This code separates the differents clusters
    '''
    # get the axis if not passed
    if not ax: ax=plt.gca()
    
    # concatenate the x and y arrays
    p = np.c_[x,y]
    
    # to calculate the limits of the polygon
    hull = ConvexHull(p)
    
    # create a polygon from the hull vertices
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    
    # add the patch to the axes
    ax.add_patch(poly)

# ----------------------------------------------------------------------------------------------------
# iterate over each category and plot the data. This way, every group has it's own color and marker.
# on the iteration we will calculate our hull/polygon for each group and connect specific groups
for cat, marker in zip(sorted(list(df["category"].unique())), markers):
    # filter x and the y for each category
    ar = df[df["category"] == cat]["area"]
    pop = df[df["category"] == cat]["poptotal"]
    
    # this will allow us to set a specific size for each group.
    size = pop/size_total
    
    # plot the data
    ax.scatter(ar, pop, label = cat, s = size*10000, marker = marker)
    
    try:
        # try to add a patch
        encircle(ar, pop, ec = "k", alpha=0.1)
    except:
        # if we don't have enough poins to encircle just pass
        pass

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# eliminate 2/4 spines (lines that make the box/axes) to make it more pleasant
ax.spines["top"].set_color("None")
ax.spines["right"].set_color("None")

# set a specific label for each axis
ax.set_xlabel("Area")
ax.set_ylabel("Population")

# change the lower limit of the plot, this will allow us to see the legend on the left
ax.set_xlim(-0.01) 
ax.set_title("Bubble plot with encircling")
ax.legend(loc = "upper left", fontsize = 10);

#%% Scatter plot with linear regression line of best fit

PATH = 'datos/mpg_ggplot2.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# filter only 2 clases to separate it more easily on the plot
df = df[df["cyl"].isin([4,8])]

# ----------------------------------------------------------------------------------------------------
# plot the data using seaborn
sns.lmplot("displ", "hwy", df, hue = "cyl")

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# since we are using seaborn and this library uses matplotlib behind the scenes
# you can call plt.gca (get current axes) and use all the familiar matplotlib commands
ax = plt.gca()

# change the upper limit of the plot to make it more pleasant
ax.set_xlim(0, 10)
ax.set_ylim(0, 50)

# set title
ax.set_title("Scatter plot with regression");

#%% Scatter plot with linear regression line of best fit 2
# get the data
PATH = 'datos/mpg_ggplot2.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# filter only 2 clases to separate it more easily on the plot
df = df[df["cyl"].isin([4,8])]


# ----------------------------------------------------------------------------------------------------
# plot the data using seaborn
axes = sns.lmplot("displ", 
                  "hwy", 
                  df, 
                  hue = "cyl", 
                  col = "cyl" # by specifying the col, seaborn creates several axes for each group
                 )

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the upper limit of the plot to make it more pleasant
axes.set( xlim = (0.5, 7.5), ylim = (0, 50))

# set title for all axes using plt
plt.suptitle("Scatter plot with regression lines on different axes", fontsize = 10);

#%% Jittering with stripplot
# Useful for:
# Draw a scatterplot where one variable is categorical. 
# This is useful to see the distribution of the points of each category.

# More info: 
# https://seaborn.pydata.org/generated/seaborn.stripplot.html
PATH = 'datos/mpg_ggplot2.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# separate x and y variables
x = df["cty"]
y = df["hwy"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
plt.figure(figsize = (10, 7))

# ----------------------------------------------------------------------------------------------------
# plot the data using seaborn
ax = sns.stripplot(x, y)
# ----------------------------------------------------------------------------------------------------
# prettify the plot

# set title
ax.set_title("Jitter plot");

#%% COUNT PLOTS
# Useful for:
# Draw a scatterplot where one variable is categorical. 
# In this plot we calculate the size of overlapping points in each category and for each y.
# This way, the bigger the bubble the more concentration we have in that region.

# More info: 
# https://seaborn.pydata.org/generated/seaborn.stripplot.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = 'datos/mpg_ggplot2.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting

# we need to make a groupby by variables of interest
gb_df = df.groupby(["cty", "hwy"]).size().reset_index(name = "counts")

# sort the values
gb_df.sort_values(["cty", "hwy", "counts"], ascending = True, inplace = True)

# create a color for each group. 
# there are several way os doing, you can also use this line: 
# colors = [plt.cm.gist_earth(i/float(len(gb_df["cty"].unique()))) for i in range(len(gb_df["cty"].unique()))]
colors = {i:np.random.random(3,) for i in sorted(list(gb_df["cty"].unique()))}

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# iterate over each category and plot the data. This way, every group has it's own color and sizwe.
for x in sorted(list(gb_df["cty"].unique())):
    
    # get x and y values for each group
    x_values = gb_df[gb_df["cty"] == x]["cty"]
    y_values = gb_df[gb_df["cty"] == x]["hwy"]
    
    # extract the size of each group to plot
    size = gb_df[gb_df["cty"] == x]["counts"]
    
    # extract the color for each group and covert it from rgb to hex
    color = matplotlib.colors.rgb2hex(colors[x])
    
    # plot the data
    ax.scatter(x_values, y_values, s = size*10, c = color)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# set title
ax.set_title("Count plot");

#%% MARGINA HISTOGRAM
# Useful for:
# This plot is a combination of 2 plots.
# On one side we have a normal scatter plot that is helpful to see the relationship between data (x and y axis)
# But we also add a histogram that is useful to see the concentration/bins and the distribution of a series.

# More info: 
# https://en.wikipedia.org/wiki/Histogram

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = 'datos/mpg_ggplot2.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# separate x and y
x = df["displ"]
y = df["hwy"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 5))
# in this case we use gridspec.
# check the basics section of this kernel if you need help.
gs = fig.add_gridspec(5, 5)
ax1 = fig.add_subplot(gs[:4, :-1])

# ----------------------------------------------------------------------------------------------------
# plot the data

# main axis: scatter plot
# this line is very nice c = df.manufacturer.astype('category').cat.codes
# since it basically generate a color for each category
ax1.scatter(x, y, c = df.manufacturer.astype('category').cat.codes) 

# set the labels for x and y
ax1.set_xlabel("Dist")
ax1.set_ylabel("Hwy")

# set the title for the main plot
ax1.set_title("Scatter plot with marginal histograms")

# prettify the plot
# get rid of some of the spines to make the plot nicer
ax1.spines["right"].set_color("None")
ax1.spines["top"].set_color("None")

# using familiar slicing, get the bottom axes and plot
ax2 = fig.add_subplot(gs[4:, :-1])
ax2.hist(x, 40, orientation = 'vertical', color = "pink")

# invert the axis (it looks up side down)
ax2.invert_yaxis()

# prettify the plot
# set the ticks to null
ax2.set_xticks([])
ax2.set_yticks([])
# no axis to make plot nicer
ax2.axison = False

# using familiar slicing, get the left axes and plot
ax3 = fig.add_subplot(gs[:4, -1])
ax3.hist(y, 40, orientation = "horizontal", color = "pink")

# prettify the plot
# set the ticks to null
ax3.set_xticks([])
ax3.set_yticks([])
# no axis to make plot nicer
ax3.axison = False

# make all the figures look nicier
fig.tight_layout()
 
#%% MARGINAL BOXPLOT
# Useful for:
# A box plot or boxplot is a method for graphically depicting groups of numerical data through their quartiles.
# It helps to see the dispersion of a series, thanks to the whiskers

# More info: 
# https://en.wikipedia.org/wiki/Box_plot

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = 'datos/mpg_ggplot2.csv'
df = pd.read_csv(PATH)


# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
x = df["displ"]
y = df["hwy"]

# in this plot we create the colors separatly
colors = df["manufacturer"].astype("category").cat.codes

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 5))
# in this case we use gridspec.
# check the basics section of this kernel if you need help.
gs = fig.add_gridspec(6, 6)
ax1 = fig.add_subplot(gs[:4, :-1])

# ----------------------------------------------------------------------------------------------------
# plot the data

# main axis: scatter plot
# this line is very nice c = df.manufacturer.astype('category').cat.codes
# since it basically generate a color for each category
ax1.scatter(x, y, c = df.manufacturer.astype('category').cat.codes) 

# set the labels for x and y
ax1.set_xlabel("Dist")
ax1.set_ylabel("Hwy")

# set the title for the main plot
ax1.set_title("Scatter plot with marginal histograms")

# prettify the plot
# get rid of some of the spines to make the plot nicer
ax1.spines["right"].set_color("None")
ax1.spines["top"].set_color("None")

# using familiar slicing, get the left axes and plot
ax2 = fig.add_subplot(gs[4:, :-1])
ax2.boxplot(x, 
            vert = False,  
            whis = 0.75 # make the boxplot lines shorter
           )
# prettify the plot
# set the ticks to null
ax2.set_xticks([])
ax2.set_yticks([])

# left plot
ax3 = fig.add_subplot(gs[:4, -1])
ax3.boxplot(y,  
            whis = 0.75 # make the boxplot lines shorter
           )
# prettify the plot
# set the ticks to null
ax3.set_xticks([])
ax3.set_yticks([])

# make all the figures look nicier
fig.tight_layout()

#%% CORRELOGRAM, CORRELATION PLOT
# Useful for:
# The correlation plot helps us to comparte how correlated are 2 variables between them

# More info: 
# https://en.wikipedia.org/wiki/Covariance_matrix#Correlation_matrix

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = 'datos/mtcars.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
################## MATPLOTLIB
# instanciate the figure
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot()

# plot using matplotlib
ax.imshow(df.corr(), cmap = 'viridis', interpolation = 'nearest')
# set the title for the figure
ax.set_title("Heatmap using matplotlib");

################### SEABORN
# prepare the data for plotting
# calculate the correlation between all variables
corr = df.corr()
# create a mask to pass it to seaborn and only show half of the cells 
# because corr between x and y is the same as the y and x
# it's only for estetic reasons
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 5))

# plot the data using seaborn
ax = sns.heatmap(corr, 
                 mask = mask, 
                 vmax = 0.3, 
                 square = True,  
                 cmap = "viridis")
# set the title for the figure
ax.set_title("Heatmap using seaborn");

#%% Pairplot
# Useful for:
# Plot pairwise relationships in a dataset. 
# Helps you to see in a glance of an eye all distribution and correlation of variables.

# More info: 
# https://seaborn.pydata.org/generated/seaborn.pairplot.html

# ----------------------------------------------------------------------------------------------------
# get the data
df = sns.load_dataset('iris')

# plot the data using seaborn
sns.pairplot(df, 
             hue = "species" # helps to separate the values by specios
            );
# plot the data using seaborn
sns.pairplot(df, 
             kind = "reg", # make a regression line for eac hue and each variables
             hue = "species"
            );

#%% Diverging bars

# Useful for:
# Based on a metric to compare, this plot helps you to see the divergence of the a value 
# to that metric (it could be mean, median or others).

# More info: 
# https://blog.datawrapper.de/divergingbars/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mtcars.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# here we standarize the data
# More info:
# https://statisticsbyjim.com/glossary/standardization/
df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()

# sort value and reset the index
df.sort_values("x_plot", inplace = True)
df.reset_index(inplace = True)

# create a color list, where if value is above > 0 it's green otherwise red
colors = ["red" if x < 0 else "green" for x in df["x_plot"]]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot()
# plot using horizontal lines and make it look like a column by changing the linewidth
ax.hlines(y = df.index, xmin = 0 , xmax = df["x_plot"],  color = colors, linewidth = 5)

# ----------------------------------------------------------------------------------------------------
# prettify the plot
# set x and y axis
ax.set_xlabel("Mileage")
ax.set_ylabel("Car Name")

# set a title
ax.set_title("Diverging plot in matplotlib")

# make a grid to help separate the lines
ax.grid(linestyle='--', alpha=0.5)

# change the y ticks
# first you set the yticks
ax.set_yticks(df.index)

# then you change them using the car names
# same can be achived using plt.yticks(df.index, df.cars)
ax.set_yticklabels(df.cars);


#%% Diverging lines with text
# Useful for:
# This plot is really useful to show the different performance of deviation of data.
# We use text to annotate the value and make more easy the comparison.

# More info: 
# https://blog.datawrapper.de/divergingbars/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mtcars.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# here we standarize the data
# More info:
# https://statisticsbyjim.com/glossary/standardization/
df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()

# sort value and reset the index
df.sort_values("x_plot", inplace = True)
df.reset_index(inplace=True)

# create a color list, where if value is above > 0 it's green otherwise red
colors = ["red" if x < 0 else "green" for x in df["x_plot"]]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot()

# plot horizontal lines that go from zero to the value
# here we make the linewidht very thin.
ax.hlines(y = df.index, xmin = 0 , color = colors,  xmax = df["x_plot"], linewidth = 1)

# ----------------------------------------------------------------------------------------------------
# plot the data
# iterate over x and y and annotate text and plot the data
for x, y in zip(df["x_plot"], df.index):
    # annotate text
    ax.text(x - 0.1 if x < 0 else x + 0.1, 
             y, 
             round(x, 2), 
             color = "red" if x < 0 else "green",  
             horizontalalignment='right' if x < 0 else 'left', 
             size = 10)
    # plot the points
    ax.scatter(x, 
                y, 
                color = "red" if x < 0 else "green", 
                alpha = 0.5)

# ----------------------------------------------------------------------------------------------------
# prettify the plot
# set title
ax.set_title("Diverging plot in matplotlib")
# change x lim
ax.set_xlim(-3, 3)

# set labels
ax.set_xlabel("Mileage")
ax.set_ylabel("Car Name")

# make a grid
ax.grid(linestyle='--', alpha=0.5)

# instead of y = 1, 2, 3...
# put the car makers on the y axis
ax.set_yticks(df.index)
ax.set_yticklabels(df.cars)

# change the spines to make it nicer
ax.spines["top"].set_color("None")
ax.spines["left"].set_color("None")

# with this line, we change the right spine to be in the middle
# as a vertical line from the origin
ax.spines['right'].set_position(('data',0))
ax.spines['right'].set_color('black')

#%% Diverging dot plot
# Useful for:
# This plot is really useful to show the different performance of deviation of data.
# We use text to annotate the value and make more easy the comparison.
# This plot is very similar to the previous 2
# But here we don't draw any lines and just play with the size of each point and make it a little bigger

# More info: 
# https://blog.datawrapper.de/divergingbars/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mtcars.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# here we standarize the data
# More info:
# https://statisticsbyjim.com/glossary/standardization/
df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()

# sort value and reset the index
df.sort_values("x_plot", inplace = True)
df.reset_index(inplace=True)

# create a color list, where if value is above > 0 it's green otherwise red
colors = ["red" if x < 0 else "green" for x in df["x_plot"]]


# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
# iterate over x and y and annotate text and plot the data
for x, y in zip(df["x_plot"], df.index):
    
    # make a horizontal line from the y till the x value
    # this doesn't appear in the original 50 plot challenge
    ax.hlines(y = y, 
               xmin = -3,  
               xmax = x, 
               linewidth = 0.5,
               alpha = 0.3,
               color = "red" if x < 0 else "green")
    
    # annotate text
    ax.text(x, 
             y, 
             round(x, 2), 
             color = "black",
             horizontalalignment='center', 
             verticalalignment='center',
             size = 8)
    
    # plot the points
    ax.scatter(x, 
                y, 
                color = "red" if x < 0 else "green", 
                s = 300,
                alpha = 0.5)

# ----------------------------------------------------------------------------------------------------
# prettify the plot
# set title
ax.set_title("Diverging plot in matplotlib")

# change x lim
ax.set_xlim(-3, 3)

# set labels
ax.set_xlabel("Mileage")
ax.set_ylabel("Car Name")

# instead of y = 1, 2, 3...
# put the car makers on the y axis
ax.set_yticks(df.index)
ax.set_yticklabels(df.cars)

# change the spines to make it nicer
ax.spines["top"].set_color("None")
ax.spines["left"].set_color("None")

# with this line, we change the right spine to be in the middle
# as a vertical line from the origin
ax.spines['right'].set_position(('data',0))
ax.spines['right'].set_color('grey')

#%% Diverging Lollipop Chart with Markers

# Useful for:
# This plot is really useful to show the different performance of deviation of data.
# In this plot we use rectagles and matplotlib patches to draw the attention to specific points

# More info: 
# https://blog.datawrapper.de/divergingbars/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mtcars.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# here we standarize the data
# More info:
# https://statisticsbyjim.com/glossary/standardization/
df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()

# sort value and reset the index
df.sort_values("x_plot", inplace = True)
df.reset_index(inplace = True)

# we plot everything with a black color except a specific Fiat model
# this way we visually communicate something to the user
df["color"] = df["cars"].apply(lambda car_name: "orange" if car_name == "Fiat X1-9" else "black")


# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (8, 12))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
# plot horizontal lines from the origin to each data point
ax.hlines(y = df.index, 
          xmin = 0,
          xmax = df["x_plot"],
          color = df["color"],
          alpha = 0.6)

# plot the dots
ax.scatter(x = df["x_plot"],
          y = df.index,
          s = 100,
          color = df["color"],
          alpha = 0.6)

# add patches
# with this piece of code, we can draw pretty much any patch or shape
# since we are interested in a rectangle, we must submit a list with the 
# coordinates
def add_patch(verts, ax, color):
    '''
    Takes the vertices and the axes as argument and adds the patch to our plot.
    '''
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    path = Path(verts, codes)
    pathpatch = PathPatch(path, facecolor = color, lw = 2, alpha = 0.3)
    ax.add_patch(pathpatch)

# coordinates for the bottom shape
verts_bottom = [
   (-2.5, -0.5),  # left, bottom
   (-2.5, 2),  # left, top
   (-1.5, 2),  # right, top
   (-1.5, -0.5),  # right, bottom
   (0., 0.),  # ignored
]

# coordinates for the upper shape
verts_upper = [
   (1.5, 27),  # left, bottom
   (1.5, 33),  # left, top
   (2.5, 33),  # right, top
   (2.5, 27),  # right, bottom
   (0., 0.),  # ignored
]

# use the function to add them to the existing plot
add_patch(verts_bottom, ax, color = "red")
add_patch(verts_upper, ax, color = "green")

# annotate text
ax.annotate('Mercedes Models', 
            xy = (0.0, 11.0), 
            xytext = (1.5, 11), 
            xycoords = 'data', 
            fontsize = 10, 
            ha = 'center', 
            va = 'center',
            bbox = dict(boxstyle = 'square', fc = 'blue', alpha = 0.1),
            arrowprops = dict(arrowstyle = '-[, widthB=2.0, lengthB=1.5', lw = 2.0, color = 'grey'), color = 'black')

# ----------------------------------------------------------------------------------------------------
# prettify the plot
# set title
ax.set_title("Diverging Lollipop of Car Mileage")

# autoscale
ax.autoscale_view()

# change x lim
ax.set_xlim(-3, 3)

# set labels
ax.set_xlabel("Mileage")
ax.set_ylabel("Car Name")

# instead of y = 1, 2, 3...
# put the car makers on the y axis
ax.set_yticks(df.index)
ax.set_yticklabels(df.cars)

# change the spines to make it nicer
ax.spines["right"].set_color("None")
ax.spines["top"].set_color("None")

# add a grid
ax.grid(linestyle='--', alpha=0.5);

#%% Area Chart
# Useful for:
# Area chart is really useful when you want to drawn the attention about when a series is below a certain point.
# The area between axis and line are commonly emphasized with colors, textures and hatchings. 
# Commonly one compares two or more quantities with an area chart.

# More info: 
# https://en.wikipedia.org/wiki/Area_chart

# ----------------------------------------------------------------------------------------------------
# get the data

PATH = "datos/economics.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# create the variation between 2 consecutive rows
df["pce_monthly_change"] = (df["psavert"] - df["psavert"].shift(1))/df["psavert"].shift(1)

# convert todatetime
df["date_converted"] = pd.to_datetime(df["date"])

# filter our df for a specific date
df = df[df["date_converted"] < np.datetime64("1975-01-01")]

# separate x and y 
x = df["date_converted"]
y = df["pce_monthly_change"]

# calculate the max values to annotate on the plot
y_max = y.max()

# find the index of the max value
x_ind = np.where(y == y_max)

# find the x based on the index of max
x_max = x.iloc[x_ind]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (15, 10))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
ax.plot(x, y, color = "black")
ax.scatter(x_max, y_max, s = 300, color = "green", alpha = 0.3)

# annotate the text of the Max value
ax.annotate(r'Max value',
             xy = (x_max, y_max), 
             xytext = (-90, -50), 
             textcoords = 'offset points', 
             fontsize = 16,
             arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3,rad=.2")
           )
# ----------------------------------------------------------------------------------------------------
# prettify the plot
# fill the area with a specific color
ax.fill_between(x, 0, y, where = 0 > y, facecolor='red', interpolate = True, alpha = 0.3)
ax.fill_between(x, 0, y, where = 0 <= y, facecolor='green', interpolate = True, alpha = 0.3)

# change the ylim to make it more pleasant for the viewer
ax.set_ylim(y.min() * 1.1, y.max() * 1.1)

# change the values of the x axis
# extract the first 3 letters of the month
xtickvals = [str(m)[:3].upper() + "-" + str(y) for y,m in zip(df.date_converted.dt.year, df.date_converted.dt.month_name())]

# this way we can set the ticks to be every 6 months.
ax.set_xticks(x[::6])

# change the current ticks to be our string month value
# basically pass from this: 1967-07-01
# to this: JUL-1967
ax.set_xticklabels(xtickvals[::6], rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})

# add a grid
ax.grid(alpha = 0.3)

# set the title
ax.set_title("Monthly variation return %");

#%% Ordered Bar Chart
# Useful for:
# This is a normal bar chart but ordered in a specific way.
# From the lowest to the highest values
# It's useful to show comparisons among discrete categories.

# More info: 
# https://en.wikipedia.org/wiki/Bar_chart

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# groupby and create the target x and y
gb_df = df.groupby(["manufacturer"])["cyl", "displ", "cty"].mean()
gb_df.sort_values("cty", inplace = True)
# fitler x and y
x = gb_df.index
y = gb_df["cty"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
for x_, y_ in zip(x, y):
    # this is very cool, since we can pass a function to matplotlib
    # and it will plot the color based on the result of the evaluation
    ax.bar(x_, y_, color = "red" if y_ < y.mean() else "green", alpha = 0.3)
    
     # add some text
    ax.text(x_, y_ + 0.3, round(y_, 1), horizontalalignment = 'center')

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# Add a patch below the x axis line to enphasize that they are below the mean
# I had to copy and paste this code, since I didn't manage to figure it out.
# red one
p2 = patches.Rectangle((.124, -0.005), width = .360, height = .13, alpha = .1, facecolor = 'red', transform = fig.transFigure)
fig.add_artist(p2)

# green one
p1 = patches.Rectangle((.124 + .360, -0.005), width = .42, height = .13, alpha = .1, facecolor = 'green', transform = fig.transFigure)
fig.add_artist(p1)

# rotate the x ticks 90 degrees
ax.set_xticklabels(x, rotation=90)

# add an y label
ax.set_ylabel("Average Miles per Gallon by Manufacturer")

# set a title
ax.set_title("Bar Chart for Highway Mileage");


#%% Lollipop chart

# Useful for:
# The purpose of this kind of chart is the same as a normal bar chart.
# The lollipop chart is often claimed to be useful compared to a normal bar chart, 
# if you are dealing with a large number of values and when the values are all high, such as in the 80-90% range (out of 100%). 
# Then a large set of tall columns can be visually aggressive.

# More info: 
# https://datavizproject.com/data-type/lollipop-chart/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# groupby and create the target x and y
gb_df = df.groupby(["manufacturer"])["cyl", "displ", "cty"].mean()
gb_df.sort_values("cty", inplace = True)
# fitler x and y
x = gb_df.index
y = gb_df["cty"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
for x_, y_ in zip(x, y):
    # make a scatter plot
    ax.scatter(x_, y_, color = "red" if y_ < y.mean() else "green", alpha = 0.3, s = 100)
    
    # add vertical lines to connect them to the data point (head of the lollipop)
    ax.vlines(x_, ymin = 0, ymax = y_, color = "red" if y_ < y.mean() else "green", alpha = 0.3)
    
    # add text with the data
    ax.text(x_, y_ + 0.5, round(y_, 1), horizontalalignment='center')
    
# ----------------------------------------------------------------------------------------------------
# prettify the plot
# change the ylim
ax.set_ylim(0, 30)

# rotate the x ticks 90 degrees
ax.set_xticklabels(x, rotation = 90)

# add an y label
ax.set_ylabel("Average Miles per Gallon by Manufacturer")

# set a title
ax.set_title("Lollipop Chart for Highway Mileage");


#%% Dotplot

# Useful for:
# This plot is the same as the diverging dot plot but here we don't add the line.
# # This plot is really useful to show the different performance of deviation of data.

# More info: 
# https://www.mathsisfun.com/data/dot-plots.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# groupby and create the target x and y
gb_df = df.groupby(["manufacturer"])["cyl", "displ", "cty"].mean()
gb_df.sort_values("cty", inplace = True)
# fitler x and y
x = gb_df.index
y = gb_df["cty"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
for x_, y_ in zip(x, y):
    ax.scatter(y_, x_, color = "red" if y_ < y.mean() else "green", alpha = 0.3, s = 100)
    
# ----------------------------------------------------------------------------------------------------
# prettify the plot
# change the xlim
ax.set_xlim(8, 27)

# add an y label
ax.set_xlabel("Average Miles per Gallon by Manufacturer")

# set the title
ax.set_title("Dot Plot for Highway Mileage")

# create the grid only for the y axis
ax.grid(which = 'major', axis = 'y', linestyle = '--');


#%% Slope Chart

# Useful for:
# This chart is very useful to show the variation of some kind of data
# between two points in time (you can expand it for more points though).

# More info: 
# https://datavizproject.com/data-type/slope-chart/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH  = 'datos/gdppercap.csv'
df = pd.read_csv(PATH)

# create a column with the colors, since we will be iterating and changing the value based on their performance
# if the value at the starting point is bigger than the ending, green color
# otherwise, red color
df["color"] = df.apply(lambda row: "green" if row["1957"] >= row["1952"] else "red", axis = 1)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (8, 12))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
for cont in df["continent"]:
    
    # prepare the data for plotting
    # extract each point and the color
    x_start = df.columns[1]
    x_finish = df.columns[2]
    y_start = df[df["continent"] == cont]["1952"]
    y_finish = df[df["continent"] == cont]["1957"]
    color = df[df["continent"] == cont]["color"]
    
    # plot eac point
    ax.scatter(x_start, y_start, color = color, s = 200)
    ax.scatter(x_finish, y_finish, color = color, s = 200*(y_finish/y_start))
    
    # connect the starting point and the ending point with a line
    # check the bouns section for more
    ax.plot([x_start, x_finish], [float(y_start), float(y_finish)], linestyle = "-", color = color.values[0])
    
    # annotate the value for each continent
    ax.text(ax.get_xlim()[0] - 0.05, y_start, r'{}:{}k'.format(cont, int(y_start)/1000), horizontalalignment = 'right', verticalalignment = 'center', fontdict = {'size':8})
    ax.text(ax.get_xlim()[1] + 0.05, y_finish, r'{}:{}k'.format(cont, int(y_finish)/1000), horizontalalignment = 'left', verticalalignment = 'center', fontdict = {'size':8})

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# get the x and y limits
x_lims = ax.get_xlim()
y_lims = ax.get_ylim()

# change the x and y limits programmaticaly
ax.set_xlim(x_lims[0] - 1, x_lims[1] + 1);

# add 2 vertical lines
ax.vlines(x_start, 0, y_lims[1], color = "grey", alpha = 0.3, lw = 0.5)
ax.vlines(x_finish, 0, y_lims[1], color = "grey", alpha = 0.3, lw = 0.5)

# for each vertical line, add text: BEFORE and AFTER to help understand the plot
ax.text(x_lims[0], y_lims[1], "BEFORE", horizontalalignment = 'right', verticalalignment = 'center')
ax.text(x_lims[1], y_lims[1], "AFTER", horizontalalignment = 'left', verticalalignment = 'center')

# set and x and y label
ax.set_xlabel("Years")
ax.set_ylabel("Mean GPD per Capita")

# add a title
ax.set_title("Slopechart: Comparing GDP per Capita between 1952 and 1957")

# remove all the spines of the axes
ax.spines["left"].set_color("None")
ax.spines["right"].set_color("None")
ax.spines["top"].set_color("None")
ax.spines["bottom"].set_color("None")


#%% Sumbbell plot

# Useful for:
# It's scope if very similar as a slope chart
# Dumbbell plot (also known as Dumbbell chart, Connected dot plot) is great for displaying changes between two points in time, two conditions or differences between two groups.

# More info: 
# https://www.amcharts.com/demos/dumbbell-plot/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/health.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
for i, area in zip(df.index, df["Area"]):
    
    # extract the data for each area
    start_data = df[df["Area"] == area]["pct_2013"].values[0]
    finish_data = df[df["Area"] == area]["pct_2014"].values[0]
    
    # plot the starting and ending plots
    ax.scatter(start_data, i, c = "blue", alpha = .8)
    ax.scatter(finish_data, i, c = "blue", alpha = .2)
    
    # connect them with an horizontal line
    ax.hlines(i, start_data, finish_data, color = "blue", alpha = .2)
    
# ----------------------------------------------------------------------------------------------------
# prettify the plot

# set x and y label
ax.set_xlabel("Pct change")
ax.set_ylabel("Mean GDP per Capita")

# set the title
ax.set_title("Dumbell Chart: Pct Change - 2013 vs 2014")

# add grid lines for the x axis to better separate the data
ax.grid(axis = "x")

# change the x limit programatically
x_lim = ax.get_xlim()
ax.set_xlim(x_lim[0]*.5, x_lim[1]*1.1)

# change the x ticks to be rounded pct %
x_ticks = ax.get_xticks()
ax.set_xticklabels(["{:.0f}%".format(round(tick*100, 0)) for tick in x_ticks]);


#%% Histogram continuous variable

# Useful for:
# This is one of the most fundamental plots to master
# It's shows the approximate distributin of numerical or categorical data.

# More info: 
# https://en.wikipedia.org/wiki/Histogram

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
gb_df = df[["class", "displ"]].groupby("class")
lx = []
ln = []

# handpicked colors
colors = ["#543005", "#8c510a", "#bf812d", "#80cdc1", "#35978f", "#01665e", "#003c30"]

# iterate over very groupby group and 
# append their values as a list
# THIS IS A CRUCIAL STEP
for _, df_ in gb_df:
    lx.append(df_["displ"].values.tolist())
    ln.append(list(set(df_["class"].values.tolist()))[0])

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data

# hist returns a tuple of 3 values
# let's unpack it
n, bins, patches = ax.hist(lx, bins = 30, stacked = True, density = False, color = colors)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change x lim
ax.set_ylim(0, 25)
# set the xticks to reflect every third value
ax.set_xticks(bins[::3])

# set a title
ax.set_title("Stacked Histogram of displ colored by class")

# add a custom legend wit class and color
# you have to pass a dict
ax.legend({class_:color for class_, color in zip(ln, colors)})

# set the y label
ax.set_ylabel("Frequency");

#%% Histogram Categorical variable

# Useful for:
# This is one of the most fundamental plots to master
# It's shows the approximate distributin of numerical or categorical data.

# More info: 
# https://en.wikipedia.org/wiki/Histogram

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
gb_df = df[["class", "manufacturer"]].groupby("class")
lx = []
ln = []

# handpicked colors
colors = ["#543005", "#8c510a", "#bf812d", "#80cdc1", "#35978f", "#01665e", "#003c30"]

# iterate over very groupby group and 
# append their values as a list
# THIS IS A CRUCIAL STEP
for _, df_ in gb_df:
    lx.append(df_["manufacturer"].values.tolist())
    ln.append(list(set(df_["class"].values.tolist()))[0])

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot()

# hist returns a tuple of 3 values
# let's unpack it
n, bins, patches = ax.hist(lx, bins = 30, stacked = True, density = False, color = colors)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# rotate the x axis label
ax.tick_params(axis = 'x', labelrotation = 90)

# add a custom legend wit class and color
# you have to pass a dict
ax.legend({class_:color for class_, color in zip(ln, colors)})

# add a title
ax.set_title("Stacked histogram of manufacturer colored by class")

# set an y label
ax.set_ylabel("Frequency");


#%% Density plot
# Useful for:
# A density plot is a representation of the distribution of a numeric variable. 
# It uses a kernel density estimate to show the probability density function of the variable

# More info: 
# https://www.data-to-viz.com/graph/density.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 8))

# ----------------------------------------------------------------------------------------------------
# plot the data
# the idea is to iterate over each class
# extract their data ad plot a sepate density plot
for cyl_ in df["cyl"].unique():
    # extract the data
    x = df[df["cyl"] == cyl_]["cty"]
    # plot the data using seaborn
    sns.kdeplot(x, shade=True, label = "{} cyl".format(cyl_))

# set the title of the plot
plt.title("Density Plot of City Mileage by n_cilinders");



#%% Density curves with histogram

# Useful for:
# A density plot is a representation of the distribution of a numeric variable. 
# It uses a kernel density estimate to show the probability density function of the variable
# This variation plots the histogram aswel

# More info: 
# https://www.data-to-viz.com/graph/density.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 8))

# ----------------------------------------------------------------------------------------------------
# plot the data
# the idea is to iterate over each class
# extract their data ad plot a sepate density plot
# with their histogram
for class_ in ["compact", "suv", "minivan"]:
    # extract the data
    x = df[df["class"] == class_]["cty"]
    # plot the data using seaborn
    sns.distplot(x, kde = True, label = "{} class".format(class_))
    
# set the title of the plot
plt.title("Density Plot of City Mileage by vehicle type");

#%% Joyplot

# Useful for:
# Joyplot are one of the favorites. 
# Joyplots are essentially just a number of stacked overlapping density plots, that look like a mountain ridge, if done right.

# More info: 
# https://sbebo.github.io/posts/2017/08/01/joypy/
# http://sigmaquality.pl/data-plots/perfect-plots-joyplot-plot/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
plt.figure(figsize = (16,10), dpi = 80)
# plot the data using joypy
fig, axes = joypy.joyplot(df, 
                          column = ['hwy', 'cty'], # colums to be plotted.
                          by = "class", # separate the data by this value. Creates a separate distribution for each one.
                          ylim = 'own', 
                          figsize = (14,10)
                         )

# ----------------------------------------------------------------------------------------------------
# prettify the plot
# add a title
plt.title('Joy Plot of City and Highway Mileage by Class', fontsize = 22);


#%% Distributed dotplot

# Useful for:
# This plot is very cool if you want to show the distribution of some categorical values
# and mark some interesting value, like median, mean of max values with a specific color

# More info: 
# https://www.statisticshowto.com/what-is-a-dot-plot/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)
# sort the values
df.sort_values(["manufacturer", "cty"], inplace = True)
lc = []

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot()

# iterate over each car manufacturer
for i, car in enumerate(df["manufacturer"].unique()):
    # prepare the data for plotting
    # get x and y
    x = df[df["manufacturer"] == car]["cty"]
    y = [car for i_ in range(len(x))]
    
    # calculate the median value
    x_median = np.median(x)
    
    # plot the data
    ax.scatter(x, y, c = "white", edgecolor = "black", s = 30)
    ax.scatter(x_median, i, c = "red",  edgecolor = "black", s = 80)
    
    # add some horizontal line so we can easily track each manufacturer with their distribution
    ax.hlines(i, 0, 40, linewidth = .1)
    
    # append the car name 
    # we need this to change the y labels
    lc.append(car)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change x and y label
ax.set_xlim(5, 40)
ax.set_ylim(-2, 16)

# change the ylabel fontsize
ax.tick_params(axis = "y", labelsize = 12)

# set a title
ax.set_title("Distribution of City Mileage by maker", fontsize = 12)

# annotate some text that will be placed below the legend
ax.text(35, 5.5, "$red \; dots \; are \; the \: median$", fontdict={'size':8}, color='firebrick')

# create a custom legend
# a red circe for the median
red_patch = plt.plot([],[], marker = "o", ms = 10, ls = "", mec = None, color = 'firebrick', label = "Median")

# add the patch and render the legend
plt.legend(handles = red_patch, loc = 7, fontsize = 12)

# remove 3 spines to make a prettier plot
ax.spines["right"].set_color("None")
ax.spines["left"].set_color("None")
ax.spines["top"].set_color("None");


#%% Boxplot

# Useful for:
# Boxplot is a fundamenta chart in statistics.
# It helps to show the distribution of categorical data through quartiles.
# It helps also to see the dispersion of a series, thanks to the whiskers

# More info: 
# https://en.wikipedia.org/wiki/Box_plot

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
plt.figure(figsize = (10, 10), dpi = 80)
# plot the data using seaborn
ax = sns.boxplot(x = "class", y = "hwy", data = df)


# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the font of the x and y ticks (numbers on the axis)
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# set and x and y label
ax.set_xlabel("Class", fontsize = 14)
ax.set_ylabel("HWY", fontsize = 14)

# set a title
ax.set_title("Boxplot", fontsize = 14);


#%% Boxplot 2
# Useful for:
# Boxplot is a fundamenta chart in statistics.
# It helps to show the distribution of categorical data through quartiles.
# It helps also to see the dispersion of a series, thanks to the whiskers.
# This plot adds annotation for each box to add additional information to the plot.

# More info: 
# https://en.wikipedia.org/wiki/Box_plot

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting

# vectors to plot
vects = [df[df["class"] == car]["hwy"] for car in df["class"].unique()]

# labels for the x axis
labels = [class_ for class_ in df["class"].unique()]

# handpicked colors
colors = ["#543005", "#8c510a", "#bf812d", "#80cdc1", "#35978f", "#01665e", "#003c30"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (16, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data using matplotlib
plot1 = ax.boxplot(vects,
    notch = False, 
    vert = True,
    meanline = True,
    showmeans = True,
    patch_artist=True
)

# iterate over every box and add some annotations
for box, color, vect, label, tick in zip(plot1["boxes"], # using this line, you can iterate over every box
                                         colors, 
                                         vects, 
                                         labels, 
                                         ax.get_xticks()):
    # change the color of the box
    box.set(facecolor = color)
    # add text
    ax.annotate("{} obs".format(len(vect)), 
                xy = (tick, np.median(vect)),
               xytext = (15, 50),
               textcoords = "offset points",
               arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3,rad=.2"),
               fontsize = 12)

# ----------------------------------------------------------------------------------------------------
# prettify the plot
# change the x labels
ax.set_xticklabels(labels = labels)

# change the rotation and the size of the x ticks (numbers of x axis)
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

# set the title for the plot
ax.set_title("Box plor of Highway Mileage by Vehicle Class", fontsize = 16);

#%% Dot+Boxplot

# Useful for:
# This plot is very cool, since it allows you to have on the same plot
# a box plot and a dot plot. 
# This way it allows you to have more information to analyze.
# Using seaborn we can also pass the hue to differentiate between classes.

# More info: 
# https://en.wikipedia.org/wiki/Box_plot
# https://en.wikipedia.org/wiki/Dot_plot_(statistics)

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
plt.figure(figsize = (10, 10), dpi= 80)
# plot the data using seaborn
# since we don't create a specific separete plot
# everything will be rendered on the same axes
sns.boxplot(x = "class", y = "hwy", data = df, hue = "cyl")
#sns.stripplot(x = 'class', y = 'hwy', data = df, color = 'black', size = 3, jitter = 1)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(df["hwy"]), color = "grey", alpha = .1)

# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Class", fontsize = 14)
ax.set_ylabel("HWY", fontsize = 14)

# add a title and put the legend on a specific location
ax.set_title("Boxplot and stripplot on the same figure", fontsize = 14)
ax.legend(loc = "lower left", fontsize = 14);

#%% Violin Plot

# Useful for:
# Violin plot is another fundamental plot in statistics
# It helps you see the probability density of the data at different values.

# More info: 
# https://en.wikipedia.org/wiki/Violin_plot

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
plt.figure(figsize = (10, 10), dpi= 80)
sns.violinplot(x = "class", 
               y = "hwy", 
               data = df, 
               scale = 'width', 
               inner = 'quartile'
              )

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(df["hwy"]), color = "grey", alpha = .1)
    
# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Class", fontsize = 14)
ax.set_ylabel("HWY", fontsize = 14)

# set title
ax.set_title("Violinplot", fontsize = 14);

#%% Violin 2
# Useful for:
# Violin plot is another fundamental plot in statistics
# It helps you see the probability density of the data at different values.

# More info: 
# https://en.wikipedia.org/wiki/Violin_plot

# ----------------------------------------------------------------------------------------------------
# get the data
tips = sns.load_dataset("tips")

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
plt.figure(figsize = (10, 10), dpi= 80)

# plot the data using seaborn
# the cool thing is that we put split = True and have 4 violin plots instead of 8
ax = sns.violinplot(x = "day", y = "total_bill", hue = "sex", split = True, data = tips)

# ----------------------------------------------------------------------------------------------------
# prettify the plot
# set a title and change the legend location
ax.set_title('Distribution of total bill amount per day', fontsize = 16)
ax.legend(loc = "upper left", fontsize = 10);

#%% Population pyramid

# Useful for:
# The population chart is a type of funnel chart.
# It really helps out to see the gain/loss of certain amount at every stage in a process.

# More info: 
# https://en.wikipedia.org/wiki/Population_pyramid

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = 'datos/email_campaign_funnel.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
gb_df = df.groupby(["Stage", "Gender"])["Users"].sum().to_frame().reset_index()
gb_df.set_index("Stage", inplace = True)

# separate the different groups to be plotted
x_male = gb_df[gb_df["Gender"] == "Male"]["Users"]
x_female = gb_df[gb_df["Gender"] == "Female"]["Users"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
ax.barh(x_male.index, x_male, color = "red", alpha = 0.3, label = "Male pyramid")
ax.barh(x_female.index, x_female, color = "green", alpha = 0.3, label = "Female pyramid")

# ----------------------------------------------------------------------------------------------------
# prettify the plot
# add the legend to a specific location
ax.legend(loc = "upper left", fontsize = 12)
# set xlabel
ax.set_xlabel("Users", fontsize = 12)
# set the title for the plot
ax.set_title("Population Pyramid", fontsize = 14)
# change the x and y ticks to a smaller size
ax.tick_params(axis = 'y', labelsize = 12)
ax.tick_params(axis = 'x', labelsize = 12)


#%% Categorical plot

# Useful for:
# This is a normal barplot (we show count of each classes)
# But seabron makes it really easy to plot this effortlessly

# More info: 
# https://seaborn.pydata.org/tutorial/categorical.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "/kaggle/input/titanic/train.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
# plot the data using seaborn
fig = plt.figure(figsize = (12, 6))
ax = sns.catplot("Survived", 
                 col = "Pclass", 
                 data = df, 
                 kind = "count",  
                 palette = 'tab20',  
                 aspect = .8
                );

#%% Categorical plot 2

# Useful for:
# This is a normal barplot (we show count of each classes)
# But seabron makes it really easy to plot this effortlessly
# In this plot we add an additional filter to separate even more the data

# More info: 
# https://seaborn.pydata.org/tutorial/categorical.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "/kaggle/input/titanic/train.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
# plot the data using seaborn
fig = plt.figure(figsize = (12, 6))
ax = sns.catplot(x = "Age",
                 y = "Embarked",
                 col = "Pclass",
                 hue = "Sex",
                 data = df, 
                 kind = "violin",  
                 palette = 'tab20',  
                 aspect = .8
                );
#%% Waffle chart

# Useful for:
# Waffle charts are very useful to show the composition of a certain column
# of different categories

# More info: 
# https://datavizproject.com/data-type/percentage-grid/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# create a dictionary of each class and their totals
values = {k:v for k,v in zip(df["class"].value_counts().index, df["class"].value_counts().values)}

# ----------------------------------------------------------------------------------------------------
# plot the data using pywaffle
plt.figure(
    FigureClass = Waffle,
    rows = 7,
    columns = 34,
    values = values,
    legend = {'loc': 'upper left', 'bbox_to_anchor': (1, 1), "fontsize": "12"},
    figsize = (20, 7)
)

# ----------------------------------------------------------------------------------------------------
# prettify the plot
# set a title
plt.title("Waffle chart using pywaffle", fontsize = 12);

#%% Waffle Chart 2
# Useful for:
# Waffle charts are very useful to show the composition of a certain column
# of different categories
# This plot tries to replicate a waffle chart using only matplotlib

# More info: 
# https://datavizproject.com/data-type/percentage-grid/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# we need this when we will create the axes
# and the colors for each column
rows = 10
columns = 7

ncats = len(df["class"].value_counts().index)
colors = [plt.cm.inferno_r(i/float(columns)) for i in range(columns)]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 5))
# create 70 smaller axes to change their color in the next loop
axes = fig.subplots(nrows = rows,
                    ncols = columns)

# ----------------------------------------------------------------------------------------------------
# plot the data
# iterate over all rows and columns
# check the basic part of this kernel if you need help
for col in range(columns):
    for row in range(rows):
        # get every axes we created
        ax = axes[row, col]
        # get the corresponding color
        color = colors[col]
        # change the background color of each axes
        ax.set_facecolor(color)
        # get rid of the x and y ticks (no numbers on x and y axis)
        ax.set_xticks([])
        ax.set_yticks([])

# add a title to the FIGURE
# Note: that matplotlib always plots on the last axes
# if we do it by ax.set_title, we will add a title to the 70'th axes
# and we don't want that
fig.suptitle("Waffle chart using raw matplotlib", fontsize = 14)

# create a legend for each category
legend_elements = [Patch(facecolor = color, 
                         edgecolor = 'white', 
                         label = str(i)) for i, color in enumerate(colors)]

# add the lgend and the patch to the figure
fig.legend(handles = legend_elements, loc = 'lower left', bbox_to_anchor = (0.0, 1.01), ncol = 2, borderaxespad = 0, frameon = False);

#%% PIE chart

# Useful for:
# A pie chart (or a circle chart) is a circular statistical graphic, which is divided into slices to illustrate numerical proportion

# More info: 
# https://en.wikipedia.org/wiki/Pie_chart

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# create a dictionary of classes and their totals
d = df["class"].value_counts().to_dict()

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (18, 6))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data using matplotlib
ax.pie(d.values(), # pass the values from our dictionary
       labels = d.keys(), # pass the labels from our dictonary
       autopct = '%1.1f%%', # specify the format to be plotted
       textprops = {'fontsize': 10, 'color' : "white"} # change the font size and the color of the numbers inside the pie
      )

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# set the title
ax.set_title("Pie chart")

# set the legend and add a title to the legend
ax.legend(loc = "upper left", bbox_to_anchor = (1, 0, 0.5, 1), fontsize = 10, title = "Vehicle Class");

#%% Nested Pie chart

# Useful for:
# A pie chart (or a circle chart) is a circular statistical graphic, which is divided into slices to illustrate numerical proportion
# Nested pie chart goes one step further and separate every outer level of the pie chart
# with the composition on a lower level

# More info: 
# https://en.wikipedia.org/wiki/Pie_chart

# ----------------------------------------------------------------------------------------------------
# get the data
size = 0.3
vals = np.array([[60., 32.], [37., 40.], [29., 10.]])

# create the outer and inner colors
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig, ax = plt.subplots()

# ----------------------------------------------------------------------------------------------------
# plot the data
# outer level
ax.pie(vals.sum(axis = 1), # plot the total [60., 32.] = 92
        radius = 1, 
        colors = outer_colors,
        wedgeprops = dict(width = size, edgecolor = 'w'))

# inner level
ax.pie(vals.flatten(), # using flatten we plot 60, 32 separetly
       radius = 1 - size, 
       colors = inner_colors,
       wedgeprops = dict(width = size, edgecolor = 'w'))

# set the title for the plot
ax.set(aspect = "equal", title = 'Nested pie chart');

#%% Tree map

# Useful for:
# Treemap is very cool and can be used mane different contexts
# usually we want to show the composition of some totals by groups
# very often, smaller groups tend to be very small squares

# More info: 
# https://en.wikipedia.org/wiki/Treemapping

# ----------------------------------------------------------------------------------------------------
# get the data

PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# get the values
label_value = df["class"].value_counts().to_dict()

# create the labels using a list comprehesion
labels = ["{} has {} obs".format(class_, obs) for class_, obs in label_value.items()]

# create n colors based on the number of labels we have
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
plt.figure(figsize = (20, 10))

# ----------------------------------------------------------------------------------------------------
# plot the data using squarify
squarify.plot(sizes = label_value.values(), label = labels,  color = colors, alpha = 0.8)
# ----------------------------------------------------------------------------------------------------
# prettify the plot
# add a title to the plot
plt.title("Treemap using external libraries");

#%% Barchart

# Useful for:
# A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent

# More info: 
# https://en.wikipedia.org/wiki/Bar_chart

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mpg_ggplot2.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# get a dictionary with x and y from a dictionary
d = df["manufacturer"].value_counts().to_dict()

# create n colors based on the number of labels we have
colors = [plt.cm.Spectral(i/float(len(d.keys()))) for i in range(len(d.keys()))]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data using matplotlib
ax.bar(d.keys(), d.values(), color = colors)

# iterate over every x and y and annotate the value on the top of the barchart
for i, (k, v) in enumerate(d.items()):
    ax.text(k, # where to put the text on the x coordinates
            v + 1, # where to put the text on the y coordinates
            v, # value to text
            color = colors[i], # color corresponding to the bar
            fontsize = 10, # fontsize
            horizontalalignment = 'center', # center the text to be more pleasant
            verticalalignment = 'center'
           )

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the size of the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# change the ylim
ax.set_ylim(0, 45)

# set a title for the plot
ax.set_title("Number of Vehicles per Manufacturer", fontsize = 14);

#%% TimeSeries

# Useful for:
# Timeseries is a special type of plots where the time component is present.

# More info: 
# https://study.com/academy/lesson/time-series-plots-definition-features.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = 'datos/AirPassengers.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# helper function to transform and work with the time column
def create_date_tick(df):
    '''
    Converts dates from this format: Timestamp('1949-01-01 00:00:00')
    To this format: 'Jan-1949'
    '''
    df["date"] = pd.to_datetime(df["date"]) # convert to datetime
    df["month_name"] = df["date"].dt.month_name() # extracts month_name
    df["month_name"] = df["month_name"].apply(lambda x: x[:3]) # passes from January to Jan
    df["year"] = df["date"].dt.year # extracts year
    df["new_date"] = df["month_name"].astype(str) + "-" + df["year"].astype(str) # Concatenaes Jan and year --> Jan-1949

# create the time column and the xtickslabels column
create_date_tick(df)

# get the y values (the x is the index of the series)
y = df["value"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data using matplotlib
ax.plot(y, color = "red", alpha = .5, label = "Air traffic")

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# set the gridlines
ax.grid(alpha = .3)

# change the ylim
ax.set_ylim(0, 700)

# get the xticks and the xticks labels
xtick_location = df.index.tolist()[::6]
xtick_labels = df["new_date"].tolist()[::6]

# set the xticks to be every 6'th entry
# every 6 months
ax.set_xticks(xtick_location)

# chage the label from '1949-01-01 00:00:00' to this 'Jan-1949'
ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})

# change the size of the font of the x and y axis
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# set the title and the legend of the plot
ax.set_title("Air Passsengers Traffic (1949 - 1969)", fontsize = 12)
ax.legend(loc = "upper left", fontsize = 10);

#%% TimeSeries annotations
# Useful for:
# Timeseries is a special type of plots where the time component is present.
# This plot is a continuation of the previous one.
# Here we use scatter markers and text to annotate relevant events.
# In our case, the local maxima and minima

# More info: 
# https://study.com/academy/lesson/time-series-plots-definition-features.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = 'datos/AirPassengers.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# helper function to transform and work with the time column
def create_date_tick(df):
    '''
    Converts dates from this format: Timestamp('1949-01-01 00:00:00')
    To this format: 'Jan-1949'
    '''
    df["date"] = pd.to_datetime(df["date"]) # convert to datetime
    df["month_name"] = df["date"].dt.month_name() # extracts month_name
    df["month_name"] = df["month_name"].apply(lambda x: x[:3]) # passes from January to Jan
    df["year"] = df["date"].dt.year # extracts year
    df["new_date"] = df["month_name"].astype(str) + "-" + df["year"].astype(str) # Concatenaes Jan and year --> Jan-1949

# create the time column and the xtickslabels column
create_date_tick(df)

# get the y values (the x is the index of the series)
y = df["value"]

# find local maximum INDEX using scipy library
max_peaks_index, _ = find_peaks(y, height=0) 

# find local minimum INDEX using numpy library
doublediff2 = np.diff(np.sign(np.diff(-1*y))) 
min_peaks_index = np.where(doublediff2 == -2)[0] + 1

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot()

# plot the data using matplotlib
ax.plot(y, color = "blue", alpha = .5, label = "Air traffic")

# we have the index of max and min, so we must index the values in order to plot them
ax.scatter(x = y[max_peaks_index].index, y = y[max_peaks_index].values, marker = "^", s = 90, color = "green", alpha = .5, label = "Peaks")
ax.scatter(x = y[min_peaks_index].index, y = y[min_peaks_index].values, marker = "v", s = 90, color = "red", alpha = .5, label = "Troughs")

# iterate over some max and min in order to annotate the values
for max_annot, min_annot in zip(max_peaks_index[::3], min_peaks_index[1::5]):
    
    # extract the date to be plotted for max and min
    max_text = df.iloc[max_annot]["new_date"]
    min_text = df.iloc[min_annot]["new_date"]
    
    # add the text
    ax.text(df.index[max_annot], y[max_annot] + 50, s = max_text, fontsize = 8, horizontalalignment = 'center', verticalalignment = 'center')
    ax.text(df.index[min_annot], y[min_annot] - 50, s = min_text, fontsize = 8, horizontalalignment = 'center', verticalalignment = 'center')

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the ylim
ax.set_ylim(0, 700)

# get the xticks and the xticks labels
xtick_location = df.index.tolist()[::6]
xtick_labels = df["new_date"].tolist()[::6]

# set the xticks to be every 6'th entry
# every 6 months
ax.set_xticks(xtick_location)

# chage the label from '1949-01-01 00:00:00' to this 'Jan-1949'
ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})

# change the size of the font of the x and y axis
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# set the title and the legend of the plot
ax.set_title("Air Passsengers Traffic (1949 - 1969)", fontsize = 12)
ax.legend(loc = "upper left", fontsize = 10);

#%% ACF and PACF TimeSeries

# Useful for:
# This plot are fundamental in timeseries analysis.
# Basically here we compare the a series again itself but with some lags.
# These are plots that graphically summarize the strength of a relationship with an observation in a time series with observations at prior time steps.

# More info: 
# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = 'datos/AirPassengers.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# ----------------------------------------------------------------------------------------------------
# plot the data using the built in plots from the stats module
plot_acf(df["value"], ax = ax1, lags = 50)
plot_pacf(df["value"], ax = ax2, lags = 15);


#%% Cross correlation plot

# Useful for:
# The cross correlation plot compares two series to see if there have a correlation.
# Remmember correlation not casuality

# More info: 
# https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9?gi=abf39ccba21b

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mortality.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# using this solution to calculate the cross correlation of 2 series
# https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas
def crosscorr(datax, datay, lag=0):
    """ 
    Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

# get the cross correlation
xcov_monthly = [crosscorr(df["mdeaths"], df["fdeaths"], lag = i) for i in range(70)]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data using matplotlib
# notice that this is a regular barchart but the width is really small
ax.bar(x = np.arange(len(xcov_monthly)), height = xcov_monthly, width = .3)

# add some vertical lines that represent the significance level
# you have to calculate them apart
ax.hlines(0.25, 0, len(xcov_monthly), alpha = .3)
ax.hlines(0, 0, len(xcov_monthly), alpha = .3)
ax.hlines(-0.25, 0, len(xcov_monthly), alpha = .3)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# set the title
ax.set_title("Cross Correlation Plot. mdeaths vs fdeasths", fontsize = 14)

# change the x and y ticks size
ax.tick_params(axis = 'x', labelsize = 10)
ax.tick_params(axis = 'y', labelsize = 10);

#%% TimeSeries Decomposition plot
# Useful for:
# The theory behind timeseries, says that a series can be decomposed into 3 parts
# The trend
# The seasonal part
# And the residual
# This plots shows how to do this

# More info: 
# https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = 'datos/AirPassengers.csv'
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# helper function to transform and work with the time column
def create_date_tick(df):
    '''
    Converts dates from this format: Timestamp('1949-01-01 00:00:00')
    To this format: 'Jan-1949'
    '''
    df["date"] = pd.to_datetime(df["date"]) # convert to datetime
    df.set_index("date", inplace = True)
    df["date"] = df.index
    df["month_name"] = df["date"].dt.month_name() # extracts month_name
    df["month_name"] = df["month_name"].apply(lambda x: x[:3]) # passes from January to Jan
    df["year"] = df["date"].dt.year # extracts year
    df["new_date"] = df["month_name"].astype(str) + "-" +df["year"].astype(str) # Concatenaes Jan and year --> Jan-1949

# create the time column and the xtickslabels column    
create_date_tick(df)

# decompose the series using stats module
# results in this case is a special class 
# whose attributes we can acess
result = seasonal_decompose(df["value"])

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
# make the subplots share teh x axis
fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex = True, figsize = (12,10))

# ----------------------------------------------------------------------------------------------------
# plot the data
# using this cool thread:
# https://stackoverflow.com/questions/45184055/how-to-plot-multiple-seasonal-decompose-plots-in-one-figure
# This allows us to have more control over the plots

# plot the original data
result.observed.plot(ax = axes[0], legend = False)
axes[0].set_ylabel('Observed')
axes[0].set_title("Decomposition of a series")

# plot the trend
result.trend.plot(ax = axes[1], legend = False)
axes[1].set_ylabel('Trend')

# plot the seasonal part
result.seasonal.plot(ax = axes[2], legend = False)
axes[2].set_ylabel('Seasonal')

# plot the residual
result.resid.plot(ax = axes[3], legend = False)
axes[3].set_ylabel('Residual')


# ----------------------------------------------------------------------------------------------------
# prettify the plot

# get the xticks and the xticks labels
xtick_location = df.index.tolist()[::6]
xtick_labels = df["new_date"].tolist()[::6]

# set the xticks to be every 6'th entry
# every 6 months
ax.set_xticks(xtick_location)

# chage the label from '1949-01-01 00:00:00' to this 'Jan-1949'
ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'});

#%% Multiple TimeSeries
# Useful for:
# Multiple timeseries is a special case when we plot 2 series and see their performance over time

# More info: 
# https://study.com/academy/lesson/time-series-plots-definition-features.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mortality.csv"
df = pd.read_csv(PATH)

# set the date column to be the index
df.set_index("date", inplace = True)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data using matplotlib
ax.plot(df["mdeaths"], color = "red", alpha = .5, label = "mdeaths")
ax.plot(df["fdeaths"], color = "blue", alpha = .5, label = "fdeaths")

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# get the xticks and the xticks labels
xtick_location = df.index.tolist()[::6]
xtick_labels = df.index.tolist()[::6]

# set the xticks to be every 6'th entry
# every 6 months
ax.set_xticks(xtick_location)
ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'});

# change the x and y ticks to be smaller
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add legend, a title and grid to make it look nicer
ax.legend(loc = "upper left", fontsize = 10)
ax.set_title("Mdeaths and fdeaths over time", fontsize = 14)
ax.grid(axis = "y", alpha = .3)

#%% Secondary axis

# Useful for:
# Multiple timeseries is a special case when we plot 2 series and see their performance over time
# However, here since the data is on a different scale, we will add a secondary y axis

# More info: 
# https://study.com/academy/lesson/time-series-plots-definition-features.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/economics.csv"
df = pd.read_csv(PATH)

# set the date column to be the index
df.set_index("date", inplace = True)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# get the x and y values
x_1 = df["psavert"]
x_2 = df["unemploy"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (14, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data using matplotlib
# here is the main axis
ax.plot(x_1, color = "red", alpha = .3, label = "Personal savings rate")

# suing twinx we can create a secondary axis
ax2 = ax.twinx()
# plot the data on the secondary axis
ax2.plot(x_2, color = "blue", alpha = .3, label = "Unemployment rate")

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# get the xticks and the xticks labels (every 12 entry)
xtick_location = df.index.tolist()[::12]
xtick_labels = df.index.tolist()[::12]

# set the xticks to be every 12'th entry
# every 12 months
ax.set_xticks(xtick_location)
ax.set_xticklabels(xtick_labels, rotation = 90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'});

# change the x and y ticks to be smaller for the main axis and for the secondary axis
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)
ax2.tick_params(axis = 'y', labelsize = 12)

# set a title and a grid
ax.set_title("Personal savings rate vs Unemployed rate: 2 axis", fontsize = 16)
ax.grid(axis = "y", alpha = .3)

#%% TimeSeries with bands
# Useful for:
# This is a regular timeseries plot, but we add some confidence level/bands to the main series
# We can add +- 5% values or we can compute the errors and add error bands

# More info: 
# https://study.com/academy/lesson/time-series-plots-definition-features.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/user_orders_hourofday.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# create a groupby df calculating the mean for each group
gb_df = df.groupby(["order_hour_of_day"])["quantity"].mean().to_frame()

# separete x and calculate the upper and lower bands
x = gb_df["quantity"]
x_lower = x*0.95
x_upper = x*1.05

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data using matplotlib
ax.plot(x, color = "white", lw = 3)
ax.plot(x_lower, color = "#bcbddc")
ax.plot(x_upper, color = "#bcbddc")

# fill the area between the 3 lines
ax.fill_between(x.index, x, x_lower, where = x > x_lower, facecolor='#bcbddc', interpolate = True)
ax.fill_between(x.index, x, x_upper, where = x_upper > x, facecolor='#bcbddc', interpolate = True)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the ylim
ax.set_ylim(0, 25)

# set the x and y labels
ax.set_xlabel("Hour of day")
ax.set_ylabel("# Orders")

# get the xticks and the xticks labels
xtick_location = gb_df.index.tolist()[::2]
xtick_labels = gb_df.index.tolist()[::2]

# set the xticks to be every 2'th entry
# every 2 months
ax.set_xticks(xtick_location)
ax.set_xticklabels(xtick_labels, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})

# change the x and y tick size
ax.tick_params(axis = 'x', labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add a title and a gridline
ax.set_title("Mean orders +- 5% interval ", fontsize = 16)
ax.grid(axis = "y", alpha = .3)



#%% TimeSeries with bands 2
# Useful for:
# This is a regular timeseries plot, but we add some confidence level/bands to the main series
# We can add +- 5% values or we can compute the errors and add error bands
# In this plot we will calculate the error bands using stats module

# More info: 
# https://study.com/academy/lesson/time-series-plots-definition-features.html

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/orders_45d.csv"
df_raw = pd.read_csv(PATH, parse_dates = ['purchase_time', 'purchase_date'])

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# Prepare Data: Daily Mean and SE Bands
df_mean = df_raw.groupby('purchase_date').quantity.mean()
df_se = df_raw.groupby('purchase_date').quantity.apply(sem).mul(1.96)


# prepare the xticks in a specific format
x = [d.date().strftime('%Y-%m-%d') for d in df_mean.index]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12,6), dpi = 80)
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
ax.plot(x, df_mean, color = "white", lw = 2)
ax.fill_between(x, df_mean - df_se, df_mean + df_se, color = "#3F5D7D") 

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the spines to make it look cleaner
ax.spines["top"].set_alpha(0)
ax.spines["bottom"].set_alpha(1)
ax.spines["right"].set_alpha(0)
ax.spines["left"].set_alpha(1)

# get the xticks and the xticks labels
xtick_location = x[::6]
xtick_labels = [str(d) for d in x[::6]]

# set the xticks to be every 4'th entry
# every 4 week
ax.set_xticks(xtick_location)
ax.set_xticklabels(xtick_labels, fontdict = {'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})

# add a title
ax.set_title("Daily Order Quantity of Brazilian Retail with Error Bands (95% confidence)", fontsize = 14)

# change the x and y limit
s, e = ax.get_xlim()
ax.set_xlim(s, e-2)
ax.set_ylim(4, 10)

# set the y label for the plot
ax.set_ylabel("# Daily Orders", fontsize = 12) 

# change the size of the x and y ticks
ax.tick_params(axis = 'x', labelsize = 12, rotation = 90)
ax.tick_params(axis = 'y', labelsize = 12)

# add some horizontal lines to make the plot look nicer
for y in range(5, 10, 1):    
    ax.hlines(y, xmin = s, xmax = e, colors = 'black', alpha = 0.5, linestyles = "--", lw = 0.5)

#%% Stacked Area chart

# Useful for:
# A stacked area chart is the extension of a basic area chart to display the evolution of the value of several groups on the same graphic. 
# The values of each group are displayed on top of each other. It allows to check on the same figure the evolution of both the total of a numeric variable, and the importance of each group. 
# If only the relative importance of each group interests you, you should probably draw a percent stacked area chart. 
# Note that this chart becomes hard to read if too many groups are displayed and if the patterns are really different between groups. In this case, think about using faceting instead.

# More info: 
# https://python-graph-gallery.com/stacked-area-plot/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/nightvisitors.csv"
df = pd.read_csv(PATH)
# set the data as index of the df
df.set_index("yearmon", inplace = True)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting

# get the x and y 
x = df.index
y = [df[col].values for col in df.columns]

# get the name of each group for the labels
labels = df.columns

# prepare some colors for each group to be ploted
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (14, 10))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
ax.stackplot(x,y, labels = labels, colors = colors)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# get the xticks and the xticks labels
xtick_location = df.index.tolist()[::3]
xtick_labels = df.index.tolist()[::3]

# set the xticks to be every 3'th entry
# every 3 entry
ax.set_xticks(xtick_location)
ax.set_xticklabels(xtick_labels, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})

# change the x and y ticks to smaller size
ax.tick_params(axis = 'x', labelsize = 10, rotation = 90)
ax.tick_params(axis = 'y', labelsize = 10)

# set the x and y label
ax.set_xlabel("Date", fontsize = 12)
ax.set_ylabel("Visitors", fontsize = 12)

# change the ylim
ax.set_ylim(0, 90000)

# set a title and a legend
ax.set_title("Night visitors in Australian Regions", fontsize = 16)
ax.legend(fontsize = 8);

#%% Area chart unstacked
# Useful for:
# An area chart is really similar to a line chart, except that the area between the x axis and the line is filled in with color or shading.
# This draws the attention to the specific area.

# More info: 
# http://python-graph-gallery.com/area-plot/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/economics.csv"
df = pd.read_csv(PATH)

# set the date as the index of the df
df.set_index("date", inplace = True)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting

# get x and y
x = df["psavert"]
y = df["uempmed"]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (14, 8))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
ax.plot(x, color = "blue", alpha = .3, label = "Personal savings rate")
ax.plot(y, color = "red", alpha = .3, label = "Unemployment rate")

# fill the areas between the plots and the x axis
# this can create overlapping areas between lines
ax.fill_between(x.index, 0, x, color = "blue", alpha = .2)
ax.fill_between(x.index, 0, y, color = "red", alpha = .2)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# set the title
ax.set_title("Personal savings rate vs Unemployed rate", fontsize = 16)

# get the xticks and the xticks labels
xtick_location = df.index.tolist()[::12]
xtick_labels = df.index.tolist()[::12]

# set the xticks to be every 3'th entry
# every 3 entry
ax.set_xticks(xtick_location)
ax.set_xticklabels(xtick_labels, rotation = 90, fontdict = {'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})

# change the x and y ticks to smaller size
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# more the spines of the axes
ax.spines["right"].set_color("None")
ax.spines["top"].set_color("None")

# set a legend and the y grid for the plot
ax.legend(fontsize = 10)
ax.grid(axis = "y", alpha = .3);

#%% Calendar heatmap
# Useful for:
# This is a very common plot you see everytime you connect to GitHub or Kaggle.
# Display the activity of a person for a certain period of time (usually a year)
# With a calendarmap

# More info: 
# https://pythonhosted.org/calmap/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/yahoo.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
df["date"] = pd.to_datetime(df["date"])
# the data must be a series with a datetime index
df.set_index("date", inplace = True)
x = df[df["year"] == 2014]["VIX.Close"]

# ----------------------------------------------------------------------------------------------------
# plot the data using calmap
calmap.calendarplot(x, fig_kws={'figsize': (16,10)}, yearlabel_kws={'color':'black', 'fontsize':14}, subplot_kws={'title':'Yahoo Stock Prices'});

#%% Seasonal plot
# Useful for:
# Seasonal plots are a regular lineplot but where we represent a lot of data/seasons.
# If the data is increasing year after year, we can see the evolution of the variable very nicely
# in a smaller plot

# More info: 
# https://python-graph-gallery.com/line-chart/

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/AirPassengers.csv"
df = pd.read_csv(PATH)

# prepare the data for plotting

# first of all. 
# We have the following format of our df
#         date	value
# 0	1949-01-01	  112
# 1	1949-02-01	  118
# 2	1949-03-01	  132
# 3	1949-04-01	  129
# 4	1949-05-01	  121

# Basically a lot of rows with each year month data
# In order to plot the data into a seasonal chart, we need the data in this format
# where each column is the year
# index_ 1949	1950	1951	1952	1953	1954	1955	1956	1957	1958	1959	1960
# 1	      112	 115	 145	 171	 196	 204	 242	 284	 315	 340	 360	 417
# 2	      118	 126	 150	 180	 196	 188	 233	 277	 301	 318	 342	 391

# To do so, we must create a repeating index (12  months) for each year

# create a repeating index of [1, 2, 3, .. 12] months x 12 times (12 years)
index_ = [i for i in range(1, 13)]*12

# set the index into the dataframe
df["index_"] = index_

# create a dictionary with the months name (we will use this later to change the x axis)
months_ = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
d = {k:v for k,v in zip(index_[:12], months_)}

# convert to datetime the date column
df["date"] = pd.to_datetime(df["date"])

# extract the year using pandas datatime (dt)
df["year"] = df["date"].dt.year

# drop the date
df.drop("date", inplace = True, axis = 1)

# create a pivot table
# traspose the rows into columns, where the columns name are the year to plot
df = df.pivot(values = "value", columns = "year", index = "index_")

# create n colors for each season
colors = [plt.cm.gist_earth(i/float(len(df.columns))) for i in range(len(df.columns))]

# get the x to plot
# since we are extracting it from our new df
# it has 12 values, one for each month
x = df.index

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
# iterate over every column in the dataframe and plot the data
for col, color in zip(df.columns, colors):
    # get the y to plot
    y = df[col]
    
    # plot the data using seaborn
    ax.plot(x, y, label = col, c = color)
    
    # get the x and y to annotate
    x_annotate = x[-1]
    y_annotate = df.iloc[11][col]
    
    # annotate at the end of each line some values
    ax.text(x_annotate + 0.3, y_annotate, col, fontsize = 8, c = color)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# set the x and y label
ax.set_xlabel("Months", fontsize = 13)
ax.set_ylabel("Air traffic", fontsize = 13)

# extract the x ticks location
xtick_location = df.index.tolist()

# using our dictionary, create a list of new xlabels
# basically instead of numbers, strings of months
months = [d[tick] for tick in xtick_location]

# change the x ticks with our new x ticks labels
ax.set_xticks(xtick_location)
ax.set_xticklabels(months, rotation = 90, fontdict = {'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})

# change the y ticks font size
ax.tick_params(axis = 'y', labelsize = 12)

# change the y limit to make the plot a little bigger
ax.set_ylim(0, 700)

# get rid of spines from our plot
ax.spines["right"].set_color("None")
ax.spines["top"].set_color("None")

# add a grid to the plot
ax.grid(axis = "y", alpha = .3)

# set the title for the plot
ax.set_title("Monthly seasonal plot of air traffic (1949 - 1969)", fontsize = 15);

#%% Dendogram
# Useful for:
# A dendrogram is a diagram representing a tree.
# It's very useful to represent hierarchy in a dataset

# More info: 
# https://en.wikipedia.org/wiki/Dendrogram

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/USArrests.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 7))

# ----------------------------------------------------------------------------------------------------
# plot the data using the scipy package
dend = shc.dendrogram(shc.linkage(df[['Murder', 'Assault', 'UrbanPop', 'Rape']], method = 'ward'), 
                      labels = df["State"].values, 
                      color_threshold = 100)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# using plt.gca() we get the current figure
ax = plt.gca()

# set x and y label
ax.set_xlabel("County level")
ax.set_ylabel("# of incidents")

# change the x and y ticks size
ax.tick_params("x", labelsize = 10)
ax.tick_params("y", labelsize = 10)

# set a title
ax.set_title("US Arrests dendograms");

#%% Cluster plot
# Useful for:
# A cluster plots, help encircle data from a specific cluster, to help separte it more easily
# Before drawing the plot, you must first cluster the data into similar groups.

# More info: 
# https://en.wikipedia.org/wiki/Cluster_analysis

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/USArrests.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting

# get the x and y to plot
x = df["Murder"]
y = df["Assault"]

# first, create out cluster using the AgglomerativeClustering from sklearn
cluster = AgglomerativeClustering(n_clusters = 5, # notice that we specify the number of "optimal" clusters
                                  affinity = 'euclidean', # use the euclidean distance to compute similarity. The closer the better.
                                  linkage = 'ward'
                                 )  

# fit and predict the clusters based on this data
cluster.fit_predict(df[['Murder', 'Assault', 'UrbanPop', 'Rape']])  


# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot()

# ----------------------------------------------------------------------------------------------------
# plot the data
ax.scatter(x, y)

# Encircle
def encircle(x,y, ax = None, **kw):
    '''
    Takes an axes and the x and y and draws a polygon on the axes.
    This code separates the differents clusters
    '''
    # get the axis if not passed
    if not ax: ax=plt.gca()
    
    # concatenate the x and y arrays
    p = np.c_[x,y]
    
    # to calculate the limits of the polygon
    hull = ConvexHull(p)
    
    # create a polygon from the hull vertices
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    
    # add the patch to the axes
    ax.add_patch(poly)

# use our cluster fitted before to draw the clusters borders like we did at the beginning of the kernel
# basically go over each cluster and add a patch to the axes
encircle(df.loc[cluster.labels_ == 0, 'Murder'], df.loc[cluster.labels_ == 0, 'Assault'], ec = "k", fc = "gold", alpha = 0.2, linewidth = 0)
encircle(df.loc[cluster.labels_ == 1, 'Murder'], df.loc[cluster.labels_ == 1, 'Assault'], ec = "k", fc = "tab:blue", alpha = 0.2, linewidth = 0)
encircle(df.loc[cluster.labels_ == 2, 'Murder'], df.loc[cluster.labels_ == 2, 'Assault'], ec = "k", fc = "tab:red", alpha = 0.2, linewidth = 0)
encircle(df.loc[cluster.labels_ == 3, 'Murder'], df.loc[cluster.labels_ == 3, 'Assault'], ec = "k", fc = "tab:green", alpha = 0.2, linewidth = 0)
encircle(df.loc[cluster.labels_ == 4, 'Murder'], df.loc[cluster.labels_ == 4, 'Assault'], ec = "k", fc = "tab:orange", alpha = 0.2, linewidth = 0)

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the size of x and y ticks
ax.tick_params("x", labelsize = 10)
ax.tick_params("y", labelsize = 10)

# set an x and y label
ax.set_xlabel("Murder", fontsize = 12)
ax.set_ylabel("Assault", fontsize = 12)

# set a title for the plot
ax.set_title("Agglomerative clustering of US arrests (5 Groups)", fontsize = 14);

#%% Andrews curves

# Useful for:
# Andrews curves allow one to plot multivariate data as a large number of curves that are created using the attributes of samples as coefficients for Fourier series.

# More info: 
# https://en.wikipedia.org/wiki/Andrews_plot

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/mtcars.csv"
df = pd.read_csv(PATH)

# ----------------------------------------------------------------------------------------------------
# prepare the data for plotting
# get the first 2 columns from our dataframe
X = df[list(df.columns)[:-2]]

# ----------------------------------------------------------------------------------------------------
# instanciate the figure 
fig = plt.figure(figsize = (12, 6))

# ----------------------------------------------------------------------------------------------------
# plot the data using pandas capabilities
ax = andrews_curves(X, 'cyl', colormap = 'Set1')

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the x and y font size
ax.tick_params("x", labelsize = 10)
ax.tick_params("y", labelsize = 10)

# no gridlines
ax.grid(False)

# set legend and title to the plot
ax.legend(loc = "upper left", fontsize = 10,  title = "cyl")
ax.set_title("Andrews curves", fontsize = 14);

#%%
from sklearn.datasets import load_iris
iris = load_iris()
iris=load_iris()
datos=pd.DataFrame(iris.data)
datos['species']=iris.target
datos.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
datos.dropna(how="all", inplace=True) # remove any empty lines
datos["Species"]=datos["Species"].replace(0, iris.target_names[0])
datos["Species"]=datos["Species"].replace(1, iris.target_names[1])
datos["Species"]=datos["Species"].replace(2, iris.target_names[2])
iris=datos.copy()
ax = andrews_curves(iris, 'Species', colormap = 'Set1')

#%% Parallel coordinates

# Useful for:
# Parallel coordinates allows one to see clusters in data and to estimate other statistics visually. 
# Using parallel coordinates points are represented as connected line segments. Each vertical line represents one attribute. 
# One set of connected line segments represents one data point. Points that tend to cluster will appear closer together.

# More info: 
# https://en.wikipedia.org/wiki/Parallel_coordinates

# ----------------------------------------------------------------------------------------------------
# get the data
PATH = "datos/diamonds_filter.csv"
df = pd.read_csv(PATH)
andrews_curves(df, 'cut', colormap = 'Set1')
# ----------------------------------------------------------------------------------------------------
# get the data
fig = plt.figure(figsize = (12, 6))

# ----------------------------------------------------------------------------------------------------
# plot the data using pandas capabilities
ax = parallel_coordinates(df, 'cut', colormap = "Dark2")

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the x and y font size
ax.tick_params("x", labelsize = 10)
ax.tick_params("y", labelsize = 10)

# no gridlines
ax.grid(False)

# set legend and title to the plot
ax.legend(loc = "upper left", fontsize = 10,  title = "Diamond type")
ax.set_title("Parallel coordinates", fontsize = 14);

#%% lines to connect points
plt.plot([1, 1], [1, 2], linestyle="--", color = "black", label = "black")
plt.plot([0, 1], [2, 2], linestyle="-", color = "blue", label = "blue")
plt.plot([1, 1], [3, 2], linestyle="-.", color = "red", label = "red")
plt.plot([1, 2], [2, 2], linestyle=":", color = "magenta", label = "magenta")
plt.plot([1,2,3])
plt.plot([3, 2, 1])
plt.legend(loc = "upper left");

