# source("libreria.R")
dir()
#library(libreria)
import os
import csv
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 50)

# setwd("C:/Users/laguila/Desktop")
os.chdir("C:/Users/laguila/Google Drive/Programacion/Python/PythonBasico")
print(os.getcwd())
os.listdir()
os.mkdir("newdir")
os.rename("newdir", "newdor")
os.rmdir("newdor")
os.makedirs("my_first_directory/my_second_directory")
os.removedirs("my_first_directory/my_second_directory")

#%% LECTURA Y ESCRITURA
# https://www.kaggle.com/rohanrao/tutorial-on-reading-large-datasets
# read.csv(file = "archivo.csv", header = F, sep = ",", row.names = T, col.names = T, skip = 10, nrows = 1000)
# data.table::fread("archivo.csv", stringsAsFactors = F)
np.loadtxt('test.csv', delimiter="," , skiprows=1, usecols=[0,1])
np.genfromtxt('test.csv', delimiter = ',', names=True)
np.recfromtxt('test.csv', delimiter = ',', names=True)

dtypes = {"row_id": "int64", "timestamp": "int64","user_id": "int32"}
pd.read_csv('test.csv', header = None, nrows = 5, comment='#', na_values=['Nothing'], delimiter=",", dtype = dtypes)

import datatable as dt
dt.fread("test.csv").to_pandas()

data = pd.ExcelFile('test.xlsx')
data.parse("hoja1", names = ["time", "open_channels", "open_channels2"])

# write.csv(dataframe, "archivo.csv")
fichero = "a"
fichero.to_csv('nombre.csv')
fichero.to_excel('dir/myDataFrame.xlsx', sheet_name='Sheet1')

#%% VECTORES Y MATRICES CON NUMPY
# a=c(1, 2, 3)
%%time
a=np.array([1,2,3])
# a[1]
a[1]
# a[1:2]
a[1:3]
# runif(n = 4, min = 0, max = 10)
from random import *
sample(range(10), 4)
choice(range(10))
# sample(10)
sample(population=range(10), k=10)
#seq(0,10,length.out = 4)
np.linspace(0,10,4)
# a[1]=abs(-5)
a[0]=abs(-5)
#Otras funciones
np.zeros(4)
np.ones(5)
np.arange(2, 8, 2)
np.full((2, 3), 7)
np.eye(4)
np.random.random((2,3))
np.random.normal(2)
np.random.randint(2, 3)
np.empty((2, 3))
randrange(0, 10, 1)

b = np.array([(1.5,2,3), (4,5,6)], dtype = float)
b.flatten()

#Uniones, intersecciones...
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
ser1[~ser1.isin(ser2)] #elementos de 1 que no estan en 2
ser_u = pd.Series(np.union1d(ser1, ser2))  # union
ser_i = pd.Series(np.intersect1d(ser1, ser2))  # intersect
ser_u[~ser_u.isin(ser_i)] #filas que no son comunes


# matriz=matrix(data = 1:100, nrow = 10, byrow=T)
matriz = np.arange(100).reshape(10,10)
np.matrix('1 2; 3 4')
# matriz[1:5,1]=4
matriz[0:4,0]=4
# t(matrix)
np.transpose(matriz)
matriz.T
#matriz*matriz
matriz*matriz
#matrix %*% matriz
matriz @ matriz

#%% LISTAS
# lista=list(a, 2*a, 4)
lista=[1, 2, 3, 4]
# lista[[1]]
lista[1]

lista.reverse()

#%% DATAFRAMES CON PANDAS

# aux=data.frame(unos=rep(1,10), otros=1:10)
df = pd.DataFrame({"unos": np.repeat(1,10), 'col2': np.arange(10)})
# aux$unos
df["unos"]
df.unos

# data(iris)
from sklearn.datasets import load_iris
iris = load_iris()
datos=pd.DataFrame(iris.data)
datos['species']=iris.target
datos.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
datos.dropna(how="all", inplace=True) # remove any empty lines
datos["Species"]=datos["Species"].replace(0, iris.target_names[0])
datos["Species"]=datos["Species"].replace(1, iris.target_names[1])
datos["Species"]=datos["Species"].replace(2, iris.target_names[2])
iris=datos.copy()


#%% Estadistica basica

# str(iris)
iris.dtypes
# table(iris$Sepal.Length, iris$Sepal.Width)
datos['Species'].value_counts()
# head(iris)
iris.head()
# tail(iris)
iris.tail()
# summary(iris)
iris.describe()
# dim(iris)
iris.shape
# sum((iris$Sepal.Length))
iris["SepalLength"].sum()
# min(iris$Sepal.Length)
iris["SepalLength"].min()
# max(iris$Sepal.Length)
iris["SepalLength"].max()
# mean(iris$Sepal.Length)
iris["SepalLength"].mean()
# median(iris$Sepal.Length)
iris["SepalLength"].median()
# sd(iris$Sepal.Length)
iris["SepalLength"].std()
# quantile(x = iris$Sepal.Length, 0.35)
iris['SepalLength'].quantile([0.25, 0.4])
# cor(iris[,1:4])
iris[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].corr(method="pearson")
#unique(iris$Species)
np.unique(iris['Species'])
iris.Species.unique()

#%% Partes de un DataFrame
# which(data$Sepal.Length<6)
np.where(iris["SepalLength"]<6)
# data[1,4]=0.3
iris.iloc[0, 3]=0.3 
iris.iloc[:, :]
iris.iloc[:,np.arange(3)]
iris.loc[0, ['PetalWidth']]=0.3
# data[which(data$Sepal.Length<5.2),1]=NA
iris.loc[iris['SepalLength']<5.2,['SepalLength']]=np.NaN
# data[is.na(data)]=5.2
iris=iris.fillna(5.2)
iris.ffill(inplace=True)
iris.bfill(inplace=True)
# data=na.omit(data)
iris=iris.dropna(how='any').reset_index()
iris.dropna(how='all')
# data$Sepal.Length[str_detect(data$Species, pattern = "set")]="esta"
iris.loc[iris['Species'].str.contains('set'), 'Species']='esta'
# str_replace(data$Species, "set", "sat")
iris["Species"].str.replace("set", "sat")

# variable="Sepal.Length"
# match(variable, colnames(iris))
# colnames(iris)[colnames(iris) %in% variable]
variable='SepalLength'
variable in iris.columns #True
iris.iloc[:,iris.columns==variable]


# data(iris)
# data=iris %>% 
# select(Petal.Length, Petal.Width, Sepal.Length, Sepal.Width, Species)
iris[['Species', 'PetalLength']]
iris.drop('SepalLength', axis=1) #quitar esa columna
iris.drop(5, axis=0) #quitar la sexta fila


# data$ceros=0
iris['ceros']=0
# data$ceros=NULL
del(iris['ceros'])

# data=iris %>% 
# filter(Petal.Length>1 & Petal.Length<100)
iris[(iris['PetalLength']>5) & (iris['PetalLength']<6)]


#%% Operaciones con DataFrame
# data=iris %>% 
# dplyr::group_by(Species) %>%
# summarise(media=mean(Petal.Length)) 
iris.groupby(["Species"]).get_group("setosa")
iris.groupby(["Species"]).sum()
iris[iris.PetalLength < 5].groupby(["Species"]).sum()
iris.groupby(["Species"])["PetalLength"].agg([np.sum, np.mean, np.median, np.min, np.max, np.std, np.var])
iris.groupby(["Species"])["PetalLength"].agg(suma = "sum", mean = "mean")
iris.groupby(["Species"])["PetalLength"].apply(lambda x: 2*sum(x))
iris.groupby(["Species"])["PetalLength"].transform(lambda x: 2*x.min()-1)

# data=iris %>% 
# mutate(total=Sepal.Length+Petal.Length, otro=ifelse(Petal.Length>2, "grande", "pequeÃ±o"))
iris.assign(redondeado = lambda x: x.PetalLength.round(), redondeado2 = lambda x: x.SepalLength.round())

# data=iris %>% 
# distinct(Species, Sepal.Length, .keep_all = T)
iris.drop_duplicates()
iris.drop_duplicates(subset='PetalLength')
iris.duplicated()
iris.drop_duplicates()
iris = iris.drop_duplicates(iris.columns[~iris.columns.isin(['SepalLength', 'Species'])], keep='first')

# #ordenando
# data=iris %>% 
# arrange(Sepal.Length, Sepal.Width)
iris.sort_values("PetalLength", ascending=False)

# data=data.frame(x1=rep(1,10), x2=rep(2, 10))
data = pd.DataFrame({'x1':np.ones(10), 'x2':np.repeat(2, 10)}) #np.full(10, -1)
# data2=data.frame(x3=rep(3,10), x4=rep(4, 10))
data2 = pd.DataFrame({'x3':np.repeat(3, 10), 'x4':np.repeat(4, 10)})
# data3=data.frame(x1=rep(1,10), x5=rep(5, 10))
data3 = pd.DataFrame({'x1':np.ones(10), 'x5':np.repeat(5, 10)})
# total=left_join(data, data3, by="x1")
# total2=merge(data, data2, by="x1")
data.merge(data3, on='x1', how='left')
pd.merge(data, data3, on='x1', how='left')
pd.concat([data, data3], keys=['x', 'y']) #el x e y son para diferenciarlos
pd.concat({'x':data, 'y':data3})
# total3=cbind(data, data2)
total3=pd.concat([data, data3], axis=1)
# total4=rbind(data, data2)
total4=pd.concat([data, data3], axis=0)
total4=data.append(data3)

# data=spread(data, key=Species, value=Sepal.Length)
spread=pd.pivot_table(iris, values='SepalLength', index=['SepalWidth', 'PetalLength', 'PetalWidth'], columns='Species').reset_index()
# data=gather(data, key=Species, value=Sepal.Length, 4:6)
iris2=pd.melt(iris, id_vars=['SepalWidth', 'PetalLength', 'PetalWidth'], var_name='Species', value_name='SepalLength').dropna(how='any').reset_index()

#Desplazar un valor
iris.PetalLength.shift(1)
iris.PetalLength.shift(-1)

#HAcer un ranking
iris.PetalLength.rank()

#maximo, minimo, suma acumulada...
iris.PetalLength.cummax()
iris.PetalLength.cumsum()

#Rolling
iris.PetalLength.rolling(3).mean()


#%% Modificar DataFrame
# #renombrar filas y columnas
# colnames(iris)=c("n1", "n2", "n3", "n4")
# rownames(iris)=1:nrow(iris)
iris.columns=['SapalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
iris.rename(columns={'Species': 'Especies'},
            index={0:'cero',1:'uno'})


# #Formatear columnas
# iris$Sepal.Length=as.integer(iris$Sepal.Length)
iris.SepalLength.map(lambda x: int(x))
# iris$Sepal.Length=as.numeric(iris$Sepal.Length)
iris.SepalLength.map(lambda x: float(x))
# iris$Species=as.character(iris$Species)
iris.Species.map(lambda x: str(x))
# iris$Species=as.factor(iris$Species)

iris.SepalLength = iris.SepalLength.astype(np.int32)
iris.convert_dtypes() #Convertir automaticamente
iris.SepalLength = pd.to_numeric(iris.SepalLength)

#%% math

import math
# iris$Petal.Length=round(iris$Petal.Length,2)
round(iris['PetalLength'], 1)
# iris$Petal.Length=ceiling(iris$Petal.Length,2)
iris['PetalLength'].map(lambda x: math.ceil(x))
# iris$Petal.Length=floor(iris$Petal.Length,2)
iris['PetalLength'].map(lambda x: math.floor(x))

#%% Cadenas de caracteres
from re import *
# str_detect(as.character(iris$Species), string = "setosa")
iris['Species'].str.match('set')
# substr(as.character(iris$Species), 1, 4)
iris['Species'].str.remove()
# str_replace(as.character(iris$Species), pattern = "a", replacement = "e")
iris['Species'].str.replace('set', 'sat')
# str_remove(as.character(iris$Species), pattern = "a")
iris['Species'].map(lambda x: x.replace('set', ''))


#Columnas que empiezan con S
iris.filter(regex = "^S", axis=1)

#Columnas que terminan con h
iris.filter(regex = "h$", axis=1)

#Regular expressions
string = "esto es una prueba, 567"
'''
\d = any digit
\D = anything but a digit
\s = space
\S = anything but a space
\w = any letter
\W = anything but letters
\b = any character except new line
\n = new line
\s = space

\d{1,4} = any digit [1,5]
\w+ = more than 1 
\$ = end of a string
\^ = beginning of a string

'''
findall(r"\s", string)
split(r'\s', string)

string="hello, world"
print(list(string))
print(string.upper())
print(string.lower())
print(string.title())
print(string.split()) #split en una lista de dos palabras. split('\n') separa por lineas
print("".join(string)) #el espacio del principio es la separacion ,pero puede ser ", " o hasta '\n'
print(string.replace("h", "J"))
print("       Hello World   ".strip()) #tambien se pueden quitar otros caracteres del principio o final, con .strip(!)
print(string.find('w'))
print(string.rfind('w'))
print(string.index("w")) == print(string.find("w"))
print(string.capitalize())
print(string.endswith("rld"))
print(string.startswith("hel"))


ord('w') == chr(119)

#%% Datetime
from datetime import *
import time
df = pd.read_csv("AirPassengers.csv")

#Pasar a datetime
df.dtypes
df['date'] = pd.to_datetime(df['date'])
pd.date_range('2000-1-1', periods=6, freq = 'M')

#Obtener mes, año... de una fecha
birthday = datetime(1991, 12, 31, 17, 00, 12) #año, mes, dia, hora, min, seg,... date, time
birthday.year #o month, day, hour, weekday, timestamp
birthday.timestamp()

#funciones datetime
datetime.now()
datetime(2018, 1, 1)-datetime(2017, 12, 12)
timestamp = time.time()
date.fromtimestamp(1326244364)
datetime.today()
date.today()
d = date.fromisoformat('2019-11-04')
d = date(1991, 2, 5)
d = d.replace(year=1992, month=1, day=16)

#timedelta
t = timedelta(days = 5, hours = 1, seconds = 33, microseconds = 233423)
t.total_seconds()

#formato datetime
print(datetime.strptime("Jan 15, 2018", "%b %d, %Y"))
print(datetime.strftime(datetime.now(), "%b %d, %Y"))

#Others
time.sleep(seconds)

#Calendar
import calendar
print(calendar.calendar(2020))   
print(calendar.month(2020, 11))     
print(calendar.weekday(2020, 12, 24))                
print(calendar.isleap(2019))


#%% SCIPY
     
from scipy.special import * #-----------------------------------------------
#Find cubic root of 27 & 64 using cbrt() function
cbrt([27, 64])
comb(5, 2, exact = False, repetition=True)                           
perm(5, 2, exact = True)


from scipy import linalg #--------------------------------------------------
matriz = np.matrix('4 5; 3 2')
matriz = np.array([ [4,5], [3,2] ])
#Determinante
linalg.det(matriz)                           

#Matriz inversa
linalg.inv(matriz)

#eigenvalues, eigenvectors
eg_val, eg_vect = linalg.eig(matriz)                           

#Solve linear equations
#x+3y+5z = 10
#2x+5y+z = 8
#2x+3y+8z = 3
a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
b = np.array([2, 4, -1])
x = linalg.solve(a, b)
                           

import scipy.integrate #----------------------------------------------------
from numpy import exp
f= lambda x:exp(-x**2)
i = scipy.integrate.quad(f, 0, 1)


#Interpolar y usar splines
import numpy as np
from scipy.interpolate import * #---------------------------------------------
import matplotlib.pyplot as plt
x = np.linspace(0, 4, 12)
y = np.cos(x**2/3+4)

plt.plot(x, y)
plt.show()

#Interpolacion
f1 = interp1d(x, y,kind = 'linear')
f2 = interp1d(x, y, kind = 'cubic')

plt.plot(x, y, 'o', x, f1(x), '-', x, f2(x), '--')
plt.legend(['data', 'linear', 'cubic','nearest'], loc = 'best')
plt.show()

#Splines
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * np.random.randn(50)
plt.plot(x, y, 'ro', ms = 5)


spl = UnivariateSpline(x, y)
xs = np.linspace(-3, 3, 1000)
plt.plot(xs, spl(xs))

spl.set_smoothing_factor(0.01)
plt.plot(xs, spl(xs), 'b', lw = 3)
plt.show()


#%%

#Comprehensions
matrix = [[col for col in range(5)] for row in range(5)]
celsius = [0, 10, 15, 32, -5, 27, 3] #Para pasar a fahrenheit
fahrenheit = [temp*(9/5) + 32 for temp in celsius]

#Diccionario comprehensions
grados = {key:value for key, value in zip(celsius, fahrenheit)} #una forma de hacer diccionario
grados = dict(zip(celsius, fahrenheit))

#Comprimido
a=zip(celsius, fahrenheit)
print(*a) #* para descomprimir

#Iteradores
celsius2 = [c for c in celsius if c >= 12]
result=(c for c in celsius)
next(result)
result2 = iter(celsius)
next(result2)














#%% library(dplyr)
import dplython as dp
iris = dp.DplyFrame(iris)
from dplython import (DplyFrame, X, diamonds, select, sift,
  sample_n, sample_frac, head, arrange, mutate, group_by,
  summarize, DelayFunction)
from dplython import *

iris >> dp.select(X.Species) >> dp.head()
iris >> dp.sift(X.PetalLength>5)
iris >> dp.group_by(X.Species) >> dp.summarize(media=X.PetalLength.mean())
iris >> dp.mutate(redondeado=X.PetalLength.round(), redondeado2=X.SepalLength.round())
iris >> dp.distinct(X.SepalLength)
iris >> dp.arrange(X.PetalLength)


#%% timeit

import timeit  
  
# code snippet to be executed only once  
mysetup = "from math import sqrt"
  
# code snippet whose execution time is to be measured  
mycode = '''  
def example():  
    mylist = []  
    for x in range(100):  
        mylist.append(sqrt(x))  
'''
  
# timeit statement  
print (timeit.timeit(setup = mysetup, 
                     stmt = mycode, 
                     number = 10000)) 

#%%
import datatable as dt
import numpy as np
#Lectura
df= dt.fread("data.csv")

np.random.seed(1)
df = dt.Frame(np.random.randn(1000000))
pandas = df.to_pandas()
df = dt.Frame(pandas)
#Conversiones
df.to_numpy()
df.to_pandas()

#propiedades
df.shape
df.names[:5]
df.stypes[:5]
df.head(10)

df.sum()      
df.nunique()
df.sd()       
df.max()
df.mode()     
df.min()
df.nmodal()   
df.mean()

#Data manipulation

df[:,'funded_amnt'] #seleccionar columna
df[:5,:3] #seleccionar fila
df.sort('funded_amnt_inv') #ordenar
del df[:, 'member_id'] #borrar columnas


df[dt.f.loan_amnt>dt.f.funded_amnt,"loan_amnt"] #filtering.   .f para referirnos al dataframe sobre el que operamos
filtro = (f.loan_amnt>f.funded_amnt)
df[filtro,:]
df[:, dt.sum(dt.f.funded_amnt), dt.by(dt.f.grade)] #group by grade, seleccionar funded_amt y hacer la suma. 

df.to_csv('output.csv')