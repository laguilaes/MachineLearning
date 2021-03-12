
library(reticulate)
#Crear nuevo environmwent conda create -n yourenvname anaconda
#Activar el environment conda activate yourenvname
#Instalacion de paquetes conda install -n yourenvname pandas
#desactivar environment: conda deactivate
#borrar conda remove -n yourenvname -all
use_condaenv("C:/Users/laguila/Anaconda3/envs/myenv")

#Usar funciones directamente
datasets <- import("sklearn.datasets")
data = datasets$load_iris(return_X_y = TRUE)
head(iris$data)

#funciones en la consola
repl_python()

#Desde scripts
source_python("flights.py") #archivo python con la funcion
flights <- read_flights("flights.csv") #funcion en python que devuelve el dataset