# DATA TABLE
# ----------------------------------------------------------------------
library(data.table)

# Que es un data.table
class(as.data.table(mtcars))
# resultado: "data.table" "data.frame"

# convertir un data.frame en data.table
dt <- as.data.table(mtcars)
# crear un data.table inline
dt <- data.table(a = 1:6, b = 7:12, c = 13:18)
# READ.CSV vs FREAD(data.table function)
# ----------------------------------------------------------------------
# Tiempos de READ.CSV para leer un csv de 38MB
system.time(df<-read.csv("c:/data/dataset_mt.csv"))
# resultados:
#   user  system elapsed 
#  15.44    0.32   15.78 

# Tiempos de FREAD para leer un csv de 38MB
system.time(dt<-fread("c:/data/dataset_mt.csv",showProgress=F))
# resultados:
#   user  system elapsed 
#  11.71    0.01   11.78


# WRITE.CSV vs FWRITE(data.table function)
# ----------------------------------------------------------------------
# Tiempos de WRITE.CSV para escribir un csv de 160MB en disco
df = mtcars[sample(32,1000000,T),]
system.time(write.csv(df,file="c:/data/dataset_mt.csv",row.names=F))
# resultados:
#   user  system elapsed 
#  62.41    0.49   63.86 

# Tiempos de FWRITE para escribir un csv de 160MB en disco
dt = as.data.table(mtcars[sample(32,1000000,T),])
system.time(fwrite(dt,file="c:/data/dataset_mt.csv",showProgress = F))
# resultados:
#   user  system elapsed 
#   4.36    0.06    1.48 


# GROUP BY en Data.Table
# ----------------------------------------------------------------------
# GROUP BY 
iris_dt = as.data.table(iris)
# promedio de Sepal.Length agrupado por Species
iris_dt[, .(media=mean(Sepal.Length)), by = Species]
# promedio de Sepal.Length agrupado por Species solo casos donde sepal.length >=6 
iris_dt[Sepal.Length >= 6, .(media=mean(Sepal.Length)), by = Species]
iris_dt[Sepal.Length >= 6, .(media=mean(Sepal.Length)), keyby = Species] #ordenado
#Registros por Species donde Sepal.Length>=6
iris_dt[Sepal.Length >= 6, .(.N), by = .(Species)] #by usar .() para varias columnas

# BORRAR COLUMNAS Y NAs en Data.Table
# ----------------------------------------------------------------------
# DROP. Borrar una columna 'vs' y 'am' del data.table mtcars_dt
mtcars_dt <- as.data.table(mtcars)
mtcars_dt[, c('vs','am') := NULL]

# NAs. Eliminar las filas que contienen valores NA en un data.table
mtcars_dt = na.omit(mtcars_dt)


# JOINS en Data.Table
# ----------------------------------------------------------------------
# crea data.tables
dt1 = data.table(id_user=c(1,2,3,4), edad=c(42,52,62,66))
dt2 = data.table(id_user=c(1,2,3), cantidad=c(4,5,6))
# right outer join unkeyed data.tables - use `on` argument
dt1[dt2, on = "id_user"]
# join para "not in"
dt1[!dt2, on = "id_user"]
# inner join
merge(dt1, dt2, by = "id_user")
# full outer join
merge(dt1, dt2, by = "id_user", all = TRUE)


# FILTROS en Data.Table
# ----------------------------------------------------------------------
dt <- as.data.table(iris)
setosas <- dt[Species == 'setosa']
head <- dt[1:10]

#ORDENAR
#------------------------------------------------------------------------
dt <- as.data.table(iris)
ordenado <- dt[order(Petal.Width, -Sepal.Length)]

#SELECCIONAR COLUMNAS
#-----------------------------------------------------------------------
dt <- as.data.table(iris)
selected <- dt[, .(Petal.Width, Sepal.Length)]

columns = c("Petal.Width", "Sepal.Length")
dt[,..columns]

#Para seleccionar un slice
dt[, Sepal.Length:Petal.Length]

#QUITAR COLUMNAS
#----------------------------------------------------------------------
dt <- as.data.table(iris)
dt[, -c("Petal.Width", "Sepal.Length")]
dt[, !c("Petal.Width", "Sepal.Length")]

columns = c("Petal.Width", "Sepal.Length")
dt[,-..columns]

dt[, -(Sepal.Length:Petal.Length)]
#RENOMBRAR COLUMNAS
#-----------------------------------------------------------------------
renamed <- dt[, .(PW = Petal.Width, SL = Sepal.Length)]

#OTRAS
#-----------------------------------------------------------------------
#Registros con esas condiciones
iris_dt[Sepal.Length >= 6, .N]

#Concatenar %>%
iris_dt[Sepal.Length <= 6][Sepal.Length >= 1]

#.SD se refiere a si mismo
iris_dt[,print(.SD)]
iris_dt[,head(.SD), by=Species]

#Operacion multiples columnas
iris_dt[,-("Species")][,lapply(.SD, mean)]
iris_dt[,lapply(.SD, mean), .SDcols=c('Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width')]
