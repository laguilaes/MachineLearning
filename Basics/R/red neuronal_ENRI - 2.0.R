x11(width = 115,height = 60)

###################################################################################################
###                                                                                             ###
###                   SCRIPT COMPLETO PARA CREAR UNA RED NEURONAL CON SUMAS                     ###
###                                                                                             ###
###################################################################################################
##### ----- Función para crear un círculo --------------------------------------------- #####
#La función crea un círculo según:
#x:número de puntos
#R: Radio
#centroX y centroY son el centro del circulo
circulo <- function(x, R, centroX=0, centroY=0){
  r = R * sqrt(runif(x))
  theta = runif(x) * 2 * pi
  x = centroX + r * cos(theta)
  y = centroY + r * sin(theta)
  
  z = data.frame(x = x, y = y)
  return(z)
}
#Devuelve las coordenadas cartesianas de cada punto


##### ----- Creamos la clase neurona del tipo R5 -------------------------------------- #####

neurona <- setRefClass(
  Class="neurona",
  fields = list(
    fun_act = "list",
    numero_conexiones = "numeric",
    numero_neuronas = "numeric",
    W = "matrix",
    b = "numeric"
  ),
  methods = list(
    initialize = function(fun_act, numero_conexiones, numero_neuronas)
    {
      fun_act <<- fun_act
      numero_conexiones <<- numero_conexiones
      numero_neuronas <<- numero_neuronas
      W <<- matrix(runif(numero_conexiones*numero_neuronas),
                   nrow = numero_conexiones)
      b <<- runif(numero_neuronas)
    }
  )
)

##### ----- Creamos las funciones de activación que estará disponibles ---------------- #####
sigmoid = function(x) {
  y = list() 
  y[[1]] <- 1 / (1 + exp(-x))
  y[[2]] <- x * (1 - x)
  return(y)
}

relu <- function(x){
  y <- list()
  y[[1]] <- ifelse(x<0,0,x)
  y[[2]] <- ifelse(x<0,0,1)
  return(y)
}

##### ----- Creamos las funciones de coste mediante las que optimizaremos el error----- #####
coste <- function(Yp,Yr){
  y <- list()
  y[[1]] <- mean((Yp-Yr)^2)
  y[[2]] <- (Yp-Yr)
  return(y)
}

##### ----- Creamos las funcion de propagación de la red neuronal---------------------- #####
red_neuronal <- function(red, X,Y, coste,lr = 0.05){
  ## Front Prop
  out = list()
  out[[1]] <- append(list(matrix(0,ncol=2,nrow=1)), list(X))
  
  for(i in c(1:(length(red)))){
    z = list((out[[length(out)]][[2]] %*% red[[i]]$W + red[[i]]$b))
    a = list(red[[i]]$fun_act[[1]](z[[1]])[[1]])
    out[[i+1]] <- append(z,a)
  }
  
  
  ## Backprop & Gradient Descent
  delta <- list() 
  
  for (i in rev(1:length(red))){
    z = out[[i+1]][[1]]
    a = out[[i+1]][[2]]
    
    if(i == length(red)){
      delta[[1]] <- coste(a,Y)[[2]] * red[[i]]$fun_act[[1]](a)[[2]]
    } else{
      delta <- list(delta[[1]] %*% W_temp * red[[i]]$fun_act[[1]](a)[[2]],delta)
    }
    
    W_temp = t(red[[i]]$W)
    
    red[[i]]$b <- red[[i]]$b - mean(delta[[1]]) * lr
    red[[i]]$W <- red[[i]]$W - t(out[[i]][[2]]) %*% delta[[1]] * lr
    
  }
  #return(out)
  return(out)
  
}


###################################################################################################
###                                                                                             ###
###                                     Desarrollo                                              ###
###                                                                                             ###
###################################################################################################
##### ----- Creamos el reto para la ANN----------------------------------------------- #####
datos1 <- circulo(500,0.5)
datos2 <- circulo(500,1.5)
##### ----- Definimos las soluciones-------------------------------------------------- #####
datos1$Y <- 1
datos2$Y <- 0
datos <- rbind(datos1,datos2)
##### ----- Representamos el reto----------------------------------------------------- #####
library(ggplot2)
ggplot(datos,aes(x,y, col = as.factor(Y))) + geom_point()
##### ----- Definición de variables de entrada y variables objetivo------------------- #####
X <- as.matrix(datos[,1:2])
Y <- as.matrix(datos[,3])

##### ----- Visualización de funciones de activación disponibles---------------------- #####
x <- seq(-5, 5, 0.01)
plot(x, sigmoid(x)[[2]], col='blue')
plot(x, relu(x)[[1]], col='blue')

##### ----- Definimos las características de la red neuronal-------------------------- #####
n = ncol(X) #NÃºm de neuronas en la primera capa
capas = c(n, 4, 8, 1) # NÃºmero de neuronas en cada capa. La ultima es una, porque es una regresion
funciones = list(sigmoid, sigmoid, sigmoid) # FunciÃ³n de activaciÃ³n en cada capa

##### ----- Creación automática de la red inicial con las características anteriores-- #####
red <- list()

for (i in 1:(length(capas)-1)){
  red[[i]] <- neurona$new(funciones[i],capas[i], capas[i+1])
}

##### ----- Entrenamiento de la red--------------------------------------------------- #####
ws=list()
bs=list()
salidas=data.frame()
for(i in seq(5000)){

  Yt = red_neuronal(red, X,Y, coste, lr=0.01)
  
  if(i %% 10 == 0){
    if(i == 10){
      iteracion <- i
      error <- coste(Yt[[length(Yt)]][[2]],Y)[[1]]
      salidas=data.frame(valor=Yt[[length(Yt)]][[2]], iter=i, elem=1:nrow(Yt[[length(Yt)]][[2]]))
      
      for (k in 1:length(red)){
        ws[[k]]=data.frame(matrix(red[[k]]$W, nrow = 1))
        bs[[k]]=data.frame(matrix(red[[k]]$b, nrow = 1))
      }

      
    }else{
      iteracion <- c(iteracion,i)
      error <- c(error,coste(Yt[[length(Yt)]][[2]],Y)[[1]]) 
      salidas=rbind(salidas, data.frame(valor=Yt[[length(Yt)]][[2]], iter=i, elem=1:nrow(Yt[[length(Yt)]][[2]])))
      
      for (k in 1:length(red)){
        ws[[k]]=rbind(ws[[k]], data.frame(matrix(red[[k]]$W, nrow = 1)))
        bs[[k]]=rbind(bs[[k]], data.frame(matrix(red[[k]]$b, nrow = 1)))
      }
      
    }
  }
}


###################################################################################################
###                                                                                             ###
###                           Visualización de Resultados                                       ###
###                                                                                             ###
###################################################################################################
##### ----- Selección de carpetas para guardar resultados----------------------------- #####
dir_escritorio="C:/Users/laguila/Desktop/"
nombre_carpeta="Pruebas_ANN"
Tipologia_pruebas="Pruebas_iniciales"
##### ----- Seteo de escritorios seleccionados---------------------------------------- #####
setwd(dir_escritorio)
suppressWarnings(dir.create(nombre_carpeta))
dir = paste0(dir_escritorio,nombre_carpeta,"/")
setwd(dir)
suppressWarnings(dir.create(Tipologia_pruebas))
setwd(paste0(dir,Tipologia_pruebas,"/"))

##### ----- Seteamos el tema de las gráficas------------------------------------------ #####
theme_set(theme_minimal() +
            theme(axis.title.x = element_text(size = 15, hjust = 1),
                  axis.title.y = element_text(size = 15),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12),
                  legend.text = element_text(size = 12,color="black"),
                  legend.title =element_text(size = 14,color="red",face = "bold.italic"),
                  legend.background = element_rect("gray"),
                  panel.grid.major = element_line(linetype = 2),
                  panel.grid.minor = element_line(linetype = 2),
                  panel.background = element_rect("gray99"),
                  plot.title = element_text(size = 18, colour = "grey25", face = "bold"), plot.subtitle = element_text(size = 16, colour = "grey44")))

##### ----- Visualización Nº Iteración vs Error--------------------------------------- #####
library(ggplot2)

grafico = data.frame(Iteracion = iteracion,Error = error)

ggplot(grafico,aes(iteracion, error)) + geom_line(color="red") + theme_minimal() +
  labs(title = "Evolución del error de la Red Neuronal")

ggsave(paste0( "Evolución del error de la Red Neuronal ", paste0(as.character(capas),collapse = "-"),".png"))

##### ----- Visualización Predicción vs Realidad-------------------------------------- #####

a=ggplot(datos,aes(x, y,color=as.factor(Y))) + geom_point()  +
  labs(title = "Soluciones")+
  scale_color_discrete(name="Tipo")
corte=(table(Y)/length(Y))[[1]]
pred=ifelse(Yt[[length(Yt)]][[2]]>corte,1,0)
prediccion=as.data.frame(cbind(X,pred))
hist(prediccion$y)
b=ggplot(prediccion,aes(x, y,color=as.factor(V3))) + geom_point()  +
  labs(title = "Predicción")+
  scale_color_discrete(name="Tipo")

library(gridExtra)
grid.arrange(a,b,nrow=1)

ggsave(paste0( "Predicciones ", paste0(as.character(capas),collapse = "-"),".png"))


##### ----- Visualización salidas red neuronal--------------------------------------- #####
library(ggplot2)

ggplot(salidas,aes(x=iter, y=valor, color=elem, group=elem)) + 
  geom_line() + 
  theme_minimal() +
  labs(title = "Evolución del error de la Red Neuronal")

ggsave(paste0( "Evolución del error de la Red Neuronal ", paste0(as.character(capas),collapse = "-"),".png"))


##### ----- Visualización evolucion pesos--------------------------------------- #####
library(tidyr)
pesos=ws[[1]]
for (k in 2:length(red)){
  pesos=cbind(pesos,ws[[k]])
}
pesos$iter=1:nrow(pesos)
colnames(pesos)=c(1:(ncol(pesos)-1), "iter")
pesos=gather(pesos, key=Wi,value=valor,-iter)
ggplot(pesos,aes(x=iter, y=valor, color=Wi, group=Wi)) + 
  geom_line() + 
  theme_minimal() +
  labs(title = "Evolución del error de la Red Neuronal")

ggsave(paste0( "Evolución del error de la Red Neuronal ", paste0(as.character(capas),collapse = "-"),".png"))

##### ----- Visualización salidas red neuronal--------------------------------------- #####
library(ggplot2)


bias=bs[[1]]
for (k in 2:length(red)){
  bias=cbind(bias,bs[[k]])
}
bias$iter=1:nrow(bias)
colnames(bias)=c(1:(ncol(bias)-1), "iter")
bias=gather(bias, key=bi,value=valor,-iter)
ggplot(bias,aes(x=iter, y=valor, color=bi, group=bi)) + 
  geom_line() + 
  theme_minimal() +
  labs(title = "Evolución del error de la Red Neuronal")

ggsave(paste0( "Evolución del error de la Red Neuronal ", paste0(as.character(capas),collapse = "-"),".png"))