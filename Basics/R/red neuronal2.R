circulo <- function(x, R, centroX=0, centroY=0){
  r = R * sqrt(runif(x))
  theta = runif(x) * 2 * pi
  x = centroX + r * cos(theta)
  y = centroY + r * sin(theta)
  
  z = data.frame(x = x, y = y)
  return(z)
}
datos1 <- circulo(100,0.5)
datos2 <- circulo(100,1.5)

datos1$Y <- 1
datos2$Y <- 0
datos <- rbind(datos1,datos2)

#rm(datos1,datos2, circulo)
library(ggplot2)
ggplot(datos,aes(x,y, col = as.factor(Y))) + geom_point()
X <- as.matrix(datos[,1:2])
Z<-(X[,1]^2+X[,2]^2)
#X=cbind(X, Z)
Y <- as.matrix(datos[,3])

#rm(datos)

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

sigmoid = function(x) {
  y = list() 
  y[[1]] <- (1 / (1 + exp(-x)))
  y[[2]] <- x * (1 - x)
  #y[[2]] <- (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x))))
  return(y)
}
s<- seq(-10, 10, 0.01)
library(gridExtra)
ggplot()+geom_line(aes(s, sigmoid(s)[[1]]), col='blue')+
            geom_line(aes(s, sigmoid(s)[[2]]), col='green')

relu <- function(x){
  y <- list()
  y[[1]] <- ifelse(x<0,0,x)
  y[[2]] <- ifelse(x<0,0,1)
  return(y)
}

plot(x, relu(x)[[2]], col='blue')

n = ncol(X) #Núm de neuronas en la primera capa
capas = c(n, 8, 1) # Número de neuronas en cada capa. La ultima es una, porque es una regresion
funciones = list( relu,relu) # Función de activación en cada capa

red <- list()

for (i in 1:(length(capas)-1)){
  red[[i]] <- neurona$new(funciones[i],capas[i], capas[i+1])
}

red



coste <- function(Yp,Yr){
  y <- list()
  y[[1]] <- mean((Yp-Yr)^2)
  y[[2]] <- (Yp-Yr)
  return(y)
}


kk=1
rrr=t(data.frame(c(kk,Y)))
red_neuronal <- function(red, X,Y, coste,lr = 0.05){
  for(kk in 1:1000){
    ## Front Prop
    out = list()
    out[[1]] <- append(list(matrix(0,ncol=2,nrow=1)), list(X))
    
    for(i in c(1:(length(red)))){
      z = list((out[[length(out)]][[2]] %*% red[[i]]$W + red[[i]]$b))
      a = list(red[[i]]$fun_act[[1]](z[[1]])[[1]])
      out[[i+1]] <- append(z,a)
    }
    
    
    rrr=as.data.frame(rbind(rrr,c(kk,t(as.data.frame(a[[1]])))))
    pop=gather(rrr,key="iter",value="number",2:201)
    #,coste(a,Y)[[2]], red[[i]]$fun_act[[1]](z)[[2]]), t(out[[i]][[2]]
    ggplot()+
    geom_point(aes(x=pop$V1,y=pop$number,color=as.factor(pop$iter),group=pop$iter),show.legend = F)
    #   scale_y_continuous(limits = c(0.5,0.8))
    #   geom_point(aes(x=1:length(copia$V1),y=copia$V1),color="black")+
    #   geom_point(aes(x=1:length(rr$V1),y=rr$V2),color="green")
    #   geom_point(aes(x=1:length(rr$V1),y=rr$V3),color="red")
    #   geom_point(aes(x=1:length(rr$V1),y=rr$V4),color="black")
    # geom_point(aes(x=1:length(rr$V1),y=rr$V5),color="brown")
    #   geom_point(aes(x=1:length(rr$V1),y=rr$V4*rr$V3),color="orange")
    #   scale_colour_manual(values = c("blue"="prediccion"))
    
    # rrr=as.data.frame(cbind( red[[i]]$b, red[[i]]$w))
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
    kk=kk+1
  }
 
  #return(out)
  return(out[[length(out)]][[2]])
  
}

resultado <- red_neuronal(red, X,Y, coste)
dim(resultado)

#Reiniciamos la neurona
red <- list()
n = ncol(X) #Núm de neuronas en la primera capa
capas = c(n, 5, 5, 1) # Número de neuronas en cada capa. La ultima es una, porque es una regresion
funciones = list(sigmoid, sigmoid, sigmoid) # Función de activación en cada capa
for (i in 1:(length(capas)-1)){
  red[[i]] <- neurona$new(funciones[i],capas[i], capas[i+1])
}

#entrenamos
for(i in seq(50000)){
  
  Yt = red_neuronal(red, X,Y, coste, lr=0.08)
  
  if(i %% 25 == 0){
    if(i == 25){
      iteracion <- i
      error <- coste(Yt,Y)[[1]]
    }else{
      iteracion <- c(iteracion,i)
      error <- c(error,coste(Yt,Y)[[1]])      
    }
  }
}

library(ggplot2)

grafico = data.frame(Iteracion = iteracion,Error = error)
ggplot(grafico,aes(iteracion, error)) + geom_line() + theme_minimal() +
  labs(title = "Evolución del error de la Red Neuronal")+
  scale_y_continuous(limits=c(0,0.25))

#Usando librerias
library(nnet)
# fit model
fit <- nnet(Y~X, maxit=1000, decay=0.05, size=8)
fit
summary(fit)
# make predictions
predictions <- predict(fit, X, type="raw")


#analisis de resultados
hist(Yt)
hist(predictions)
d=as.numeric(rrr[1000,2:201])
dd=as.data.frame(d)%>%dplyr::mutate(pred=ifelse(d>0.5,1,0))
datos=data.frame(x=X[,1], y=X[,2], Y=Y, pred=dd$pred)
#/datos$pred=ifelse(datos$pred>median(datos$pred), 1, 0)
#datos$nnet=ifelse(predictions>median(predictions), 1, 0)



ggplot(datos,aes(x,y, size = as.factor(Y), col=as.factor(pred))) + geom_point()


x11(width = 115,height = 60)
setwd("C:/Users/ldelaguila/Desktop/red_neuronal")
ggsave("relu_sigmoid_281-0.08-evolucion.png")
