library(Metrics)     #libreria para calcular el weighted kappa


#Funcion Summary para mostrar el Weighted Kappa
summaryKappa<-function (data, lev = NULL, model = NULL,...) 
{ 
  # adaptation of twoClassSummary
  require(Metrics)
  
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))){ 
    stop("levels of observed and predicted data do not match")
  }

  Kappa <- try(Metrics::ScoreQuadraticWeightedKappa(rater.a = as.numeric(data$pred), rater.b = as.numeric(data$obs), min.rating = 0, max.rating = 10))

  ret <- if (class(Kappa)[1] == "try-error") {
    NA
  } else {Kappa   
  }
  out<-ret 
  names(out) <- c("WeightedKappa")
  out 
}



ctrl <-
  trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 5,
    summaryFunction = summaryKappa,  #se le indica la funcion summary de arriba
    classProbs = TRUE
  )
set.seed(123)

svmTune <-
  train(
    Class ~ .,
    data = training,
    method = "svmRadial",
    trControl = ctrl,
    metric = "WeightedKappa",   #y la metrica
    verbose = FALSE
  )

