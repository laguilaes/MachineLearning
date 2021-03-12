rfRFE <-  list(
  summary = summaryKappa,
  fit = function (x, y, first, last, ...) {
    train(x, y, method="xgbLinear", ...)
  },
  pred = function (object, x) {
    tmp <- predict(object, x)
    if (is.factor(object$y)) {
      out <- cbind(data.frame(pred = tmp), as.data.frame(predict(object, x, type = "prob")))
    }
    else
      out <- tmp
    out
  },
  rank = function (object, x, y) {
    vimp <- varImp(object)
    if (is.factor(y)) {
      if (all(levels(y) %in% colnames(vimp))) {
        avImp <- apply(vimp[, levels(y), drop = TRUE], 1, mean)
        vimp$Overall <- avImp
      }
    }
    print(vimp$model)
    vimp=vimp$importance
    #vimp <- vimp[order(vimp$Overall, decreasing = TRUE), drop = FALSE]
    if (ncol(x) == 1) {
      vimp$var <- colnames(x)
    }
    else
      vimp$var <- rownames(vimp)
    
    vimp
  },
  selectSize = function (x, metric, maximize) {
    best <- if (maximize)
      which.max(x[, metric])
    else
      which.min(x[, metric])
    min(x[best, "Variables"])
  },
  selectVar = pickVars
)


ctrl <- rfeControl(functions = rfRFE, #puede ser lmFuncs, rfFuncs, nbFuncs o treebagFuncs
                   method = "cv", 
                   number = 2, 
                   returnResamp = "all", 
                   verbose = FALSE)

set.seed(342)


results <- caret::rfe(x = as.matrix(iris[,1:4]), y=as.factor(iris[,5]),
                      sizes = seq(from=1, to=4, by=1),
                      metric = "WeightedKappa",
                      rfeControl = ctrl)



