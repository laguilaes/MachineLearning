rfRFE <-  list(summary = summaryKappa,
               fit = function(x, y, lev, last, classProbs, ...) { 
                 xgb.train = xgb.DMatrix(data=x,label=y)
                 model=xgboost::xgb.train(data = xgb.train,
                                           nrounds = 50,
                                           ...)
                 #print("--------")
                 #print(model)
                 model
               },
               pred = function (object, x){
                 #print("a")
                 tmp <- round(predict(object, x))
                 #print(tmp)
                 if (is.factor(object$y)) {
                    #print("aqui")
                    out <- cbind(data.frame(pred = tmp), as.data.frame(predict(object, x, type = "prob")))
                 }
                 else out <- tmp
                 table(tmp)
                 out
               },
               rank = function (object, x, y){
                 #print("b")
                 xgb.train = xgb.DMatrix(data=x,label=y)
                 vimp <- as.data.frame(xgb.importance(colnames(xgb.train), model = object))%>%
                   arrange(desc(Gain))
                 colnames(vimp)[1:2]=c("var", "Overall")
                 vimp=vimp[,c(2,1)]
                 #print(vimp, max = 10)
                 vimp
               },
               selectSize = function (x, metric, maximize){
                 #print(x)
                 #print(metric)
                 #print(maximize)
                 best <- if (maximize) 
                   which.max(x[, metric])
                 else which.min(x[, metric])
                 min(x[best, "Variables"])
               },
               selectVar = pickVars)

ctrl <- rfeControl(functions = rfRFE, #puede ser lmFuncs, rfFuncs, nbFuncs o treebagFuncs
                   method = "cv", 
                   number = 5, 
                   # returnResamp = "all", 
                   verbose = FALSE)

set.seed(342)


results2 <- caret::rfe(x = as.matrix(iris[,1:4]), y=as.factor(as.numeric(iris[,5])),
                      sizes = c(1,2,3,4),
                      metric = "WeightedKappa",
                      rfeControl = ctrl)
plot(results)





