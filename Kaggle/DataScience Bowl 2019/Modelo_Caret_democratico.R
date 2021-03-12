library(tidyverse) # metapackage with lots of helpful functions
library(tidyverse)
library(scales)
library(lubridate)
library(C50)
library(xgboost)
library(data.table)
library(stringr)
library(dplyr)
library(crayon)
library(rsample)
library(caret)


####PRE-------------
train <- data.table::fread("C:/Users/ldelaguilar/Google Drive/Kaggle/datos iniciales/train.csv", stringsAsFactors = F)
train_labels <- data.table::fread("C:/Users/ldelaguilar/Google Drive/Kaggle/datos iniciales/train_labels.csv", stringsAsFactors = F)
# train <- data.table::fread("../input/data-science-bowl-2019/train.csv", stringsAsFactors = F)
# train_labels <- data.table::fread("../input/data-science-bowl-2019/train_labels.csv", stringsAsFactors = F)
print("Training datasets leidos exitosamente")

#specs <- data.table::fread("../input/data-science-bowl-2019/specs.csv")
indice=first(which(train$installation_id=="8123bc13"))
#quitar columnas que no queremos, y ordenar por tiempo
get_accuracy <- function(df, game ,code){
  tmp <- df %>%
    filter(str_detect(title , pattern = game)) %>%
    filter(event_code == code) %>%
    mutate(event_data = gsub('"', '', event_data)) %>%
    mutate(Status =  ifelse(str_detect(event_data, pattern = 'correct:true'),  'correct',
                            ifelse(str_detect(event_data, pattern = 'correct:false'),  'incorrect',
                                   NA))) %>%
    group_by(installation_id, game_session,indice_assessment, title) %>%
    summarise(num_correct = sum(Status %in% 'correct'),
              num_incorrect = sum(Status %in% 'incorrect')) %>%
    ungroup %>%
    mutate(accuracy = num_correct/(num_correct+num_incorrect)) %>%
    mutate(accuracy_group = ifelse(accuracy == 0,  0L,
                                   ifelse(accuracy> 0 & accuracy < 0.5,  1L,
                                          ifelse(accuracy >= 0.5 & accuracy < 1,  2L,
                                                 3L
                                          ))))
  return(tmp)
}
extractor_par_train=function(train,train_labels){
  #train_=train%>%dplyr::select(-event_data, -event_code)#%>%arrange(installation_id, timestamp)
  train_=train
  #quitar installation_id sin assessment
  ids=as.data.frame(unique(train_[str_detect(train_$type, "Assessment"),"installation_id"]))
  colnames(ids)="installation_id"
  train3=merge(train_, ids, by="installation_id", keep.all=T)%>%arrange(installation_id, timestamp)
  
  
  indices=which(train3$type=="Assessment" & train3$game_time==0)[!duplicated(substr(as.character(train3[which(train3$type=="Assessment" & train3$game_time==0),]$timestamp),1,19))]
  indices=indices[diff(indices)>2]
  train3$indice_assessment=NA
  train3$indice_assessment[indices]=1:length(indices)
  
  #cuentas=as.data.frame(train%>%group_by(installation_id)%>%filter(indice_assessment>0)%>%summarise(count=n()))$count
  
  
  # train%>%group_by(installation_id)%>%dplyr::
  #   mutate(order(unique(indice_assessment))=1:length(order(unique(indice_assessment))))
  # rain[train$installation_id=="0006a69f",]$indice_assessment`[order(unique(train[train$installation_id=="0006a69f",]$indice_assessment))]
  #
  train2=train3%>%group_by(installation_id)%>%fill(indice_assessment, .direction = "up")
  #
  # seq(1,cuentas,cuentas)
  
  train4=train2%>%dplyr::group_by(installation_id)%>%mutate(indice_assessment=indice_assessment-min(indice_assessment, na.rm=T)+1)
  
  aa=diff(train4$game_time)
  train4$diftime=c(0,aa)
  train4[train4$diftime<0,"diftime"]=0
  
  
  tiempo_total=train4%>%group_by(installation_id, indice_assessment)%>%summarise(tiempo_total=sum(diftime), world=last(world), game_session=last(game_session), title_assessment=last(title))
  tiempo_type=train4%>%group_by(installation_id, indice_assessment, type)%>%summarise(tiempo_total=sum(diftime))
  tiempo_title=train4%>%group_by(installation_id, indice_assessment, title)%>%summarise(tiempo_total=sum(diftime))
  BM=train4%>%group_by(installation_id, indice_assessment)%>%get_accuracy(.,'Bird Measurer' , '4110')
  colnames(BM)[4:8]=paste0(colnames(BM)[4:8],"_",BM$title[1])
  CB=train4%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Cart Balancer", '4100')
  colnames(CB)[4:8]=paste0(colnames(CB)[4:8],"_",CB$title[1])
  CF=train4%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Cauldron Filler", "4100")
  colnames(CF)[4:8]=paste0(colnames(CF)[4:8],"_",CF$title[1])
  CS=train4%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Chest Sorter", "4100")
  colnames(CS)[4:8]=paste0(colnames(CS)[4:8],"_",CS$title[1])
  MS=train4%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Mushroom Sorter", "4100")
  colnames(MS)[4:8]=paste0(colnames(MS)[4:8],"_",MS$title[1])
  total=train4%>%group_by(installation_id, indice_assessment)%>%filter(row_number()==n())
  arte_1=right_join(BM[,c(1,3,5:6)],total[,c("installation_id","indice_assessment")],by=c("installation_id","indice_assessment"))
  arte_1=right_join(CB[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  arte_1=right_join(CF[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  arte_1=right_join(CS[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  arte_1=right_join(MS[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  for (i in 1:length(arte_1)){
    arte_1[is.na(arte_1[,i]),i]=0
  }
  ##Sacar IQ installation_id
  tiempo_total=merge(arte_1,tiempo_total,by=c("installation_id","indice_assessment"))
  ##Elapsed Time
  tiempo_final=train4%>%group_by(installation_id, indice_assessment)%>%summarise(tiempo_final=max(timestamp))
  tiempo_inicial=train4%>%group_by(installation_id)%>%summarise(tiempo_inicial=min(timestamp))
  ttt=merge(tiempo_inicial,tiempo_final,by="installation_id",keep.all=T)
  ttt$elapsed_time=difftime(ymd_hms(ttt$tiempo_final),ymd_hms(ttt$tiempo_inicial),units = "secs")
  tiempo_total=merge(tiempo_total,ttt[,c("installation_id","indice_assessment","elapsed_time")],by=c("installation_id","indice_assessment"))
  
  ###Madrugada_mañana_Tarde
  tiempo_final$momento=as.numeric(substr(tiempo_final$tiempo_final,12,13))
  tiempo_final$momento_c=0
  tiempo_final$momento_c[tiempo_final$momento<8 | tiempo_final$momento>23]=0
  tiempo_final$momento_c[tiempo_final$momento>=8 & tiempo_final$momento<16]=1
  tiempo_final$momento_c[tiempo_final$momento>=16 & tiempo_final$momento<24]=2
  tiempo_total=merge(tiempo_total,tiempo_final[,c("installation_id","indice_assessment","momento_c")],by=c("installation_id","indice_assessment"))
  
  ###Tiempo dedicado en los ultimos 5 game_session
  e=train4%>%group_by(installation_id,indice_assessment,game_session)%>%summarise(tiempo_gs=max(game_time),max_time=max(timestamp))
  ee=e%>%group_by(installation_id,indice_assessment)%>%summarise(tiempo_gs=sum(tail(tiempo_gs,5)))
  tiempo_total=merge(tiempo_total,ee[,c("installation_id","indice_assessment","tiempo_gs")],by=c("installation_id","indice_assessment"))
  
  
  
  
  tiempotype2=spread(tiempo_type, key=type, value=tiempo_total)
  tiempotitle2=spread(tiempo_title, key=title, value=tiempo_total)
  
  df1=merge(tiempo_total, tiempotype2, by=c("installation_id", "indice_assessment"))
  
  num_event_id=train4%>%group_by(installation_id, indice_assessment, event_id)%>%dplyr::summarise(a = n())
  num_event_id2=spread(num_event_id, key=event_id, value=a)
  
  df2=merge(tiempotitle2, num_event_id2, by=c("installation_id", "indice_assessment"))
  df3=merge(df1, df2, by=c("installation_id", "indice_assessment"))
  df3[is.na(df3)]=0
  df4=df3[df3$indice_assessment!=0,]
  colnames(df4)[ colnames(df4)=="title_assessment"]="title"
  df4=df4%>%arrange(installation_id,as.numeric(indice_assessment))
  #mutate los valores acumulados
  variables_extr=c("installation_id","indice_assessment" , "world" ,"game_session" ,"title","title_assessment","elapsed_time","momento_c","tiempo_gs"  )
  variables_porfin=setdiff(colnames(df4),variables_extr)
  for (i in 2:max(df4$indice_assessment)){
    if (length(table(df4$indice_assessment==i))>1){
      indices=which(df4$indice_assessment==i)
      print(i)
      #for (k in indices){
      df4[indices,variables_porfin]=df4[indices-1,variables_porfin]+df4[indices,variables_porfin]
      #}
      
    }
  }
  
  df4$elapsed_time=as.numeric(df4$elapsed_time)
  df4$momento_c=as.numeric(df4$momento_c)
  df4$tiempo_gs=as.numeric(df4$tiempo_gs)
  # onteo=df4%>%group_by(game_session)%>%summarise(count=n())
  # onteo[onteo$count>1,]
  # View(df4[df4$game_session=="b07a285929f9d7b9",])
  # View(train_labels[train_labels$game_session=="b07a285929f9d7b9",])
  df5=merge(df4, train_labels, by=c("title", "game_session","installation_id"), keep.all=T)
  df6=df5[!duplicated(df5$game_session),]
  df6=df6%>%dplyr::arrange(installation_id, indice_assessment)
  return(df6)
}
train_par_ind1=extractor_par_train(train[1:(indice-1),],train_labels)
train_par_ind2=extractor_par_train(train[indice:length(train$event_id),],train_labels)
rm(train)
columnas=intersect(colnames(train_par_ind1),colnames(train_par_ind2))
train_par=rbind(train_par_ind1[,columnas],train_par_ind2[,columnas])
rm(train_par_ind1,train_par_ind2)
rm(train_labels)
print("Training datasets tratados exitosamente")
saveRDS(train_par,"train_biblia.rds")




###### TEST ###########
# read in required data
test <- data.table::fread("../input/data-science-bowl-2019/test.csv", stringsAsFactors = F)
test1=split(test,test$installation_id)
print("Test_dividido_correctamente")
extractor_par_test=function(test){
  # test=test%>%dplyr::select(-event_data, -event_code)#%>%arrange(installation_id, timestamp)
  # 
  # #quitar installation_id sin assessment
  # ids=as.data.frame(unique(test[str_detect(test$type, "Assessment"),"installation_id"]))
  # colnames(ids)="installation_id"
  # test=merge(test, ids, by="installation_id", keep.all=T)%>%arrange(installation_id, timestamp)
  
  # 
  # indices=which(test$type=="Assessment" & test$game_time==0)[!duplicated(substr(as.character(test[which(test$type=="Assessment" & test$game_time==0),]$timestamp),1,19))]
  # # indices=indices[diff(indices)!=1]
  # test$indice_assessment=NA
  # test$indice_assessment[indices]=1:length(indices)
  
  #cuentas=as.data.frame(train%>%group_by(installation_id)%>%filter(indice_assessment>0)%>%summarise(count=n()))$count
  
  
  # train%>%group_by(installation_id)%>%dplyr::
  #   mutate(order(unique(indice_assessment))=1:length(order(unique(indice_assessment))))
  # rain[train$installation_id=="0006a69f",]$indice_assessment`[order(unique(train[train$installation_id=="0006a69f",]$indice_assessment))]
  # 
  # test=test%>%group_by(installation_id)%>%fill(indice_assessment, .direction = "up")
  # 
  # seq(1,cuentas,cuentas)
  # a=test%>%dplyr::group_by(installation_id)%>%summarise(max=max(indice_assessment,na.rm = T))
  # test=test%>%dplyr::group_by(installation_id)%>%mutate(indice_assessment=indice_assessment-min(indice_assessment, na.rm=T)+1)
  
  # 
  aa=diff(test$game_time)
  test$diftime=c(0,aa)
  test[test$diftime<0,"diftime"]=0
  
  
  tiempo_total=test%>%group_by(installation_id)%>%summarise(tiempo_total=sum(diftime), world=last(world), game_session=last(game_session), title_assessment=last(title))
  tiempo_type=test%>%group_by(installation_id, type)%>%summarise(tiempo_total=sum(diftime))
  tiempo_title=test%>%group_by(installation_id, title)%>%summarise(tiempo_total=sum(diftime))
  
  tiempotype2=spread(tiempo_type, key=type, value=tiempo_total)
  tiempotitle2=spread(tiempo_title, key=title, value=tiempo_total)
  
  df1=merge(tiempo_total, tiempotype2, by=c("installation_id"))
  
  num_event_id=test%>%group_by(installation_id, event_id)%>%dplyr::summarise(a = n())
  num_event_id2=spread(num_event_id, key=event_id, value=a)
  
  df2=merge(tiempotitle2, num_event_id2, by=c("installation_id"))
  df3=merge(df1, df2, by=c("installation_id"))
  df3[is.na(df3)]=0
  df4=df3
  colnames(df4)[5]="title"
  #mutate los valores acumulados
  # variables_extr=c("installation_id" , "world" ,"game_session" ,"title"  )
  # variables_porfin=setdiff(colnames(df4),variables_extr)
  # for (i in 2:max(df4$indice_assessment)){
  #   if (length(table(df4$indice_assessment==i))>1){
  #     indices=which(df4$indice_assessment==i)
  #     # print(i)
  #     #for (k in indices){
  #     df4[indices,variables_porfin]=df4[indices-1,variables_porfin]+df4[indices,variables_porfin]
  #     #}
  #     
  #   }
  # }
  # df44=df4%>%group_by(installation_id)%>%filter(indice_assessment==max(indice_assessment,na.rm = T))
  return(df4)
}

vector=c("installation_id" ,"indice_assessment" , "tiempo_total" ,      
         "world", "game_session" ,"title"    ,                    
         "Activity", "Assessment", "Clip" ,                        
         "Game" ,
         unique(test$event_id),
         unique(test$title))
A=matrix(nrow = 20000,ncol = length(vector))
colnames(A)=vector

for (k in  1:length(test1)){
  print(k)
  a=extractor_par_test(test1[[k]])
  A[k,vector]=0
  A[k,colnames(a)]=as.matrix(a)
  
}
test=A[!is.na(A[,1]),]
print("Parametros_test_creados")
# HOMOGEINIZACION-----------------------------------------------------------

variables= colnames(test)

tratar_test=function(test_par,variables){
  BB=as.data.frame(test_par)
  BB$title=as.integer(as.factor(BB$title))
  BB$world=as.integer(as.factor(BB$world))
  BB=BB%>%dplyr::select(-game_session) 
  BBB=BB%>%dplyr::select( -installation_id) 
  for (i in 1:length(BBB)) { 
    BBB[,i]=as.numeric(as.character(BBB[,i])) 
    } 
  test.label=as.matrix(BB[,"installation_id"]) 
  test.data=as.matrix(BBB[,intersect(colnames(BBB),variables)]) 
  return(list(test.data,test.label))
}
test.data_1=tratar_test(test,variables)
colnames(test.data_1[[1]])


















#MOODELO-------------
library(doParallel)
cores=detectCores()-2
cl <- makePSOCKcluster(cores) #Usar 2 nucleos del procesador en paralelo
registerDoParallel(cl)


source("auxiliares.R")
#setwd("C:/Users/ldelaguilar/Google Drive/Kaggle")
setwd("C:/Users/laguila/Google Drive/Kaggle")


train_par=as.data.frame(readRDS("train_biblia.rds"))
#quitar algunas columnas
train_par=train_par%>%select( -num_correct, -num_incorrect, -accuracy, -game_session)
train_par$title=as.numeric(as.factor(train_par$title))
train_par$world=as.numeric(as.factor(train_par$world))
train_par$installation_id=as.numeric(as.factor(train_par$installation_id))
train_par$accuracy_group=as.numeric(train_par$accuracy_group)


#quitar columnas que valen 0------
train_par=train_par[,-as.numeric(which(colSums(train_par[,-length(train_par)])==0))]


#Preparar DataFrame

train_par=correlacionadas(train_par, coef=0.95)

train_par=corr_obj(train_par, num=100)[[1]]
correlaciones=corr_obj(train_par, num=100)[[2]]

train_par=process(train_par, tipo="corr", numpca=200, corr=0.95)

train_par=process(train_par, tipo="pca", numpca=200, corr=0.95)

train_par=process(train_par, tipo="nzv", numpca=200, corr=0.95)


#Dividir train y test

split_strat  <- initial_split(train_par, prop = 0.8, strata = "accuracy_group") #esta es la variable descompensada
train.data  <- training(split_strat)
test.data   <- testing(split_strat)
train.label=as.factor(train.data$accuracy_group)
train.data$accuracy_group=NULL

#Preparar modelo

fitControl <- trainControl(
  method = "repeatedcv", ## 10-fold CV
  number = 5,
  repeats=2,
  summaryFunction = summaryKappa,
  search = "random")



#Entrenar el modelo

model1 <- train(x = train.data, y=train.label, 
                  method = "xgbLinear", 
                  trControl = fitControl,
                  verbose = T, 
                  metric="WeightedKappa", 
                  tuneLength=5,
                  allowParallel=TRUE)

model2 <- train(x = train.data, y=train.label, 
                method = "xgbLinear", 
                trControl = fitControl,
                verbose = T, 
                metric="WeightedKappa", 
                tuneLength=2,
                allowParallel=TRUE)

model3 <- train(x = train.data, y=train.label, 
                method = "xgbLinear", 
                trControl = fitControl,
                verbose = T, 
                metric="WeightedKappa", 
                tuneLength=2,
                allowParallel=TRUE)

model4 <- train(x = train.data, y=train.label, 
                method = "xgbLinear", 
                trControl = fitControl,
                verbose = T, 
                metric="WeightedKappa", 
                tuneLength=2,
                allowParallel=TRUE)

model5 <- train(x = train.data, y=train.label, 
                method = "xgbLinear", 
                trControl = fitControl,
                verbose = T, 
                metric="WeightedKappa", 
                tuneLength=2,
                allowParallel=TRUE)
model6 <- train(x = train.data, y=train.label, 
                method = "xgbLinear", 
                trControl = fitControl,
                verbose = T, 
                metric="WeightedKappa", 
                tuneLength=5,
                allowParallel=TRUE)

model7 <- train(x = train.data, y=train.label, 
                method = "xgbLinear", 
                trControl = fitControl,
                verbose = T, 
                metric="WeightedKappa", 
                tuneLength=2,
                allowParallel=TRUE)

model8 <- train(x = train.data, y=train.label, 
                method = "xgbLinear", 
                trControl = fitControl,
                verbose = T, 
                metric="WeightedKappa", 
                tuneLength=2,
                allowParallel=TRUE)

model9 <- train(x = train.data, y=train.label, 
                method = "xgbLinear", 
                trControl = fitControl,
                verbose = T, 
                metric="WeightedKappa", 
                tuneLength=2,
                allowParallel=TRUE)

model10 <- train(x = train.data, y=train.label, 
                method = "xgbLinear", 
                trControl = fitControl,
                verbose = T, 
                metric="WeightedKappa", 
                tuneLength=2,
                allowParallel=TRUE)

pred=data.frame(p1=predict(model1, test.data),
                p2=predict(model2, test.data),
                p3=predict(model3, test.data),
                p4=predict(model4, test.data),
                p5=predict(model5, test.data),
                p6=predict(model6, test.data),
                p7=predict(model7, test.data),
                p8=predict(model8, test.data),
                p9=predict(model9, test.data),
                p10=predict(model10, test.data))

library(modeest)
for (i in 1:length(pred$p1)){
  moda=mlv(as.numeric(pred[i,1:5]), method = "mfv")-1
  pred$final1[i]=moda
  moda=mlv(as.numeric(pred[i,]), method = "mfv")-1
  pred$final2[i]=moda
  pred$sd[i]=sd(as.numeric(pred[i,1:10])-1)
}
pred$real=test.data$accuracy_group
pred$error=(pred$real-pred$final1)**2

ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p1)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p2)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p3)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p4)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p5)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p6)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p7)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p8)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p9)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = (as.numeric(pred$p10)-1),rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = pred$final1,rater.b = test.data$accuracy_group)
ScoreQuadraticWeightedKappa(rater.a = pred$final2,rater.b = test.data$accuracy_group)

pred=pred%>%filter(sd>0)%>%arrange(error)
pred$ind=1:length(pred$p1)


ggplot(pred)+
  geom_line(aes(x=pred$ind, y=0.2*pred$error), color="blue")+
  geom_point(aes(x=pred$ind, y=pred$sd), color="red")



#Ver importancia d elas variables
importancia=data.frame(imp=c(imp0$importance[0:50,], imp1$importance[0:50,], imp2$importance[0:50,], imp3$importance[0:50,]))
importancia$nombres=c(rownames(imp0$importance)[0:50],rownames(imp1$importance)[0:50],rownames(imp2$importance)[0:50],rownames(imp3$importance)[0:50])
importancia=importancia%>%dplyr::group_by(nombres)%>%dplyr::summarise(suma=sum(imp))


pred$real=test.data$accuracy_group
pred$fallos=(pred$real-pred$final)**2
table(pred$real, pred$final)
ScoreQuadraticWeightedKappa(rater.a = pred$final,rater.b = pred$real)




stopCluster(cl)
