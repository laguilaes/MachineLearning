#https://github.com/zachmayer/caretEnsemble/tree/master/R
#Setting inputs
ejecucion="kernell"

library(vctrs)
library(tidyverse)
library(scales)
library(lubridate)
library(xgboost)
library(data.table)
library(stringr)
library(dplyr)
library(crayon)
library(rsample)
library(caret)
print("Librerias cargadas exitosamente")

if (ejecucion=="kernel"){
  train <- data.table::fread("../input/data-science-bowl-2019/train.csv", stringsAsFactors = F)
  train_labels <- data.table::fread("../input/data-science-bowl-2019/train_labels.csv", stringsAsFactors = F)
}else{
  train <- data.table::fread("C:/Users/laguila/Google Drive/Kaggle/datos iniciales/train.csv", stringsAsFactors = F)
  train_labels <- data.table::fread("C:/Users/laguila/Google Drive/Kaggle/datos iniciales/train_labels.csv", stringsAsFactors = F)
}
print("Training datasets leidos exitosamente")

#Funci?n que obtiene los accuracy
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


#Funci?n extractora de par?metros del train
extractor_par_train = function (train,train_labels) {
  
  #Quitamos installation_ids sin assessment
  ids=as.data.frame(unique(train[str_detect(train$type, "Assessment"),"installation_id"]))
  colnames(ids)="installation_id"
  train=merge(train, ids, by="installation_id", keep.all=T)%>%arrange(installation_id, timestamp)
  
  #Identificamos principios de assessments
  indices=which(train$type=="Assessment" & train$game_time==0)[!duplicated(substr(as.character(train[which(train$type=="Assessment" & train$game_time==0),]$timestamp),1,19))]
  indices=indices[diff(indices)>2]
  train$indice_assessment=NA
  train$indice_assessment[indices]=1:length(indices)
  train=train%>%group_by(installation_id)%>%fill(indice_assessment, .direction = "up")
  train=train%>%dplyr::group_by(installation_id)%>%mutate(indice_assessment=indice_assessment-min(indice_assessment, na.rm=T)+1)
  
  #C?lculamos espacios temporales
  aa=diff(train$game_time)
  train$diftime=c(0,aa)
  train[train$diftime<0,"diftime"]=0
  
  #Calculamos todo tipo de par?metros
  tiempo_total=train%>%group_by(installation_id, indice_assessment)%>%summarise(tiempo_total=sum(diftime), world=last(world), game_session=last(game_session), title_assessment=last(title),count_id=n())
  tiempo_e_code=train%>%group_by(installation_id, indice_assessment,event_code)%>%summarise(count_event_code=n())
  tiempo_type_1=train%>%group_by(installation_id, indice_assessment, type)%>%summarise(tiempo_total=sum(diftime))
  tiempo_type_2=train%>%group_by(installation_id, indice_assessment, type)%>%summarise(count_type=n())
  tiempo_title_1=train%>%group_by(installation_id, indice_assessment, title)%>%summarise(tiempo_total=sum(diftime))
  tiempo_title_2=train%>%group_by(installation_id, indice_assessment, title)%>%summarise(count_title=n())
  BM=train%>%group_by(installation_id, indice_assessment)%>%get_accuracy(.,'Bird Measurer' , '4110')
  colnames(BM)[4:8]=paste0(colnames(BM)[4:8],"_",BM$title[1])
  CB=train%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Cart Balancer", '4100')
  colnames(CB)[4:8]=paste0(colnames(CB)[4:8],"_",CB$title[1])
  CF=train%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Cauldron Filler", "4100")
  colnames(CF)[4:8]=paste0(colnames(CF)[4:8],"_",CF$title[1])
  CS=train%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Chest Sorter", "4100")
  colnames(CS)[4:8]=paste0(colnames(CS)[4:8],"_",CS$title[1])
  MS=train%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Mushroom Sorter", "4100")
  colnames(MS)[4:8]=paste0(colnames(MS)[4:8],"_",MS$title[1])
  total=train%>%group_by(installation_id, indice_assessment)%>%filter(row_number()==n())
  arte_1=right_join(BM[,c(1,3,5:6)],total[,c("installation_id","indice_assessment")],by=c("installation_id","indice_assessment"))
  arte_1=right_join(CB[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  arte_1=right_join(CF[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  arte_1=right_join(CS[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  arte_1=right_join(MS[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  for (i in 1:length(arte_1)){
    arte_1[is.na(arte_1[,i]),i]=0
  }
  tiempo_total=merge(arte_1,tiempo_total,by=c("installation_id","indice_assessment"))
  
  ##Elapsed Time
  tiempo_final=train%>%group_by(installation_id, indice_assessment)%>%summarise(tiempo_final=max(timestamp))
  tiempo_inicial=train%>%group_by(installation_id)%>%summarise(tiempo_inicial=min(timestamp))
  ttt=merge(tiempo_inicial,tiempo_final,by="installation_id",keep.all=T)
  ttt$elapsed_time=difftime(ymd_hms(ttt$tiempo_final),ymd_hms(ttt$tiempo_inicial),units = "secs")
  tiempo_total=merge(tiempo_total,ttt[,c("installation_id","indice_assessment","elapsed_time")],by=c("installation_id","indice_assessment"))
  
  ###Madrugada_ma?ana_Tarde
  tiempo_final$momento=as.numeric(substr(tiempo_final$tiempo_final,12,13))
  tiempo_final$momento_c=0
  tiempo_final$momento_c[tiempo_final$momento<8 | tiempo_final$momento>23]=0
  tiempo_final$momento_c[tiempo_final$momento>=8 & tiempo_final$momento<16]=1
  tiempo_final$momento_c[tiempo_final$momento>=16 & tiempo_final$momento<24]=2
  tiempo_total=merge(tiempo_total,tiempo_final[,c("installation_id","indice_assessment","momento_c")],by=c("installation_id","indice_assessment"))
  
  ###Tiempo dedicado en los ultimos 5 game_session
  e=train%>%group_by(installation_id,indice_assessment,game_session)%>%summarise(tiempo_gs=max(game_time),max_time=max(timestamp))
  ee=e%>%group_by(installation_id,indice_assessment)%>%summarise(tiempo_gs=sum(tail(tiempo_gs,5)))
  tiempo_total=merge(tiempo_total,ee[,c("installation_id","indice_assessment","tiempo_gs")],by=c("installation_id","indice_assessment"))
  
  
  
  
  tiempotype2=spread(tiempo_type_1, key=type, value=tiempo_total)
  tiempotype3=spread(tiempo_type_2, key=type, value=count_type)
  colnames(tiempotype3)[3:dim(tiempotype3)[2]]=paste0(colnames(tiempotype3)[3:dim(tiempotype3)[2]],"_contador")
  tiempotitle2=spread(tiempo_title_1, key=title, value=tiempo_total)
  tiempotitle3=spread(tiempo_title_2, key=title, value=count_title)
  colnames(tiempotitle3)[3:dim(tiempotitle3)[2]]=paste0(colnames(tiempotitle3)[3:dim(tiempotitle3)[2]],"_contador")
  tiempocode=spread(tiempo_e_code, key=event_code, value=count_event_code)
  df1=merge(tiempo_total, tiempotype2, by=c("installation_id", "indice_assessment"))
  df1=merge(df1, tiempotype3, by=c("installation_id", "indice_assessment"))
  df1=merge(df1, tiempocode, by=c("installation_id", "indice_assessment"))
  
  num_event_id=train%>%group_by(installation_id, indice_assessment, event_id)%>%dplyr::summarise(a = n())
  num_event_id2=spread(num_event_id, key=event_id, value=a)
  
  df2=merge(tiempotitle2, num_event_id2, by=c("installation_id", "indice_assessment"))
  df2=merge(df2, tiempotitle3, by=c("installation_id", "indice_assessment"))
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
  df5=merge(df4, train_labels, by=c("title", "game_session","installation_id"), keep.all=T)
  df6=df5[!duplicated(df5$game_session),]
  df6=df6%>%dplyr::arrange(installation_id, indice_assessment)
  return(df6)
}
indice=first(which(train$installation_id=="8123bc13"))
train_par_ind1=extractor_par_train(train[1:(indice-1),],train_labels)
train_par_ind2=extractor_par_train(train[indice:length(train$event_id),],train_labels)
rm(train)
columnas=intersect(colnames(train_par_ind1),colnames(train_par_ind2))
train_par=rbind(train_par_ind1[,columnas],train_par_ind2[,columnas])
rm(train_par_ind1,train_par_ind2)
rm(train_labels)
print("Training datasets tratados exitosamente")
saveRDS(train_par,"train_biblia_25_11.rds")


if (ejecucion=="kernel"){
  test <- data.table::fread("../input/data-science-bowl-2019/test.csv", stringsAsFactors = F)
  train_labels <- data.table::fread("../input/data-science-bowl-2019/train_labels.csv", stringsAsFactors = F)
}else{
  test <- data.table::fread("C:/Users/laguila/Google Drive/Kaggle/datos iniciales/test.csv")
  train_labels <- data.table::fread("C:/Users/laguila/Google Drive/Kaggle/datos iniciales/train_labels.csv")
}
print("Test_leido_correctamente")


extractor_par_test=function(test,train_labels){
  #train_=train%>%dplyr::select(-event_data, -event_code)#%>%arrange(installation_id, timestamp)
  train_=test
  #quitar installation_id sin assessment
  ids=as.data.frame(unique(train_[str_detect(train_$type, "Assessment"),"installation_id"]))
  colnames(ids)="installation_id"
  train3=merge(train_, ids, by="installation_id", keep.all=T)%>%arrange(installation_id, timestamp)
  
  
  indices=which(train3$type=="Assessment" & train3$game_time==0)[!duplicated(substr(as.character(train3[which(train3$type=="Assessment" & train3$game_time==0),]$timestamp),1,19))]
  if (length(indices)>1){
    indices=indices[diff(indices)!=1]
  }
  
  train3$indice_assessment=NA
  train3$indice_assessment[indices]=1:length(indices)
  
  
  train2=train3%>%group_by(installation_id)%>%fill(indice_assessment, .direction = "up")
  
  train4=train2%>%dplyr::group_by(installation_id)%>%mutate(indice_assessment=indice_assessment-min(indice_assessment, na.rm=T)+1)
  
  aa=diff(train4$game_time)
  train4$diftime=c(0,aa)
  train4[train4$diftime<0,"diftime"]=0
  
  test=train4
  
  tiempo_total=test%>%group_by(installation_id,indice_assessment)%>%summarise(tiempo_total=sum(diftime), world=last(world), game_session=last(game_session), title_assessment=last(title))
  tiempo_e_code=test%>%group_by(installation_id, indice_assessment,event_code)%>%summarise(count_event_code=n())
  tiempo_type_1=test%>%group_by(installation_id, indice_assessment,type)%>%summarise(tiempo_total=sum(diftime))
  tiempo_type_2=test%>%group_by(installation_id, indice_assessment,type)%>%summarise(count_type=n())
  tiempo_title_1=test%>%group_by(installation_id, title,indice_assessment)%>%summarise(tiempo_total=sum(diftime))
  tiempo_title_2=test%>%group_by(installation_id, title,indice_assessment)%>%summarise(count_title=n())
  
  BM=test%>%group_by(installation_id, indice_assessment)%>%get_accuracy(.,'Bird Measurer' , '4110')
  colnames(BM)[5:8]=paste0(colnames(train_labels)[4:7],"_","Bird Measurer (Assessment)")
  CB=test%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Cart Balancer", '4100')
  colnames(CB)[5:8]=paste0(colnames(train_labels)[4:7],"_","Cart Balancer (Assessment)")
  CF=test%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Cauldron Filler", "4100")
  colnames(CF)[5:8]=paste0(colnames(train_labels)[4:7],"_","Cauldron Filler (Assessment)")
  CS=test%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Chest Sorter", "4100")
  colnames(CS)[5:8]=paste0(colnames(train_labels)[4:7],"_","Chest Sorter (Assessment)")
  MS=test%>%group_by(installation_id, indice_assessment)%>%get_accuracy(., "Mushroom Sorter", "4100")
  colnames(MS)[5:8]=paste0(colnames(train_labels)[4:7],"_","Mushroom Sorter (Assessment)")
  total=test%>%group_by(installation_id, indice_assessment)%>%filter(row_number()==n())
  arte_1=right_join(BM[,c(1,3,5:6)],total[,c("installation_id","indice_assessment")],by=c("installation_id","indice_assessment"))
  arte_1=right_join(CB[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  arte_1=right_join(CF[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  arte_1=right_join(CS[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  arte_1=right_join(MS[,c(1,3,5:6)],arte_1,by=c("installation_id","indice_assessment"))
  for (i in 1:length(arte_1)){
    arte_1[is.na(arte_1[,i]),i]=0
  }
  
  tiempo_total=merge(arte_1,tiempo_total,by=c("installation_id","indice_assessment"))
  ##Elapsed time
  tiempo_final=test%>%group_by(installation_id,indice_assessment)%>%summarise(tiempo_final=max(timestamp))
  tiempo_inicial=test%>%group_by(installation_id)%>%summarise(tiempo_inicial=min(timestamp))
  ttt=merge(tiempo_inicial,tiempo_final,by="installation_id",keep.all=T)
  ttt$elapsed_time=difftime(ymd_hms(ttt$tiempo_final),ymd_hms(ttt$tiempo_inicial),units = "secs")
  tiempo_total=merge(tiempo_total,ttt[,c("installation_id","indice_assessment","elapsed_time")],by=c("installation_id","indice_assessment"))
  
  ###Madrugada_ma?ana_Tarde
  tiempo_final$momento=as.numeric(substr(tiempo_final$tiempo_final,12,13))
  tiempo_final$momento_c=0
  tiempo_final$momento_c[tiempo_final$momento<8 | tiempo_final$momento>23]=0
  tiempo_final$momento_c[tiempo_final$momento>=8 & tiempo_final$momento<16]=1
  tiempo_final$momento_c[tiempo_final$momento>=16 & tiempo_final$momento<24]=2
  tiempo_total=merge(tiempo_total,tiempo_final[,c("installation_id","indice_assessment","momento_c")],by=c("installation_id","indice_assessment"))
  
  ###Tiempo dedicado en los ultimos 5 game_session
  
  e=test%>%group_by(installation_id,indice_assessment,game_session)%>%summarise(tiempo_gs=max(game_time),max_time=max(timestamp))
  ee=e%>%group_by(installation_id,indice_assessment)%>%summarise(tiempo_gs=sum(tail(tiempo_gs,5)))
  tiempo_total=merge(tiempo_total,ee[,c("installation_id","indice_assessment","tiempo_gs")],by=c("installation_id","indice_assessment"))
  
  
  tiempotype2=spread(tiempo_type_1, key=type, value=tiempo_total)
  tiempotype3=spread(tiempo_type_2, key=type, value=count_type)
  colnames(tiempotype3)[3:dim(tiempotype3)[2]]=paste0(colnames(tiempotype3)[3:dim(tiempotype3)[2]],"_contador")
  tiempotitle2=spread(tiempo_title_1, key=title, value=tiempo_total)
  tiempotitle3=spread(tiempo_title_2, key=title, value=count_title)
  colnames(tiempotitle3)[3:dim(tiempotitle3)[2]]=paste0(colnames(tiempotitle3)[3:dim(tiempotitle3)[2]],"_contador")
  tiempocode=spread(tiempo_e_code, key=event_code, value=count_event_code)
  
  df1=merge(tiempo_total, tiempotype2, by=c("installation_id", "indice_assessment"))
  df1=merge(df1, tiempotype3, by=c("installation_id", "indice_assessment"))
  df1=merge(df1, tiempocode, by=c("installation_id", "indice_assessment"))
  
  num_event_id=test%>%group_by(installation_id, indice_assessment, event_id)%>%dplyr::summarise(a = n())
  num_event_id2=spread(num_event_id, key=event_id, value=a)
  
  df2=merge(tiempotitle2, num_event_id2, by=c("installation_id", "indice_assessment"))
  df2=merge(tiempotitle3, df2, by=c("installation_id", "indice_assessment"))
  df3=merge(df1, df2, by=c("installation_id", "indice_assessment"))
  df3[is.na(df3)]=0
  df4=df3[df3$indice_assessment!=0,]
  colnames(df4)[ colnames(df4)=="title_assessment"]="title"
  df4=df4%>%arrange(installation_id,as.numeric(indice_assessment))
  df4$elapsed_time=as.numeric(df4$elapsed_time)
  df4$momento_c=as.numeric(df4$momento_c)
  df4$tiempo_gs=as.numeric(df4$tiempo_gs)
  
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
  # df44=df4%>%group_by(installation_id)%>%filter(indice_assessment==max(indice_assessment,na.rm = T))
  return(df4)
}
test1=split(test,test$installation_id)
print("Test_dividido_correctamente")
vector=c("installation_id" ,"indice_assessment" , "tiempo_total" ,      
         "world", "game_session","elapsed_time","momento_c","tiempo_gs" ,"title"    ,                    
         "Activity", "Assessment", "Clip" ,                        
         "Game" ,
         "Activity_contador", "Assessment_contador", "Clip_contador" ,                        
         "Game_contador" ,
         unique(test$event_id),
         unique(test$title),
         unique(test$event_code),
         paste0(  unique(test$title),"_contador"),
         colnames(train_par)[str_detect(colnames(train_par),"correct_")])
A=matrix(nrow = 50000,ncol = length(vector))
colnames(A)=vector
F_t=matrix(nrow = 50000,ncol = length(vector))
colnames(F_t)=vector
inst=1
for (k in  1:length(test1)){
  print(k)
  a=extractor_par_test(test1[[k]],train_labels)
  
  A[k,vector]=0
  A[k,colnames(a[dim(a)[1],])]=as.matrix(a[dim(a)[1],])
  
  if (dim(a)[1]>1){
    F_t[inst:(dim(a)[1]-1+inst),vector]=0
    F_t[inst:(dim(a)[1]-1+inst),colnames(a[dim(a)[1],])]=as.matrix(a[1:(dim(a)[1]),])
    inst=(dim(a)[1]+inst)
  }
  
  
  
}
test=A[!is.na(A[,1]),]
test_pasa_a_train=F_t[!is.na(F_t[,1]),]


vector=c()
total=c()
for (i in which(str_detect(colnames(test_pasa_a_train),"_correct_"))){
  print(i)
  vector=as.numeric(as.character(test_pasa_a_train[,i]))
  total=cbind(total,vector)
  num_correct=apply(total, 1, sum)
}
vector=c()
total=c()
for (i in which(str_detect(colnames(test_pasa_a_train),"_incorrect_"))){
  print(i)
  vector=as.numeric(as.character(test_pasa_a_train[,i]))
  total=cbind(total,vector)
  num_incorrect=apply(total, 1, sum)
}
test_pasa_a_train=as.data.frame(test_pasa_a_train)
test_pasa_a_train$num_correct_hist= num_correct

test_pasa_a_train$num_incorrect_hist=num_incorrect
test_pasa_a_train$num_correct_t_l=
  -c((test_pasa_a_train$num_correct_hist))+
  c((test_pasa_a_train$num_correct_hist[2:(length(test_pasa_a_train$num_correct_hist))]),0)
test_pasa_a_train$num_incorrect_t_l=
  -c((test_pasa_a_train$num_incorrect_hist))+
  c((test_pasa_a_train$num_incorrect_hist[2:(length(test_pasa_a_train$num_incorrect_hist))]),0)

test_pasa_a_train$installation_id_num=as.integer(as.factor(test_pasa_a_train$installation_id))
train_artificial=test_pasa_a_train[!(test_pasa_a_train$num_incorrect_t_l<0 | test_pasa_a_train$num_correct_t_l<0),]
train_artificial$accuracy_group=0
train_artificial$accuracy_group[train_artificial$num_correct_t_l<1]=0
train_artificial$accuracy_group[train_artificial$num_correct_t_l==1 & (train_artificial$num_incorrect_t_l+train_artificial$num_correct_t_l)==1]=3
train_artificial$accuracy_group[train_artificial$num_correct_t_l==1 & (train_artificial$num_incorrect_t_l+train_artificial$num_correct_t_l)==2]=2
train_artificial$accuracy_group[train_artificial$num_correct_t_l==1 & (train_artificial$num_incorrect_t_l+train_artificial$num_correct_t_l)>2]=1

saveRDS(train_artificial,"train_artificial.rds")
print("Parametros_test_creados")

vars_train=colnames(train_par)
vars_train_artif=colnames(train_artificial)
vars_test=colnames(test)

a=as.vector(train_par$installation_id)
b=as.vector(train_artificial$installation_id)
c=as.vector(test[,1])
diccionario=as.data.frame(c(a,b,c))
diccionario$num=as.integer(as.factor(diccionario$`c(a, b, c)`))
colnames(diccionario)=c("installation_id","inst_id_num")
diccionario=diccionario[!duplicated(diccionario$installation_id),]
train_par_def=rbind(train_par[,intersect(colnames(train_artificial),colnames(train_par))],train_artificial[,intersect(colnames(train_artificial),colnames(train_par))])

train_par_def=merge(train_par_def,diccionario,by="installation_id")

test=merge(test,diccionario,by="installation_id")
copia_1=train_par_def
copia_2=test
saveRDS(copia_1,"train_bonico2.rds")
saveRDS(copia_2,"test_bonico2.rds")


train_par=as.data.frame(read.csv("train_bonico_17000.csv", ))
train_par=readRDS("train_bonico_17000.rds")

test.data=as.data.frame(readRDS("test_bonico2.rds"))

#------------------------------------------

tratar<-function(df, dummies){
  df$game_session=NULL
  df2=df%>%dplyr::select(-world, -title, -installation_id)
  for (i in 1:length(df2)){
    df2[,i]=as.numeric(as.character(df2[,i]))
  }
  
  df2$world=as.factor(df$world)
  df2$title=as.factor(df$title)
  #One-hot encoding
  if(dummies){
    #df2$installation_id=as.factor(df$installation_id)
    dummies <- dummyVars(accuracy_group ~ ., data = df2)
    df3=as.data.frame(predict(dummies, newdata = df2))
    df3$installation_id=df$installation_id
    return(df3)
  }else{
    #Si se prefiere hacer target encoding
    df2$installation_id=as.factor(df$installation_id)
    return(df2)
  }
}
train.labels=train_par$accuracy_group
test.data$accuracy_group=-1
train.data=tratar(train_par, dummies=T)
test.data=tratar(test.data, dummies=T)
train.data=train.data[,intersect(names(train.data), names(test.data))]
test.data=test.data[,intersect(names(train.data), names(test.data))]
train.data$sep=0
test.data$sep=1

#target encoding
world_encod=targetencod(df=train.data, col="world", target="accuracy_group", test=test.data)
train.data$worldencoded=world_encod[[1]]
test.data$worldencoded=world_encod[[2]]

title_encod=targetencod(df=train.data, col="title", target="accuracy_group", test=test.data)
train.data$titleencoded=world_encod[[1]]
test.data$titleencoded=world_encod[[2]]




#----------

ids=test.data$installation_id
test.data$installation_id=NULL
train.data$installation_id=NULL
datos=rbind(train.data, test.data)
datos$world=NULL
datos$title=NULL
saveRDS(datos,"datos.rds")
saveRDS(train.labels,"train.labels.rds")
saveRDS(ids,"ids.rds")

#-------------------------------------------------------------------------------------------------------------------------------------------






####MODELO--------
setwd("C:/Users/laguila/Google Drive/Kaggle")
source("auxiliares.R")

#Parallel processing
library(doParallel)
cores=detectCores()-1
cl <- makePSOCKcluster(cores) #Usar 2 nucleos del procesador en paralelo
registerDoParallel(cl)

#Lectura datos
#datos=readRDS("datos_dummies_id.rds")
#datos=readRDS("datos_target.rds")
datos=readRDS("datos.rds")
datos$inst_id_num=NULL
train.labels=readRDS("train.labels.rds")
ids=readRDS("ids.rds")
sepp=datos$sep

#Transformar TrainLabels
cop=train.labels
train.labels[cop==3]=1
train.labels[cop==2]=10
train.labels[cop==1]=50
train.labels[cop==0]=200


#Eliminacion variables
datos=process(datos, tipo="zv", numpca=200, corr=0.95)
datos=process(datos, tipo="corr", numpca=200, corr=0.98)
datos=process(datos, tipo="pca", numpca=200, corr=0.95)
datos=process(datos, tipo="nzv", numpca=200, corr=0.95)

#Comprobar si train y test son diferenciables
check=datos[sample(nrow(datos)),]
split_strat  <- initial_split(check, prop = 0.8, strata = "sep") #esta es la variable descompensada
train  <- training(split_strat)
test   <- testing(split_strat)
label=as.factor(train$sep)
train$sep=NULL
ylabel=test$sep
test$sep=NULL

fitControl <- trainControl(
  method = "cv", ## 10-fold CV
  number = 5,
  search="random")#,

modely <- train(x = train, y=label,
                method = "xgbLinear",
                trControl = fitControl,
                verbose = T,
                tuneLength=2)

pred1=predict(modely, test)

table(pred1, ylabel)
ScoreQuadraticWeightedKappa(rater.a = as.numeric(pred1)-1,rater.b = ylabel)


#medias variables entre train y test
datostrain=datos[datos$sep==0,]
datostest=datos[datos$sep==1,]
medias_train=apply(datostrain, 2, mean)
medias_test=apply(datostest, 2, mean)
ratios=medias_train/medias_test
hist(ratios, breaks = 1000)
vars=names(ratios[ratios>5 | ratios<0.2])
datos=datos%>%dplyr::select(-vars)
length(vars)


#Escalar

datos=process(datos, tipo="scale", numpca=200, corr=0.95)
datos$sep=sepp

#Dividir train y test
test.data=datos%>%dplyr::filter(sep==1)%>%dplyr::select(-sep)
train.data=datos%>%dplyr::filter(sep==0)%>%dplyr::select(-sep)


# Recursive Feature Elimination
results=rfe_manual(method="xgb", train=train.data, label=train.labels, size=c(5, 10, 15, 20, 40, 80, 120, 180, 230, 300))
plot(results)
predictors=predictors(results)
train.data=train.data[, predictors]
test.data=test.data[,predictors]

#Varable aleatoria
train.data$accuracy_group=train.labels
train.data$random=runif(-1, 1, n = nrow(train.data))
lim=cor(train.data$random, train.data$accuracy_group)
cors=as.data.frame(cor(train.data,  train.data$accuracy_group))
cors$names=rownames(cors)
vars=cors[cors$V1<lim,]$names
test.data=test.data%>%dplyr::select(-vars)
train.data=train.data%>%dplyr::select(-vars)
train.data$random=NULL
length(vars)

#Crear train y test
split_strat  <- initial_split(train.data, prop = 0.8, strata = "accuracy_group") #esta es la variable descompensada
train  <- training(split_strat)
test   <- testing(split_strat)
label=train$accuracy_group
train$accuracy_group=NULL
ylabel=test$accuracy_group
test$accuracy_group=NULL

#Visualizar variables

featurePlot(x = train$world, 
            y = label, 
            plot = "box")

featurePlot(x = train$world, 
            y = label, 
            plot = "density")

#Elegir modelos a entrenar
num=10
tipo="Regression"
basemodel="xgbLinear"
quemodelo(tipo, num, basemodel)


#entrenar modelos
Grid=expand.grid(usekernel=F,
                 laplace=0,
                 adjust=c(0.000000001, 1))

fitControl <- trainControl(
  method = "cv", ## 10-fold CV
  number = 4, 
  search="random")#,
  #summaryFunction = summaryKappa,
  #search="random")




#modelos: gbm, bayesglm, plsRglm
xgb <- train(x = train, y=label, 
                  method = "xgbLinear", 
                  trControl = fitControl,
                tuneLength=5)
saveRDS(xgb, "xgb.rds")
modely

 # modely=copia
 # cual <- tolerance(modely$results, metric = "RMSE", maximize = TRUE)
 # modely$results=modely$results[9,]

pred1=predict(xgb, test)
hist(pred1, breaks=200)
a=table(as.numeric(label))
a=a/sum(a)
a0=quantile(pred1, a[1] )
a1=quantile(pred1, a[1]+a[2] )
a2=quantile(pred1, a[1]+a[2]+a[3] )

pred1[pred1<=a0]=1
pred1[pred1>a0 & pred1<=a1]=10
pred1[pred1>a1 & pred1<=a2]=50
pred1[pred1>a2]=200

table(pred1, ylabel)
library(Metrics)
ScoreQuadraticWeightedKappa(rater.a = pred1,rater.b = ylabel)


#MODELOS COMBINADOS----------------------------------------------------------------------------------------------------------
setwd("C:/Users/ldelaguila/Google Drive/Kaggle")
source("auxiliares.R")

#library(caretEnsemble)
source("caretEnsemble2.R")
train.data=read.csv("train.data.csv")
train.data$V1=NULL

#Crear train y test
train=train.data
test =test.data
label=train$accuracy_group
train$accuracy_group=NULL

#FitControl
fitControl2 <- trainControl(
  method = "cv",  
  number = 5,
  savePredictions="final",
  classProbs=TRUE,
  returnResamp = "all",
  index=createResample(label),
  search="random")

#CaretList c("glmnet" ,"xgbLinear", "xgbTree", "ranger", "gbm", "bayesglm", "plsRglm")

#Crear modelos por separado
xgb <- train(x = train, y=label, 
             method = "xgbLinear", 
             trControl = fitControl2,
             tuneLength=10)
saveRDS(xgb, "xgb.rds")
tree <- train(x = train, y=label, 
              method = "xgbTree", 
              trControl = fitControl2,
              tuneLength=5)
saveRDS(tree, "tree.rds")
gbm <- train(x = train, y=label, 
              method = "gbm", 
              trControl = fitControl2,
              tuneLength=3)
saveRDS(gbm, "gbm.rds")
bayesglm <- train(x = train, y=label, 
             method = "bayesglm", 
             trControl = fitControl2,
             tuneLength=10)
saveRDS(bayesglm, "bayesglm.rds")
glmnet <- train(x = train, y=label, 
             method = "glmnet", 
             trControl = fitControl2,
             tuneLength=10)
saveRDS(glmnet, "glmnet.rds")
plsRglm <- train(x = train, y=label, 
                method = "plsRglm", 
                trControl = fitControl2,
                tuneLength=3)
saveRDS(plsRglm, "plsRglm.rds")
ranger <- train(x = train, y=label, 
              method = "ranger", 
              trControl = fitControl2,
              tuneLength=5)
saveRDS(ranger, "ranger.rds")
cforest <- train(x = train, y=label, 
                method = "cforest", 
                trControl = fitControl2,
                tuneLength=3)
saveRDS(cforest, "cforest.rds")
cubist <- train(x = train, y=label, 
                 method = "cubist", 
                 trControl = fitControl2,
                 tuneLength=3)
saveRDS(cubist, "cubist.rds")



multimodel <- list(xgboost = xgb, bayes = bayesglm)
class(multimodel) <- "caretList"

#Directamente 
multi <- caretList(x = train, y=label,
                        trControl = fitControl2,
                   tuneList=list(
                     xgb=caretModelSpec(method="bayesglm", tuneLength=5)
                   )
                        #verbose = T
)


#Ver resultados de los modelos
res <- resamples(multi)
summary(res)

#Ver compatibilidad modelos
xyplot(resamples(multi_mod3))#Para combinar dos modelos, ver que no estan correlacionados y tienen alta eficiencia
modelCor(resamples(multi_mod3))


#CaretEnsemble

fitControl3 <- trainControl(
  method = "repeatedcv",  
  number = 10,
  repeats=10,
  savePredictions="final",
  classProbs=TRUE,
  returnResamp = "all",
  index=createResample(label))


ensemble <- caretEnsemble(
  multimodel, 
  trControl=fitControl3
    )
summary(ensemble)


#Stack Models

stack <-
  caretStack(multi,
             method = "glmnet",
             trControl = fitControl2)


library(pbapply)
# Predict
pred1=predict(multi, test)
hist(pred1, breaks=200)
a=table(as.numeric(label))
a=a/sum(a)
a0=quantile(pred1, a[1] )
a1=quantile(pred1, a[1]+a[2] )
a2=quantile(pred1, a[1]+a[2]+a[3] )

pred1[pred1<=a0]=1
pred1[pred1>a0 & pred1<=a1]=10
pred1[pred1>a1 & pred1<=a2]=50
pred1[pred1>a2]=200

table(pred1, ylabel)
ScoreQuadraticWeightedKappa(rater.a = pred1,rater.b = ylabel)



#Importancia variables
importancia=varImp(ensemble, scale=F)
importancia$nombres=rownames(importancia)
importancia=arrange(importancia, desc(overall))



#Comparar modelos manualmente
pred=data.frame(p1=predict(model1, test.data),
                p2=predict(model2, test.data),
                p3=predict(model3, test),
                p4=predict(model4, test.data),
                p5=predict(model5, test.data),
                p6=predict(model6, test.data),
                p7=predict(model7, test.data),
                p8=predict(model8, test.data),
                p9=predict(model9, test.data),
                p10=predict(model10, test.data))

#Prediccion democratica
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



stopCluster(cl)


library(rJava)
library(mailR)

send.mail(from = "laguilaes@gmail.com",
          to = c("laguilaes@gmail.com"), # Colocar correos
          subject = "Modelos!",
          body = paste("Todo hecho =)"),
          smtp = list(host.name = "smtp.gmail.com", port = 465,
                      user.name = "laguilaes@gmail.com",
                      passwd = pass, ssl=T),  # Cambiar contrase?a
          authenticate = TRUE,
          send = TRUE,
          debug = TRUE)

