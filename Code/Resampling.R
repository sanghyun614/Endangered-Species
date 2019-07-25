setwd("C:/Users/Admin/Desktop/iucn")

# Libraries ------------------------------------------------------
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(tm)
library(slam)
library(stringr)
library(topicmodels)
library(ROSE)
library(xgboost)
library(caret)
library(DMwR)
library(rBayesianOptimization)

# Read File -------------------------------------------------------
animal <- fread("animal_after_embedding.csv")

# Customize Metric -----------------------------------------------
customf1 <- function(preds, dtrain) {
  preds <- as.integer(preds)
  labels <- getinfo(dtrain, "label")
  class0rc <- sum(labels==0 & preds==0) / (sum(labels==0 & preds==0) + sum(labels==0 & preds!=0))
  class1rc <- sum(labels==1 & preds==1) / (sum(labels==1 & preds==1) + sum(labels==1 & preds!=1))
  class2rc <- sum(labels==2 & preds==2) / (sum(labels==2 & preds==2) + sum(labels==2 & preds!=2))
  class3rc <- sum(labels==3 & preds==3) / (sum(labels==3 & preds==3) + sum(labels==3 & preds!=3))
  class4rc <- sum(labels==4 & preds==4) / (sum(labels==4 & preds==4) + sum(labels==4 & preds!=4))
  class5rc <- sum(labels==5 & preds==5) / (sum(labels==5 & preds==5) + sum(labels==5 & preds!=5))
  
  class0pr <- sum(labels==0 & preds==0) / (sum(labels==0 & preds==0) + sum(labels!=0 & preds==0))
  class1pr <- sum(labels==1 & preds==1) / (sum(labels==1 & preds==1) + sum(labels!=1 & preds==1))
  class2pr <- sum(labels==2 & preds==2) / (sum(labels==2 & preds==2) + sum(labels!=2 & preds==2))
  class3pr <- sum(labels==3 & preds==3) / (sum(labels==3 & preds==3) + sum(labels!=3 & preds==3))
  class4pr <- sum(labels==4 & preds==4) / (sum(labels==4 & preds==4) + sum(labels!=4 & preds==4))
  class5pr <- sum(labels==5 & preds==5) / (sum(labels==5 & preds==5) + sum(labels!=5 & preds==5))
  
  customf1 <- 8 / ( (1/class0rc)+(1/class1rc)+(1/class2rc)+(1/class3rc)+(1/class4rc)+(1/class5rc)+
                      (1/class0pr)+(1/class1pr)+(1/class2pr)+(1/class3pr)+(1/class4pr)+(1/class5pr))
  return(list(metric = "f1", value = customf1))
}

# Split Data Deficient -------------------------------------------- 
train <- 
  animal %>% 
  filter(redlistCategory!="DD")

test <-
  animal %>% 
  filter(redlistCategory=="DD")

train %<>% mutate_if(is.character, as.factor)

# LC undersampling ------------------------------------------------
# -----------------------------------------------------------------

tune_grid <- c(0.8, 0.775, 0.75, 0.725, 0.7, 0.675, 0.65)

fitAssessmentLst = list()
lstPos = 0

for (tune_grid in tune_grid) {
  
  train_LC <-
    train %>% mutate(redlistCategory_binary=ifelse(redlistCategory=="LC", 1, 0))
  
  train_rus <- ovun.sample(redlistCategory_binary ~ . , data=train_LC, method="under", p=tune_grid, na.action = na.pass, seed=614)$data %>% 
    select(-redlistCategory_binary)
  train_rus$redlistCategory <- as.factor(train_rus$redlistCategory)
  train_rus <- cbind(train_rus %>% select(redlistCategory), 
                     dummyVars(" ~ .", data=train_rus %>% select(-redlistCategory)) %>% 
                       predict(., newdata=train_rus %>% select(-redlistCategory)) %>% as.data.frame)
                     
  train_rus$redlistCategory = as.integer(train_rus$redlistCategory) - 1
  # Full data set
  data_variables <- train_rus %>% select(-redlistCategory) %>% as.matrix # index of column 'label'
  data_label <- as.matrix(train_rus[,"redlistCategory"])
  
  # split train data and make xgb.DMatrix
  train_matrix <- xgb.DMatrix(data = data_variables, label = data_label)
  
  # Fit cv.nfold * cv.nround XGB models and save OOF predictions
  lstPos = lstPos + 1
  
  cv_model = xgb.cv(params=list(objective="multi:softmax",
                                subsample=0.8,
                                colsample_bytree=0.8,
                                eta=0.1,
                                max_depths=6,
                                num_class=6),
                    data=train_matrix,
                    nrounds=1000,
                    nfold=4,
                    feval=customf1,
                    maximize=TRUE,
                    verbose=FALSE,
                    prediction=TRUE,
                    early_stopping_rounds=100,
                    nthread=4)
  
  fitAssessmentLst[[lstPos]]=list(rus_ratio=tune_grid, assessmentTbl=cv_model)
  print(lstPos)
} # 0.75 with 0.3127969

train_LC <-
  train %>% mutate(redlistCategory_binary=ifelse(redlistCategory=="LC", 1, 0))

train_rus <- ovun.sample(redlistCategory_binary ~ . , data=train_LC, method="under", p=0.75, na.action = na.pass, seed=614)$data %>% 
  select(-redlistCategory_binary)

# SMOTE --------------------------------------------------------
# --------------------------------------------------------------

glimpse(train_rus)

train_rus %<>% mutate_at(colnames(train_rus)[8:27], as.factor)

xgb_cv_bayes <- function(CE_param, En_param, Ex_param, NT_param) {
  
  # CE Set
  train_CE <-
    train_rus %>% mutate(redlistCategory=factor(ifelse(redlistCategory=="CE", "CE", "Other")))
  set.seed(614)
  train_CE_smote <- SMOTE(redlistCategory ~ ., data=train_CE, perc.over = CE_param, perc.under=300) %>% filter(redlistCategory == "CE")
  
  # En Set
  train_En <-
    train_rus %>% mutate(redlistCategory=factor(ifelse(redlistCategory=="En", "En", "Other")))
  set.seed(614)
  train_En_smote <- SMOTE(redlistCategory ~ ., data=train_En, perc.over = En_param, perc.under=300) %>% filter(redlistCategory == "En")
  
  # Ex Set
  train_Ex <-
    train_rus %>% mutate(redlistCategory=factor(ifelse(redlistCategory=="Ex", "Ex", "Other")))
  set.seed(614)
  train_Ex_smote <- SMOTE(redlistCategory ~ ., data=train_Ex, perc.over = Ex_param, perc.under=300) %>% filter(redlistCategory == "Ex")
  
  # NT Set
  train_NT <-
    train_rus %>% mutate(redlistCategory=factor(ifelse(redlistCategory=="NT", "NT", "Other")))
  set.seed(614)
  train_NT_smote <- SMOTE(redlistCategory ~ ., data=train_NT, perc.over = NT_param, perc.under=300) %>% filter(redlistCategory == "NT")
  
  # Smote Set
  train_smote <- rbind(train_CE_smote,
                       train_En_smote,
                       train_Ex_smote,
                       train_NT_smote,
                       train_rus %>% filter(redlistCategory == "V"),
                       train_rus %>% filter(redlistCategory == "LC"))
  # One-hot encoding
  train_smote$redlistCategory <- as.factor(as.character(train_smote$redlistCategory))
  train_smote <- cbind(train_smote %>% select(redlistCategory),
                       
                       dummyVars(" ~ .", data=train_smote %>% select(-redlistCategory)) %>% 
                         predict(., newdata=train_smote %>% select(-redlistCategory)) %>% as.data.frame)
  
  train_smote$redlistCategory = as.integer(train_smote$redlistCategory) - 1
  
  # Make xgb.DMatrix
  data_variables <- train_smote %>% select(-redlistCategory) %>% as.matrix # index of column 'label'
  data_label <- as.matrix(train_smote[,"redlistCategory"])
  train_matrix <- xgb.DMatrix(data = data_variables, label = data_label)
  
  # CV
  cv <- xgb.cv(params=list(objective="multi:softmax",
                           subsample=0.8,
                           colsample_bytree=0.8,
                           eta=0.1,
                           max_depths=6,
                           num_class=6),
               data=train_matrix,
               nrounds=1000,
               nfold=4,
               feval=customf1,
               maximize=TRUE,
               verbose=0,
               prediction=TRUE,
               early_stopping_rounds=100,
               nthread=4)
  list(Score = cv$evaluation_log[,max(test_f1_mean)],
       Pred = cv$pred)
}

OPT_Res3 <- BayesianOptimization(xgb_cv_bayes,
                                 bounds = list(CE_param = c(20L, 120L),
                                               En_param = c(20L, 120L),
                                               Ex_param = c(100L, 500L),
                                               NT_param = c(20L, 120L)),
                                 init_points = 10, n_iter = 40,
                                 acq = "ucb", kappa = 2.576, eps = 0.0,
                                 verbose = TRUE) #CE 120, En 96, Ex 500, NT 120 with F1 4324

# Make Final Dataset ----------------------------------------
# -----------------------------------------------------------

# CE Set
train_CE <-
  train_rus %>% mutate(redlistCategory=factor(ifelse(redlistCategory=="CE", "CE", "Other")))
set.seed(614)
train_CE_smote <- SMOTE(redlistCategory ~ ., data=train_CE, perc.over = 120, perc.under=300) %>% filter(redlistCategory == "CE")

# En Set
train_En <-
  train_rus %>% mutate(redlistCategory=factor(ifelse(redlistCategory=="En", "En", "Other")))
set.seed(614)
train_En_smote <- SMOTE(redlistCategory ~ ., data=train_En, perc.over = 96, perc.under=300) %>% filter(redlistCategory == "En")

# Ex Set
train_Ex <-
  train_rus %>% mutate(redlistCategory=factor(ifelse(redlistCategory=="Ex", "Ex", "Other")))
set.seed(614)
train_Ex_smote <- SMOTE(redlistCategory ~ ., data=train_Ex, perc.over = 500, perc.under=300) %>% filter(redlistCategory == "Ex")

# NT Set
train_NT <-
  train_rus %>% mutate(redlistCategory=factor(ifelse(redlistCategory=="NT", "NT", "Other")))
set.seed(614)
train_NT_smote <- SMOTE(redlistCategory ~ ., data=train_NT, perc.over = 120, perc.under=300) %>% filter(redlistCategory == "NT")

# Smote Set
train_smote <- rbind(train_CE_smote,
                     train_En_smote,
                     train_Ex_smote,
                     train_NT_smote,
                     train_rus %>% filter(redlistCategory == "V"),
                     train_rus %>% filter(redlistCategory == "LC"))

# Delete Minority Class -------------------------------------------
#table(train_smote$phylum) %>% sort(decreasing=T) 

#table(train_smote$class) %>% sort(decreasing=T)
train_smote$class[train_smote$class %in% c("Cephalopoda", "Holothuroidea")] <- "Other"

#table(train_smote$populationTrend)
#table(train_smote$threats) %>% sort(decreasing = T)
#table(train_smote$useTrade) %>% sort(decreasing = T)
#table(train_smote$conservationAction) %>% sort(decreasing = T)

sum_fac <- function(x) {
  x <- as.numeric(x)
  sum(x)
}

#apply(train_smote[,8:27], 2, sum_fac) %>% sort(decreasing=T)

# write file
write.csv(rbind(train_smote,test), "animal_smote.csv", row.names=F)
