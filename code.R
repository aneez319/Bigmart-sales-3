setwd("E:/project/Bigmart")

train=read.csv("Train.csv",na.strings = c(""," "))
test=read.csv("Test.csv",na.strings = c(""," "))

str(train)
str(test)

train$flag_train=1
test$flag_train=0

test$Item_Outlet_Sales=NA

full=rbind(train,test)

str(full)

#Data Exploration

table(full$Item_Fat_Content)
table(full$Item_Type)
table(full$Outlet_Identifier)
table(full$Outlet_Size)
table(full$Outlet_Type)
table(full$Outlet_Establishment_Year)
table(full$Outlet_Location_Type)

table(full$Outlet_Size,full$Outlet_Location_Type)
table(full$Outlet_Size,full$Outlet_Identifier)
table(full$Outlet_Size,full$Outlet_Type)
table(full$Outlet_Size,full$Outlet_Establishment_Year)

table(full$Outlet_Type,full$Outlet_Location_Type)
table(full$Outlet_Type,full$Outlet_Establishment_Year)
table(full$Outlet_Type,full$Outlet_Identifier)

table(full$Outlet_Location_Type,full$Outlet_Identifier)
table(full$Outlet_Location_Type,full$Outlet_Establishment_Year)

table(full$Outlet_Identifier,full$Outlet_Establishment_Year)

 
 
#Imputing missing values of size column

temptype=vector(mode = "character",length=0)
tempouttype=vector(mode = "character",length=0)
tempid=vector(mode = "character",length=0)
tempyear=vector(mode = "character",length=0)

sapply(full,function(x){sum(is.na(x))})

full$Outlet_Size=as.character(full$Outlet_Size)
j=1

for(i in 1:14204)
{
  if(is.na(full$Outlet_Size[i])){

    temptype[j]=as.character(full$Outlet_Location_Type[i])
    tempouttype[j]=as.character(full$Outlet_Type[i])
    tempid[j]=as.character(full$Outlet_Identifier[i])
    tempyear[j]=as.character(full$Outlet_Establishment_Year[i])
    j=j+1
  }
}
table(temptype)
table(tempouttype)
table(tempid)
table(tempyear)

#After observing the 4 vectors,we come to know that the missing value should be Small

for(i in 1:14204)
{
  if(is.na(full$Outlet_Size[i]))
    
  {
    full$Outlet_Size[i]="Small"
  }
  
}

full$Outlet_Size=as.factor(full$Outlet_Size)

#Modifying fat content variable

full$Item_Fat_Content=as.character(full$Item_Fat_Content)

for(i in 1:14204)
{
  if(full$Item_Fat_Content[i]=="LF"|full$Item_Fat_Content[i]=="low fat")
  {
    full$Item_Fat_Content[i]= "Low Fat"
  }
  else if(full$Item_Fat_Content[i]=="reg")
  {
    full$Item_Fat_Content[i]="Regular"
  }
}
table(full$Item_Fat_Content)
full$Item_Fat_Content=as.factor(full$Item_Fat_Content)

#Modifying year


full$Outlet_Establishment_Year=2013-full$Outlet_Establishment_Year

#Looking at Item visibility


for(i in 1:14204)
{
  if(full$Item_Visibility[i]==0)
  {
    full$Item_Visibility[i]=NA
  }
}
#Looking at identifier

id=substring(full$Item_Identifier,1,2)
unique(id)

#Creating a new variable Item_Id

full$Item_Id=NA

for(i in 1:14204)
{
  full$Item_Id[i]=substring(full$Item_Identifier[i],1,2)
}

table(full$Item_Id)

typeof(full$Item_Id)

full$Item_Id=as.factor(full$Item_Id)


#Imputing weight column

library(mlr)

rpart_imp <- impute(full[-c(1,13)], target = "Item_Outlet_Sales",
                    classes = list(numeric = imputeLearner(makeLearner("regr.gbm")),
                                   factor = imputeLearner(makeLearner("classif.gbm")))
)

full=cbind(rpart_imp$data,full[c(1,13)])

sapply(full,function(x){sum(is.na(x))})


#Checking relation between categorical variables


full$Outlet_Establishment_Year=as.factor(full$Outlet_Establishment_Year)

library(GoodmanKruskal)
GK=GKtauDataframe(full[c(2,4,6,7,8,9,10,12)])

plot(GK)

#We see that outlet identifier completely predicts Outlet size,outlet type,outlet establishment year and outlet
#location type

full$Outlet_Establishment_Year=as.numeric(full$Outlet_Establishment_Year)


#Model building

full=full[-c(8,7,9,10,13)]


new_train = full[full$flag_train==1,-9]
new_test = full[full$flag_train==0,-9]


#xgboost

library(xgboost)
library(caret)

cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                                              allowParallel=T)

xgb.grid <- expand.grid(nrounds =250,
                        eta = 0.1,
                        max_depth =c(4,6,8,10),gamma=c(0,5,10,15),
                        colsample_bytree=c(0.4,0.6,0.8,1),min_child_weight=c(1,seq(10,100,by=10)),
                        subsample=c(0.5,0.8,1)
)
set.seed(45)
xgb_tune <-train(Item_Outlet_Sales~.,
                 data=df,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="RMSE"
                 
)

xgb_tune
varImp(xgb_tune)


label_train = new_train$Item_Outlet_Sales
label_test = new_test$Item_Outlet_Sales

train_matrix = model.matrix(~.+0,data = new_train[-7])
test_matrix=model.matrix(~.+0,data=new_test[-7])

train_xgb = xgb.DMatrix(data = train_matrix,label = label_train) 
test_xgb = xgb.DMatrix(data=test_matrix,label=label_test)


model=xgb.cv(data=train_xgb,nrounds =250,nfold=5,min_child_weight=90,subsample=1,
             early_stopping_rounds=30,colsample_bytree=0.8, eta=0.1,gamma=5,max_depth=6)

model=xgb.train(train_xgb,nrounds = 200,params=list(nfold=5,min_child_weight=90,subsample=1,
                colsample_bytree=0.8, eta=0.1,gamma=5,max_depth=6))


pred_xg=predict(model,test_xgb)

test$Item_Outlet_Sales=pred_xg

sub=test[c(1,7,13)]

write.csv(sub,"sub.csv",row.names = F)



