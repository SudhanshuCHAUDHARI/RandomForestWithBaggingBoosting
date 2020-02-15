extractTitle <- function(sex, name)
{
  name = as.character(name)
  
  if (sex=="male")
  {
    if (length(grep("Mr.",name))>0)
    {
      return ("Mr")
    }
    else
    {
      return ("Other-M")
    }
  }
  else
  {
    if (length(grep("Miss",name))>0)
    {
      return ("Miss")
    }
    else if (length(grep("Mrs.",name))>0)
    {
      return ("Mrs")
    }
    else
    {
      return ("Other-F")
    }
  }
}

extractAgeGroup <- function(age, prob1, prob2)
{
  if (is.na(age))
  {
    val = runif(1,0,1)
    if (val < prob1) age2=30
    else if (val < prob2) age2=60
    else age2 = 70
  }
  else age2=age
  
  if (age2 <= 15) return("Child")
  else if (age2 <= 30) return("Young")
  else if (age2 <= 60) return("Adult")
  else return ("Old")
}

extractFamily <- function(sibsp, parch)
{
  if (sibsp+parch==0) return("Single")
  else if (sibsp+parch>3) return("BigFamily")
  else return("SmallFamily")
}

#Lecture des données d'apprentissage

train <- read.csv("H:\\Downloads\\train.csv",sep=",", stringsAsFactor=FALSE)

# Transformer les 4 variables suivantes en variables nominalestrain$Survived

train$Survived <- factor(train$Survived)
train$Pclass <- factor(train$Pclass)
train$Sex <- factor(train$Sex)
train$Embarked <- factor(train$Embarked)

#Adding the variable Title

titles <- NULL
for(i in 1:nrow(train))
{
  titles <- c(titles, extractTitle(train[i,"Sex"], train[i,"Name"])) 
}
train$Title = as.factor(titles)
train
#Adding the variable AgeGroup


AgeGroups <- NULL
for(i in 1:nrow(train))
{
  AgeGroups <- c(AgeGroups, 
                 extractAgeGroup(train[i,"Age"],0.25,0.75)) 
}
train$AgeGroup = as.factor(AgeGroups)
train
#Adding the variable Family

Family <- NULL
for(i in 1:nrow(train))
{
  Family <- c(Family, extractFamily(train[i,"SibSp"],train[i,"Parch"])) 
}
train$Family = as.factor(Family)

#Adding the variable Fare


train$FareGroup <- cut(train$Fare, breaks=c(-1, 8, 15, 32, 75, 520), 
                       labels=c("Very cheep","Cheep","Intermediate","Expensive","Very expensive")) 

#The variable Cabine
#We define a new variable Cabine2 by extracting the first letter of the variable cabine.

train$Cabin2 <- substr(train$Cabin, 1, 1)

for(i in 1:nrow(train))
{
  if (train[i,"Cabin2"]=="") train[i,"Cabin2"]="X" 
}

train$Cabin2 <- as.factor(train$Cabin2)
#install.packages("randomForest")

data1 = sort(sample(nrow(train), nrow(train)*0.8))

#creating training data set by selecting the output row values
training<-train[data1,]

#creating test data set by not selecting the output row values
test<-train[-data1,]

training
library(randomForest)
#training
rf<-randomForest(Survived ~ Pclass + Sex + Title + Family + AgeGroup + FareGroup + Cabin2, data = training)
rf$confusion
#testing
res <- predict(rf,newdata = test)


#Evaluate Test accuracy
confTest <- table(res,test$Survived)
confTest

accuracyTEST = (confTest[1,1] + confTest[2,2])/(confTest[1,1] + confTest[2,2]+ confTest[1,2]+ confTest[2,1])*100
error = (confTest[1,2]+ confTest[2,1])/(confTest[1,1] + confTest[2,2]+ confTest[1,2]+ confTest[2,1])
error
accuracyTEST
#Evaluate OOB  accuracy
confOOB = rf$confusion
accuracyOOB = (confOOB[1,1] + confOOB[2,2])/(confOOB[1,1] + confOOB[2,2]+ confOOB[1,2]+ confOOB[2,1])*100
accuracyOOB

