library(ggplot2)
library(cowplot)
library(randomForest)


data <- read.csv("processed.cleveland.data", header=FALSE)

## Reformat the data so that it is 
#####################################
head(data) # you see data, but no column names

colnames(data) <- c(
  "age",
  "sex",# 0 = female, 1 = male
  "cp", # chest pain 
  # 1 = typical angina, 
  # 2 = atypical angina, 
  # 3 = non-anginal pain, 
  # 4 = asymptomatic
  "trestbps", # resting blood pressure (in mm Hg)
  "chol", # serum cholestoral in mg/dl
  "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
  "restecg", # resting electrocardiographic results
  # 1 = normal
  # 2 = having ST-T wave abnormality
  # 3 = showing probable or definite left ventricular hypertrophy
  "thalach", # maximum heart rate achieved
  "exang",   # exercise induced angina, 1 = yes, 0 = no
  "oldpeak", # ST depression induced by exercise relative to rest
  "slope", # the slope of the peak exercise ST segment 
  # 1 = upsloping 
  # 2 = flat 
  # 3 = downsloping 
  "ca", # number of major vessels (0-3) colored by fluoroscopy
  "thal", # this is short of thalium heart scan
  # 3 = normal (no cold spots)
  # 6 = fixed defect (cold spots during rest and exercise)
  # 7 = reversible defect (when cold spots only appear during exercise)
  "hd" # (the predicted attribute) - diagnosis of heart disease 
  # 0 if less than or equal to 50% diameter narrowing
  # 1 if greater than 50% diameter narrowing
)

head(data) # now we have data and column names

str(data) # this shows that we need to tell R which columns contain factors
# it also shows us that there are some missing values. There are "?"s
# in the dataset.

## First, replace "?"s with NAs.
data[data == "?"] <- NA

## Now add factors for variables that are factors and clean up the factors
## that had missing data...
data[data$sex == 0,]$sex <- "F"
data[data$sex == 1,]$sex <- "M"
data$sex <- as.factor(data$sex)

data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)

data$ca <- as.integer(data$ca) # since this column had "?"s in it (which
# we have since converted to NAs) R thinks that
# the levels for the factor are strings, but
# we know they are integers, so we'll first
# convert the strings to integiers...
data$ca <- as.factor(data$ca)  # ...then convert the integers to factor levels

data$thal <- as.integer(data$thal) # "thal" also had "?"s in it.
data$thal <- as.factor(data$thal)

## This next line replaces 0 and 1 with "Healthy" and "Unhealthy"
data$hd <- ifelse(test=data$hd == 0, yes="Healthy", no="Unhealthy")
data$hd <- as.factor(data$hd) # Now convert to a factor

str(data) ## this shows that the correct columns are factors and we've replaced
## "?"s with NAs because "?" no longer appears in the list of factors
## for "ca" and "thal"

data <- na.omit(data)

# ==================== Explore heart disease or not Gender wise =========================== #

heart = data #add labels only for plot
levels(heart$hd) = c("No disease","Disease")
levels(heart$sex) = c("female","male","")
mosaicplot(heart$sex ~ heart$hd, 
           main="Ratio of male and Female - Heart Disease / Not", shade=FALSE,color=TRUE,
           xlab="Gender", ylab="Heart disease")

# ==================== Blox plot by Age - Heart disease or not =========================== #

boxplot(heart$age ~ heart$hd,
        main=" Age wise - Heart Disease / Not ",
        ylab="Age",xlab="Heart disease")

#================================================================================#

ggplot(data, aes(x=data$sex)) +
  geom_bar() + theme(axis.text.x = element_text(angle = 90))+
  ggtitle("Gender Statistics")+ labs(y="Count(in thousand)", x = "Gender")


#####################################
##
## Now we are ready to build a random forest.
##
#####################################
set.seed(42)
#     ---------------------- Training and testing data for validation -------------------------      #

data1 <-sample(2,nrow(data),prob = c(0.7,0.3), replace = T) 

data_train <- data[data1==1,]
data_test <- data[data1==2,]

str(data_train)
str(data_test)

#generate = hd ~ age + gender + chest_pain + resting_bp + cholesterol + fbs + resting_electroc + max_hrt_rate + exang + oldpeak + slope + no_major_vessel + thalassemia + result

rd_model<- randomForest(hd ~ ., data = data_train, importance =TRUE)
rd_model

str(data)

#     ----------------------------------- Testing model --------------------------------      #
pred<- predict(rd_model, newdata = data_test, type = "class")
pred
table(pred,data_test$hd)
a <- pred[2:2]
a
#----------Give Gini Index - Priority of variables-----------------------#  
rd_model$importance

#----------Most optimized value of mtry of random forest-----------------------#  
bestmtry <- tuneRF(data_train, main = 'Most optimized value of mtry',
                   data_train$hd,
                   stepFactor = 1.2,improve = 0.01,trace = T,plot = T)

#     ------------------------------- Additional visualizations ------------------------------      #
plot(rd_model, main = "Random Forest errors for treees") #Plot the error rates or MSE of a randomForest object
plot(margin(rd_model,data_test$hd))

varImpPlot(rd_model, main = 'Variable Importance')

## Using different symbols for the classes:
MDSplot(rd_model, data_train$hd,k=2, palette=rep(1, 3), pch=as.numeric(data_train$hd))

hist(treesize(rd_model)) #Size of trees (number of nodes) in and ensemble


#     --------------------------------- Extracting the Tree --------------------------------      #
treevisual<-getTree(rd_model,1,labelVar = T)
t<-grow(rd_model,1)
t
treevisual


#model # gives us an overview of the call, along with...
# 1) The OOB error rate for the forest with ntree trees. 
#    In this case ntree=500 by default
# 2) The confusion matrix for the forest with ntree trees.
#    The confusion matrix is laid out like this:
#          
#                Healthy                      Unhealthy
#          --------------------------------------------------------------
# Healthy  | Number of healthy people   | Number of healthy people      |
#          | correctly called "healthy" | incorectly called "unhealthy" |
#          | by the forest.             | by the forest                 |
#          --------------------------------------------------------------
# Unhealthy| Number of unhealthy people | Number of unhealthy peole     |
#          | incorrectly called         | correctly called "unhealthy"  |
#          | "healthy" by the forest    | by the forest                 |
#          --------------------------------------------------------------

## Now check to see if the random forest is actually big enough...
## Up to a point, the more trees in the forest, the better. You can tell when
## you've made enough when the OOB no longer improves.

oob.error.data <- data.frame(
  Trees=rep(1:nrow(rd_model$err.rate), times=3),
  Type=rep(c("OOB", "Healthy", "Unhealthy"), each=nrow(rd_model$err.rate)),
  Error=c(rd_model$err.rate[,"OOB"], 
          rd_model$err.rate[,"Healthy"], 
          rd_model$err.rate[,"Unhealthy"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))
# ggsave("oob_error_rate_500_trees.pdf")

## Blue line = The error rate specifically for calling "Unheathly" patients that
## are OOB.
##
## Green line = The overall OOB error rate.
##
## Red line = The error rate specifically for calling "Healthy" patients 
## that are OOB.

## NOTE: After building a random forest with 500 tress, the graph does not make 
## it clear that the OOB-error has settled on a value or, if we added more 
## trees, it would continue to decrease.
## So we do the whole thing again, but this time add more trees.
#rd_model<- randomForest(hd ~ ., data = data_train, importance =TRUE)
model <- randomForest(hd ~ ., data=data_train, ntree=1000, proximity=TRUE)
model

oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "Healthy", "Unhealthy"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
          model$err.rate[,"Healthy"], 
          model$err.rate[,"Unhealthy"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))
# ggsave("oob_error_rate_1000_trees.pdf")

## After building a random forest with 1,000 trees, we get the same OOB-error
## 16.5% and we can see convergence in the graph. So we could have gotten
## away with only 500 trees, but we wouldn't have been sure that number
## was enough.


## If we want to compare this random forest to others with different values for
## mtry (to control how many variables are considered at each step)...
oob.values <- vector(length=10)
for(i in 1:10) {
  temp.model <- randomForest(hd ~ ., data=data_train, mtry=i, ntree=500)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values
## > oob.values
#[1] 0.1970443 0.1871921 0.1773399 0.1921182 0.1773399 0.1674877 0.2068966 0.1822660 0.1871921
#[10] 0.1822660
## The lowest value is when mtry=3, so the default setting was the best.

## Now let's create an MDS-plot to show how the samples are related to each 
## other.
##
## Start by converting the proximity matrix into a distance matrix.
distance.matrix <- as.dist(1-model$proximity)
#distance.matrix
mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)

## calculate the percentage of variation that each MDS axis accounts for...
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

## now make a fancy looking plot that shows the MDS axes and the variation:
mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Status=data_train$hd)

ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) + 
  geom_text(aes(color=Status)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using (1 - Random Forest Proximities)")
# ggsave(file="random_forest_mds_plot.pdf")