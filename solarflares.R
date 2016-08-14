# Group Project
# Solar Flares

library(dplyr)
library(randomForest)
library(glmnet)
library(leaps)
library(ggplot2)
library(class)
library(MASS, pos = 50)
library(gbm)
library(nnet)
library(tree)


setwd("C:/Users/austi/OneDrive/Documents/MSPA/422/Group Project/solar flare/solarflare/")


### Useful functions

se <- function(x) sd(x)/sqrt(length(x))


### Import data

# flare1 <- read.csv("flare.data1", skip = 1, header = FALSE, sep = " ")
# per the documentation, the first data file isn't as reliable

flare2 <- read.csv("flare.data2", skip = 1, header = FALSE, sep = " ")
flare_names <- c("zurich_class", "spot_size", "spot_distrib", "activity",
                 "evolution", "act_code", "hist_complex", "become_complex",
                 "area", "area_largest", "c_class", "m_class", "x_class")
colnames(flare2) <- flare_names
summary(flare2)
str(flare2)

# Make columns factors
flare2[, 4:9] <- lapply(flare2[, 4:9], FUN = as.factor)
str(flare2)

# Histograms of target variables
hist(flare2$c_class)
hist(flare2$m_class)
hist(flare2$x_class)

# Frequencies of target variables
flare2 %>% group_by(c_class) %>% summarise(freq = n())
flare2 %>% group_by(m_class) %>% summarise(freq = n())
flare2 %>% group_by(x_class) %>% summarise(freq = n())

# Let's only try to predict C class, so remove M and X classes
flare2 <- flare2 %>% dplyr::select(-m_class, -x_class)


### Create test and training sets
n <- dim(flare2)[1]
set.seed(1)
test <- sample(n, round(n/4)) # randomly sample 25% test
data.train <- flare2[-test, ]
data.test <- flare2[test, ]

# Create model matrices for ridge and lasso
x       <- model.matrix(c_class ~ ., data = flare2)[,-1] # 1st col is 1's
x.train <- x[-test,]          # define training predictor matrix
x.test  <- x[test,]           # define test predictor matrix
y       <- flare2$c_class     # define response variable
y.train <- y[-test]           # define training response variable
y.test  <- y[test]            # define test response variable
n.train <- dim(data.train)[1] # training sample size = 800
n.test  <- dim(data.test)[1]  # test sample size = 266

### Mean model - Just use the mean as prediction
mean.pred <- mean(y.train)
mean.mse <- mean((mean.pred - y.test)^2)
mean.mse
mean.se <- se((mean.pred - y.test)^2)
mean.se

### Linear model
lm.fit <- lm(c_class ~ ., data = data.train)
summary(lm.fit)
lm.pred <- predict(lm.fit, newdata = data.test)
lm.mse <- mean((lm.pred - data.test$c_class)^2)
lm.se <- se((lm.pred - data.test$c_class)^2)


### 10 fold cv best subset
# Predict function from textbook lab
predict.regsubsets <- function(object, newdata, id, ...){
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}

k <- 10 # Number of folds
set.seed(1)
folds <- sample(1:k, nrow(data.train), replace = TRUE)
cv.errors <- matrix(NA, k, 12, dimnames = list(NULL, paste(1:12)))
for(j in 1:k){
  best.fit <- regsubsets(c_class ~ ., 
                         data = data.train[folds != j, ], nvmax = 12)
  for(i in 1:12){
    pred <- predict(best.fit, data.train[folds == j, ], id = i)
    cv.errors[j, i] <- mean((data.train$c_class[folds == j] - pred)^2)
  }
}

mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors
which.min(mean.cv.errors)
par(mfrow = c(1, 1))
plot(mean.cv.errors, type = 'b')

# train on full train data set with selected number of predictors
bss10f.train <- regsubsets(c_class ~ ., data = data.train, 
                           nvmax = 12)
bss10f.coef <- coef(bss10f.train, 12)
bss10f.pred <- predict(bss10f.train, data.test, 12)
bss10f.mse <- mean((data.test$c_class - bss10f.pred)^2) #mean square error
bss10f.mse
bss10f.se <- se((data.test$c_class - bss10f.pred)^2) #standard error
bss10f.se

### Random Forest
rf.train <- randomForest(c_class ~ ., data = data.train, 
                      importance = TRUE, ntree = 5000, mtry = 2)
# varImpPlot(rf.train)
rf.train
rf.pred <- predict(rf.train, data.test)

rf.mse <- mean((data.test$c_class - rf.pred)^2) #mean square error
rf.mse
rf.se <- se((data.test$c_class - rf.pred)^2) #standard error
rf.se

# ### Random Forest 2
# flare2b <- flare2
# flare2b$c_class <- as.factor(flare2b$c_class)
# 
# rf2.train <- randomForest(c_class ~ ., data = flare2b[-test, ])
# 
# rf2.train
# rf2.pred <- predict(rf2.train, data.test, type = 'prob')
# 
# rf2.mse <- mean((data.test$c_class - rf2.pred)^2) #mean square error
# rf2.mse
# rf2.se <- se((data.test$c_class - rf2.pred)^2) #standard error
# rf2.se

### Ridge

ridge.mod <- glmnet(x.train, y.train, alpha = 0)
dim(coef(ridge.mod))

set.seed(1)
ridge.cv.out <- cv.glmnet(x.train, y.train, alpha = 0, nfolds = 10)

ridge.bestlam <- ridge.cv.out$lambda.min
ridge.bestlam
# ridge.bestlam1se <- ridge.cv.out$lambda.1se
# ridge.bestlam1se
ridge.pred <- predict(ridge.mod, s = ridge.bestlam, newx = x.test)
ridge.mse <- mean((y.test - ridge.pred) ^ 2) #mean square error
ridge.mse
ridge.se <- se((y.test - ridge.pred) ^ 2) #standard error
ridge.se

ridge.coef <- predict(ridge.mod, type = "coefficients", 
                      s = ridge.bestlam)[1:11, ]
predict(ridge.mod, type = "coefficients", 
        s = ridge.cv.out$lambda.min)[1:11, ]
ridge.coef
ridge.coef[ridge.coef!=0]

# ridge plots
# Cairo(800, 600, file = "ridgeplot2.png", type = "png", bg = "white", dpi = 96)
plot(ridge.mod, xvar = "lambda", label = TRUE) # colorful plot
# dev.off()
# Cairo(800, 600, file = "ridgeplot.png", type = "png", bg = "white", dpi = 96)
plot(ridge.cv.out)
# dev.off()

### Lasso

lasso.mod <- glmnet(x.train, y.train, alpha = 1)
dim(coef(lasso.mod))

set.seed(1)
lasso.cv.out <- cv.glmnet(x.train, y.train, alpha = 1, nfolds = 10)

lasso.bestlam <- lasso.cv.out$lambda.min
lasso.bestlam
# lasso.bestlam1se <- lasso.cv.out$lambda.1se
# lasso.bestlam1se
lasso.pred <- predict(lasso.mod, s = lasso.bestlam, newx = x.test)
lasso.mse <- mean((y.test - lasso.pred) ^ 2) #mean square error
lasso.mse
lasso.se <- se((y.test - lasso.pred) ^ 2) #standard error
lasso.se

lasso.coef <- predict(lasso.mod, type = "coefficients", 
                      s = lasso.bestlam)[1:11, ]
predict(lasso.mod, type = "coefficients", 
        s = lasso.cv.out$lambda.min)[1:11, ]
lasso.coef
lasso.coef[lasso.coef!=0]

# Lasso plots
# Cairo(800, 600, file = "lassoplot2.png", type = "png", bg = "white", dpi = 96)
plot(lasso.mod, xvar = "lambda", label = TRUE) # colorful plot
# dev.off()
# Cairo(800, 600, file = "lassoplot.png", type = "png", bg = "white", dpi = 96)
plot(lasso.cv.out)
# dev.off()


# KNN
knn.pred <- as.numeric(as.character(knn(x.train, x.test, y.train, k = 5)))
knn.pred
knn.mse <- mean((y.test - knn.pred) ^ 2) #mean square error
knn.mse
knn.se <- se((y.test - knn.pred) ^ 2) #standard error
knn.se

### Boosting model
set.seed(1)
boost.flare <- gbm(c_class ~ ., data = data.train, n.trees = 5000, interaction.depth = 4)
summary(boost.flare)
par(mfrow=c(1,2))
plot(boost.flare, i = "zurich_class")
plot(boost.flare, i = "spot_size")
yhat.boost <- predict(boost.flare, newdata = data.test, n.trees = 5000)
boost.mse <- mean((y.test - yhat.boost)^2) 
boost.mse # [1] 0.5634364807
boost.se <- se((y.test - yhat.boost)^2)
boost.se # [1] 0.1202052845

## fit using only zurich_class and spot_size since they are the most important variables based on summary
boost.flare <- gbm(c_class ~ zurich_class + spot_size, data = data.train, n.trees = 5000, interaction.depth = 4)
yhat.boost <- predict(boost.flare, newdata = data.test, n.trees = 5000)
boost.mse <- mean((y.test - yhat.boost)^2) 
boost.mse # [1] 0.5486130138
boost.se <- se((y.test - yhat.boost)^2)
boost.se # [1] 0.1162284558
# slight improvement, but not much

### Artificial neural network model
nnet.fit <- nnet(c_class ~ ., data = data.train, size = 2)
nnet.predict <- predict(nnet.fit, data.test)
ann.mse <- mean((y.test - nnet.predict)^2)
ann.mse # [1] 0.564971
ann.se <- se((y.test - nnet.predict)^2)
ann.se # [1] 0.1233341
# plot(data.test$c_class, nnet.predict, main = "Artificial Neural Network Predictions vs Actual",
#      xlab = "Actual", ylab = "Predictions", pch = 19, col = "blue")



### Tree model
tree.flare <- tree(c_class ~ ., data = data.train)
summary(tree.flare) # zurich_class, spot_size, spot_distrib, and activity are included in tree
plot(tree.flare)
text(tree.flare, pretty = 0)
tree.flare
yhat.tree <- predict(tree.flare, data.test)
tree.mse <- mean((y.test - yhat.tree)^2)
tree.mse # [1] 0.5562123343
tree.se <- se((y.test - yhat.tree)^2)
tree.se # [1] 0.1573765894

# prune the tree to see if we get better results
set.seed(1)
cv.flare <- cv.tree(tree.flare)
plot(cv.flare$size, cv.flare$dev, type = "b")
prune.flare <- prune.tree(tree.flare, best = 2)
plot(prune.flare)
text(prune.flare, pretty = 0)
yhat.prune <- predict(prune.flare, data.test)
prune.mse <- mean((y.test - yhat.prune)^2)
prune.mse # [1] 0.5724697791
prune.se <- se((y.test - yhat.prune)^2)
prune.se # [1] 0.1573765894
# unfortunately the model did not improve when pruning




# Model Evaluations
results.mse <- c(mean.mse, lm.mse, bss10f.mse, ridge.mse, lasso.mse, rf.mse, 
                 knn.mse, boost.mse, ann.mse, tree.mse)
results.se <- c(mean.se, lm.se, bss10f.se, ridge.se, lasso.se, rf.se, knn.se,
                boost.se, ann.se, tree.se)
results.model <- c("Mean", "Least Squares", "Best Subset 10-Fold CV",
                   "Ridge", "Lasso", "Random Forest", "KNN", "Boosting",
                   "Neural Net", "Decision Tree")
results <- data.frame(results.model, results.mse, results.se)
colnames(results) <- c("Model", "MSE", "SE")
results <- arrange(results, MSE)
results # %>% write.csv("modresults.csv", row.names = F)

ggplot(results, aes(x = reorder(Model, desc(MSE)), y = MSE)) +
  geom_bar(position = position_dodge(), stat = "identity", width = .8, 
           fill = "lightblue") +
  geom_errorbar(aes(ymin = MSE - SE, ymax = MSE + SE), width = .4) +
  geom_label(label = round(results$MSE, 3)) +
  labs(x = "Model")
ggsave(filename = "msebarplot.png", width = 10, height = 6)
