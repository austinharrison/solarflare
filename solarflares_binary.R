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
library(caret)
library(pROC)
library(Cairo)    # Save nicer plots
library(gridExtra)


setwd("C:/Users/austi/OneDrive/Documents/MSPA/422/Group Project/solar flare/solarflare/")
par(mfrow = c(1, 1))

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

# Create binary variable to indicate if a flare of any class occurred
flare2$flare_occurred <- (flare2$c_class > 0 | flare2$m_class > 0 | flare2$x_class > 0 )
flare2$flare_occurred <- ifelse(flare2$flare_occurred, 1, 0)

# Let's only try to predict C class, so remove M and X classes
flare2 <- flare2 %>% dplyr::select(-m_class, -x_class, -c_class)

flare2$flare_occ_factor <- as.factor(flare2$flare_occurred)
# bar chart for zurich class
p1 <- ggplot(flare2, aes(x = zurich_class, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

p2 <- ggplot(flare2, aes(x = spot_size, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

p3 <- ggplot(flare2, aes(x = spot_distrib, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

p4 <- ggplot(flare2, aes(x = activity, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

p5 <- ggplot(flare2, aes(x = evolution, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

p6 <- ggplot(flare2, aes(x = act_code, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

p7 <- ggplot(flare2, aes(x = hist_complex, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

p8 <- ggplot(flare2, aes(x = become_complex, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

p9 <- ggplot(flare2, aes(x = area, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

p10 <- ggplot(flare2, aes(x = area_largest, fill = flare_occ_factor)) +
  geom_bar(position = "dodge") +
  guides(fill = F)

grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, ncol = 5)

### Create test and training sets
n <- dim(flare2)[1]
set.seed(1)
test <- sample(n, round(n/4)) # randomly sample 25% test
data.train <- flare2[-test, ]
data.test <- flare2[test, ]

# Create model matrices for ridge and lasso
x       <- model.matrix(flare_occurred ~ ., data = flare2)[,-1] # 1st col is 1's
x.train <- x[-test,]          # define training predictor matrix
x.test  <- x[test,]           # define test predictor matrix
y       <- flare2$flare_occurred     # define response variable
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

mean.roc <- pROC::roc(y.test, rep(mean.pred, length(y.test)))
plot(mean.roc, main = "Mean")
mean.cf <- confusionMatrix(as.factor(rep(round(mean.pred), length(y.test))),
                as.factor(y.test))

mean.cf$table
mean.cf$overall[1]

### Linear model
lm.fit <- lm(flare_occurred ~ ., data = data.train)
summary(lm.fit)
lm.pred <- predict(lm.fit, newdata = data.test)
lm.mse <- mean((lm.pred - data.test$flare_occurred)^2)
lm.se <- se((lm.pred - data.test$flare_occurred)^2)

lm.roc <- pROC::roc(y.test, lm.pred)
plot(lm.roc, main = "Linear Model")
lm.cf <- confusionMatrix(as.factor(round(lm.pred)),
                           as.factor(y.test))

lm.cf$table
lm.cf$overall[1]

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
  best.fit <- regsubsets(flare_occurred ~ ., 
                         data = data.train[folds != j, ], nvmax = 12)
  for(i in 1:12){
    pred <- predict(best.fit, data.train[folds == j, ], id = i)
    cv.errors[j, i] <- mean((data.train$flare_occurred[folds == j] - pred)^2)
  }
}

mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors
which.min(mean.cv.errors)
par(mfrow = c(1, 1))
plot(mean.cv.errors, type = 'b')

# train on full train data set with selected number of predictors
bss10f.train <- regsubsets(flare_occurred ~ ., data = data.train, 
                           nvmax = 12)
bss10f.coef <- coef(bss10f.train, 8)
bss10f.pred <- predict(bss10f.train, data.test, 8)
bss10f.mse <- mean((data.test$flare_occurred - bss10f.pred)^2) #mean square error
bss10f.mse
bss10f.se <- se((data.test$flare_occurred - bss10f.pred)^2) #standard error
bss10f.se

bss10f.roc <- pROC::roc(y.test, bss10f.pred)
plot(bss10f.roc, main = "Best Subset")
bss10f.cf <- confusionMatrix(as.factor(round(bss10f.pred)),
                         as.factor(y.test))

bss10f.cf$table
bss10f.cf$overall[1]

### Random Forest
rf.train <- randomForest(flare_occurred ~ ., data = data.train, 
                      importance = TRUE, ntree = 5000, mtry = 2)
# varImpPlot(rf.train)
rf.train
rf.pred <- predict(rf.train, data.test)

rf.mse <- mean((data.test$flare_occurred - rf.pred)^2) #mean square error
rf.mse
rf.se <- se((data.test$flare_occurred - rf.pred)^2) #standard error
rf.se

rf.roc <- pROC::roc(y.test, rf.pred)
plot(rf.roc, main = "Random Forest")
rf.cf <- confusionMatrix(as.factor(round(rf.pred)),
                             as.factor(y.test))

rf.cf$table
rf.cf$overall[1]

# ### Random Forest 2
# flare2b <- flare2
# flare2b$flare_occurred <- as.factor(flare2b$flare_occurred)
# 
# rf2.train <- randomForest(flare_occurred ~ ., data = flare2b[-test, ])
# 
# rf2.train
# rf2.pred <- predict(rf2.train, data.test, type = 'prob')
# 
# rf2.mse <- mean((data.test$flare_occurred - rf2.pred)^2) #mean square error
# rf2.mse
# rf2.se <- se((data.test$flare_occurred - rf2.pred)^2) #standard error
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

ridge.roc <- pROC::roc(y.test, ridge.pred)
plot(ridge.roc, main = "Ridge")
ridge.cf <- confusionMatrix(as.factor(round(ridge.pred)),
                         as.factor(y.test))

ridge.cf$table
ridge.cf$overall[1]


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

lasso.roc <- pROC::roc(y.test, lasso.pred)
plot(lasso.roc, main = "LASSO")
lasso.cf <- confusionMatrix(as.factor(round(lasso.pred)),
                            as.factor(y.test))

lasso.cf$table
lasso.cf$overall[1]

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

knn.roc <- pROC::roc(y.test, knn.pred)
plot(knn.roc, main = "KNN")
knn.cf <- confusionMatrix(as.factor(round(knn.pred)),
                            as.factor(y.test))

knn.cf$table
knn.cf$overall[1]

### Artificial neural network model
ann.fit <- nnet(flare_occurred ~ ., data = data.train, size = 10)
ann.pred <- predict(ann.fit, data.test)
ann.mse <- mean((y.test - ann.pred)^2)
ann.mse # [1] 0.564971
ann.se <- se((y.test - ann.pred)^2)
ann.se # [1] 0.1233341

ann.roc <- pROC::roc(y.test, ann.pred)
plot(ann.roc, main = "Neural Net")
ann.cf <- confusionMatrix(as.factor(round(ann.pred)),
                          as.factor(y.test))

ann.cf$table
ann.cf$overall[1]


### Tree model
tree.flare <- tree(flare_occurred ~ ., data = data.train)
summary(tree.flare) # zurich_class, spot_size, spot_distrib, and activity are 
# included in tree
plot(tree.flare)
text(tree.flare, pretty = 0)
tree.flare
yhat.tree <- predict(tree.flare, data.test)
tree.mse <- mean((y.test - yhat.tree)^2)
tree.mse # [1] 0.5562123343
tree.se <- se((y.test - yhat.tree)^2)
tree.se # [1] 0.1573765894

tree.roc <- pROC::roc(y.test, yhat.tree)
plot(tree.roc, main = "Decision Tree")
tree.cf <- confusionMatrix(as.factor(round(yhat.tree)),
                          as.factor(y.test))

tree.cf$table
tree.cf$overall[1]

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

prune.roc <- pROC::roc(y.test, yhat.prune)
plot(prune.roc, main = "Pruned Tree")
prune.cf <- confusionMatrix(as.factor(round(yhat.prune)),
                           as.factor(y.test))

prune.cf$table
prune.cf$overall[1]




# Model Evaluations
results.mse <- c(mean.mse, lm.mse, bss10f.mse, ridge.mse, lasso.mse, rf.mse, 
                 knn.mse, ann.mse, tree.mse, prune.mse)
results.se <- c(mean.se, lm.se, bss10f.se, ridge.se, lasso.se, rf.se, knn.se,
                ann.se, tree.se, prune.se)
results.accuracy <- c(mean.cf$overall[1], lm.cf$overall[1], 
                      bss10f.cf$overall[1], ridge.cf$overall[1], 
                      lasso.cf$overall[1], rf.cf$overall[1], knn.cf$overall[1], 
                      ann.cf$overall[1], tree.cf$overall[1], 
                      prune.cf$overall[1])
results.sensitivity <- c(mean.cf$byClass[1], lm.cf$byClass[1], 
                      bss10f.cf$byClass[1], ridge.cf$byClass[1], 
                      lasso.cf$byClass[1], rf.cf$byClass[1], knn.cf$byClass[1], 
                      ann.cf$byClass[1], tree.cf$byClass[1], 
                      prune.cf$byClass[1])
results.specificity <- c(mean.cf$byClass[2], lm.cf$byClass[2], 
                         bss10f.cf$byClass[2], ridge.cf$byClass[2], 
                         lasso.cf$byClass[2], rf.cf$byClass[2], knn.cf$byClass[2], 
                         ann.cf$byClass[2], tree.cf$byClass[2], 
                         prune.cf$byClass[2])
results.auc <- c(mean.roc$auc, lm.roc$auc, bss10f.roc$auc, ridge.roc$auc,
                      lasso.roc$auc, rf.roc$auc, knn.roc$auc, 
                      ann.roc$auc, tree.roc$auc, prune.roc$auc)
results.model <- c("Mean", "Least Squares", "Best Subset 10-Fold CV",
                   "Ridge", "Lasso", "Random Forest", "KNN", "Neural Net", 
                   "Decision Tree", "Pruned Tree")
results.fill <- rep("normal", length(results.model))
results.fill[1] <- "highlight"
results <- data.frame(results.model, results.mse, results.se, results.accuracy, 
                      results.sensitivity, results.specificity, results.auc, 
                      results.fill)
colnames(results) <- c("Model", "MSE", "SE", "Accuracy", "Sensitivity", "Specificity",
                       "AUC", "fillcolor")
results <- arrange(results, MSE)
results # %>% write.csv("modresults.csv", row.names = F)

ggplot(results, aes(x = reorder(Model, desc(MSE)), y = MSE, fill = fillcolor)) +
  geom_bar(position = position_dodge(), stat = "identity", width = .8) +
  geom_errorbar(aes(ymin = MSE - SE, ymax = MSE + SE), width = .4) +
  geom_label(label = round(results$MSE, 3), fill = "white") +
  labs(x = "Model") +
  guides(fill = F)
ggsave(filename = "msebarplot-logistic.png", width = 10, height = 6)

# Accuracy
results <- arrange(results, Accuracy)
results # %>% write.csv("modresults.csv", row.names = F)

ggplot(results, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = fillcolor)) +
  geom_bar(position = position_dodge(), stat = "identity", width = .8) +
  geom_label(label = round(results$Accuracy, 3), fill = "white") +
  labs(x = "Model") +
  guides(fill = F)
ggsave(filename = "accuracybarplot-logistic.png", width = 10, height = 6)

# AUC
results <- arrange(results, AUC)
results # %>% write.csv("modresults.csv", row.names = F)

ggplot(results, aes(x = reorder(Model, AUC), y = AUC, fill = fillcolor)) +
  geom_bar(position = position_dodge(), stat = "identity", width = .8) +
  geom_label(label = round(results$AUC, 3), fill = "white") +
  labs(x = "Model") +
  guides(fill = F)
ggsave(filename = "aucbarplot-logistic.png", width = 10, height = 6)

# ROC curves
Cairo(800, 800, file = "roccurves.png", type = "png", bg = "white", dpi = 96)

par(mfrow = c(3, 3)) #mean.mse, lm.mse, bss10f.mse, ridge.mse, lasso.mse, rf.mse, 
#knn.mse, ann.mse, tree.mse
plot(mean.roc, main = "Mean")
plot(lm.roc, main = "Linear Model")
plot(bss10f.roc, main = "Best Subset")
plot(ridge.roc, main = "Ridge")
plot(lasso.roc, main = "LASSO")
plot(rf.roc, main = "Random Forest")
plot(knn.roc, main = "KNN")
plot(ann.roc, main = "Neural Net")
plot(tree.roc, main = "Decision Tree")

dev.off()

# TODO - Remove all cases where no flares occurred and see if we can predict the
# type.

