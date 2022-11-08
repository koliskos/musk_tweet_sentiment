setwd("~")
setwd("/Users/skylarkolisko/Desktop:/DS_Capstone/musk_tweet_sentiment")
getwd()

library(tidyverse)



sentis<- read.csv("only_yes_keys_combined_sentis.csv", stringsAsFactors = FALSE) 
colnames(sentis)

# Make training set and testing set
set.seed(12) 
nrow(sentis)
indicesTrainingSet<-sample(1:nrow(sentis), 530, replace=FALSE)
train<-sentis[indicesTrainingSet,]
test<-sentis[-indicesTrainingSet,]
nrow(train) 
nrow(test) 


lm_model <-lm(fav_to_follower_ratio~prediction_creative+prediction_adventurous+prediction_generous+prediction_decisive+prediction_diplomatic+prediction_affectionate+prediction_hardworking+prediction_observant+prediction_helpful+prediction_optimistic+prediction_honest+prediction_funny+prediction_amusing+prediction_calm+prediction_polite+prediction_charismatic+prediction_romantic+prediction_friendly+prediction_clever+prediction_compassionate+prediction_aggressive+prediction_cynical+prediction_grumpy+prediction_nervous+prediction_defensive+prediction_arrogant+prediction_impulsive+prediction_patronizing+prediction_pessimistic+prediction_sullen+prediction_stubborn+prediction_egotistical+prediction_rude+prediction_mean+prediction_secretive+prediction_bossy+prediction_sarcastic+prediction_irresponsible+prediction_lazy+prediction_selfish, data = train)



summary(lm_model)
# R^2 = 0.2282
# Adjusted R^2 = 0.1456
# To perform linear regression, there must be equal variance within the residuals
# Since our Adjusted R^2 is similar to our R^2, we can infer that our model is not over fitting. We would be suspicious of overfitting if our R^2 was very large (meaning little residuals) and our Adjusted R^2 was much smaller, we could see that our R^2 was probably only large because it was over fitted. 


# Calculate MSE (mean squared error)
mean(summary(lm_model)$residuals^2) # equals 1.49782. This means our model tends to be biased towards over estimating the like value.

# boxplot(sentis $fav_to_follower_ratio ~ all_names$Assigned_Sex + all_names$Pronoun, ylim=c(0,1), horizontal=TRUE)

# Plot the residuals vs fitted vals to check that there is no heteroscedasticity.
# The residuals vs fitted vals plot seems to show mostly homoscedasticity.
plot(lm_model$fitted.values, lm_model$resid)
set.seed(44)
fit.jit<-lm_model $fitted.values + rnorm(nrow(sentis), 0, 1)
res.jit<-lm_model $residuals + rnorm(nrow(sentis), 0, 1)
plot(fit.jit, res.jit)
abline(h=0)
# add spline
spout<-smooth.spline(lm_model $residuals~ lm_model $fitted.values)
points(spout$x, spout$y, col="red",type="l")

# Normality of error seems to hold because the spline of the residuals is near the y=0 line, meaning that there is almost equal probability of the residuals being above as below the y=0 line, ie the distribution of residuals is centered at about zero (the MSE is 1.49 so this amkes sense).


trimmed_lm_model <-lm(fav_to_follower_ratio~prediction_diplomatic+prediction_observant+prediction_compassionate+prediction_arrogant+prediction_egotistical+prediction_rude, data = train)

summary(trimmed_lm_model)
# R^2 = 0.14
# Adjusted R^2 = 0.1273
mean(summary(trimmed_lm_model)$residuals^2) # 1.668992

# Plot the residuals vs fitted vals to check that there is no heteroscedasticity.
# There definitely does not seem to be perfect homoscedasticity, but perhaps it is acceptable.
plot(trimmed_lm_model $fitted.values, trimmed_lm_model $resid)
set.seed(44)
fit.jit<-trimmed_lm_model$fitted.values + rnorm(nrow(sentis), 0, 1)
res.jit<-trimmed_lm_model$residuals + rnorm(nrow(sentis), 0, 1)
plot(fit.jit, res.jit)
abline(h=0)
# add spline
spout<-smooth.spline(trimmed_lm_model$residuals~ trimmed_lm_model$fitted.values)
points(spout$x, spout$y, col="red",type="l")

# Predict new tweet's value
test_predictions<- predict(trimmed_lm_model,  new=test)
print(test_predictions)
rmse_test <- sqrt(sum((test$fav_to_follower_ratio - test_predictions)^2)/nrow(test))
rmse_test