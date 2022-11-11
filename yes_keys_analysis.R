setwd("~")
setwd("/Users/skylarkolisko/Desktop:/DS_Capstone/musk_tweet_sentiment")
getwd()

library(partykit)

library(tidyverse)



sentis<- read.csv("FINAL_data_used/ALL_COMBINED_FINAL.csv", stringsAsFactors = FALSE) 
colnames(sentis)

# Make training set and testing set
set.seed(12) 
nrow(sentis)
indicesTrainingSet<-sample(1:nrow(sentis), 530, replace=FALSE)
train<-sentis[indicesTrainingSet,]
test<-sentis[-indicesTrainingSet,]
nrow(train) 
nrow(test) 


# all vars un trimmed model in predicting fave score 
lm_model <-lm(fav_to_follower_ratio ~prediction_creative+prediction_adventurous+prediction_generous+prediction_decisive+prediction_diplomatic+prediction_affectionate+prediction_hardworking+prediction_observant+prediction_helpful+prediction_optimistic+prediction_honest+prediction_funny+prediction_amusing+prediction_calm+prediction_polite+prediction_charismatic+prediction_romantic+prediction_friendly+prediction_clever+prediction_compassionate+prediction_aggressive+prediction_cynical+prediction_grumpy+prediction_nervous+prediction_defensive+prediction_arrogant+prediction_impulsive+prediction_patronizing+prediction_pessimistic+prediction_sullen+prediction_stubborn+prediction_egotistical+prediction_rude+prediction_mean+prediction_secretive+prediction_bossy+prediction_sarcastic+prediction_irresponsible+prediction_lazy+prediction_selfish, data = train)
summary(lm_model)

# lm model made to predict fav score
lm_fav <-lm(fav_to_follower_ratio ~prediction_creative+prediction_adventurous+prediction_generous+prediction_decisive+prediction_diplomatic+prediction_affectionate+prediction_observant+prediction_optimistic+prediction_funny+prediction_calm+prediction_polite+prediction_charismatic+prediction_romantic+prediction_friendly+prediction_clever+prediction_compassionate+prediction_defensive+prediction_arrogant+prediction_impulsive+prediction_pessimistic+prediction_sullen+prediction_stubborn+prediction_egotistical+prediction_rude+prediction_mean+prediction_irresponsible, data = train)
summary(lm_fav) # Muliple R^2 of .1915 and Adjusted R^2 of 0.1497


# Calculate MSE (mean squared error)
mean(summary(lm_fav)$residuals^2)# 1.56803


# model trimmed to predict retweet value
lm_retweet <-lm(retweet_to_follower_ratio ~prediction_creative+prediction_adventurous+prediction_generous+prediction_decisive+prediction_diplomatic+prediction_affectionate+prediction_hardworking+prediction_observant+prediction_optimistic+prediction_funny+prediction_calm+prediction_polite+prediction_charismatic+prediction_romantic+prediction_friendly+prediction_clever+prediction_compassionate+prediction_grumpy+prediction_arrogant+prediction_impulsive+prediction_pessimistic+prediction_sullen+prediction_stubborn+prediction_egotistical+prediction_rude+prediction_mean+prediction_irresponsible, data = train)

summary(lm_retweet)
# Multiple R^2 = 0.235
# Adjusted R^2 = 0.1939
# To perform linear regression, there must be equal variance within the residuals
# Since our Adjusted R^2 is similar to our R^2, we can infer that our model is not over fitting. We would be suspicious of overfitting if our R^2 was very large (meaning little residuals) and our Adjusted R^2 was much smaller, we could see that our R^2 was probably only large because it was over fitted. 

retweet_tree_model<-ctree(retweet_to_follower_ratio ~prediction_creative+prediction_adventurous+prediction_generous+prediction_decisive+prediction_diplomatic+prediction_affectionate+prediction_hardworking+prediction_observant+prediction_optimistic+prediction_funny+prediction_calm+prediction_polite+prediction_charismatic+prediction_romantic+prediction_friendly+prediction_clever+prediction_compassionate+prediction_grumpy+prediction_arrogant+prediction_impulsive+prediction_pessimistic+prediction_sullen+prediction_stubborn+prediction_egotistical+prediction_rude+prediction_mean+prediction_irresponsible, data = train)
plot(tree_model)

# Calculate MSE (mean squared error)
mean(summary(lm_retweet)$residuals^2) # equals 1.736. This means our model tends to be biased towards over estimating the like value.

rmse.lm_retweet <-sqrt(sum((train$fav_to_follower_ratio-predict(lm_retweet))^2)/nrow(train))
rmse.lm_retweet # 3.130922
rmse.retweet_tree_model <-sqrt(sum((train$fav_to_follower_ratio-predict(retweet_tree_model))^2)/nrow(train))
rmse.retweet_tree_model # 3.1309
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

# The final model uses the predictions for the following seven adjectives: diplomatic, observant, compassionate, generous, polite, egotistical, and rude. 
trimmed_lm_model <-lm(retweet_to_follower_ratio ~prediction_diplomatic+prediction_observant+prediction_compassionate+prediction_generous+prediction_polite+prediction_egotistical+prediction_rude, data = train)
# trimmed_lm_model <-lm(fav_to_follower_ratio~prediction_diplomatic+prediction_observant+prediction_compassionate+prediction_arrogant+prediction_egotistical+prediction_rude, data = train)

summary(trimmed_lm_model)
# R^2 = 0.09668
# Adjusted R^2 = 0.08456
mean(summary(trimmed_lm_model)$residuals^2) # 1.751884

# Plot the residuals vs fitted vals to check that there is no heteroscedasticity.
# There definitely does not seem to be perfect homoscedasticity, but perhaps it is acceptable.
plot(trimmed_lm_model $fitted.values, trimmed_lm_model $resid)
set.seed(44)
# fit.jit<-trimmed_lm_model$fitted.values + rnorm(nrow(sentis), 0, 1)
# res.jit<-trimmed_lm_model$residuals + rnorm(nrow(sentis), 0, 1)
# plot(fit.jit, res.jit)
abline(h=0)
# add spline
spout<-smooth.spline(trimmed_lm_model$residuals~ trimmed_lm_model$fitted.values)
points(spout$x, spout$y, col="red",type="l")
title(main="Model Residuals vs Fitted Values")

# Predict new tweet's value
test_predictions<- predict(trimmed_lm_model,  new=test)
print(test_predictions)
rmse_test <- sqrt(sum((test$fav_to_follower_ratio - test_predictions)^2)/nrow(test))
rmse_test