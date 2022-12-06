setwd("~")
setwd("/Users/skylarkolisko/Desktop:/DS_Capstone/musk_tweet_sentiment")
getwd()

sentis<- read.csv("FINAL_data_used/pysentis.csv", stringsAsFactors = FALSE) 
sentis_with_ln<- read.csv("FINAL_data_used/pysentis_ln_retweets.csv", stringsAsFactors = FALSE) 

summary(sentis)
head(sentis)

sentis$'small_retweet'<-sentis$'small_retweet'/10

# Make training set and testing set
set.seed(12) 
nrow(sentis)
indicesTrainingSet<-sample(1:nrow(sentis), 530, replace=FALSE)
train<-sentis[indicesTrainingSet,]
test<-sentis[-indicesTrainingSet,]

lm_model <-lm(retweet_score ~ joy+others+surprise+anger+fear+sadness+disgust, data = sentis)
summary(lm_model)

# "Others" totally dominates. R2 of 0.04
lm_model_ln <-lm(retweet_ln ~ joy+others+surprise+anger+fear+sadness+disgust, data = sentis_with_ln)
summary(lm_model_ln)

# So now without "others". Still R2 of 0.04
lm_model_ln <-lm(retweet_ln ~ joy+surprise+anger+fear+sadness+disgust, data = sentis_with_ln)
summary(lm_model_ln)