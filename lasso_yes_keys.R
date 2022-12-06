# Load glmnet library
library(glmnet)
library(ggplot2)

setwd("~")
setwd("/Users/skylarkolisko/Desktop:/DS_Capstone/musk_tweet_sentiment")
getwd()


# Load data 
sentis<- read.csv("FINAL_data_used/ALL_COMBINED_FINAL.csv", stringsAsFactors = FALSE) 
colnames(sentis)



sentis_x<- subset(sentis, select= -c(tweet_text, tweet_id, retweets, favorites, date, fav_to_follower_ratio, retweet_to_follower_ratio))


sentis_y <- subset(sentis, select= c(retweet_to_follower_ratio))
dtype(sentis_y)
print(sentis_y)



# set seed for reproducible results
set.seed(22)

# Non-cross validated model
glmnet_fit <- glmnet(x= as.matrix(sentis_x), y=as.matrix(sentis_y))
par(mar= c(5, 5, 6, 5))
plot(glmnet_fit, label=TRUE, main='Number of Coefficents at Lambda Progression > 0', cex.main=1, font.main=1)
title("LASSO Variable Traceable Plot", line=4.5)

# need to input family = 'gaussian' so that glmnet uses the gaussian linear model (least squares) as the model to apply LASSO to.
# COMMENTING OUT PART USED TO MAKE NNEW CV GLMNET SO DONT LOSE CURRENT ONE
# crossv_glmnet_fit <- cv.glmnet(x= as.matrix(sentis_x), y=as.matrix(sentis_y), family='gaussian', type.measure='mse',nfolds=5)
print(crossv_glmnet_fit)
# We can here see the mean square error of the model for each log(lambda) attempted.
plot(crossv_glmnet_fit)
title("MSE for k=5 Fold Cross Validation by Lambda Value")

# We will work with the lambda that gives the set of coefficents that gives minimum mean cross-validated error.
crossv_glmnet_fit$lambda.min

# Next, we can take the chosen-as-best lambda, in this case the lambda that minimizes the minimum cross-validated error, and get the newly regularized coefficents from that lambda's model into a dataframe.
# We will write a function to do this
make_coef_df <- function(glmnet_fit, lam){
	coefficents<- coef(glmnet_fit, lam)
	coef_df <- data.frame(name=coefficents@Dimnames[[1]][coefficents@i+1], coefficent = coefficents@x)
	return(coef_df[order(coef_df$coefficent, decreasing=T) , ])
}

coef_df <- make_coef_df(glmnet_fit= crossv_glmnet_fit, lam='lambda.1se')
coef_df
colnames(coef_df)
# Dropping intercept since its magnitude is so large. Want to highlight the differeces between the eotion's coefficents.
coef_df_no_intercept <- coef_df[-c(11) , ]
coef_df_no_intercept

# Now to make a plot to show coefficents
ggplot(data = coef_df_no_intercept) +
geom_col(aes(x=name, y=coefficent, fill={coefficent>0})) +
xlab(label = "Emotion")+
# the "paste" is used to get the lambda symbol
ggtitle(expression(paste("Lasso Coefficents with ",lambda, " = 0.03971")))+
theme(axis.text.x = element_text(angle=60, hjust = 1), legend.position = 'none', plot.margin=unit(c(0.5,1,0.4,1), 'cm'), plot.title = element_text(size = 25), axis.text=element_text(size=10), axis.title=element_text(size=15))

# Get the "R^2 values" for the cv glmnet model. Quotes because it really is the deviance explained by the model
devianceexplained <- crossv_glmnet_fit$glmnet.fit$dev.ratio[which(crossv_glmnet_fit$glmnet.fit$lambda == crossv_glmnet_fit$lambda.min)]


devianceexplained
