
print(getwd()) 

setwd("DS_Capstone")
print(getwd()) 
trade_data <-read.csv("insider_trade_deltas.csv")



boxplot(trade_data$Delta_dir ~ trade_data$Transaction, ylim=c(0,1), horizontal=TRUE)


delta_by_transaction_anova<-aov(trade_data$Delta_dir ~ trade_data$Transaction)
summary(delta_by_transaction_anova)
