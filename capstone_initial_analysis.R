
print(getwd())
setwd("Desktop:/DS_Capstone")
stock_prices<- read.csv('TSLAprice.csv')


print(head(stock_prices))

stock_prices$change<- stock_prices$Open - stock_prices$Close
stock_prices$percent_change<- (stock_prices$Open - stock_prices$Close)/stock_prices$Open

print(head(stock_prices))
print(head(stock_prices$percent_change))

print(max(stock_prices$percent_change))

greater_than_10 <- stock_prices[stock_prices$percent_change >= 0.1, ]
print(nrow(greater_than_10))
print(greater_than_10$Date)
# "2020-02-05" "2021-11-09" "2022-01-27" "2022-04-26"
# Days to include in graph: between "2021-11-09 and 2022-04-26"
# get row number of 2021-11-09
row_21.11.9 <- match("2021-11-09", stock_prices$Date)
print(row_21.11.9) #1045
# get row number of 2022-04-26
row_22.4.26 <- match("2022-04-26", stock_prices$Date)
print(row_22.4.26) #1160
# get rows for dates between "2021-11-09 minus a month and 2022-04-26 plus a month"
start<- row_21.11.9 - 30
end<- row_22.4.26+30
days_since_oct_10_21<- start:end
included_dates<- stock_prices[start:end ,]
print(head(included_dates))
included_dates_close<- included_dates $Close
print(included_dates_close)
included_dates_dates <- included_dates$Date
print(included_dates_dates)
plot(days_since_oct_10_21, included_dates_close, type="l")

