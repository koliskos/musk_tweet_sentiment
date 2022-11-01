import csv
import pandas as pd
import numpy as np
import sys
import datetime

def open_f(in_csv):
    with open(in_csv, "r") as f:
        reader = csv.DictReader(f)
        list_dicts = list(reader)
    return list_dicts

# given the list of dicts of the rows from the stock price data, organize the
# data into a dictionary where the data, in format of yyyy-mm-dd, is the key and
# the directio of change of the stock price over the day is 0 for negative
# change and 1 for positive change.
def price_change_as_dict(prices_dict_list):
    deltas = {}
    for dict in prices_dict_list:
        # print()
        date = dict["Date"]
        openp = float(dict["Open"])
        closep = float(dict["Close"])
        delta = closep - openp # positive if close>open
        delta_direction = 1 if delta>0 else 0
        deltas[date] = [delta_direction,openp, closep] # save delta direction to deltas dictionary
    return deltas

# Add the stock price's delta to each insider trade's dictionary. Also handle weekend
# trades that are not in the og stock price data. Outputs a dict of dicts, where trade
# date is key and dict about the trade is the value
def add_delta_to_intrade(insider_trades, delta_by_date):
    insider_trade_by_dates={}
    for dict in insider_trades:
        intrade_date_str = dict["Date"]
        date_pieces = intrade_date_str.split("-")

        date_as_dt = datetime.datetime(int(date_pieces[0]),int(date_pieces[1]),int(date_pieces[2]))
        # check if insider trade was made on the weekend
        print(date_as_dt)
        open_dt = date_as_dt
        close_dt = date_as_dt
        if intrade_date_str not in delta_by_date:
            print("date not in delta_by_date is "+date_as_dt.strftime("%Y-%m-%d"))
            # find last close adjacent to the trade's date. Use that as OPEN for
            # this weekend trade date

            while open_dt.strftime("%Y-%m-%d") not in delta_by_date:
                open_dt -= datetime.timedelta(days=1)
                #print("deduction gives "+open_dt.strftime("%Y-%m-%d"))
            # find next open adjacent to the trade's date. Use that as CLOSE for
            # this weekend trade date
            while close_dt.strftime("%Y-%m-%d") not in delta_by_date:
                close_dt += datetime.timedelta(days=1)

            # get weekend delta
            #print('here')
            openp = delta_by_date[open_dt.strftime("%Y-%m-%d")][2] # get the CLOSE that started of weekend. Index 2 holds the close of that date
            closep = delta_by_date[close_dt.strftime("%Y-%m-%d")][1] # get the OPEN of the day after the weekend. Index 1 holds the open of that date
            delta = closep-openp
            dict["Delta"] = 1 if delta>0 else 0
        else: # if trade NOT made on weekend:
            delta = delta_by_date[intrade_date_str][0] # 0th index holds the delta direction val
            dict["Delta"] = delta

        dict["Open_Date"] =open_dt.strftime("%Y-%m-%d")
        dict["Close_Date"]= close_dt.strftime("%Y-%m-%d")
        dict["Trade_Date"] = dict["Date"] # rename to "Trade_Date"
        del dict["Date"]
        insider_trade_by_dates[dict["Trade_Date"]] = dict # add to dict  being output
    return insider_trade_by_dates

# add insider sale column to delta_by_date
def prices_and_trade_dates(insider_trades_with_dates, delta_by_date):
    output_list = []
    for date in delta_by_date.keys():
        if date in insider_trades_with_dates:
            if insider_trades_with_dates[date]["Transaction"] == 'Option Exercise':
                delta_by_date[date].append("opt_ex") # if fourth index in list held at
                #delta_by_date[date] is a 1, indicates option exercise occurred.
            elif insider_trades_with_dates[date]["Transaction"] == 'Sale':
                delta_by_date[date].append("sale") # if fourth index in list held at delta_by_date[date] is a 1, indicates option exercise occurred.
        else:
            delta_by_date[date].append("no_trade") # zero indicates no trade occurred on this date
        # convert dict entry into a dict
        current_dict = {"Date":date, "Delta_dir":delta_by_date[date][0], "Open_price":delta_by_date[date][1],"Close_price":delta_by_date[date][2], "Transaction":delta_by_date[date][3]}
        output_list.append(current_dict)
    return output_list

# convert insider_trades_with_dates list into a csv and output the csv
def export(output_name,insider_trades_with_dates):
    with open(output_name, 'w') as out_csvfile:
        print("Exporting data")

        header_names = ["Date","Delta_dir", "Open_price","Close_price", "Transaction"]
        writer = csv.DictWriter(out_csvfile, fieldnames= header_names)
        writer.writeheader()
        for dict in insider_trades_with_dates:
            writer.writerow(dict)

def main():
    insider_trades = open_f(sys.argv[1])
    print(insider_trades[0])
    prices = open_f(sys.argv[2])
    print(prices[0])
    delta_by_date = price_change_as_dict(prices)
    print(delta_by_date["2017-09-18"])
    insider_trades_with_dates = add_delta_to_intrade(insider_trades, delta_by_date)
    output_name = sys.argv[3]
    to_output = prices_and_trade_dates(insider_trades_with_dates, delta_by_date)
    export(output_name,to_output)
main()
