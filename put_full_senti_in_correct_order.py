import sys
import csv
import itertools
import pandas as pd
import numpy as np


def open_f(in_csv):
    with open(in_csv, "r") as f:
        reader = csv.DictReader(f)
        dict_list = list(reader)
    return dict_list

def export(output_name,dict_list):
    with open(output_name, 'w') as out_csvfile:
        print("Exporting data")
        header_names = dict_list[0].keys()
        writer = csv.DictWriter(out_csvfile, fieldnames= header_names)
        writer.writeheader()
        writer.writerows(dict_list)

def main(): # python3 put_full_senti_in_correct_order.py FINAL_data_used/FINAL_full_sentis.csv
    in_d_list = open_f(sys.argv[1])
    combined_dict_list = []
    i = 0
    while i < 25400:
        current_tweet = in_d_list[i]['tweet_text']
        print(current_tweet)
        # remake dict, excluding adj and sentiment
        new_dict = {'tweet_text':current_tweet, 'tweet_id':in_d_list[i]['tweet_id'],
        'retweets':in_d_list[i]['retweets'], 'favorites':in_d_list[i]['favorites'],
        'date':in_d_list[i]['date'],'fav_to_follower_ratio':in_d_list[i]['fav_to_follower_ratio'],
        'retweet_to_follower_ratio':in_d_list[i]['retweet_to_follower_ratio']}
        j = i
        for j in range(i+40):

            another_of_same_tweet = in_d_list[j]['tweet_text']

            current_yes_key = "prediction_"+in_d_list[j]["adj"] # make key prediction_adj
            # current_not_key = "prediction_not_"+in_d_list[j]["adj"] # make key prediction_not_adj

            new_dict[current_yes_key] = in_d_list[j]["Prediction_A"]
            # new_dict[current_not_key] =  in_d_list[j]["Prediction_B"]

            print(j)
            if j == i+39:
                print("ending with:")
                print(current_tweet)
                print("j = "+str(j))
                print(another_of_same_tweet)
                print("\n")
        combined_dict_list.append(new_dict)
        i = j+1
        print("i = "+str(i))

    export(sys.argv[2],combined_dict_list)

main()
