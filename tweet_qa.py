# # from Desktop:/proj232/232_final_project/task2/pt4_qa
# from get_text_tweet import * # CREDIT TO : https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Tweet-Lookup/get_tweets_with_bearer_token.py
## twitter API
import sys
import csv

# get_text_using_id(1546641991597006851) # from get_text_tweet

import itertools
import pandas as pd
import numpy as np

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaForQuestionAnswering, RobertaForMultipleChoice, RobertaForCausalLM
from transformers import RobertaConfig





def predict_and_record(input_data_df, counter):
    choice_dict = {
        "answer_a":0,
        "answer_b":1
    }

# pd df has the following cols:
# ["tweet_text", "tweet_id",
# 'retweets','favorites','date','adj','fav_to_follower_ratio',
# 'retweet_to_follower_ratio','question','answer_a', 'answer_b',
# 'Prediction_A', 'Prediction_B','Dif_Scores_A.B']

    context = input_data_df.iat[counter, 0]
    adj = input_data_df.iat[counter, 5]
    q = input_data_df.iat[counter, 8]
    opt1 = input_data_df.iat[counter,9]
    opt2 = input_data_df.iat[counter,10]
    options = [opt1, opt2]
    predictions= get_multiple_choice_answers(context,q,options)

    #record prediction for qa in row of qa for version A input
    input_data_df.iat[counter, 11] = predictions[0]#prediction for option a
    input_data_df.iat[counter, 12] = predictions[1]#prediction for option b

    #record difference between prediction for A and B
    input_data_df.iat[counter, 13] = predictions[0] - predictions[1]

def get_multiple_choice_answers(context,question,options):
    tokenizer = AutoTokenizer.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
    model = RobertaForMultipleChoice.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
    questions = []
    for ending_idx, ending in enumerate(options):
        if question.find("_") != -1:
            # fill-in-the-blank questions
            question_option = question.replace("_", ending)
        else:
            question_option = question + " " + ending
        inputs = tokenizer(context,question_option,add_special_tokens=True,padding="max_length",truncation=True,return_overflowing_tokens=False, return_tensors="pt")
        questions.append(question_option)
    encoding = tokenizer([context for o in options], questions, return_tensors="pt", padding=True)
    output = model(**{k: v.unsqueeze(0) for k, v in encoding.items()})
    logits = output.logits
    predictions = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()[0]
    predstr = ' '.join([f"| p(Answer {i})={predictions[i]:.4f}" for i in range(len(options))])
    # print("predstr")
    # print(predstr)
    # print("logits")
    # print(logits)
    # print("predictions")
    # print(predictions)
    return predictions

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

def main(): #python3 tweet_qa.py input_to_model.csv first_run_sentis.csv
    qa_file = open(sys.argv[1])
    input_data_df = pd.read_csv(qa_file, names=["tweet_text", "tweet_id",
    'retweets','favorites','date','adj','fav_to_follower_ratio',
    'retweet_to_follower_ratio','question','answer_a', 'answer_b',
    'Prediction_A', 'Prediction_B','Dif_Scores_A.B'])
    num_rows=len(input_data_df)
    # adjectives_list contains 20 positive adjectives and 20 negative adjectives for describing a person
    counter = 0
    num_rows=len(input_data_df)
    portion_to_run = sys.argv[2] # 1, 2, 3, 4 to run 1/4 of the data at a time
    #testing
    # while counter<182:
    if portion_to_run == "first": # tmux: musk1 , call: python3 tweet_qa.py UPDATED_input_to_qa_model.csv first first_sentis.csv
        while counter< num_rows/4:
            print("p&r again")
            predict_and_record(input_data_df, counter)
            counter+=1
    if portion_to_run == "second": # tmux: musk2 , call: python3 tweet_qa.py UPDATED_input_to_qa_model.csv second second_sentis.csv
        while counter< num_rows/2:
            if counter>= num_rows/4:
                print("p&r again")
                predict_and_record(input_data_df, counter)
            counter+=1
    if portion_to_run == "third": # tmux: musk3 , call: python3 tweet_qa.py UPDATED_input_to_qa_model.csv third third_sentis.csv
        while counter< 3*num_rows/4:
            if counter>= num_rows/2:
                print("p&r again")
                predict_and_record(input_data_df, counter)
            counter+=1
    if portion_to_run == "fourth": # tmux: musk4 , call: python3 tweet_qa.py UPDATED_input_to_qa_model.csv fourth fourth_sentis.csv
        while counter< num_rows:
            if counter>= 3*num_rows/4:
                print("p&r again")
                predict_and_record(input_data_df, counter)
            counter+=1
    # export
    input_data_df.to_csv(sys.argv[3])

# tmux session is musk. output to full_run_sentis.csv
main() # python3 tweet_qa.py UPDATED_input_to_qa_model.csv full_run_sentis.csv

# run for

#
#
# adjectives_list =[]
#
# # itertools.combinations(adjectives_list, 4)
# # Will run the q & a for each of the adjectives for tweet x.
# # Each run will have andswer A be "This tweet is {}.format(current_adjective)"
# # and "This tweet is not {}.format(current_adjective)".
#
# # ie tweetx <- df[df$tweet==x],
# # tweetxA <- dplyr::filter(tweetx, grepl('Funny', optionA))
# # grepl info: https://stackoverflow.com/questions/22850026/filter-rows-which-contain-a-certain-string
# # tweetxB <- dplyr::filter(tweetx, grepl('Not Funny', optionB))
#
#
# # funny_average_for_tweetx will want to be a log prob Im guessing
#
# # Then will liken the above calculated data to the percent change in tesla stock
# # for the next 24 hours proceeding the tweetx. Use the tweet_number column here!
#
# # We'd then have funny_prob_for_tweetx and delta_stock_by_tweet_x.
#
# # Funny will now be a var like "ftfem" ie a value between 0 and 1.
# # Then just copy the procedure for making the model in 260 for predicting fttrump.
#
# # ??????????????We'd then need funny_prob_for_tweetx and delta_stock_x for all x in Tweets.
# # ?????????We would take av_delta_stock for sentiment_funny by adding together the
# # ??????funny_prob_for_tweetx for all x in Tweets and then divide by |x|. Same for all
# # ??????????delta_stock_x (so same value there for all tweets).
#
# # We now would have a delta_stock value associated with funny_prob?
#
# #sent 1 is the text of A, sent 2 is the text of B.
#
