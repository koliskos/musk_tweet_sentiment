import sys
import csv

# makes the dicts that will be passed into the qa model
def make_input_to_model(preprocessed_tweet_dicts,adjs_list):
    print(adjs_list)
    list_inputs_for_model = []
    for tweet_dict in preprocessed_tweet_dicts:
        for adj in adjs_list:
            new_dict = {}
            new_dict["tweet_text"] = tweet_dict['text']
            new_dict["tweet_id"] = tweet_dict['id']
            new_dict["retweets"] = tweet_dict["retweets"]
            new_dict["favorites"] = tweet_dict["favorites"]
            new_dict["date"] = tweet_dict["date"]
            new_dict["adj"] = adj
            new_dict["question"] = "How can this statement be described?"
            new_dict["answer_a"] = "{}".format(adj)
            new_dict["answer_b"] = "not {}".format(adj)
            list_inputs_for_model.append(new_dict)
    return list_inputs_for_model


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

def main():
    preprocessed_tweet_dicts = open_f(sys.argv[1]) #preprocessed_tweets.csv
    # adjectives_list contains 20 positive adjectives and 20 negative adjectives for describing a person
    adjs_list = ["creative","adventurous","generous","decisive","diplomatic", 'affectionate','hardworking','observant','helpful','optimistic','honest','funny','amusing','calm','polite','charismatic', 'romantic','friendly','clever','compassionate',"aggressive",'cynical','grumpy','nervous','defensive','arrogant','impulsive','patronizing','pessimistic','sullen','stubborn','egotistical','rude','mean','secretive','bossy','sarcastic','irresponsible','lazy','selfish']
    input_to_model = make_input_to_model(preprocessed_tweet_dicts,adjs_list)
    export(sys.argv[2], input_to_model)

main()
