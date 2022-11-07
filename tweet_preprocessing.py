import sys
import csv
import get_text_tweet

def open_f(in_csv):
    with open(in_csv, "r") as f:
        reader = csv.DictReader(f)
        dict_list = list(reader)
    return dict_list

def preproc_for_dates(dict_list):
    new_dict_list = []
    for dict in dict_list:
        date = dict['date'][0:10] #get just the date not the time
        dict['date'] = date
        new_dict_list.append(dict)
    return new_dict_list

# takes in a tweet dict and returns the full text
def preproc_for_truncated(truncated_tweet_dict):
    truncated_tweets_id = truncated_tweet_dict['id']
    full_tweet_text = get_text_tweet.get_text_using_id(truncated_tweets_id)
    return full_tweet_text

# gets full text for a truncated tweet, scrubs out identifying information
def preprocess(dict_list): # Stripping identifying information, not exactly for the sake of privacy, but instead so that the sentiment is based strictly on Elon's words and is not swayed by the usernames of others, since some usernames contain words with strong associations.
    print(len(dict_list))
    new_dict_list = []
    limit_reached = True
    counter = 0
    for tweet_dict in dict_list:
        counter+=1
        if counter>500: # to get the tweets which in the past run were skipped
        # due to the 500 tweet per call access limit
            limit_reached = False
        # if tweet_dict["text"] == "Falcon arching to orbit https://t.co/m7grug8FV9":
        #     limit_reached = True # since we have reached the limit of preprocessing
        split_text = tweet_dict["text"].split(" ") #splitting on white space
        if len(split_text[-1])>12 and split_text[-1][:12] == "https://t.co":
            if limit_reached != True:
                print('here')
                from_twitter_api = preproc_for_truncated(tweet_dict)
                print("from_twitter_api")
                print(from_twitter_api)
                split_text = from_twitter_api.split(" ")
            else:
                pass


#scrubbing out identifying information. CREDIT TO ROBERTA HUGGINGFACE PAGE
        new_tweet = []
        for word in split_text:
            word = '@user' if word.startswith('@') and len(word) > 1 else word #doesnt want to run on real twitter handles so masking htose
            word = 'http' if word.startswith('http') else word
            new_tweet.append(word)
        scrubbed_from_api = " ".join(new_tweet)
        tweet_dict["text"] = scrubbed_from_api
        new_dict_list.append(tweet_dict)
    return dict_list

def export(output_name,dict_list):
    with open(output_name, 'w') as out_csvfile:
        print("Exporting data")
        header_names = dict_list[0].keys()
        writer = csv.DictWriter(out_csvfile, fieldnames= header_names)
        writer.writeheader()
        writer.writerows(dict_list)

def main(): # python3 tweet_preprocessing.py original_data/elon_musk_tweets.csv after_500_full_tweets.csv
    print("opening")
    og_dict_list = open_f(sys.argv[1])
    good_dates_dict_list = preproc_for_dates(og_dict_list)
    preprocessed_dict_list =  preprocess(good_dates_dict_list)
    export(sys.argv[2], preprocessed_dict_list)
main()
