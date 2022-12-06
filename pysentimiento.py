import pysentimiento
import sys
import csv
"""
Emotion Analysis in English
"""


# emotion_analyzer = pysentimiento.create_analyzer(task="emotion", lang="en")
#
#
# with open('input_to_model.csv', "r") as f:
#     reader = csv.DictReader(f)
#     dict_list = list(reader)
# tweets=[]
# prediction_dicts = []
# for d in dict_list:
#     tweet = d['text']
#     tweets.append(tweet)
#
# for tweet in tweets:
#     preds = emotion_analyzer.predict(tweet)
#     print(prepreds["probas"])
#     prediction_dicts.append(prepreds["probas"])
#
# with open('pysentis.csv', 'w') as csvfile:
#     fieldnames = ['joy', 'others', 'surprise', 'anger', 'fear', 'sadness', 'disgust', 'surprise']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(prediction_dicts)

# returns AnalyzerOutput(output=joy, probas={joy: 0.723, others: 0.198, surprise: 0.038, disgust: 0.011, sadness: 0.011, fear: 0.010, anger: 0.009})
