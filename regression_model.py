import sys
import torch
import csv
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import re
from collections import Counter

nlp = English()
tokenizer = nlp.tokenizer

EPOCHS = 40
BATCH_SIZE = 30
LR = 5e-2

device = "cuda" if torch.cuda.is_available() else "cpu"

#####################################################

def add_feature(feats,data,fn,names,name):
	for i,f in enumerate(feats):
		print("data[i]")
		print(data[i])
		print("type data[i]")
		print(type(data[i]))

		f.append(fn(data[i]))
	names.append(name)

# Function to determine if lyric contains the word fuck
# Telling because Beyoncé uses the word the most, followed by TS
# takes in d, which will be in add_feature data[i] (as is for
# all feature functions)


def feat_fcn_they(d):
	return -1*avg_word(d, "they")

def feat_fcn_friends(d):
	return avg_word(d, "friends")

def feat_fcn_talk(d):
	return -1*avg_word(d, "talk")

def feat_fcn_gon(d):
	occurs = re.findall("gon'", str(d), re.I)
	text_toks = [token.orth_ for token in d]
	return len(occurs)

def feat_fcn_call(d):
	return avg_word(d, "call")

def feat_fcn_rep(d):
	print(d)
	print(avg_word(d, "reputation"))
	return -1*avg_word(d, "reputation")

# def feat_fcn_rep(d):
# 	return avg_words(d, [""])


def feat_fcn_chick(d):
	occurs = re.findall("chick", str(d), re.I)
	text_toks = [token.orth_ for token in d]
	return len(occurs)

def feat_fcn_money(d):
	occurs = re.findall("money", str(d), re.I)
	text_toks = [token.orth_ for token in d]
	return 10*len(occurs)

def feat_fcn_daddy(d):
	occurs = re.findall("daddy", str(d), re.I)
	# text_toks = [token.orth_ for token in d]
	return 5*len(occurs)



def feat_fcn_boy_comma(d):
	occurs = re.findall("boy,", str(d),re.I)
	return len(occurs)

def feat_fcn_repeated_words(lyric):
	repeats = re.findall(r'\b(\w+)\b(?=.*\b\1\b)', str(lyric), re.I)
	text_toks = [token.orth_ for token in lyric]
	return len(repeats)/len(text_toks)


def feat_fcn_pronoun_order(lyric):
	total = 0
	occurs = re.search("I.*you", str(lyric), re.I)
	text_toks = [token.orth_ for token in lyric]
	if occurs!= None:
		total+= -1/len(text_toks)
	occurs = re.search("you.*me", str(lyric), re.I)
	if occurs!= None:
		total+= -1/len(text_toks)
	return total

def feat_fcn_word_comma(lyric):
	occurs = re.search("^[a-zA-Z],",str(lyric))
	text_toks = [token.orth_ for token in lyric]
	if occurs!= None:
		return 1/len(text_toks)
	else:
		return 0/len(text_toks)

def feat_fcn_profane(d):
	f = re.findall("fuck", str(d), re.I)
	c = re.findall("cunt", str(d), re.I)
	b = re.findall("bitch", str(d), re.I)
	p = re.findall("pimp", str(d), re.I)
	h = re.findall("hoe", str(d), re.I)
	n = re.findall("nigga", str(d), re.I)
	text_toks = [token.orth_ for token in d]
	return 10*(len(f)+len(b)+len(c)+len(p)+len(n)+len(h))

def feat_fcn_bey_conjugations(d):
	occurs_ima = re.findall("I'ma", str(d), re.I)
	text_toks = [token.orth_ for token in d]
	occurs_yall = re.findall("ya'll", str(d), re.I)
	# occurs_imgon = re.findall("I'm gon'", str(d), re.I)
	# occurs_aint = re.findall("ain't'", str(d), re.I)
	return 10*(len(occurs_ima)+ len(occurs_yall))

# Word Count Helper Function- helper function that counts occurences of a word within lyric
def avg_word(d, word):
	text_toks = [token.orth_ for token in d] #changes tokens to strings
	occurences = text_toks.count(word)
	print(text_toks)
	print(occurences)
	return occurences/len(text_toks)

# Wordlist Helper Function
def avg_words(d, word_list):
	occurences = 0
	text_toks = [token.orth_ for token in d] #changes tokens to strings
	for word in word_list:
		occurences += text_toks.count(word)
	print(text_toks)
	print(occurences)
	return occurences/len(text_toks)

def avg_word_special_words(d, word):
	occurences = text_toks.count(word)
	print(occurences)
	return occurences



def make_features(data,words_to_add):
	feats = [[] for d in data]
	names = []
	for word in words_to_add:
		if word != "yaka-yaka":
			def fcn(data):
				return avg_word(data, word)
			add_feature(feats,data,fcn,names,str(word))
	add_feature(feats,data,len,names,'Length')

	# add_feature(feats,data,len,names,'Length') # with only length: Accuracy: 56.8%, Avg loss: 0.937050
	# add_feature(feats,data,feat_fcn_f,names,'F-word')
	# add_feature(feats,data,feat_fcn_b,names,'bitch')
	# add_feature(feats,data,feat_fcn_c,names,'c-word')
	# add_feature(feats,data,feat_fcn_chick,names,'chick')
	add_feature(feats,data,feat_fcn_money,names,'money')
	add_feature(feats,data,feat_fcn_daddy,names,'daddy')
	add_feature(feats,data,feat_fcn_they,names,'They')
	add_feature(feats,data,feat_fcn_boy_comma,names,'boy,')
	#add_feature(feats,data,feat_fcn_friends,names,'Friends')
	add_feature(feats,data,feat_fcn_friends,names,'Talk')
	# add_feature(feats,data,feat_fcn_call,names,'Call')
	add_feature(feats,data,feat_fcn_gon,names,'gon')
	add_feature(feats,data,feat_fcn_rep,names,'reputation')
	add_feature(feats,data,feat_fcn_repeated_words,names,'repeated_words')
	add_feature(feats,data,feat_fcn_pronoun_order,names,'pronoun_order')

	add_feature(feats,data,feat_fcn_bey_conjugations,names,'bey_conjugations')
	add_feature(feats,data,feat_fcn_profane,names,'profane')
	add_feature(feats,data,feat_fcn_word_comma,names,'word_comma')


	return [torch.tensor(f, dtype=torch.float32) for f in feats], names

#####################################################

def print_coefficients(model,train_features, feat_names,label_map):
	spacing = max([len(f) for f in feat_names])
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("Model weights (rows=features; columns=categories)")
	print("Feature: "+' '*(spacing-7)+'\t'.join(label_map.keys()))

	params =  list(model.parameters())
	weights = params[0]
	biases = params[1]

	#print weights
	for i,f in enumerate(feat_names):
		ws = [float(v[i]) for v in weights]
		print(f+': '+' '*(spacing-len(f))+'\t'.join(map(lambda x:str(round(x,3)),ws)))

	#print biases
	ws = [float(v) for v in list(biases)]
	print("Bias: "+' '*(spacing-4)+'\t'.join(map(lambda x:str(round(x,3)),ws)))

#####################################################
def lyric_counter_by_label(dataset_file,label_map, artist_num_wanted):
	lyric_counter_of_artist = Counter()
	with open(dataset_file,'r') as in_csv:
		reader = csv.reader(in_csv)
		for row in reader:
			# row_as_list = row.split(',')
			lyric = row[3].lower()
			lyric_as_list = lyric.split(' ')
			lyric_as_list = [l.replace(',', '') for l in lyric_as_list]
			lyric_as_list = [l.replace('.', '') for l in lyric_as_list]
			artist = row[0]
			artist_number = label_map[artist]
			if artist_number == artist_num_wanted:
				lyric_counter_of_artist.update(lyric_as_list)
	return lyric_counter_of_artist

def make_data(dataset_file,label_map):
	lyric_list = []
	label_list = []
	with open(dataset_file,'r') as in_csv:
		reader = csv.reader(in_csv)
		for row in reader:
			# row_as_list = row.split(',')
			lyric = row[3].lower()
			tokenized_l = tokenizer(lyric)
			print("type(tokenized_l)")
			print(type(tokenized_l))
			artist = row[0]
			artist_number = label_map[artist]
			lyric_list.append(tokenized_l)
			label_list.append(artist_number)
	return lyric_list, [torch.tensor(l, dtype=torch.long) for l in label_list]

# prints a summary of accuracy by category
def print_performance_by_class(labels, predictions):
	# INSERT CODE TO GET THE COMPARISON OF LABEL AND HIGHEST PROB FOR EACH OF THE TENSORS IN PRED (ie each row) [-0.5567, -1.1329, -2.2555] vs index in y)
	ind = 0
	# instances_of_x is the count o fht etotal number of songs by artist x
	# in the data set. Collected by considering the labels
	instances_of_0 =0
	instances_of_1 =0
	instances_of_2 =0

	# correct_x is the number of correctly predicted instances of songs by artist x
	correct_0 = 0
	correct_1 = 0
	correct_2 = 0
	any_predicted_nonzero = False
	ind = 0 # for position in labels vector to correspond with correct pred tensor
	for tens in predictions:
		print("tens")
		print(tens)
		 # tens is the individual tensor returned for a song title. It consists of 3 vals: 0th index
	# holds pred for TS, 1th index holds pred for Bey, 2nd holds pred for mitski
		highest_predicted_artist = torch.argmax(tens).item() # highest_pred is the INDEX of the highest value on the prediction tensor
		label = labels[ind].item()
		if label !=0:
			any_predicted_nonzero =True
		print('label')
		print(label)

		print('predicted artist')
		print(highest_predicted_artist)
		if label == 0:
			instances_of_0 +=1
			if highest_predicted_artist== label:
				correct_0+=1

		elif label == 1:
			instances_of_1 +=1
			if highest_predicted_artist== label:
				correct_1+=1
		else:
			instances_of_2 +=1
			if highest_predicted_artist== label:
				correct_2+=1
		ind+=1
	print("Accuracy by Category:")
	print("Category 0 : "+str(float(correct_0/instances_of_0)))
	print("Category 1 : "+str(float(correct_1/instances_of_1)))
	print("Category 2 : "+str(float(correct_2/instances_of_2)))
	print(any_predicted_nonzero)

class Regression(nn.Module):
    def __init__(self):
        super(Regression,self).__init__()
        self.layers = nn.Linear(28,3)
        # 3 is the number of artists, so output is a weight for each
        self.out= nn.LogSoftmax()

    def forward(self,x):
        print("x",x)
        scores = self.layers(x)
        probs = self.out(scores)
        print("probs", probs)
        return probs


def train(dataloader, model):
	loss_fn = nn.NLLLoss()
	optimizer=  torch.optim.SGD(model.parameters(), lr = LR)
	size = len(dataloader.dataset)
	model.train()
	for batch, (X,y) in enumerate(dataloader):
		current_xy = (X,y)
		cur_x = current_xy[0]
		cur_y = current_xy[1]
		# print("cur_x")
		# print(cur_x)
		# print("cur_y")
		# print(cur_y)
		X = cur_x.to(device)
		y = cur_y.to(device)
		print("Model input", X, y)
		# X=X.swapaxes(0,1) # added this swap because x was 24x2 instead of 2x24
		pred = model(X)
		print("\n")
		print("Model output", pred)
		loss = loss_fn(pred,y)
		print("Loss", loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

def test(dataloader, model):
	loss_fn = nn.NLLLoss()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for X,y in dataloader:
			X,y = X.to(device), y.to(device)
			pred = model(X)
			print('x at 181 is ')
			print(X)
			print('y is ')
			print(y)
			print('pred is ')
			print(pred)

			test_loss += loss_fn(pred,y).item()
			correct += (pred.argmax(1)==y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"Test error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

def predict(data,model):
	predictions = []
	labels=[]
	dataloader = DataLoader(data,batch_size=1)

	with torch.no_grad():
		for (X,y) in dataloader:
			print("X")
			print(X) #[tensor([[7., 0., 0., 0., 0., 0.]]), tensor([1])]... so really (X,y)
			X = X.to(device)
			pred = model(X)
			print("pred")
			print(pred)
			predictions.append(pred)
	return predictions


# CLASS CODE'S MAIN():
# def main():
#     with open("goodreads_titles.txt",'r') as of:
#         titles= of.readlines()
#     with open("goodreads_langs.txt", "r") as of:
#         langs = of.readlines()
#
#     print(titles[:3])
#     print(langs[:3])
#
#     features = make_features(titles)
#     labels, lang_map = make_labels(langs)
#     dataset = list(zip(features,labels))
#     train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE)
#     model = Regression().to(device)
#     for e in range(EPOCHS):
#         train(train_dataloader, model)
#         print("EPOCH",e)
#         test(train_dataloader, model)
#####################################################
def main():

	train_f = 'train_swift_bey_mitski.csv'
	test_f = 'test_swift_bey_mitski.csv'
	label_map = {"Taylor Swift":0,"Beyoncé":1,"Mitski":2,}
	train_lyrics, train_labels = make_data(train_f,label_map)
	test_lyrics, test_labels = make_data(test_f,label_map)

	print("type(train_lyrics)")
	print(type(train_lyrics))

	print(train_lyrics[0])
	print(train_labels[1].item())

	print("type(train_labels)")
	print(type(train_labels))

	ts_lyrics = lyric_counter_by_label(train_f,label_map, 0)
	# print('ts_lyrics')
	# print(ts_lyrics)
	# print("\n")
	# print("\n")
	# print("\n")

	bey_lyrics = lyric_counter_by_label(train_f,label_map, 1)
	# print("bey_lyrics")
	# print(bey_lyrics)
	# print("\n")
	# print("\n")
	# print("\n")

	mitski_lyrics = lyric_counter_by_label(train_f,label_map, 2)
	# print("mitski_lyrics")
	# print(mitski_lyrics)

	bey_words_not_in_ts = set(bey_lyrics.keys()).difference(set(ts_lyrics.keys()))
	more_than_seven_bey_words = [l for l in bey_words_not_in_ts if bey_lyrics[l]>15]
	print(bey_words_not_in_ts)

	mitski_words_not_in_ts = set(mitski_lyrics.keys()).difference(set(ts_lyrics.keys()))
	more_than_seven_mitski_words = [l for l in mitski_words_not_in_ts if mitski_lyrics[l]>15]
	print("\n")
	print("\n")
	print("\n")
	print(mitski_words_not_in_ts)

	more_than_seven_words_not_in_taylor = list(set(more_than_seven_bey_words)) + list(set(more_than_seven_mitski_words) - set(more_than_seven_bey_words))

	print(len(more_than_seven_words_not_in_taylor))

	# ts_words_not_in_bey = set(ts_lyrics.keys()).difference(set(bey_lyrics.keys()))
	# ts_words_not_in_mitski = set(set(ts_lyrics.keys()).difference(set(mitski_lyrics.keys()))
	#
	# more_than_seven_NONbey_words = [l for l in ts_words_not_in_bey if ts_lyrics[l]>15]
	# more_than_seven_NONmitski_words = [l for l in ts_words_not_in_mitski if ts_lyrics[l]>15]

	# more_than_ten_words_in_taylor = list(set(more_than_seven_NONbey_words)) + list(set(more_than_seven_NONmitski_words) - set(more_than_seven_NONbey_words))


	print("type(train_lyrics)")
	print(type(train_lyrics))

	print(train_lyrics[0])


	train_feats, train_names = make_features(train_lyrics, more_than_seven_words_not_in_taylor)
	test_feats, test_names = make_features(test_lyrics,more_than_seven_words_not_in_taylor)



	train_feats_dataset = list(zip(train_feats,train_labels))
	test_feats_dataset = list(zip(test_feats,test_labels))



	print(len(train_feats_dataset))

	train_dataloader = DataLoader(train_feats_dataset,batch_size = BATCH_SIZE)
	test_dataloader = DataLoader(test_feats_dataset,batch_size = BATCH_SIZE)

	model = Regression().to(device)
	for e in range(EPOCHS):
		train(train_dataloader, model)
		print("EPOCH",e)
		test(test_dataloader, model)
		predictions = predict(test_feats_dataset,model)
		print('\n')
		print("predictions")
		print(predictions)
		print_performance_by_class(test_labels, predictions)



	# Which features are most informative for each class?
	print_coefficients(model,train_feats,train_names,label_map)

main()
