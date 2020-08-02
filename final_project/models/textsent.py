import csv
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

truncated = pd.read_csv("/home/ec2-user/new_truncated.csv")

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

n = 0

trump_indices = []
biden_indices = []
read = True
with open('/home/ec2-user/trump.txt', 'r') as trump:
    for line in trump:
        stripped_line = line.strip()
        trump_indices.append(stripped_line)
with open('/home/ec2-user/biden.txt', 'r') as biden:
    for line in biden:
        stripped_line = line.strip()
        biden_indices.append(stripped_line)

def getText(index):
    #get the article text from the other spreadsheet (from the index)
    row = truncated.iloc[index,0]
    return(row)

def textsent(text):
    #get text sentiment of an article text, somehow
    return(sid.polarity_scores(text))

current_docname = None
last_sentiment = None
topic_weights = {}

trump_total_sentiments = {}
trump_topic_counts = {}

biden_total_sentiments = {}
biden_topic_counts = {}

#now that text sent model has been trained, iterate through news articles
with open('/home/ec2-user/doc-topics.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    first_row = next(reader)

    for row in reader:
        docname = (row[0].split(",")[0])

        if docname==current_docname:
            if read:
            #still on the same doc, just different topics
                try:
                    topic_weights[float(row[0].split(",")[1])] += float(row[0].split(",")[2])
                except KeyError:
                    topic_weights[float(row[0].split(",")[1])] = float(row[0].split(",")[2])
        else:
            #new doc!
            if n == 0:
                current_docname = docname
            #check if in trump or biden
            if current_docname.split(":")[-1] in trump_indices:
                for topic in topic_weights.keys():
                    try:
                        trump_topic_counts[topic] = trump_topic_counts[topic] + topic_weights[topic]
                    except KeyError:
                        trump_topic_counts[topic] = topic_weights[topic]
                    
                    pos_sentiment = topic_weights[topic] * last_sentiment['pos']
                    neg_sentiment = topic_weights[topic] * last_sentiment['neg']
                    neu_sentiment = topic_weights[topic] * last_sentiment['neu']
                    compound_sentiment = topic_weights[topic] * last_sentiment['compound']

                    try:
                        trump_total_sentiments[topic]['pos'] += pos_sentiment
                        trump_total_sentiments[topic]['neg'] += neg_sentiment
                        trump_total_sentiments[topic]['neu'] += neu_sentiment
                        trump_total_sentiments[topic]['compound'] += compound_sentiment

                    except KeyError:
                        trump_total_sentiments[topic] = {'pos':pos_sentiment, 'neg':neg_sentiment, 'neu':neu_sentiment, 'compound':compound_sentiment}


            elif current_docname.split(":")[-1] in biden_indices:
                for topic in topic_weights.keys():
                    try:
                        biden_topic_counts[topic] = biden_topic_counts[topic] + topic_weights[topic]
                    except KeyError:
                        biden_topic_counts[topic] = topic_weights[topic]
                    
                    pos_sentiment = topic_weights[topic] * last_sentiment['pos']
                    neg_sentiment = topic_weights[topic] * last_sentiment['neg']
                    neu_sentiment = topic_weights[topic] * last_sentiment['neu']
                    compound_sentiment = topic_weights[topic] * last_sentiment['compound']

                    try:
                        biden_total_sentiments[topic]['pos'] += pos_sentiment
                        biden_total_sentiments[topic]['neg'] += neg_sentiment
                        biden_total_sentiments[topic]['neu'] += neu_sentiment
                        biden_total_sentiments[topic]['compound'] += compound_sentiment

                    except KeyError:
                        biden_total_sentiments[topic] = {'pos':pos_sentiment, 'neg':neg_sentiment, 'neu':neu_sentiment, 'compound':compound_sentiment}


            #update larger dicts with sentiment and topic weights
            
            #reset variables
            topic_weights = {}
            current_docname = row[0].split(",")[0]
            
            #calculate sentiment of new thing
            text_index = str(row[0].split(":")[-1])[0]
            if ((text_index in trump_indices) or (text_index in biden_indices)):
                text = getText(int(text_index))
                last_sentiment = textsent(text)
                print(str(n) + ": " + str(last_sentiment))
                read = True
            else:
                last_sentiment = {'pos':0, 'neg':0, 'neu':0, 'compound':0}
                read = False
            n+=1

    #write total_sentiment divided by total_weight (topic_counts)
with open("/home/ec2-user/trump_output.txt","a") as output:
    string = "Topic,Pos,Neg,Neu,Compound\n"
    print(string)
    output.write(string)
    for key in trump_total_sentiments.keys():
        pos = trump_total_sentiments[key]['pos']/trump_topic_counts[key]
        neg = trump_total_sentiments[key]['neg']/trump_topic_counts[key]
        neu = trump_total_sentiments[key]['neu']/trump_topic_counts[key]
        compound = trump_total_sentiments[key]['compound']/trump_topic_counts[key]
        string = str(int(key)) + "," + str(pos) + "," + str(neg) + "," + str(neu) + "," + str(compound) + "\n"
        print(string)
        output.write(string)

with open("/home/ec2-user/biden_output.txt","a") as output:
    string = "Topic,Pos,Neg,Neu,Compound\n"
    print(string)
    output.write(string)
    for key in biden_total_sentiments.keys():
        pos = biden_total_sentiments[key]['pos']/biden_topic_counts[key]
        neg = biden_total_sentiments[key]['neg']/biden_topic_counts[key]
        neu = biden_total_sentiments[key]['neu']/biden_topic_counts[key]
        compound = biden_total_sentiments[key]['compound']/biden_topic_counts[key]
        string = str(int(key)) + "," + str(pos) + "," + str(neg) + "," + str(neu) + "," + str(compound) + "\n"
        print(string)
        output.write(string)
