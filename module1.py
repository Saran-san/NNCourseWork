
tp = int(input("TP?"))
tn = int(input("TN?"))
fp = int(input("FP?"))
fn = int(input("FN?"))

precision = tp/(tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn)/(tp + tn + fp + fn)

print (precision, recall, accuracy)

fMetric = (2 * precision *recall)/(precision + recall)

print (fMetric)


beta = int(input("beta?"))
fMeasure = (((beta * beta) + 1) * precision *recall)/((beta * beta) * (precision + recall))

import feedparser
NewsFeed = feedparser.parse("https://ir.thomsonreuters.com/rss/news-releases.xml?items=15")
entry = NewsFeed.entries[1]

print(entry.keys())

for entry in NewsFeed.entries:
    print(entry.title)
    print(entry.link)
    print("--------------------------------------")
