docs = ["France wins world cup in football",
        "Manchester United beats (someone) in France",
        "Bill Gates invests in Bitcoin.",
        "Coventry University ranked 24th in ....",
        "Student at Coventry University achieves ...."]

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

sw = stopwords.words('english')
print(sw)

nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()
filtered_docs = []
for doc in docs:
    tokens = word_tokenize(doc)
    tmp = ""
    for w in tokens:
        if w not in sw:
            tmp += ps.stem(w) + " "
    filtered_docs.append(tmp)

print(filtered_docs)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(filtered_docs)
print(X.todense())
 
from sklearn.cluster import KMeans
K = 3 
model = KMeans(n_clusters=K)#, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("cluster no. of input documents, in the order they received:")
print(model.labels_)

Y = vectorizer.transform(["I like football"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Bitcoin is great."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["I study at Coventry University"])
prediction = model.predict(Y)
print(prediction)

test_doc = ["I like football London",
             "Bitcoin is great.",
             "I study at Coventry University"
           ]
filtered_test_docs = []
for doc in test_doc:
    tokens = word_tokenize(doc)
    tmp = ""
    for w in tokens:
        if w not in sw:
            tmp += ps.stem(w) + " "
    filtered_test_docs.append(tmp)

print(filtered_test_docs)

Y = vectorizer.transform([filtered_test_docs[0]])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform([filtered_test_docs[1]])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform([filtered_test_docs[2]])
prediction = model.predict(Y)
print(prediction)
