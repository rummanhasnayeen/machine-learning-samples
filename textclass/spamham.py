import pandas as pd
import numpy as np


filepath = '/home/rumman/Downloads/data sets/spam_ham_dataset.csv'
df = pd.read_csv(filepath, header=None, names=['dummy', 'label', 'text', 'label_num'], sep=',')

#print(df.head())
df = df.iloc[1:]
df = df.drop('dummy', 1)
# print(df.head())
# df.text = df.text.astype(str)

#df['text'] = df['text'].astype('|S')
# a = (df['text'].tolist())
# b=[]
# for text in a:
#     texts=text.split(":")
#     m_str=""
#     for i in range(len(texts)):
#         if i==0:
#             continue
#         elif i==1:
#             m_str=m_str+texts[i]
#         else:
#             m_str=m_str+":"+texts[i]
#     b.append(m_str)
#
# print(b[0])
#
# df2 = pd.DataFrame({'text': b})
#
# df2['label_num'] = df['label_num']
# df2= df2.iloc[1:]
# print(df2.head())
#
# df=df2
#
# s = df.label_num

# op = df.to_numeric(s)
# print(df.label_num.dtype)

# print(df.label_num.shape)
# print(df.head())

# np.concatenate(df.label_num).astype(int)


# print(df.text.dtype)


from sklearn.model_selection import train_test_split

df_yelp = df
sentences = df_yelp['text'].values
y = df_yelp['label_num'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
# print(X_train)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score)


from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model_sum = model.summary()
print(model_sum)

history = model.fit(X_train, y_train, epochs=100, verbose=False, validation_data=(X_test, y_test), batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

