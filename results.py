# import
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sn

# init list and variables for the computations
labels_fb = ['open facebook', 'user profile selection', 'post button selection', 'send message selection',
             'menu message selection', 'status post selection', 'status selection', 'other']
labels_tw = ['open Twitter', 'tweets selection', 'direct messages selection', 'send selection',
             'contact selection', 'back to home', 'writing tweet', 'other']
labels_gmail = ['open gmail', 'sending mail', 'reply selection', 'sending mail reply', 'chats selection',
                'delete selection', 'inbox selection', 'other']
labels_tumblr = ['open tumblr', 'refresh home', 'search page', 'user page', 'user likes', 'following page',
                 'new post', 'other']

labels_fb_conf = ['open Facebook', 'user profile\n selection', 'post button\n selection', 'send message\n selection',
                  'menu message\n selection', 'status post\n selection', 'status\n selection', 'other']
labels_tw_conf = ['open Twitter', 'tweets selection', 'DM selection', 'send\n selection',
                  'contact\n selection', 'back to home', 'writing\n tweet', 'other']
labels_gmail_conf = ['open Gmail', 'sending\n mail', 'reply\n selection', 'sending mail\n reply', 'chats\n selection',
                     'delete\n selection', 'inbox\n selection', 'other']
labels_tumblr_conf = ['open Tumblr', 'refresh home', 'search page', 'user page', 'user likes', 'following\n page',
                      'new post', 'other']


# read data from txt
def convert_to_int(sequence):
    sequence = sequence.split(',')
    seq_arr = []
    for char in sequence:
        seq_arr.append(int(char))
    return seq_arr


f = open('target_facebook.txt', 'r')
target = []
for line in f:
    target.append(line[:-2])
f.close()

e = open('fv_facebook.txt', 'r')
fv = []
for line in e:
    line = line[1:-3]
    line = convert_to_int(line)
    fv.append(line)
e.close()

# different labels for each app
label = labels_fb
labels_conf = labels_fb_conf


# RANDOM FOREST
print('random forest')
# split dataset in test and train
x_train, x_test, y_train, y_test = train_test_split(fv, target, test_size=0.3)

# rf plot
estimators_arr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
score_arr = []
for x in range(len(estimators_arr)):
    rf = RandomForestClassifier(n_estimators=estimators_arr[x])
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    score = rf.score(x_test, y_test)
    score_arr.append(score)

# actual plotting
fig = plt.figure()
ax = plt.axes()
ax.plot(estimators_arr, score_arr, '-ok', color='c')
plt.xticks(estimators_arr, estimators_arr)
plt.ylabel('Average precision score', fontsize=12)
plt.xlabel('Estimator number', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# normalized confusion matrix
rf = RandomForestClassifier(n_estimators=30)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred, labels=label)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# classification metrics
precision_fb = precision_score(y_test, y_pred, labels=label, average=None)
recall_fb = recall_score(y_test, y_pred, labels=label, average=None)
precision_score_fb = f1_score(y_test, y_pred, labels=label, average=None)
print('metrics')
print(precision_fb)
print(recall_fb)
print(precision_score_fb)

# confusion matrix plot
fig, ax = plt.subplots(figsize=(15, 15))
sn.heatmap(cm, annot=True, cbar=False, square=True, center=0, ax=ax)
ax.set_xticklabels(labels_conf, rotation='vertical')
ax.tick_params(axis='both', which='major', labelsize=9)
ax.set_yticklabels(labels_conf, rotation='horizontal')
plt.ylabel('Real actions     Predicted actions', fontsize=18)
plt.show()


# DNN
print('cnn')
# split dataset in test and train
x_train, x_test, y_train, y_test = train_test_split(fv, target, test_size=0.3)

# training model
activation_arr = ['identity', 'logistic', 'tanh', 'relu']
for x in range(len(activation_arr)):
    nn = MLPClassifier(activation=activation_arr[x], max_iter=400)
    nn.fit(x_train, y_train)
    y_pred = nn.predict(x_test)
    print(nn.score(x_test, y_test))

# confusion matrix
nn = MLPClassifier(activation='relu', max_iter=400)
nn.fit(x_train, y_train)
y_pred = nn.predict(x_test)
cm = confusion_matrix(y_test, y_pred, labels=label)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# classification metrics
precision_fb = precision_score(y_test, y_pred, labels=label, average=None)
recall_fb = recall_score(y_test, y_pred, labels=label, average=None)
precision_score_fb = f1_score(y_test, y_pred, labels=label, average=None)
print('metrics')
print(precision_fb)
print(recall_fb)
print(precision_score_fb)

# confusion matrix plot
fig, ax = plt.subplots(figsize=(15, 15))
sn.heatmap(cm, annot=True, cbar=False, square=True, center=0, ax=ax)
ax.set_xticklabels(labels_conf, rotation='vertical')
ax.tick_params(axis='both', which='major', labelsize=9)
ax.set_yticklabels(labels_conf, rotation='horizontal')
plt.ylabel('Real actions     Predicted actions', fontsize=18)
plt.show()
