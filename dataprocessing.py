# import
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import cityblock
from fastdtw import fastdtw

# init constant values and lists
maxlen = 12
clusters = 200
actions_set_fb = ['open facebook', 'user profile selection', 'post button selection', 'send message selection',
                  'menu message selection', 'status post selection', 'status selection']
actions_set_tw = ['open Twitter', 'tweets selection', 'direct messages selection', 'send selection',
                  'contact selection', 'back to home', 'writing tweet']
actions_set_gmail = ['open gmail', 'sending mail', 'reply selection', 'sending mail reply', 'chats selection',
                     'delete selection', 'inbox selection']
actions_set_tumblr = ['open tumblr', 'refresh home', 'search page', 'user page', 'user likes', 'following page',
                      'new post']


# DATASET PREPROCESSING
# processing of the flows data, in order to be further edited later

def intsequence(sequence):
    # limit and convert time sequences to *maxlen* elements and fill 0s the ones shorter
    sequence = sequence[1:-1].split(',')
    seq_arr = []
    for char in sequence:
        if len(seq_arr) < maxlen:
            seq_arr.append(int(char))
        else:
            break
    while len(seq_arr) < maxlen:
        seq_arr.append(0)
    return seq_arr


def dataimporting(nameapp):
    # import the dataset: take only data about one single app and the packet length column
    init_dataset = pd.read_csv("apps_total_plus_filtered.csv", encoding='utf-8')
    dataset = init_dataset.loc[init_dataset['app'] == nameapp]['packets_length_total']
    # create a np array of np arrays of integers, which contains the dataset
    ds = []
    for row in dataset:
        ds.append(intsequence(row))

    print('flow dataset done')
    return np.array(ds)


# CLUSTERING
# find leaders using clustering based on dtw

def metr(f1, f2):
    # create the metric, easy implementation, without weights and different time sequences
    distance, path = fastdtw(f1, f2, dist=cityblock)
    return distance


def clustering(data, databound):
    # creation of linkage matrix
    linkagematr = hac.linkage(data[0:databound], method='average', metric=metr)
    print('linkage matrix done')

    # actual clustering
    cluster = hac.fcluster(linkagematr, t=clusters, criterion='maxclust')
    print('clustering done')

    # for each cluster take all the flows
    leaders = []
    for x in range(1, clusters + 1):
        temp = []
        max_sum = 0
        leader = []
        for k in range(len(cluster)):
            if cluster[k] == x:
                temp.append(data[k])
        # find the flow that minimize the obj function, that is the leader
        for element1 in temp:
            s = 0
            for element2 in temp:
                s = s + metr(element1, element2)
            if s >= max_sum:
                max_sum = s
                leader = element1
        leaders.append(leader)

    print('leaders done')
    return leaders


# CREATE ACTIONS DATASET
# create dataset with actions and time sequence

def actiondata(nameapp, actions_set_app):
    # retrive raw data and take actions and flows column
    init_dataset = pd.read_csv("apps_total_plus_filtered.csv", encoding='utf-8')
    fvdata = np.array(init_dataset.loc[init_dataset['app'] == nameapp].iloc[:, [1, 11]])

    # init fist values inside lists
    fvd = []
    flowset = []
    tempset = []
    flowset = [intsequence(fvdata[0][1])]
    tempset = [fvdata[0][0]]

    # create a list where in each entry there are an action and a list of flows
    for x in range(1, len(fvdata)):
        if fvdata[x][0] == fvdata[x - 1][0]:
            flowset.append(intsequence(fvdata[x][1]))
        else:
            tempset.append(flowset)
            fvd.append(tempset)
            flowset = []
            tempset = []
            flowset = [intsequence(fvdata[x][1])]
            tempset = [fvdata[x][0]]

    # all not important actions are labeled other
    for x in range(len(fvd)):
        if fvd[x][0] not in actions_set_app:
            fvd[x][0] = 'other'

    print('action dataset done')
    return fvd


# COMPUTING FEATURE VECTORS
# compute feature vectors

def findcluster(flow, leaders_list):
    # find correct cluster for each flow of an action
    mindist = pow(10, 10)
    cluster_index = -1
    for x in range(len(leaders_list)):
        distance = metr(flow, leaders_list[x])
        if distance <= mindist:
            mindist = distance
            cluster_index = x
    return cluster_index


def findsinglevector(action_list, leaders_list):
    # find the cluster for each flow inside belonging to an action
    featvec = [0] * clusters
    for flow in action_list:
        featvec[findcluster(flow, leaders_list)] = featvec[findcluster(flow, leaders_list)] + 1
    return featvec


def find_fvs_targets(dataset_actions, leaders_list, bound):
    # compute *bound* features vectors and targets
    featvectors = []
    targetvector = []
    for x in range(bound):
        featvec = findsinglevector(dataset_actions[x][1], leaders_list)
        print(x)
        featvectors.append(featvec)
    for k in range(bound):
        targetvector.append(dataset_actions[k][0])
    print('fv and targets done')
    return featvectors, targetvector


# ACTUAL DATA PROCESSING
# using previous function in order to get cluster leaders and then feature vectors
'''
leaders = clustering(dataimporting('twitter'), 1000)
actionds = actiondata('facebook', actions_set_fb)
fv, target = find_fvs_targets(actionds, leaders, 1000)

# writing data in txt files
out_fv = ''
for x in range(len(fv)):
    out_fv = out_fv + str(fv[x]) + ',' + '\n'

text_file = open("fv_facebook.txt", "w")
text_file.write(out_fv)
text_file.close()

out_target = ''
for x in range(len(target)):
    out_target = out_target + str(target[x]) + ',' + '\n'

text_file = open("target_facebook.txt", "w")
text_file.write(out_target)
text_file.close()
'''
