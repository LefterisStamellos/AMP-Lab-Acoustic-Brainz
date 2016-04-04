
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import math

import ipdb

import os

from sklearn import metrics,cross_validation

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFE,VarianceThreshold,SelectKBest,f_classif

from sklearn.svm import SVC,LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import scale

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

import pandas2arff


# In[3]:

#discardNan is given a dataframe column transposed into a row and its name, deletes all nans and returns the result
#as a row 
def discardNan(name,index):
    criterion = index.T[name].map(lambda x: str(x) != 'nan')
    index = index.T[criterion].T
    
    return index

#dictSeek is given a dataframe row and returns the dictionaries found in it in a dataframe row form 
def dictSeek(name,index):
    tmp = pd.DataFrame
    criterion = index.T[name].map(lambda x: type(x) == dict)
    tmp = index.T[criterion].T
    #mfcc and gfcc are a bit different from other lowlevel features contained in dictionaries,
    #we'll take care of them later
    if name == 'lowlevel':
        del tmp['gfcc']
        del tmp['mfcc']

    return tmp

#meanzVarzFeatz is given a dataframe containing only dictionaries and returns a list with values whose keys  are 'mean'
#and 'var', along with corresponding column label (feature name)
def meanzVarzFeatz(tmp):
    meanz=[y for y in [tmp[x][0]['mean'] for x in tmp.T.index] if type(y)!=list]
    varz=[y for y in [tmp[x][0]['var'] for x in tmp.T.index] if type(y)!=list]
    featz = [x for x in tmp.T.index if type(tmp[x][0]['mean'])!=list]
    
    return meanz,varz,featz

#from those features stored in dictionaries, keep only their mean and var values and discard the rest
def dictAggr(index,meanz,varz,featz):
    for i in range(len(meanz)):
        name = featz[i] + '_mean'
        index[name] = meanz[i]
    for i in range(len(varz)):
        name = featz[i] + '_var'
        index[name] = varz[i]
    for f in featz:
        del index[f]
        
    return index

#create three new single row DataFrames from the feature vectors' json file, each one containing descriptors of the 
#feature "family" corresponding to its name: lowlevel, tonal and rhythm (and no NaNs!!!) 
def featureExtract(data):
    
    data_df = pd.read_json(data)
    
    #(the initial dataframe produced by read_json is structured in such a way that every feature family will contain
    #all features, including those from other families. For example there will be an average_loudness index in tonal 
    #column. All of those elements will be nans. They're useless and they will be discarded).
    #Also, metadata is not usefull for our task so we cast it aside. 
    lowlevel = discardNan('lowlevel',data_df.iloc[:,:1].T)
    rhythm = discardNan('rhythm',data_df.iloc[:,2:3].T)
    tonal = discardNan('tonal',data_df.iloc[:,3:4].T)

    #tonal.chords_histogram and tonal.thpcp are in a list form which is inconvenient. Also, they are seen as irrelevant
    #for genre classification. They are discarded.
    criterion = tonal.T['tonal'].map(lambda x: type(x) == list)
    tmp = tonal.T[criterion].T
    for x in tmp:
        del tonal[x]
        
    #rhythm.beats_position is in a list form which is inconvenient. Also, it is seen as irrelevant
    #for genre classification. It is discarded.
    criterion = rhythm.T['rhythm'].map(lambda x: type(x) == list)
    tmp = rhythm.T[criterion].T
    for x in tmp:
        del rhythm[x]
    
    tmp = dictSeek('rhythm',rhythm)
    meanz,varz, featz = meanzVarzFeatz(tmp)
    rhythm = dictAggr(rhythm,meanz,varz,featz)

    tmp = dictSeek('lowlevel',lowlevel)
    meanz,varz,featz = meanzVarzFeatz(tmp)
    lowlevel = dictAggr(lowlevel,meanz,varz,featz)

    tmp = dictSeek('tonal',tonal)
    meanz,varz,featz = meanzVarzFeatz(tmp)
    tonal = dictAggr(tonal,meanz,varz,featz)

    #hpcp is in a very inconvenient form and is also seen as irrelevant for the genre classification task so
    #we discard it
    del tonal['hpcp']
    
    #mfcc and gfcc differ from other lowlevel features stored in dictionaries in that they do not containt a 'var'
    #key, but they do contain a 'mean' key, thus, they are treated differentlys
    for i in range(len(lowlevel['gfcc'][0]['mean'])):
        name = 'gfcc_mean_' + str(i)
        lowlevel[name] = lowlevel['gfcc'][0]['mean'][i]
    del lowlevel['gfcc']

    for i in range(len(lowlevel['mfcc'][0]['mean'])):
        name = 'mfcc_mean_' + str(i)
        lowlevel[name] = lowlevel['mfcc'][0]['mean'][i]
    del lowlevel['mfcc']

    #Now, there are more lowlevel features we need to take care of, namely barkbands,erbbands,melbands,
    #spectral_contrast_coeffs and spectral_contrast_valleys
    #Those have different bands, coefficients or valleys. Each of those means and vars will be added as a separate
    #unique feature. 
    low_dict = [x for x in lowlevel.T.index if type(lowlevel[x][0])==dict]

    for desc in low_dict:
        for i in range(len(lowlevel[desc][0]['mean'])):
            name = desc + '_mean_' + str(i)
            lowlevel[name] = lowlevel[desc][0]['mean'][i]
            name = desc + '_var_' + str(i)
            lowlevel[name] = lowlevel[desc][0]['var'][i]
        del lowlevel[desc] 

    #Again, in the case of rhythm features, band based ones will be "flattened".
    for i in range(len(rhythm['beats_loudness_band_ratio'][0]['mean'])):
        name = 'beats_loudness_band_ratio_mean_' + str(i)
        rhythm[name] = rhythm['beats_loudness_band_ratio'][0]['mean'][i]
        name = 'beats_loudness_band_ratio_var_' + str(i)
        rhythm[name] = rhythm['beats_loudness_band_ratio'][0]['var'][i]
    del rhythm['beats_loudness_band_ratio']
    
    return lowlevel,rhythm,tonal

#This function receives the FULL PATH of one json with the feature vectors (data) and another one (target)
#with the ground truth (genre mapping) of the classification task to follow (genre classification).
#It combines the two files in one pandas DataFrame. Also, some features which cover different bands and/or are 
#summarized statistically over frames are stored into dictionaries. These are "flattenned", i.e. each value of the 
#dictionary is treated as a separate, new descriptor in the occuring DataFrame. There are also some descriptors stored 
#in lists, these are mostly discarded.
#It is this DataFrame that we need for later tasks, but the final thing this function does is to write the DataFrame 
#into a csv file. The reason is that we want to avoid running the whole thing again, since it takes too long!!!
def twoJsonsToOneNeatCsv(data,target):
    #create from the feature vectors' json three DataFrames, each one containing descriptors of the feature "family"
    #corresponding to its name
    lowlevel,rhythm,tonal = featureExtract(data)

    #read genre mapping json into a DataFrame
    #for some reason, this needs to be loaded as a 'series' type
    target_sr = pd.read_json(target,typ='series')

    #turn that into a frame, it might be more useful this way
    target_df = target_sr.to_frame()

    #assign genre as the name to the column containing genres
    target_df.columns = ['genre']
    
    '''
    #For a general case, the code should be as seen below:
    df = pd.DataFrame()

    for i in range(len(target)):
        print i
        tmp = target.iloc[i:i+1,:1]
        data = pd.read_json(path + os.listdir(path)[i])
        lowlevel,rhythm,tonal = featureExtract(data)

        for feat in lowlevel.T.index:
            tmp[feat] = lowlevel[feat][0]
        for feat in rhythm.T.index:
            tmp[feat] = rhythm[feat][0]
        for feat in tonal.T.index:
            tmp[feat] = tonal[feat][0]
        df = pd.concat([df,tmp])
    
    #Unfortunately, in our case, there is a specific file, numbered 5287, which throws an error so we do this thing
    #in the following manner'''

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for i in range(5287):
        print i
        tmp = target_df.iloc[i:i+1,:1]
        lowlevel,rhythm,tonal = featureExtract(data)

        for feat in lowlevel.T.index:
            tmp[feat] = lowlevel[feat][0]
        for feat in rhythm.T.index:
            tmp[feat] = rhythm[feat][0]
        for feat in tonal.T.index:
            tmp[feat] = tonal[feat][0]
        df1 = pd.concat([df1,tmp])

    for i in range(5287,len(target)):
        print i
        tmp = target_df.iloc[i:i+1,:1]
        lowlevel,rhythm,tonal = featureExtract(data)

        for feat in lowlevel.T.index:
            tmp[feat] = lowlevel[feat][0]
        for feat in rhythm.T.index:
            tmp[feat] = rhythm[feat][0]
        for feat in tonal.T.index:
            tmp[feat] = tonal[feat][0]
        df2 = pd.concat([df2,tmp])

    df = pd.concat([df1,df2])

    df.to_csv(path_or_buf = 'AMP_Lab_fullSongAnalysis.csv')

#scikit learn functions generaly need to have floats as inputs, strings throw errors.
#This function turns any string values inside data and target matrices into numbers.
def mapToNumbers(sk_data,sk_target):
    #just the names of the genres
    genres = np.unique(sk_target)

    #mapping genres in sk_target to numbers
    tmp = np.array([])
    for g in sk_target:
        tmp = np.append(tmp,np.where(genres == g))  

    sk_target = tmp

    #indices of string elements, i.e. keys and modes
    keys_ind,modes_ind = [x for x in range(sk_data.shape[1]) if type(sk_data[:,x][0]) == str]

    #keys includes key names, needed for later on mapping them to numbers
    keys = np.unique(sk_data[:,keys_ind])

    #modes includes mode names, needed for later on mapping them to numbers
    modes = np.unique(sk_data[:,modes_ind])

    #mapping keys in corresponding columns of sk_data to numbers
    tmp = np.array([])
    for k in sk_data[:,keys_ind]:
        tmp = np.append(tmp,np.where(keys == k)) 
    sk_data[:,keys_ind] = tmp

    #mapping modes in corresponding columns of sk_data to numbers
    tmp = np.array([])
    for m in sk_data[:,modes_ind] :
        tmp = np.append(tmp,np.where(modes == m)) 
    sk_data[:,modes_ind] = tmp
    
    return sk_data,sk_target    

#discard values with low variance
def fitVarianceThreshold(X,df,thr):

    to_discard_ind = []
    
    thr = thr*(1-thr)
    #selector = VarianceThreshold(.8 * (1 - .8))
    selector = VarianceThreshold(thr)
    selector.fit_transform(X)
    to_discard_ind = [x for x in np.arange(X.shape[1]) if x not in selector.get_support(indices = True)]

    #add 2 to the indices to be dropped in order to account for the first two columns, filename and genre
    to_discard_ind = np.add(2,np.array(to_discard_ind))

    features = df.columns.values 

    to_drop = np.array([])
    for i in to_discard_ind:
        to_drop = np.append(to_drop,features[i])

    df = df.drop(to_drop,axis=1)
    
    return df

def dfToMatrices(df):
    #convert data dataframe to 2-d array to use in sklearn feature selection and classification algorithms,
    #discarding filename and genre columns 
    sk_data = df.iloc[:,2:df.shape[1]].as_matrix()

    #same for the target
    sk_target = df.iloc[:,1:2].as_matrix()

    #target genres in a single line array
    sk_target = np.ravel(sk_target)
    
    return sk_data,sk_target

#This function preprocesses the data, trains a classifier and returns the results of a 3 fold cross validation
def trainAndTestClassifier(df,thr = 0,k = 30,classifier = 'svm',scaled = 'False'):
    #preprocess data: first off fit variance threshold...
    sk_data,sk_target = dfToMatrices(df)
    sk_data,sk_target = mapToNumbers(sk_data,sk_target)
    df = fitVarianceThreshold(sk_data,df,thr)

    #...and then select k Best.
    #At this point, if scaled = 'True', you can also scale data to have zero mean and unit variance
    sk_data,sk_target = dfToMatrices(df)
    sk_data,sk_target = mapToNumbers(sk_data,sk_target)
    if scaled:
        sk_data_scaled = scale(sk_data)
        X,y = sk_data_scaled, sk_target
    else:
        X,y = sk_data, sk_target
    
    #select k Best features
    selectorKBest = SelectKBest(f_classif, k=k)
    X_new = selectorKBest.fit_transform(X, y)
    
    to_keep_ind = selectorKBest.get_support(indices = True)
    to_keep_ind = np.add(2,np.array(to_keep_ind))
    features = df.columns.values 
    to_keep = np.array([])
    for i in to_keep_ind:
        to_keep = np.append(to_keep,features[i])
    df_to_keep = pd.DataFrame
    df_to_keep = df[to_keep[0]]
    df_to_keep = df_to_keep.to_frame().join(df[to_keep[1]])
    for i in range(2,len(to_keep)):
        df_to_keep = df_to_keep.join(df[to_keep[i]])
    sk_selection = df_to_keep.as_matrix()
    
    #choose classification algorithm
    if classifier == 'svm':
        clf = LinearSVC(dual = False)
    elif classifier == 'tree':
        clf = DecisionTreeClassifier()
    elif classifier == 'kNN':
        clf = KNeighborsClassifier(n_neighbors = 5)
    else:
        raise Exception('Bad classifier type!!!')
    print 'done!'

    #evaluation with 3 fold cross vlidation
    scores = cross_validation.cross_val_score(clf, sk_selection, sk_target, cv=3)
    predict = cross_validation.cross_val_predict(clf, sk_selection, sk_target, cv=3)
    conf_mtx = confusion_matrix(sk_target, predict)
    
    return scores, predict, conf_mtx


# In[4]:

#data = 'Desktop/smc/AMP_lab/AMP_MusicBrainz/lfm-genre-ds-training/8b381c9c-4ed0-4021-bdb0-61e2f54557d8.json'
#target = 'Desktop/smc/AMP_lab/AMP_MusicBrainz/lastfm-genre-mapping-2016-03-02.json'


# In[5]:

#twoJsonsToOneNeatCsv(data,target)


# In[6]:

#read the above made file from csv, eitherwise making the df DataFrame takes about 1.5 hour
df = pd.read_csv('AMP_Lab_fullSongAnalysis.csv')

#df['chords_scale'] == df['key_scale'] is always true, so we can discard df['chords_scale'] as redundant
#df['chords_key'] == df['key_key'] is also always true, so we can discard df['chords_key'] as redundant
del df['chords_scale']
del df['chords_key']

#thr:  if thr(eshold) is set equal to 0, all features with constant value are discarded (those are in fact the ones that
#are always equal to 0)
#if, for example, a 0.8 threshold is given, then those features that have a probability of over 80% to of be equal value are
#discarded
#default: thr = 0

#k:  Choose number of best features to be chosen for classification 
#default: k = 30

#classifier:  Choose classifier. Available choices: 'kNN','svm','tree'
#default: classifier = 'svm'

#scaled:  Chose whether you want the features vectores to be scaled. This is supposed to be particularly
#useful for some kinds of svm, but not recommended here
#default: scaled = 'False'

scores, predict, conf_mtx = trainAndTestClassifier(df)


# In[ ]:




# In[ ]:



