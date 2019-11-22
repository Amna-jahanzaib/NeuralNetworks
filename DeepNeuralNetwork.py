# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:26:54 2018

@author: user
"""


# coding: utf-8

# In[42]:

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from numpy import array
from numpy import argmax
from keras.utils import to_categorical

def Style_to_integer(name):
    style = {
#       "Sexy" : 9,
#       "Casual":2,
#       "Vintage":10,
#       "Brief": 1,
#       "Cute": 3,
#       "Bohemian":0,
#       "Novelty":6,
#       "Party":8,
#       "work":11,
#       "Fashion" :4,
       "Sexy" : 7,
       "Brief": 1,
       "Cute": 3,
       "Casual":2,
       "Vintage":8,
       "Bohemian":0,
       "Novelty":5,
#       "Flare":5,
       "Party":6,
       "work":9,
#       "OL":7,
       "Fashion" :4,
	 }
    
    return style.get(name)

def Price_to_integer(name):
    price={		
       "Low":2,
       "High":1,
       "Average":0,
       "Medium":3,
       "very-high":5,
       "Null":4,}
    
    return price.get(name)
      
def NeckLine_to_integer(name):
    neckline={

       "o-neck" :7,
       "boat-neck":4,
       "v-neck":14,
       "turndowncollor":13,
       "peterpan-collor":9,
       "Sweetheart":2,
       "Null":0,
       "Scoop":1,
       "bowneck":5,
       "halter":6,
       "ruffled":10,
       "slash-neck":11,
       "open":8,
       "sqare-collor":12,
       "backless":3,
       }
    return neckline.get(name)

def SleeveLength_to_integer(name):
    sleeveLength={
       "sleeveless":8,
       "full":4,
       "butterfly":1,
       "short":7,
       "threequarter":9,
       "halfsleeve":6,
       "cap-sleeves":3,
       "turndowncollor":11,
       "half":6,
       "Null":7,
       }
    return sleeveLength.get(name)

def waiseline_to_integer(name):
    waiseline={
       
       "empire":2,
       "natural":3,
       "Null":0,
       "princess":4,
       "dropped":1,
       }
    return waiseline.get(name)

def Material_to_integer(name):
    material={
       "null":0,
       "silk":19,
       "chiffonfabric":3,
       "polyster":16,
       "cotton":4,
       "milksilk":10,
       "linen":7,
       "rayon":17,
       "lycra":8,
       "cashmere":2,
       "microfiber":9,
       "nylon":14,
       "other":15,
       "mix":11,
       "acrylic":1,
       "spandex":20,
       "lace":6,
       "modal":12,
       "viscos":21,
       "knitting":5,
       "wool":22,
       "model":13,
       "shiffon":18,}
    return material.get(name)


def FabricType_to_integer(name):
    fabricType={     
       "chiffon":4,
       "null":1,
       "broadcloth":3,
       "jersey":8,
       "batik":2,
       "flannael":6,
       "worsted":22,
       "woolen":21,
       "poplin":11,
       "dobby":5,
       "flannel":7,
       "tulle":19,
       "sattin":12,
       "organza":10,
       "lace":9,
       "Corduroy":0,
       "wollen":20,
       "shiffon":13,
       "terry":14,}
    return fabricType.get(name)



def Decoration_to_integer(name):
    decoration={ 
       "ruffles":18,
       "null":0,
       "embroidary":8,
       "bow":4,
       "lace":11,
       "beading":3,
       "sashes":19,
       "hollowout":10,
       "pockets":18,
       "sequined":20,
       "applique":2,
       "button":5,
       "Tiered":1,
       "rivet":16,
       "pocket":15,
       "flowers":9,
       "pleat":14,
       "crystal":6,
       "ruched":17,
       "draped":7,
       "tassel":24,
       "plain":13,
       "none":12,       
       }
    
    return decoration.get(name)

def PatternType_to_integer(name):
    patternType={
       "animal":1,
       "print":11,
       "dot":3,
       "solid":12,
       "null":0,
       "patchwork":9,
       "striped":13,
       "geometric":5,
       "plaid":10,
       "leopard":7,
       "floral":4,
       "character":2,
       "splice":12,
       "leapord":6,
       "none":8,
}
    return patternType.get(name)

def Size_to_integer(name):
    size = {
       
       "M":1,
       "L":0,
       "XL":3,
       "free":4,
       "S":2,
	 }
    
    return size.get(name)
       

def Season_to_integer(name):
    season = {
       
       "Summer" :2,
       "Autumn" :0,
       "Spring" :1,
       "Winter" :3,
	 }
    
    return season.get(name)

aaa=np.zeros((1,136))
def Encoding_to_binary(style,price,size,season,neck,sleeve,waise,material,fabric,decor,pattern):
    for i in range(137):
        if i==style or i==10+price or i==16+size or i==21+season or i==25+neck or i==40+sleeve or i==52+waise or i==57+material or i==80+fabric or i==99+decor or i==121+pattern:
            aaa[0][i]=1
    return aaa

def encode_Rating(rating):
    
    for i in range(y.size):
        rating[i]=round(rating[i])
        
    return rating

# In[43]:

dataset  = pd.read_csv('E:\\7th Semester\\FYP\\Dresses_Attribute_Sales\\Attribute DataSet5.csv')


# In[44]:

#dataset


# In[45]:

dataset=dataset.drop('Dress_ID', axis = 1)


# In[46]:

dataset['Rating']=pd.Series.round(dataset['Rating'])

y=dataset['Rating']

y=to_categorical(y)



# In[47]:

X=dataset.drop('Rating', axis = 1)


# In[48]:

X


# In[49]:

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[50]:

labelencoder_X_1 = LabelEncoder()


# In[51]:

dataset['Style']=labelencoder_X_1.fit_transform(dataset['Style']);


# In[52]:

dataset['Style']


# In[53]:

dataset['Price']=labelencoder_X_1.fit_transform(dataset['Price'])


# In[54]:

dataset['Size']=labelencoder_X_1.fit_transform(dataset['Size'])
dataset['Season']=labelencoder_X_1.fit_transform(dataset['Season'])
dataset['NeckLine']=labelencoder_X_1.fit_transform(dataset['NeckLine'])
dataset['SleeveLength']=labelencoder_X_1.fit_transform(dataset['SleeveLength'])
dataset['waiseline']=labelencoder_X_1.fit_transform(dataset['waiseline'])
dataset['Material']=labelencoder_X_1.fit_transform(dataset['Material'])
dataset['Decoration']=labelencoder_X_1.fit_transform(dataset['Decoration'])
dataset['FabricType']=labelencoder_X_1.fit_transform(dataset['FabricType'])
dataset['Pattern Type']=labelencoder_X_1.fit_transform(dataset['Pattern Type'])


# In[55]:

dataset


# In[56]:



# define example
data = array(dataset)
#print(data)
# one hot encode

Style=to_categorical(dataset['Style']);

Price=to_categorical(dataset['Price']);

Size=to_categorical(dataset['Size']);

Season=to_categorical(dataset['Season']);

neckline=to_categorical(dataset['NeckLine']);

SleeveLength=to_categorical(dataset['SleeveLength']);

waiseline=to_categorical(dataset['waiseline']);

Material=to_categorical(dataset['Material']);


FabricType=to_categorical(dataset['FabricType']);

Decoration=to_categorical(dataset['Decoration']);

Pattern_Type=to_categorical(dataset['Pattern Type']);

encoded3=np.concatenate((Style,Price,Size,Season,neckline,SleeveLength,waiseline,Material,FabricType,Decoration,Pattern_Type), axis=1)


X=dataset.drop('Rating',axis=1)

X=encoded3

# In[ ]:
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, :]




# In[57]:

#Now creating Dummy variables




# In[70]:

#print(X)
#X=dataset.drop('Recommendation', axis = 1)
#X=X.drop('Rating', axis = 1)

# In[59]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[60]:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[65]:

import keras
from keras.models import Sequential


# In[68]:

from keras.layers import Dense


# In[69]:

#Initializing Neural Network
classifier = Sequential()


# In[75]:

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 69, init = 'uniform', activation = 'relu', input_dim = 136))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 69, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'linear'))


# In[76]:

# Compiling Neural Network
classifier.compile(loss='mse', optimizer='adam')

# In[80]:
# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#classifier.evaluate(X_test, Y_test)



# In[ ]:

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.argmax(y_pred[5]))
a=np.zeros((76,1))
b=np.zeros((76,1))
for i in range(76):
    a[i][0]=(np.argmax(y_pred[i]))

    b[i][0]=(np.argmax(y_test[i]))

truepos=0
trueneg=0
c=0
for i in range(76):
    c=c+(a[i][0]-b[i][0])
    if(a[i][0]==b[i][0]):
        
        truepos=truepos+1
    else:
        
        trueneg=trueneg+1


Style='Casual';

Price='Average';

Size='M';

Season='Spring';

neckline='turndowncollor';

SleeveLength='short';

waiseline='empire';

Material='polyster';


FabricType='chiffon';

Decoration='sashes';

Pattern_Type='patchwork';

#Xnew=np.array(Encoding_to_binary(Style_to_integer('Party'),Price_to_integer('Low'),Size_to_integer('L'),Season_to_integer('Winter'),NeckLine_to_integer('v-neck'),SleeveLength_to_integer('sleeveless'),waiseline_to_integer('natural'),Material_to_integer('silk'),FabricType_to_integer('chiffon'),Decoration_to_integer('pleat'),PatternType_to_integer('solid'))
#)

Xnew=np.array(Encoding_to_binary(Style_to_integer(Style),Price_to_integer(Price),Size_to_integer(Size),
                                 Season_to_integer(Season),NeckLine_to_integer(neckline)
                                 ,SleeveLength_to_integer(SleeveLength),waiseline_to_integer(waiseline),
                                 Material_to_integer(Material),FabricType_to_integer(FabricType),
                                 Decoration_to_integer(Decoration),PatternType_to_integer(Pattern_Type))
)

#Xnew=np.array([[Style_to_integer('Cute'),Price_to_integer('Average'),Size_to_integer('M'),Season_to_integer('Summer'),NeckLine_to_integer('o-neck'),SleeveLength_to_integer('halfsleeve'),waiseline_to_integer('empire'),Material_to_integer('cotton'),FabricType_to_integer('chiffon'),Decoration_to_integer('ruffles'),PatternType_to_integer('animal')]])

# new instances where we do not know the answer
# new instance where we do not know the answer
# make a prediction
ynew = classifier.predict(Xnew)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (Xnew[0], np.argmax(ynew)))



baseline_errors = abs(b - a)

print('Average baseline error: ', round(np.mean(baseline_errors), 2))
mape = 100 * (baseline_errors/a)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(b, a)
#print(cm)


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = (0,1,2,3,4,5)
y_pos = np.arange(len(objects))
ind=np.argmax(ynew)
performance = [0,0,0,0,0,0]
performance[ind]=1
plt.bar(y_pos,performance , align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Frequency')
plt.title('Rating')
plt.savefig('rating.png')
plt.show()


