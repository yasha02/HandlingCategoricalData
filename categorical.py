#def scoreset function

from sklearn.ensemble import RandomForestGenerator
from sklearn.metrics import mean_absolute_error  
def score_set (x_train , x_val, y_train , y_val):
    model =RandomForestGenerator(n_estimators =100, random_state =0)
    model.fit(x_train,y_train)
    pred =model.predict(x_val)
    return mean_absolute_error(y_valid , preds)



import pandas as pd
from sklearn.mode_selection import train_test_split
#path to the data set file.
data =pd.read_csv('')

#seprate the dependent and independent variables 

y =data.Price
x = data.drop(['Price'],axis =1)

#dividing the data into training and testing data set
x_train_full , x_valid_full , y_train_full, y_valid_full = train_test_split(x,y,train_size =0.8,test_size =0.2,random_state =0)
#Drop columns with missing values

cols_with_missing_values =[col for col in x_train_full.columns if x_train_full[col].isnull.any()]
x_train_full.drop(cols_with_missing_values , axis =1 ,inplace=True)
x_valid_full.drop(cols_with_missing_values , axis =1 ,inplace=True)

low_cardinality_cols =[cname for cname in x_train_full.columns if x_train_full[cname].nunique() <10 and 
              x_train_full[cname].dtype =="object"]

 #select numerical columns

 numerical_cols =[cname for cname in x_train_full.columns if x_train_full[cname].dtype in ['int64','float64']]
 #keeping selected columns only
 new_cols =numerical_cols + low_cardinality_cols

 x_train = x_train_full[new_cols].copy()
 x_valid = x_valid_full[new_cols].copy()


 # let us now get the list of categorical variables

 s =(x_train.dtypes == 'object')
 object_cols =list(s[s].index)

 print("Categorical variables :")

#Approach 1 dropping the columns with categorical data

drop_x_train =x_train.select_dtypes(exclude =['object'])
drop_x_valid = x_train.select_dtypes(exclude =['object'])

score_dataset(drop_x_train, drop_x_valid, y_train, y_valid)

#Approach 2

from sklearn.preprocessing import LabelEncoder

label_X_train = x_train.copy()
label_X_valid = x_valid.copy()

label_encoder = LabelEncoder()

for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(x_train[col])
    label_X_valid[col] = label_encoder.transform(x_valid[col])

score_dataset(label_X_train ,label_X_valid ,y_train ,y_valid)

#Approach 3

from sklearn.preprocessing import OneHotEncoder

OH_encoder =OneHotEncoder(handle_unknown ='ignore' , sparse =False)

OH_cols_train =pd.DataFrame(OH_encoder.fit_transform(x_train[object_cols]))
OH_cols_valid =pd.DataFrame(OH_encoder.transform(x_valid[object_cols]))

OH_cols_train.index = x_train.index
OH_cols_valid.index = x_valid.index

num_X_train = x_train.drop(object_cols , axis =1)
num_X_valid = x_valid.drop(object_cols , axis=1)

OH_X_train =pd.concat([num_X_train,OH_cols_train],axis=1)
OH_X_valid =pd.concat ([num_X_valid , OH_cols_valid], axis =1)

score_dataset(OH_X_train,OH_X_valid,y_train,y_valid)


