import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle


# In[3]:


# reading the data
data=pd.read_csv('insuranceFraud.csv')


# In[4]:


# Having a look at the data
data.head()


# In[5]:


# In this dataset missing values have been denoted by '?'
# we are replacing ? with NaN for them to be imputed down the line.
data=data.replace('?',np.nan)


# In[6]:


# list of columns not necessary for pfrediction
cols_to_drop=['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year']


# In[7]:


# dropping the unnecessary columns
data.drop(columns=cols_to_drop,inplace=True)


# In[8]:


# checking the data after dropping the columns
data.head()


# In[9]:


# checking for missing values
data.isna().sum()


# In[10]:


# checking for th number of categorical and numerical columns
data.info()


# In[31]:


class CategoricalImputer:
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy

    def fit_transform(self, X):
        if self.strategy == 'most_frequent':
            self.most_frequent_values = X.mode().iloc[0]
        X_filled = X.fillna(self.most_frequent_values)
        return X_filled
#         else:
#             raise ValueError("Invalid strategy. Supported strategies: 'most_frequent'")


# In[32]:


# As the columns which have missing values, they are only categorical, we'll use the categorical imputer
# Importing the categorical imputer
# from sklearn_pandas import CategoricalImputer
imputer = CategoricalImputer()


# In[33]:


# imputing the missing values from the column

data['collision_type']=imputer.fit_transform(data['collision_type'])
data['property_damage']=imputer.fit_transform(data['property_damage'])
data['police_report_available']=imputer.fit_transform(data['police_report_available'])


# In[34]:


# Extracting the categorical columns
cat_df = data.select_dtypes(include=['object']).copy()


# In[35]:


cat_df.columns


# In[36]:


cat_df.head()


# Checking the categorical values present in the columns to decide for getDummies encode or custom mapping to convert categorical data to numeric one

# In[37]:


cat_df.columns


# In[38]:


cat_df['policy_csl'].unique()


# In[39]:


cat_df['insured_education_level'].unique()


# In[40]:


cat_df['incident_severity'].unique()


# In[41]:


#cat_df['property_damage'].unique()


# In[42]:


# custom mapping for encoding
cat_df['policy_csl'] = cat_df['policy_csl'].map({'100/300' : 1, '250/500' : 2.5 ,'500/1000':5})
cat_df['insured_education_level'] = cat_df['insured_education_level'].map({'JD' : 1, 'High School' : 2,'College':3,'Masters':4,'Associate':5,'MD':6,'PhD':7})
cat_df['incident_severity'] = cat_df['incident_severity'].map({'Trivial Damage' : 1, 'Minor Damage' : 2,'Major Damage':3,'Total Loss':4})
cat_df['insured_sex'] = cat_df['insured_sex'].map({'FEMALE' : 0, 'MALE' : 1})
cat_df['property_damage'] = cat_df['property_damage'].map({'NO' : 0, 'YES' : 1})
cat_df['police_report_available'] = cat_df['police_report_available'].map({'NO' : 0, 'YES' : 1})
cat_df['fraud_reported'] = cat_df['fraud_reported'].map({'N' : 0, 'Y' : 1})


# In[43]:


# auto encoding of categorical variables
for col in cat_df.drop(columns=['policy_csl','insured_education_level','incident_severity','insured_sex','property_damage','police_report_available','fraud_reported']).columns:
    cat_df= pd.get_dummies(cat_df, columns=[col], prefix = [col], drop_first=True)


# In[44]:


# data fter encoding
cat_df.head()


# In[45]:


# extracting the numerical columns
num_df = data.select_dtypes(include=['int64']).copy()


# In[46]:


num_df.columns


# In[47]:


num_df.head()


# In[48]:


# combining the Numerical and categorical dataframes to get the final dataset
final_df=pd.concat([num_df,cat_df], axis=1)


# In[49]:


final_df.head()


# In[50]:


# separating the feature and target columns
x=final_df.drop('fraud_reported',axis=1)
y=final_df['fraud_reported']



num_df.columns





x.columns


# In[61]:


x.drop(columns=['age','total_claim_amount'], inplace=True)


# In[62]:


# splitting the data for model training

# splitting the data into training and test set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y, random_state=355 )


# In[63]:


train_x.head()


# In[64]:


num_df=train_x[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]


# In[65]:


num_df.columns


# In[66]:


print(train_x.shape)
print(num_df.shape)


# In[67]:


# Scaling the numeric values in the dataset

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[68]:


scaled_data=scaler.fit_transform(num_df)
scaled_num_df= pd.DataFrame(data=scaled_data, columns=num_df.columns,index=train_x.index)
scaled_num_df.shape


# In[69]:


scaled_num_df.isna().sum()


# In[70]:


train_x.drop(columns=scaled_num_df.columns, inplace=True)


# In[71]:


train_x.shape


# In[72]:


train_x.head()


# In[73]:


train_x=pd.concat([scaled_num_df,train_x],axis=1)


# first using the Support vector classifier for model training
from sklearn.svm import SVC
sv_classifier=SVC()


# In[77]:


y_pred = sv_classifier.fit(train_x, train_y).predict(test_x)


# In[78]:


from sklearn.metrics import accuracy_score


# In[79]:


sc=accuracy_score(test_y,y_pred)



from sklearn.model_selection import GridSearchCV


# In[81]:


param_grid = {"kernel": ['rbf','sigmoid'],
             "C":[0.1,0.5,1.0],
             "random_state":[0,100,200,300]}


# In[82]:


grid = GridSearchCV(estimator=sv_classifier, param_grid=param_grid, cv=5,  verbose=3)


# In[83]:


grid.fit(train_x, train_y)


# In[86]:


grid.best_estimator_



xgb=XGBClassifier()


# In[97]:


y_pred = xgb.fit(train_x, train_y).predict(test_x)


# In[98]:


ac2=accuracy_score(test_y,y_pred)

param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 10, 1)}

            #Creating an object of the Grid Search class
grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5,  verbose=3,n_jobs=-1)

grid.fit(train_x, train_y)

grid.best_estimator_

with open('model.pkl', 'wb') as f:
            pickle.dump(grid, f)
model=pickle.load(open('model.pkl','rb'))


