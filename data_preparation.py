# Part 1
# Importing libreries
import pandas as pd
import numpy as np
import re

# Constant Variable
path = "/home/gourav/Desktop/Data Science/portfolio_projects/churn_prediction/Telco_Customer_Churn.csv" # relative path

# Reading csv
df = pd.read_csv(path)
#print(df)

print("Total null values : ",df.isnull().any().sum())
df.info()

# Data preparation 
df.nunique()


#Part 2
#Data Manipulation

#Replacing spaces with null values in total charges column
df['TotalCharges'] = df["TotalCharges"].replace(" ",np.nan)

#Dropping null values from total charges column which contain .15% missing data 
df = df[df["TotalCharges"].notnull()]
df = df.reset_index()[df.columns]

#convert to float type
df["TotalCharges"] = df["TotalCharges"].astype(float)


#replace 'No internet service' to No for the following columns
replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']

for i in replace_cols : 
    df[i]  = df[i].replace({'No internet service' : 'No'})

#replace 'No phone service' to No for the following columns
df['MultipleLines']  = df['MultipleLines'].replace({'No phone service' : 'No'})    

# dealing with categorical data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# cat_list = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
#        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
#        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
#        'Churn']
 
# df[cat_list] = df[cat_list].apply(LabelEncoder().fit_transform)

# # Dropping columns 
# df_2 = df.copy()
# df_2 =  df_2.drop(columns=['customerID','PaymentMethod'],axis=1)

#Part 3
#changing columns name and types
def desire_cols(df):        
    for col in df.columns:
            cols = col[0].lower() + col[1:] 
            # print(cols)
            df  = df.rename(columns={col:cols})
            df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    return df
        
    
df_new = desire_cols(df)

#cat_cols   = df_new.nunique()[df_new.nunique() < 6].keys().tolist()

df_new.to_csv('/home/gourav/Desktop/Data Science/portfolio_projects/churn_prediction/'+'clean_df.csv',index=False)
