#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:54:44 2020

@author: Geoffrey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report

def fill_mort_acc(total_acc,mort_acc):
    '''
    If mort_acc is NaN, take the mean of its corresponding total_acc value
    else, keep the value
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc
    
def ann_model (X,y):    
    model = Sequential()
    model.add(Dense(units=78,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=39,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    model.fit(x=X_train, y=y_train,epochs=600, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stop])
    return model    
    
        
if __name__ == "__main__":
    # Read the CSV file
    loan_data = pd.read_csv("/Users/Geoffrey/Desktop/PythonDSandML/Machine Learning/Ch12_DeepLearning/lending_club_loan.csv")
    
    loan_data.info() 
    print(loan_data.head())

    ''' 
    Exploratory Data Visualization
    Create plots to better understand the data at hand
    '''
    
    # Pair-wise correlations 
    plt.figure(figsize=(10,6))
    ax = sns.heatmap(loan_data.corr(), annot=True, cmap="YlGnBu")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    # Variable of interest
    plt.figure()
    sns.countplot(loan_data['loan_status'])
    # Distribution of the loan amounts
    plt.figure()
    sns.distplot(loan_data['loan_amnt'],kde=False,bins=25)
    # Relationship between two pairs of variables
    plt.figure()
    sns.countplot(x='home_ownership', hue='loan_status', data=loan_data)
    # Loan amount and loan status
    plt.figure()
    sns.boxplot(x='loan_status', y='loan_amnt', data=loan_data)
    loan_data.groupby('loan_status').describe()['loan_amnt']
    # Correlation between variable of interest and other numeric variables
    bin_status = {'Fully Paid':1, 'Charged Off':0}
    loan_data['dummy_status'] = loan_data['loan_status'].apply(lambda item: bin_status[item])
    plt.figure()
    loan_data.corr()['dummy_status'][:-1].sort_values().plot(kind='bar')


    ''' 
    Feature engineering and data pre-processing
    '''
    # Check for null variables with values in the dataset
    loan_data.isnull().sum()
    
    '''
    Deal with variables containing null instances
    '''
    loan_data["emp_title"].value_counts()
    # too many diff titles with low counts so it wouldn't bring anything to the model
    loan_data.drop("emp_title",axis=1, inplace=True)
    
    # let's compare emp_length to the output value
    plt.figure(figsize=(8,5))
    sns.countplot(x="emp_length", hue="loan_status", data=loan_data)
    emplength_df = pd.get_dummies(loan_data[["loan_status","emp_length"]], prefix=['loan_status'], columns=['loan_status']).groupby("emp_length").sum()
    emplength_df['perc'] = emplength_df["loan_status_Charged Off"]/(emplength_df["loan_status_Charged Off"]+emplength_df["loan_status_Fully Paid"])
    emplength_df 
    # emp_length doesn't influence whether the person pays their loan or not, so let's remove it 
    loan_data.drop("emp_length",axis=1, inplace=True)

    # title is dependant on the variable purpose
    loan_data.drop("title",axis=1, inplace=True) 
    
    loan_data.corr()["mort_acc"].sort_values() # highest corr with total_acc
    total_acc_avg = loan_data.groupby('total_acc').mean()['mort_acc']
    # approximate the missing value to the mean value of mort_acc correlated to the total_acc
    loan_data['mort_acc'] = loan_data.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

    # revol_util and pub_rec_bankruptcies only have < 0.5% of NaN values
    loan_data.dropna(inplace=True)
    
    '''
    Deal with categorical variables
    '''
    # 36 months or 60 months, keep the nb and remove months
    loan_data["term"] = loan_data["term"].apply(lambda term: int(term[:3])) 
    
    # Grades are A,B,C,D and sub_grades are A1,A2,...,B3, etc
    # redudancy because sub-grade has the grade in it so let's remove grade
    loan_data.drop("grade",axis=1, inplace=True) 
    
    # Get these variables as dummies
    loan_data['home_ownership'].value_counts()
    loan_data['home_ownership'] = loan_data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

    dummies = pd.get_dummies(loan_data[['sub_grade', 'home_ownership','verification_status', 'initial_list_status','purpose', 'application_type']],drop_first=True)
    loan_data = loan_data.drop(['sub_grade', 'home_ownership','verification_status', 'initial_list_status','purpose', 'application_type'],axis=1)
    loan_data = pd.concat([loan_data,dummies],axis=1)
    
    # Irrelevant data
    loan_data.drop('issue_d',axis=1, inplace=True)
    loan_data.drop('earliest_cr_line',axis=1, inplace=True)

    # Extract zip code from the address (take last 5 digits)
    loan_data['zip_code'] = loan_data['address'].apply(lambda address:address[-5:])
    loan_data.drop('address', axis=1, inplace=True)
    loan_data = pd.get_dummies(loan_data,prefix=['zip_code'], columns=['zip_code'])
    
    loan_data.drop('loan_status', axis=1, inplace=True) #keep dummy_status instead
      
    ''' 
    Deep Learning: Create Model
    '''
    X = loan_data.drop('dummy_status', axis=1).values
    y = loan_data['dummy_status'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
     
    X_train.shape
    
    MLP = ann_model(X_train, y_train)
    
    loss_var = pd.DataFrame(MLP.history.history)
    loss_var.plot()
    
    pred_test = MLP.predict_classes(X_test)
    print(classification_report(y_test,pred_test))
    print(confusion_matrix(y_test,pred_test))


