import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


st.markdown("""
# Titanic Survival Prediction
""")
# https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
train_df = pd.read_csv("train.csv")
print(train_df)

st.dataframe(train_df)


def manipulate_df(df):
    # Update sex column to numerical
    df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
    # Fill the nan values in the age column
    df['Age'].fillna(value = df['Age'].mean() , inplace = True)
    # Create a first class column
    df['FirstClass'] = df['Pclass'].map(lambda x: 1 if x == 1 else 0)
    # Create a second class column
    df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
    # Create a second class column
    df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)
    # Select the desired features
    df= df[['Sex' , 'Age' , 'FirstClass', 'SecondClass' ,'ThirdClass' , 'Survived']]
    return df

st.markdown("## Manipulated Data")
manipulated_df = manipulate_df(train_df)
print(manipulated_df)

st.dataframe(manipulated_df)


st.markdown("## train test split")
features= train_df[['Sex' , 'Age' , 'FirstClass', 'SecondClass','ThirdClass']]
survival = train_df['Survived']
X_train , X_test , y_train , y_test = train_test_split(features , survival ,test_size = 0.3)

st.markdown("## scale the feature data")
st.text("We need to scale the data. To do that, we’ll set mean = 0 and standard deviation = 1.")
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)
print(test_features)

# Create and train the model
model = LogisticRegression()
model.fit(train_features , y_train)
train_score = model.score(train_features,y_train)
test_score = model.score(test_features,y_test)
y_predict = model.predict(test_features)
print("Training Score: ",train_score)
print("Testing Score: ",test_score)