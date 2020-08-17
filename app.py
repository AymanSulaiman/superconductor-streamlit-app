import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.title('Superconductor Machine Learning App And Analysis')

st.write('''
This is an app that predicts the Critical Temperature of a Superconductor 
using XGBoost.  Shoutout to Kam Ham idieh of UPenn for donating the data to UC Irvine for providing the clean data 
so I can make this app. Here is the link to the [dataset]('https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data')
and the [Data Professor]('https://www.youtube.com/channel/UCV8e2g4IWQqK71bbzGDEI4Q') who's tutorial I followed for the streamlit app.

All you need to do is select a material that is on the left of your monitor and the Neural Network will do the rest.

**Disclaimers**

This App runs a little bit slow due to the size of the Neural Nework and the size of the data files.

This is not 100% accurate. These are just predictions and getting 100% accuracy would mean that the Neural Network would have overfitted the data.

Have a look at this [article](http://www.owlnet.rice.edu/~dodds/Files332/HiTc.pdf) to see how to obtain the temperature of Supercondutors.
''')



# Declaring paths for each of the CSV files
path_merged = os.path.join('data','merged.csv')
history_path = os.path.join('data','history.csv')

# Declaring the DataFrames with respect to the CSV file
df_merged = pd.read_csv(path_merged)
df_history = pd.read_csv(history_path)



# Start of the machine learning
X = df_merged.drop(['critical_temp'], axis=1)
y = df_merged['critical_temp'].values.reshape(-1,1)


model_path = os.path.join('my_keras_model.h5')
model = load_model(model_path)

temp_pred = model.predict(df_merged.drop(['material','critical_temp'], axis=1))
# end of the machine learning


temp_actual = df_merged.critical_temp


# start of side bar
st.sidebar.header('Select a Material')

@st.cache(suppress_st_warning=True)
def sidebar_test_materials_list():

    list_of_materials = []
    for i in df_merged.material:
        if i not in list_of_materials:
            list_of_materials.append(i)
        else:
            pass    
    return list_of_materials


materia = st.sidebar.selectbox('',(sidebar_test_materials_list()))
# end of sidebar

# start of modifyable Dataframe

actual_temp = temp_actual[df_merged[df_merged.material == materia].index]
predicted_temp = temp_pred[df_merged[df_merged.material == materia].index]

data =  {
    'Material': materia,
    'Actual Critical Temperature': actual_temp,
    'Predicted Critical Temperature': np.array([[i] for i in predicted_temp]).flatten(),
    
}

df = pd.DataFrame(data)
df
# End of modifyable Dataframe

# Modifyable material plot
def material_plot():
    fig = px.scatter(
        x = df['Actual Critical Temperature'],
        y = df['Predicted Critical Temperature'],
    )
    
    fig.update_layout(
        title=f'{materia}',
        xaxis=dict(
            title='Actual Critical Temperature (K)'
        ),
        yaxis=dict(
            title='Predicted Critical Temperature (K)'
        ),
    )

    return st.plotly_chart(fig)

material_plot()

st.write('''
# The Machine Learning Algorithm 
''')

score_test = r2_score(temp_actual, temp_pred)

st.write(f'''
## How the Model was made.
Below is the process of how the Neural Network was made and how the file was exported.
I wanted to add in more layers to make the model more accurate and to improve the model overall. 
I think an accuracy of {100*round(score_test,4)}%, however is perfectly fine.  I made this Neural Network with a CPU rather than a GPU.  I am currently hosting this model on Heroku and they do not provide the luxary of a GPU.

```
X = df.drop(['critical_temp', 'material'], axis=1)
y = df['critical_temp'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['mae', 'mse']
             )


checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)


history = model.fit(X_train, y_train, epochs=3500,
                    validation_data=(X_test, y_test),
                    batch_size=1024,
                    callbacks=[checkpoint_cb])


model = keras.models.load_model("my_keras_model.h5") # rollback to best model
mse_test = model.evaluate(X_test, y_test)

```
''')

st.write('This gives an r squared score of', 100*round(score_test,4), '%')

st.write('''
Below is the number of epochs done and the measurements performed 
''')


def history_plot():
    df_history.plot(figsize=(15,7))
    plt.ylim(0,350)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mertrics')
    plt.title('Epochs and Metrics for Neural Network')

    return st.pyplot()
history_plot()

######################################
# Showing the data and visualizations#
######################################


st.title(f'Sample of the data (tail end)')

show_df = df_merged.tail(n=round(0.2*21263))
show_df

st.title(f'Number of rows of the full dataset: {len(df_merged)}')

merged_copy = df_merged.copy()
merged_copy['pred_temp'] = temp_pred



def predicted_actual():
    fig = px.scatter(
        merged_copy,
        hover_data=['material'],
        x='critical_temp',
        y='pred_temp',
        # width=1000,
    )
    
    fig.update_layout(
        title='Actual Temperatures vs Predicted Temperatures',
        xaxis=dict(
            title='Actual Critical Temperature (K)'
        ),
        yaxis=dict(
            title='Predicted Critical Temperature (K)'
        ),
    )

    return st.plotly_chart(fig)

predicted_actual()



def mean_atomic_mass_and_critical_temperature():
    fig = px.scatter(
        df_merged,
        hover_data=['material'],
        x='mean_atomic_mass',
        y='critical_temp',
        size='critical_temp', 
        color='number_of_elements',
    )


    fig.update_layout(
        title='Mean Atomic Mass and Critical Temperature of the whole Dataset',
        xaxis=dict(
            title='Mean Atomic Mass'
        ),
        yaxis=dict(
            title='Critical Temperature (K)'
        ),
    )

    return st.plotly_chart(fig)

mean_atomic_mass_and_critical_temperature()





st.write(f'''
[GitHub Link](https://github.com/AymanSulaiman/superconductor-analysis-and-prediction)    
[Resume](https://drive.google.com/file/d/1Cic_2AMCGAVRlwc7pu28N2-KcFyDaIhl/view?usp=sharing)      
[LinkedIn](https://www.linkedin.com/in/s-ayman-sulaiman/)
''')