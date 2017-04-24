import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import time

import socketio
import eventlet
from flask import Flask, render_template, send_from_directory

from GetData import get_duration, get_published_hour, get_num_words_in_title, get_published_day_of_week, get_words

sio = socketio.Server()
app = Flask(__name__, static_url_path='')

app.config['DEBUG'] = True # enable hot reload


"""

Data Preposseccing

"""

#collect data
df = pd.read_csv('cleaned_age_added.csv')

df['SURGICAL_SPECIALTY'] = df['SURGICAL_SPECIALTY'].map(lambda x:x.upper())

def str_2_time(str):
    l = str.split(":")
    return int(l[0])*60+int(l[1])
df['Duration'] = df['Duration of Surgery (hh:mm).1'].map(str_2_time)

#convert object to float
df['SN - BM - Pre-Op INR'] = df['SN - BM - Pre-Op INR'].apply(lambda x: float(x) if x != '.' else float('nan'))

#remove outliers
df = df[np.abs(df['SN - BM - Pre-Op INR']-df['SN - BM - Pre-Op INR'].mean())<=(5*df['SN - BM - Pre-Op INR'].std())]

#fill Nan with mean value
df['SN - BM - Pre-Op INR'] = df['SN - BM - Pre-Op INR'].fillna(df['SN - BM - Pre-Op INR'].mean())
#convert object to float
df['SN - BM - Pre-Op Platelet Count'] = df['SN - BM - Pre-Op Platelet Count'].apply(lambda x: float(x) if x != '.' else float('nan'))
#outliers removal
df = df[np.abs(df['SN - BM - Pre-Op Platelet Count']-df['SN - BM - Pre-Op Platelet Count'].mean())<=(5*df['SN - BM - Pre-Op Platelet Count'].std())]    
#fill Nan with mean value
df['SN - BM - Pre-Op Platelet Count'] = df['SN - BM - Pre-Op Platelet Count'].fillna(df['SN - BM - Pre-Op Platelet Count'].mean())

df = df.drop(['Masked FIN'], 1)
df = df.drop(['Sequence No.'], 1)
df = df.drop(['Duration of Surgery (hh:mm).1'], 1)
df = df.drop(['EBL'], 1)
df = df.drop(['SN - BM - PRBC Ordered'], 1)
#since high dimensional data is not good for random forest, we drop SURG_PROCEDURE to get better results
df = df.drop(['SURG_PROCEDURE'], 1)

df = df.dropna(subset=['age','SN - BM - Pre-Op INR','SN - BM - Pre-Op Platelet Count'])
df_complete = df.copy()
df_seg2 = df_complete[df_complete['ResultAfterSurgery'].isnull()]
df_seg2 = df_seg2.drop('ResultAfterSurgery',1)
df_seg1 = df_complete.dropna()


df_toy = pd.get_dummies(df)
df_to_be_completed = df_toy[df_toy['ResultAfterSurgery'].isnull()]
df_toy = df_toy.dropna()

df_toy_target = df_toy['ResultAfterSurgery']
df_toy = df_toy.drop('ResultAfterSurgery',1)

radm = RandomForestRegressor(n_estimators = 300)
radm.fit(df_toy,df_toy_target)

#Filling missing anemia status data
df_to_be_completed = df_to_be_completed.drop('ResultAfterSurgery',1)
df_seg2['ResultAfterSurgery'] = radm.predict(pd.get_dummies(df_to_be_completed))
#concatenate two segments
frames = [df_seg1,df_seg2]
df_missing_post_anemia_filled = pd.concat(frames)


df_predict_RBC_raw = df_missing_post_anemia_filled[df_missing_post_anemia_filled['ResultAfterSurgery'] > 10]
df_predict_RBC_target = df_predict_RBC_raw['SN - BM - Red Blood Cells']
df_predict_RBC_features_not_dummified = df_predict_RBC_raw.drop(['SN - BM - Red Blood Cells','SN - BM - Cryoprecipitate','SN - BM - Fresh Frozen Plasma','SN - BM - Platelets','Duration','Allogeneic Blood Transfusion','ResultAfterSurgery'],1)
df_predict_RBC_features = pd.get_dummies(df_predict_RBC_features_not_dummified)

final_model = RandomForestRegressor(n_estimators = 300)
final_model.fit(df_predict_RBC_features,df_predict_RBC_target)

@app.route('/')
def index():
    """Serve the client-side application."""
    return render_template('index_b.html')

@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)

@sio.on('request_surgeon_data')
def request_surgeon_data(sid, data):
    specialty = data['data'].upper()
    doc_count = df[df['SURGICAL_SPECIALTY'] == specialty]['Surgeon Hash Name'].value_counts().to_json()
    sio.emit('doc_data', doc_count)

@sio.on('request_RBC_data')
def request_RBC_data(sid, data):
    print 'start predicting...'
    print data
    tmp = pd.DataFrame([data.values()],columns=data.keys())
    tmp['SURGICAL_SPECIALTY'] = tmp['SURGICAL_SPECIALTY'].map(lambda x:x.upper())
    tmp['ResultsBeforeSurgery'] = tmp['ResultsBeforeSurgery'].map(lambda x:float(x))
    tmp['SN - BM - Pre-Op INR'] = tmp['SN - BM - Pre-Op INR'].map(lambda x:float(x))
    tmp['SN - BM - Pre-Op Platelet Count'] = tmp['SN - BM - Pre-Op Platelet Count'].map(lambda x:float(x))
    tmp['age'] = tmp['age'].map(lambda x:float(x))
    dd = df_predict_RBC_features_not_dummified.append(tmp)
    print pd.get_dummies(dd).iloc[-1].values.reshape(1,-1)
    res = final_model.predict(pd.get_dummies(dd).iloc[-1].values.reshape(1,-1))
    #res=1
    print 'end predicting...'
    print res
    sio.emit('RBC_data', {'value': res[0]})
@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 3001)), app)
