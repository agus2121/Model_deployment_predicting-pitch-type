import flask
from flask import url_for
import pickle
import xgboost as xgb
import pandas as pd
# Use pickle to load in the pre-trained model.

with open(f'model/model_rf1.pkl', 'rb') as f:
    model = pickle.load(f)
#model2 = xgb.XGBClassifier()
#model2.load_model(r"F:\MySQL\Python\cobaflask\model\xgboost1.json")
app = flask.Flask(__name__, template_folder='templates')
#app = flask.Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('home - try - Copy.html'))
    if flask.request.method == 'POST':
        b_count = flask.request.form['b_count']
        s_count = flask.request.form['s_count']
        outs = flask.request.form['outs']
        inning = flask.request.form['inning']
        stand = flask.request.form['stand']
        p_throws = flask.request.form['p_throws']
        base_count = flask.request.form['base_count']
        last_pitch = flask.request.form['last_pitch']
        last_result = flask.request.form['last_result']
        input_variables = pd.DataFrame([[b_count,s_count,outs,inning,stand,p_throws,base_count,last_pitch,last_result]],
                                       columns=['b_count','s_count','outs','inning','stand','p_throws','base_count','last_pitch','last_result'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        print(b_count,s_count,outs,inning,stand,p_throws,base_count,last_pitch,last_result)
        print(prediction)
        return flask.render_template('home - try - Copy.html',
                                     original_input={'b_count':b_count,
                                                     's_count':s_count,
                                                     'outs':outs,
                                                     'inning':inning,
                                                     'stand':stand,
                                                     'p_throws':p_throws,
                                                     'base_count':base_count,
                                                     'last_pitch':last_pitch,
                                                     'last_result':last_result
                                                     }
                                                     ,result = prediction)
if __name__ == '__main__':
    app.run()   