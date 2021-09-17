import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('rfr_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    df_test = request.form['file']
    df_test = pd.read_csv(df_test)
    prediction = model.predict(df_test)
    output = str(list(prediction))
    return render_template('index1.html', prediction_text=f'You can sell these cars at {output} lakhs')


if __name__ == '__main__':
    app.run(debug=True, port=9989, use_reloader=False)
