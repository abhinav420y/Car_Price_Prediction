from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('rfr_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Fuel_Type_Diesel = 0
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Owner = int(request.form['Owner'])
        Fuel_Type_Petrol = request.form['Fuel_Type_Petrol']
        if Fuel_Type_Petrol == 'Petrol':
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        elif Fuel_Type_Petrol == 'Diesel':
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 0

        Year = 2021 - Year

        Seller_Type_Individual = request.form['Seller_Type_Individual']
        if Seller_Type_Individual == 'Individual':
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0

        Trasnmission_Manual = request.form['Transmission_Mannual']
        if Trasnmission_Manual == 'Manual':
            Trasnmission_Manual = 1
        else:
            Trasnmission_Manual = 0

        prediction = model.predict([[Present_Price,Kms_Driven,Owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual, Trasnmission_Manual]])
        output = round(prediction[0],2)
        if output < 0:
            return render_template('index.html', prediction_text='Sorry you cannot sell this car')
        else:
            return render_template('index.html', prediction_text=f'You can sell this car at {output}')

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
