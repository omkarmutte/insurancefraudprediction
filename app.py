from flask import Flask,request, url_for, redirect, render_template, Response
import pickle
import numpy as np
import pandas as pd
import csv
from io import StringIO

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

download = pd.DataFrame()

@app.route('/')
def hello_world():
    return render_template("fraud.html")

@app.route('/download_csv')
def download_csv():
    global download
    csv_data = StringIO()
    download.to_csv(csv_data, index=False, quoting=csv.QUOTE_NONNUMERIC)
    csv_data.seek(0)
    return Response(
        csv_data.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=table_data.csv'}
    )
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    global download
    data = pd.read_csv(request.form["csvfile"])
    data=data.replace('?',np.nan)
    download = data.copy()
    new=data[['policy_number','insured_sex']]
    cols_to_drop=['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year']
    data.drop(columns=cols_to_drop,inplace=True)
    class CategoricalImputer:
        def __init__(self, strategy='most_frequent'):
            self.strategy = strategy

        def fit_transform(self, X):
            if self.strategy == 'most_frequent':
                self.most_frequent_values = X.mode().iloc[0]
            X_filled = X.fillna(self.most_frequent_values)
            return X_filled

    
    
    imputer = CategoricalImputer()

    data['collision_type']=imputer.fit_transform(data['collision_type'])
    data['property_damage']=imputer.fit_transform(data['property_damage'])
    data['police_report_available']=imputer.fit_transform(data['police_report_available'])
    cat_df = data.select_dtypes(include=['object']).copy()
    cat_df['policy_csl'] = cat_df['policy_csl'].map({'100/300' : 1, '250/500' : 2.5 ,'500/1000':5})
    cat_df['insured_education_level'] = cat_df['insured_education_level'].map({'JD' : 1, 'High School' : 2,'College':3,'Masters':4,'Associate':5,'MD':6,'PhD':7})
    cat_df['incident_severity'] = cat_df['incident_severity'].map({'Trivial Damage' : 1, 'Minor Damage' : 2,'Major Damage':3,'Total Loss':4})
    cat_df['insured_sex'] = cat_df['insured_sex'].map({'FEMALE' : 0, 'MALE' : 1})
    cat_df['property_damage'] = cat_df['property_damage'].map({'NO' : 0, 'YES' : 1})
    cat_df['police_report_available'] = cat_df['police_report_available'].map({'NO' : 0, 'YES' : 1})
    for col in cat_df.drop(columns=['policy_csl','insured_education_level','incident_severity','insured_sex','property_damage','police_report_available']).columns:
        cat_df= pd.get_dummies(cat_df, columns=[col], prefix = [col], drop_first=True)
    num_df = data.select_dtypes(include=['int64']).copy()
    final_df=pd.concat([num_df,cat_df], axis=1)

    final_df.drop(columns=['age','total_claim_amount'], inplace=True)
    # 
    prediction=model.predict(final_df)
    new['fraud_reported']=prediction
    new['fraud_reported']=new['fraud_reported'].apply(lambda x: "Yes" if x==1 else "No")
    download['fraud_reported'] = new['fraud_reported']
    return render_template('data_file.html',dataframe=new)
    # if output>str(0.5):
    #     return render_template('forest.html',pred=output)
    # else:
    #     return render_template('forest.html',pred=output)


if __name__ == '__main__':
    app.run(debug=True)
