import numpy as np
import pandas as pd
from joblib import load
import skops.io as sio
import gradio as gr
import shap
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from google import genai
import os

## versions
# shap 0.49.1
# scikit-learn 1.6.1
# numpy 2.0.2
# pandas 2.2.2
# skops 0.13.0
# joblib 1.5.3
# gradio 5.38.1

## functions for shap
def predict_log_proba(z):
    p = model.predict_proba(z)
    return np.log(p[:,1] / p[:,0])

def predict_proba(z):
    p = model.predict_proba(z)
    return p[:,1]

model_path = 'model.skops'
enc_path   = 'encoder.skops'
unknown_model = sio.get_untrusted_types(file=model_path)
unknown_enc   = sio.get_untrusted_types(file=enc_path)
model = sio.load(model_path, trusted=unknown_model)
enc   = sio.load(  enc_path, trusted=unknown_enc)

explainer = load('explainer.joblib')
exist_shap = load('shap_values.joblib')

## read pdf file
pdf_path = 'SHAP_scatter_plot.pdf'

# display(model)
# display(enc)
# display(explainer)
# display(exist_shap)

api_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=api_key) ## initailize client

print("Uploading file...")
uploaded_file = client.files.upload(file=pdf_path)

while uploaded_file.state.name == "PROCESSING":
    print(".", end="", flush=True)
    time.sleep(2)
    uploaded_file = client.files.get(name=uploaded_file.name)

if uploaded_file.state.name == "FAILED":
    raise Exception("File processing failed.")

print("\nFile ready!")


## notes on input
# user might not know about debt-to-income-ratio, credit-score, interest_rate, grade_subgrade

## here is the list of dropdown inputs
gender = ['Male', 'Female', 'Prefer not to Tell']
marital = ['Single', 'Married', 'Divorced', 'Widowed']
education = ['High School', "Master's", "Bachelor's", 'PhD', 'Other']
employment = ['Employed', 'Self-employed', 'Retired', 'Student', 'Unemployed']
purpose = ['Debt consolidation', 'Home', 'Education', 'Vacation', 'Car',
           'Medical', 'Business', 'Other']
## for debt-to-income-ratio, let use enter his debt only
## grade sub-grade is based on the person's history, so here we assume C3, the most class on data
grade_sub = 'C3'
## credit score also depend on the person's history. Here we assume 682
credit = 682
## interest rate might computed under complicated matter. Here we assume 12.4
int_rest = 12.4

## this is front page
def do_this(ai, debt, lo, g, ms, el, es, lp):
    ## run additional checks here

    ## make dataframe of input
    masuk = {}
    masuk['annual_income'] = ai
    masuk['debt_to_income_ratio'] = debt/ai
    masuk['credit_score'] = credit
    masuk['loan_amount'] = lo
    masuk['interest_rate'] = int_rest
    masuk['gender'] = g
    masuk['marital_status'] = ms
    masuk['education_level'] = el
    masuk['employment_status'] = es
    masuk['loan_purpose'] = lp
    masuk['grade_subgrade'] = grade_sub
    masuk_pd = pd.DataFrame(masuk, index=[0])
    # print(masuk_pd)

    ## perform encoding
    objek = ['gender', 'marital_status', 'education_level',
             'employment_status', 'loan_purpose', 'grade_subgrade'] ## non-numericals
    X_ord = masuk_pd[objek]
    X_enc = enc.transform(X_ord)
    X_enc = pd.DataFrame(X_enc, index=masuk_pd.index,
                         columns=enc.get_feature_names_out())
    
    ## replace grade feature with encoded value
    masuk_pd = masuk_pd.drop(objek, axis=1)
    masuk_pd = masuk_pd.merge(X_enc, how='inner',
                              left_index=True, right_index=True) ## use index as keys
    # print(masuk_pd)

    ## predict
    masuk_pd = masuk_pd[model.feature_names_in_]
    pred = model.predict(masuk_pd)

    if pred > 0.5:
        pred_text = 'Accept'
    else:
        pred_text = 'Reject'

    pred_p = model.predict_proba(masuk_pd)[:,1]

    pred_pd = f"{100*float(pred_p[0]):.2f}%"

    if pred_text == 'Reject':

        ## shap values
        shap_values = explainer(masuk_pd)

        print(shap_values)

        prom = f'''
        Explain why my loan application has low chance to be accepted.
        Then, suggest me some approach to improve my chance of approval based on the result.
        Make sure that your explanation is easy to understand and avoid technical terms whenever possible.
        Here is the data given by company for you to analyze.
        Debt-to-Income Ratio: {masuk['debt_to_income_ratio']:.3f}, SHAP {shap_values.values[:,0].item():.3f}
        Credit Score: {masuk['credit_score']:.3f}, SHAP {shap_values.values[:,1].item():.3f}
        Loan Amount: {masuk['loan_amount']:.3f}, SHAP {shap_values.values[:,2].item():.3f}
        Marital Status: {masuk['marital_status']}, SHAP {shap_values.values[:,3].item():.3f}
        Employment Status: {masuk['employment_status']}, SHAP {shap_values.values[:,4].item():.3f}
        SHAP base value: {shap_values.base_values.item():.3f}
        SHAP value vs. Feature plot on all features and Basic analysis given in the document.
        Categorical features and their encoding can be seen in the document.
        '''
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=[uploaded_file, prom]
        )
    
        # rej_mes = ""
        rej_mes = response.text

    else:
        rej_mes = "**Congratulations!** Your application have a good chance to be accepted!"

    rej_mes = "**Messages:**<br>"+rej_mes
    
    return pred_text, pred_pd, rej_mes

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Loan Acceptance Prediction
    This is a simple gradio UI demo for loan applicant to predict if their loan application is accepted or not.<br>
    The prediction is constructed based on a Machine Learning model trained from [this dataset.](https://www.kaggle.com/competitions/playground-series-s5e11/overview)<br>
    **Note**: This app is not related to any institution that provide lending service. The result you obtained here may differ when you actually apply to such company.
    """)
    inputs=[
        gr.Number(label='Annual Income', show_label=True, minimum=0),
        gr.Number(label='Debt', show_label=True, minimum=0),
        gr.Number(label='Loan Amount', show_label=True, minimum=0),
        gr.Dropdown(gender, label='Gender'),
        gr.Dropdown(marital, label='Marital Status'),
        gr.Dropdown(education, label='Education Level'),
        gr.Dropdown(employment, label='Employment Status'),
        gr.Dropdown(purpose, label='Loan Purpose')
    ]
    tombol = gr.Button(value='Submit', variant='primary')
    outputs=[
        gr.Textbox(max_lines=1, label='Prediction', show_label=True),
        gr.Textbox(max_lines=1, label='Acceptance Probability', show_label=True),
        # gr.Textbox(max_lines=6, label='Rejection Message', show_label=True)
        gr.Markdown(value="**Messages:**<br>", label='Rejection Message', show_label=True, container=True)        
        ]
    tombol.click(do_this, inputs, outputs)
    gr.Markdown(
    """
    **Notes and Assumptions:**<br>
    Here we assume that you are a new applicant to a lending company, although you may have debt in other company as well. Thus, several informations required to do prediction that typically not known to you are set to a constant value. These informations are, but not limited to: Credit Score, Interest Rate, and your Grade.
    """
    )

demo.launch()
