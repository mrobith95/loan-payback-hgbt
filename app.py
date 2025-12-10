import numpy as np
import pandas as pd
from joblib import load
import gradio as gr
import shap

## versions
# numpy 1.26.4
# pandas 2.2.3
# joblib 1.5.2
# gradio 5.38.1 

## functions for shap
def predict_log_proba(z):
    p = model.predict_proba(z)
    return np.log(p[:,1] / p[:,0])

def predict_proba(z):
    p = model.predict_proba(z)
    return p[:,1]

model = load('best_model.joblib')
enc = load('encoder.joblib')
explainer = load('explainer.joblib')
# ## additionals
# categoricals = load('/kaggle/input/best-model-v2/scikitlearn/default/1/categoricals.joblib')
# features = load('/kaggle/input/best-model-v2/scikitlearn/default/1/features.joblib')
# monotonics = load('/kaggle/input/best-model-v2/scikitlearn/default/1/motonics.joblib')

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

    ## shap values
    shap_values = explainer(masuk_pd)

    ## give rejection message
    rej_mes = ""
    if pred_p[0]<0.5: ## this meant that application rejected
        ## debt-to-income ratio
        if shap_values.values[0][1]<0 and shap_values.data[0][1]>=0.2:
            rej_mes += "Your current debt is too high. Pay some of your debt first, or try to have more income.\n"
        elif shap_values.values[0][1]<0 and shap_values.data[0][1]<0.2:
            rej_mes += "Your current debt is too high for this application. Pay some of your debt first, or try to have more income.\n"
            
        ## credit-score
        if shap_values.values[0][2]<0:
            rej_mes += "Your credit score is too low for this application. Consider looking for more lenient lending services and pay your credit in time to improve your score.\n"
        
        ## loan-amount
        if shap_values.data[0][3]>38000:
            rej_mes += "Loan amount you propose is too high. Consider lowering your loan amount.\n"

        ## deafult text
        if len(rej_mes)==0:
            rej_mes = "Sorry, we have no suggestion on how to improve your application."
    else:
        rej_mes = "Congratulations! Your application has a good chance to be accepted!"
    
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
        gr.Textbox(max_lines=6, label='Message', show_label=True)
        ]
    tombol.click(do_this, inputs, outputs)
    gr.Markdown(
    """
    **Notes and Assumptions:**<br>
    Here we assume that you are a new applicant to a lending company, although you may have debt in other company as well. Thus, several informations required to do prediction that typically not known to you are set to a constant value. These informations are, but not limited to: Credit Score, Interest Rate, and your Grade.
    """
    )

demo.launch()