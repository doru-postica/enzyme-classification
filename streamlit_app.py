import pandas as pd
import streamlit as st
import joblib


def load_models():
    EC1_model = joblib.load('fit_pickles/fit_EC1_model.pkl')
    EC2_model = joblib.load('fit_pickles/fit_EC2_model.pkl')
    scaler = joblib.load('fit_pickles/fit_scaler.pkl')
    feature_names = joblib.load('fit_pickles/feature_names.pkl')
    return (EC1_model, EC2_model, scaler, feature_names)


def display_greetings():
    st.title('Enzyme Classification Model')
    st.image('https://i.kym-cdn.com/photos/images/newsfeed/000/234/739/fa5.jpg')
    st.header('Please upload your file to generate prediction:')


def provide_sample_files():
    st.text('Empty template to fill in:')
    with open('eda/empty_template.csv', 'r') as empty_template:
        st.download_button(label='Template', data=empty_template, mime='text/csv')

    st.text('Dummy data if you want to test the model:')
    with open('eda/test_data.csv', 'r') as dummy_data:
        st.download_button(label='Dummy_data', data=dummy_data, mime='text/csv')


def get_streamlit_input():
    while True:
        try:
            uploaded_file = st.file_uploader(label='Upload your csv here:', type=['csv'])
            input_dataframe = pd.read_csv(uploaded_file)
        except Exception:
            continue
        else:
            break
    return input_dataframe


def validate_column_names(input_dataframe):
    if input_dataframe.drop(['id'], axis=1).columns.tolist() != feature_names:
        st.text('Columns do not match our template. Please upload another file.')
        exit()
    return


def validate_data_types(input_dataframe):
    for column in input_dataframe.columns:
        if input_dataframe[column].dtype == 'object':
            st.text('Unexpected values found in file. Please upload another file.')
            exit()
    return


def prepare_data(input_dataframe):
    mypred_df = input_dataframe.drop(['id'], axis=1)
    mypred_df = scaler.transform(mypred_df)
    return mypred_df


def get_output(mypred_df):
    EC1 = EC1_model.predict_proba(mypred_df)[:, 1]
    EC1 = pd.Series(data=EC1, name='EC1')

    EC2 = EC2_model.predict_proba(mypred_df)[:, 1]
    EC2 = pd.Series(data=EC2, name='EC2')

    final_file = (pd.concat([input_dataframe, EC1, EC2], axis=1)).to_csv(index=False)
    st.success('Success!', icon="✅")
    st.download_button(label='Download results', data=final_file, mime='text/csv')


# Executing script:

EC1_model, EC2_model, scaler, feature_names = load_models()
display_greetings()
provide_sample_files()

input_dataframe = get_streamlit_input()
validate_column_names(input_dataframe)
validate_data_types(input_dataframe)

mypred_df = prepare_data(input_dataframe)

if st.button('Generate results:'):
    get_output(mypred_df)
