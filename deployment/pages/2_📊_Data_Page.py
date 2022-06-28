import os
import logging
import sys
import tensorflow as tf
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from VisuWeigh.lib import util, paths, static

LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title='Data Page', page_icon='icon.png')

MODEL_PATH = 'assets/sample_models'

SOURCES = ['training', 'evaluation']

@st.experimental_memo
def load_training_data():
    d = util.load_and_prep_data(os.path.join(paths.TRAINING_DATA, paths.DATASET))
    return d.drop(columns=['prediction'])


@st.experimental_memo
def load_eval_data():
    eval_names = os.listdir(paths.EVALUATION_DATA)
    LOGGER.debug(eval_names)
    d = pd.concat([util.load_all_json(os.path.join(paths.EVALUATION_DATA, name)) for name in eval_names], axis=0)
    return d.drop(columns=['prediction'])


@st.experimental_memo
def select_data(weight_range, source):

    if source == ['evaluation']:
        d = load_eval_data()
    elif source == ['training']:
        d = load_training_data()
    else:
        d = pd.concat([load_training_data(), load_eval_data()], axis=0)

    # filter range
    d = d[(d.weight < weight_range[1]) & (d.weight > weight_range[0])]

    LOGGER.debug(d.head())
    return d


@st.experimental_memo(suppress_st_warning=True)
def test_models(names, data):

    models = [tf.keras.models.load_model(os.path.join(MODEL_PATH, name)) for name in names]
    return util.test_on_image(models, data, model_labels=names, display_width=3)


st.title('The VisuWeigh Data')
st.write('The following data has been collected from internet sources and cleaned for use in the VisuWeigh project.')
st.write('Take a look through the stats, visualize the data, and try the models.')
st.write('Some of the trained models are available on the left side-bar. Select the models you would like to test.')
st.write('')

# Get input
st.markdown('''## Options
Select a data source from either training data or evaluation data. By default both sources are loaded. It is recoomended
to load only the evaluation data for testing models since the models have never seen the evaluation data during training. 
''')
in1, in2, in3 = st.columns(3)
with in1:
    source = st.multiselect('Select a data source', SOURCES)

with in2:
    w_const = paths.config['weight_constraint']

    w_range = st.slider("Select Weight Range", w_const['lower'], w_const['upper'], (w_const['lower'], w_const['upper']), 1)
    df = select_data(weight_range=w_range, source=source)

with in3:
    max_count = st.slider("Select Maximum Sample Size", 1, len(df), len(df)) if len(df) else 0

# Retrieve Data
df_limit = df.sample(n=max_count)

# Render Results
st.markdown('## Stats')
disp1, disp2 = st.columns([2,1])
with disp1:
    fig, ax = plt.subplots()
    ax = sns.histplot(data=df_limit.weight, bins=50, kde=True)
    st.pyplot(fig)

with disp2:
    st.write(df_limit.weight.describe())

st.markdown('''
## Data
The data can be sorted by clicking on the column headers''')
st.dataframe(df_limit.drop(columns=['auction', 'path']))
st.write(f'Showing {len(df_limit)} data points')

# Get model input
model_list = os.listdir('assets/sample_models')

with st.sidebar:
    st.title('Models')
    selected_models = st.multiselect('Select a few models', model_list)

if len(selected_models):

    with st.sidebar:
        sample_size = st.slider("Select the sample size to test on", 3, 21, 6, 3)
        run_test = st.button("Run Model Test")

    if run_test:
        st.markdown('## Model Test Results')

        try:
            model_test_results = test_models(selected_models, df.sample(n=sample_size))
            if model_test_results is not None:
                test_fig, results = model_test_results
                st.pyplot(test_fig)
                st.markdown('''
                Each model is tested and the main two metrics are calculated as follows: 
                
                1. Mean absolute error as an error metric: 
                ''')
                st.latex(r''' MAE =  \frac {\sum \limits _{i=0} ^{N} |y_i - \hat{y_i}|} {N}''')


                st.write('2. Mean absoluted accuracy percentage as an accuracy metric:')

                st.latex(r'''MAAP = (1 -  {\sum \limits _{i=0} ^{N} \frac {|y_i - \hat{y_i}|} {\hat{y_i}}}) * 100 \%''')


                st.dataframe(results)
        except Exception as ex:
            LOGGER.error(f"Could not test on models: {ex}")
            st.write('Error - could not perform the test.')

else:
    st.write('Select models from the sidebar to test on the data')







