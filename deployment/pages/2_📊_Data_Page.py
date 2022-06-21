import json
import os
import logging
import sys
import tensorflow as tf
import pandas as pd
sys.path.append('../scripts')
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from lib import util, paths, static

LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title='Data Page', page_icon='icon.png')

MODEL_PATH = 'assets/sample_models'

SOURCES = ['ponoka', 'westlock', 'rimbey', 'dawson_creek', 'beaverlodge']
static.initialize({"df": pd.DataFrame([])})

@st.experimental_memo
def load_data():
    d = util.load_and_prep_data(os.path.join(paths.TRAINING_DATA, paths.DATASET))
    return d.drop(columns=['prediction'])

@st.experimental_memo
def select_data(weight_range, source):
    d = load_data()
    d = d[(d.weight < weight_range[1]) & (d.weight > weight_range[0])]
    if len(source) > 0:
        d = d[d.auction.isin(source)]
    LOGGER.info(d.head())
    return d

@st.experimental_memo(suppress_st_warning=True)
def test_models(names, data):
    models = [tf.keras.models.load_model(os.path.join(MODEL_PATH, name)) for name in names]
    test_fig, results = util.test_on_image(models, data, model_labels=names, display_width=3)
    st.pyplot(test_fig)
    return results


st.title('The VisuWeigh Data')
st.write('The following data has been collected from internet sources and cleaned for use in the VisuWeigh project.')
st.write('Take a look through the stats, visualize the data, and try the models.')
st.write('Some of the trained models are available on the left side-bar. Select the models you would like to test.')
st.write('')

# Get input
in1, in2, in3 = st.columns(3)
with in1:
    source = st.multiselect('Select a data source', SOURCES)

with in2:
    w_range = st.slider("Select Weight Range", 400, 1400, (400, 1400), 1)
    df = select_data(weight_range=w_range, source=source)

with in3:
    max_count = st.slider("Select Maximum Sample Size", 1, len(df), len(df)) if len(df) else 0

# Retrieve Data
df_limit = df.sample(n=max_count)

# Render Results
disp1, disp2 = st.columns([2,1])
with disp1:
    fig, ax = plt.subplots()
    ax = sns.histplot(data=df_limit.weight, bins=50, kde=True)
    st.pyplot(fig)

with disp2:
    st.write(df_limit.describe())

st.dataframe(df_limit.drop(columns=['path']))
st.write(f'Showing {len(df_limit)} data points')

# Get model input
model_list = os.listdir('assets/sample_models')
selected_models = []
with st.sidebar:
    st.title('Models')
    selected_models = st.multiselect('Select a few models', model_list)

if len(selected_models):

    with st.sidebar:
        sample_size = st.slider("Select the sample size to test on", 3, 21, 6, 3)
        run_test = st.button("Run Model Test")

    if run_test:
        st.title('Model Test Results')
        test_models(selected_models, df.sample(n=sample_size))

else:
    st.write('Select models from the sidebar to test on the data')







