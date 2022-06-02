import streamlit as st
from cv2 import cv2
import numpy as np
from VisuWeigh import predict, save
from datetime import datetime as dt
import logging
import os
import static

# Use this to declare static variables once at the start of the session
static.initialize({
    'improve': False,
    'cow_count': 0,
    'submitted': False
})
basePath = ''
client_data_path = '../client_data'

st.set_page_config(page_title='Visual Weighing Tool', page_icon='icon.png')

@st.experimental_memo
def weigh(images):

    with open(os.path.join(basePath, 'pcount'), 'r+') as f:
        cow_count = int(f.read())
        cow_count = 1 + cow_count
        f.seek(0)
        f.write(str(cow_count))
        static.setVar('cow_count', cow_count)

    print('cow_count: '+str(static.getVar('cow_count')))
    st.session_state.submitted = False
    return predict(images=images)


st.title('Hi There!')
st.write("You've found it! This is the place to try out the revolutionary cattle weighing app!")
st.write("Just select an image to upload or take a picture with your camera, and we'll handle the rest of the heavy lifting! It's that Easy!")
st.write('')
st.write("Don't have a cow around to try it on? Try one of ours from the sidebar!")

with st.sidebar:
    st.write('Choose "Upload Images" and then drag and drop on of these images')
    paths = os.listdir('test_images')
    for path in np.random.choice(paths, 5):
        im = cv2.cvtColor(cv2.imread(os.path.join('test_images', path)), cv2.COLOR_BGR2RGB)
        st.image(im)

selection = st.selectbox('Select Input:', ['Camera', 'Upload Images'], index=0)

try:
    if selection == 'Camera':
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is None:
            img_file_buffer = []
        else:
            img_file_buffer = [img_file_buffer]

    elif selection == 'Upload Images':
        img_file_buffer = st.file_uploader('Select images for uploading', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if img_file_buffer is not None and img_file_buffer != []:
        # Read images from file buffer
        try:
            images = [cv2.imdecode(np.frombuffer(img_f.read(), np.uint8), 1) for img_f in img_file_buffer]
        except Exception as ex:
            logging.error(ex)

        #img_array = np.array(img)

        # make a prediction on the weight
        with st.spinner("Hmm, let's see ... "):
            print('weighing...')
            t1 = dt.now()
            p_crops, p_images, weights = weigh(images)
            print(f'Time {dt.now() - t1}')

        # Check the shape of img_array:
        if weights == []:
            st.write('No cows in view!')

        else:
            # Should output shape: (height, width, channels)
            for image, weight in zip(p_images, weights):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image, caption=f'Weight of cows in image: {[int(w) for w in weight]}')

            st.write('')
            st.write("Wasn't quite what you were expecting?")
            st.write("The system is still in early development, but you can help make it better!")
            st.write("Allow us to use your image to improve the algorithm!")

            improve = st.checkbox('I agree to let you use my images')

            if improve and not st.session_state.submitted:

                form = st.empty()
                with form.form(key='sub'):
                    input_id = 0
                    for crops, weight in zip(p_crops, weights):
                        for image, p_weight in zip(crops, weight):
                            input_id += 1
                            st.image(image)
                            a_weight = st.number_input("What should this animal weigh?", min_value=0,
                                                       max_value=2500, key=input_id)

                            if a_weight > 0:
                                print(f'Count: {static.getVar("cow_count")}_{input_id}, Weight: {a_weight}')
                                save(image, client_data_path, f'{static.getVar("cow_count")}_{input_id}', a_weight, p_weight)

                    st.session_state.submitted = st.form_submit_button()

                if st.session_state.submitted:
                    form.header('Thank you for contributing!')

    #st.write(st.session_state)

except Exception as ex:
    print(ex)
    logging.error(ex)


