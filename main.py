import streamlit as st
from cv2 import cv2
import numpy as np
from VisuWeigh import predict, save
from datetime import datetime as dt
import logging

sweight = 0
count = 0
IMPROVE = False

st.set_page_config(page_title='Visual Weighing Tool', page_icon='icon.png')

@st.experimental_memo
def weigh(images):
    return predict(images=images)


st.title('Hi There!')
st.write("You've found it! This is the place to try out the revolutionary cattle weighing app!")
st.write("Just select an image to upload or take a picture with your camera, and we'll handle the rest of the heavy lifting! It's that Easy!")
st.write('')
st.write("Don't have a cow around to try it on? Try one of ours!")

selection = st.selectbox('Select Input:', ['Camera', 'Upload Images', 'Use One of Ours'], index=0)

try:
    if selection == 'Camera':
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is None:
            img_file_buffer = []
        else:
            img_file_buffer = [img_file_buffer]

    elif selection == 'Upload Images':
        img_file_buffer = st.file_uploader('Select images for uploading', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    elif selection == 'Use One of Ours':
        #link = './test_images/im_28005.png'
        #html = f"<a href='{link}'><img src='data:image/png;base64,{image_base64}'></a>"
        #st.markdown(html, unsafe_allow_html=True)
        img_file_buffer = None
        pass


    if img_file_buffer is not None and img_file_buffer != []:
        # Read images from file buffer
        try:
            images = [cv2.imdecode(np.frombuffer(img_f.read(), np.uint8), 1) for img_f in img_file_buffer]
        except():
            #logging.error(ex)
            pass

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

            with open('pcount', 'r+') as f:
                count = int(f.read())
                count = 1 + count
                f.seek(0)
                f.write(str(count))

            st.write('')
            st.write("Wasn't quite what you were expecting?")
            st.write("The system is still in early development, but you can help make it better!")
            st.write("Allow us to use your image to improve the algorithm!")
            IMPROVE = st.checkbox('I agree to let you use my images')

            if IMPROVE:
                with st.form("Submit Images"):
                    input_id = 0
                    for crops, weight in zip(p_crops, weights):
                        for image, p_weight in zip(crops, weight):
                            input_id += 1
                            st.image(image)
                            a_weight = st.number_input("What should this animal weigh?", min_value=0, max_value=2500, key=input_id)
                            if a_weight > 0 and a_weight != sweight:
                                sweight = a_weight
                                print(f'Count: {count}, Weight: {a_weight}')
                                save(image, count, a_weight, p_weight)

                    st.form_submit_button()

except Exception as ex:
    print(ex)
    logging.error(ex)
