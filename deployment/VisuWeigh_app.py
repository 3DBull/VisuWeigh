
import streamlit as st

# Setup page
st.set_page_config(page_title='Visual Weighing Tool', page_icon='icon.png')

def run():
    st.markdown(
        '''
        # The VisuWeigh Project
        Hey! You found it! This is the place to try out the all new cattle weighing app!
        That's right! Weigh cows with just a single image!
        
        Check out the [Predict Page](https://visuway.tech/Weigh_Page) to try it on an image!
        
        The project is still developing and predictions are improving all the time! Check out the 
        [Data Page](https://visuway.tech/Data_Page) to see the some of the most 
        recent models and data that go into the project.
        
        '''
    )
    st.image('assets/demo.gif')

if __name__ == '__main__':
    run()