# VisuWeigh Deployment

The deployment of the VisuWeigh cattle weight estimation model will be performed using a Streamlit web app. The following list tracks the steps and progress of deployment. 

- [x] Create stand-alone code for loading and passing data to model
	
	Code needs to be reusable and well documented for handling images preprocessing, post-processing, and model prediction.


- [x] Create basic web application UI with Streamlit to pass images to the model

	The Streamlit fromework was chosen for deployment of this project due to its simplicity and intuitive usage. It is ideal for the simple interface that 	is needed for this project. The framework is straitforward for the the engineer to use and intuitive for the user. 

- [x] Test basic functionality

	The basic interface includes two methods of weighing a cow from an image. The user can eaither upload an image or take an image from a camera. 
	If there is a cow detected, the image will be passed to the CNN. JPEG, JPG, and PNG files are accepted. The interface will show bounding boxes and 	predicted weights for all cows in the image. 

	The code needs to handle cases for:</br>
		1. No cows in view</br>
		2. Corrupted image</br>
		3. Multiple images uploaded</br>
		4. Multiple cows in multiple images</br>

- [ ] Add data set viewing/browsing to the UI
	
	An evaluation data set will be available on the web app for the users to observe and interact with. The data can be used for users to try the weight 	prediction functionality. </br>
		1. Display data statistics</br>
		2. Display model performance for evaluation</br>

- [ ] Add feedback functionality

	It is always helpful to obtain user data to improve a model. Feedback will be built into the UI to allow users to upload the true weight of a cow if 		they know it. 

- [ ] Test complete functionality

	The code needs to handle cases for:</br>
		1. Fast input to data interaction (avoid hanging or laging)</br>
		2. No feedback</br>
		3. Partial Feedback</br>
		4. Invalid feedback</br>
		
- [ ] Deploy app on exposed webserver
 

