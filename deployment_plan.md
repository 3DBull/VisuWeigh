# VisuWeigh Deployment

The deployment of the VisuWeigh cattle weight estimation model will be performed using a Streamlit web app. The following list tracks the steps and progress of deployment. 

- [x] Create stand-alone code for loading and passing data to model
	
	Code needs to be reusable and well documented for handling images preprocessing, post-processing, and model prediction.


- [x] Create basic web application UI with Streamlit to pass images to the model

	The Streamlit fromework was chosen for deployment of this project due to its simplicity and intuitive usage. It is ideal for the simple interface that 	is needed for this project. The framework is straitforward for the the engineer to use and intuitive for the user. 

- [x] Test basic functionality

	The basic interface includes two methods of weighing a cow from an image. The user can eaither upload an image or take an image from a camera. 
	If there is a cow detected, the image will be passed to the CNN. JPEG, JPG, and PNG files are accepted. The interface will show bounding boxes and 	predicted weights for all cows in the image. 

	The code needs to handle cases for:
		1. No cows in view
		2. Corrupted image
		3. Multiple images uploaded
		4. Multiple cows in multiple images

- [ ] Add data set viewing/browsing to the UI
	
	An evaluation data set will be available on the web app for the users to observe and interact with. The data can be used for users to try the weight 	prediction functionality. 
		1. Display data statistics
		2. Display model performance for evaluation

- [ ] Add feedback functionality

	It is always helpful to obtain user data to improve a model. Feedback will be built into the UI to allow users to upload the true weight of a cow if 		they know it. 

- [ ] Test complete functionality

	The code needs to handle cases for:
		1. Fast input to data interaction (avoid hanging or laging)
		2. No feedback
		3. Partial Feedback
		4. Invalid feedback
		
- [ ] Deploy app on exposed webserver
 

