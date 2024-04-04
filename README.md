TCET 620 - Applied Machine Learning - Dr. Clark Hochgraf

Facial Recognition Team

DeepFaceTriggerTest.py is the standalone script used for testing implementation

DeepFaceTriggerTrue.py contains function that would be run by the main program when the admin gesture is triggered (Returns boolean value as to whether individual in frame is admin)

DeepFaceTriggerFace.py contains same function as True, but returns admin name

To add images to admin database, add .jpg file to imgDataCount directory with I followed by sequential numbers, eg. 'I1.jpg', 'I2.jpg'. Keep 'I0.jpg' for initializing purposes

Main Dependencies:

Python 3.11.0

deepface 0.0.89

tf-keras 2.16.0
