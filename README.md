TCET 620 - Applied Machine Learning - Dr. Clark Hochgraf

Facial Recognition Team

DeepFaceTriggerBound.py is functionally identical to DeepFaceTriggerTest, but only tests against the most prominent face in frame [STABLE]

Use ',' keypress to live embed new face
Use '.' keypress to capture and compare most prominent face in frame with saved embeddings
Use '/' keypress to exit script

To add images to admin database, add "NAME.jpg" file to imgDataCount directory, or use live embed function in DeepFaceTriggerBound. If using different directory, please update imgLoc const in code.


Main Dependencies:

Python 3.11.0

deepface 0.0.89

tensorflow 2.12


DeepFaceTriggerTest.py is the standalone script used for testing and implementation [DEPRECATED]

DeepFaceTriggerTrue.py contains function that would be run by the main program when the admin gesture is triggered (Returns boolean value as to whether individual in frame is admin) [DEPRECATED]

DeepFaceTriggerFace.py contains same function as True, but returns admin name [DEPRECATED]
