# ELON - The General Purpose Chatbot
   * This is a General purpose chatbot made using NLP NLTK KERAS TENSORFLOW which is serviceable for everyday usage. To this end, I trained this model with a variety of intents which are used in day to day life which make it easy for the chatbot to give appropriate and informative responses.
   * I named this chatbot after my Inspiration ELON MUSK.
# More Information about the Chatbot
  * This is achieved by creating a 3-layer deep neural network with keras sequential API. Then it is trained using the test data containing the intents to make it           fairly accurate.
  * The GUI is then created in order to load the chatbot model and take responses from the user. This leads to base interaction which leads a back and forth from the       user to the bot and vice versa for a seamless experience.
  * It supports a unique feature where the output text is converted to speech using python builtin module pyttsx3.
  * Every Statement in the above code is commented appropriately for simple understanding.
  
# Contents 
  * train_chatbot.py is the python file to train the model .
  * gui.py is the main python gui file used for interacting with the chatbot.
  * intents.json consists of the intents / Training data necessary for traning the model
  
# Installing Packages required to Build AI Chatbot
    pip install tensorflow, keras, pickle, nltk, numpy, pyttsx3
    
# How to run?
   *  Train the model by executing train_chatbot.py
   *  Run the chatbot by executing gui.py
