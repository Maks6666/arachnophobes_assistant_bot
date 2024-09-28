
# Spider classification with double-output CNN Telegram bot
![](project_images/title.jpg)

## Idea
The bite of a poisonous spider causes the death of thousands of people every year. The main problem is often that in order to administer an antidote, it is necessary to know exactly which spider bit the person. And since not every person is an expert in arachnology, 
it was decided to create a program that can, using a neural network, classify the 15 most common species of spiders and the degree of their poisonousness only from a photo.
For convenience, the created neural network model was deployed in the bot telegram format. To activate it, use the default bot - @BotFather, create your own bot with it, get a token. Paste your token into the specified location in the main.py file and launch your bot.

## Programm 
The basis of the program is a convolutional neural network, written in the Tensorflow library and having two outputs. The first of them is responsible for predicting the type of spider, the second - for the degree of its poisonousness. More details about the structure of the model can be found in the file model.ipynb.
A dataset was used for training 
https://www.kaggle.com/datasets/gpiosenka/yikes-spiders-15-species
Here you may find results of model training:

![losses.jpg](project_images%2Flosses.jpg)

*For common losses:*

![output1.jpg](project_images%2Foutput1.jpg)

*For 1st output accuracy (spider classification)*

![output1.jpg](project_images%2Foutput2.jpg)

*For 2nd output accuracy (venomous level classification)*

![matrix1.jpg](project_images%2Fmatrix1.jpg)

*Confusion matrix for spider classification*

![matrix1.jpg](project_images%2Fmatrix2.jpg)

*Confusion matrix for spider venomous leve classification*

During the training process, the model encountered overfitting and here are the best results that I was able to obtain. If you have ideas for improving performance, I'm always happy to listen.


## Bot
![content.jpg](project_images%2Fcontent.jpg)
Telegram bot is the simplest and most accessible way to deploy neural network models. Telegram is always at hand, which means a person who needs to classify a spider will always be able to do it. The principle of operation is simple - you just need to take a photo of the spider and send it to the bot - it will return its name and degree of toxicity to humans.
If the user sends the bot a text and not a photo, the bot will return a message that it only works with images. The photos that the bot accepts are first downloaded from the Telegram server. then they are processed using the cv2 library - they are given the required size and format to be transferred to the model. When the photo is processed, the model returns the result.

It is also important to note that before activating the script with the bot, it is necessary to execute the code in the model_test.py file - it will recompile the model and load ready-made, trained weights into it to prepare the model for work. Then, in the main.py file, the model is imported and begins to be used for its intended purpose.
## Authors:
- Kucher Maks (maxim.kucher2005@gmail.com)




