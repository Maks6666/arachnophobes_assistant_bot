import telebot
from model_test import model
import cv2
import numpy as np
from io import BytesIO
import requests
from PIL import Image

def prediction(image):
    danger_level = ["High", "Medium", "Low"]
    list_of_classes = ['Red Knee Tarantula', 'Black Widow', 'Blue Tarantula',
                       'Golden Orb Weaver', 'Bold Jumper', 'Peacock Spider',
                       'Yellow Garden Spider', 'Brown Recluse Spider', 'Spiny-backed Orb-weaver',
                       'Deinopis Spider', 'Hobo Spider', 'White Kneed Tarantula', 'Huntsman Spider',
                       'Ladybird Mimic Spider', 'Brown Grass Spider']

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    res = model.predict(image)
    class_pred = np.argmax(res[0])
    level_pred = np.argmax(res[1])

    fin_res = f"Looks like {list_of_classes[class_pred]} and it has {danger_level[level_pred]} danger for humans."
    return fin_res



token = 'YOUR_OWN_TOKEN'
bot = telebot.TeleBot(token)

@bot.message_handler(commands=["start"])
def main(message):

    bot.send_message(message.chat.id, f"Hello, {message.from_user.first_name}, my arachnophobic comrade. "
                                      f"Give me a photo of a monster, you've faced with...")


@bot.message_handler(content_types=['text'])
def reply(message):
    bot.send_message(message.chat.id, "Sorry, I cannot handle text, only scary photos.")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file_url = f'https://api.telegram.org/file/bot{token}/{file_info.file_path}'
    response = requests.get(file_url)

    img = Image.open(BytesIO(response.content))


    res = prediction(img)


    bot.reply_to(message, res)

if __name__ == "__main__":
    bot.polling(none_stop=True)
