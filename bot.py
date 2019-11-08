from emoji import emojize
import os
import telebot
from predict import Prediction
from flask import Flask, request

TOKEN = '1028491931:AAHHw3d43K2NWVIYLOWlQoLYf9bPmr4IDLc'
bot = telebot.TeleBot(TOKEN)
server = Flask(__name__)
model = Prediction()


@bot.message_handler(content_types=['photo'])
def predict_photo(message):
    for photo in message.photo:
        file_id = photo.file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open('temp.png', 'wb') as new_file:
            new_file.write(downloaded_file)

        res = model.predict('temp.png')
        bot.send_message(message.chat.id,'Class: ' + str(res))

@bot.message_handler(content_types=['document'])
def predict_doc(message):
    file_id = message.document.file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('temp.png', 'wb') as new_file:
        new_file.write(downloaded_file)

    res = model.predict('temp.png')
    bot.send_message(message.chat.id,'Class: ' + str(res))

@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id, 'Отправь мне картинку')

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, 'Привет, ' + message.from_user.first_name + " !")

@bot.message_handler(func=lambda message: True, content_types=['text'])
def non_image(message):
    bot.reply_to(message, 'Я не понимаю' + emojize(':pouting_face:', use_aliases=True))


@server.route('/' + TOKEN, methods=['POST'])
def get_message():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200

@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://uma-ml-bot.herokuapp.com/' + TOKEN)
    return "!", 200

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
