import os
import telebot
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('TOKEN')

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def get_start(message):
    bot.send_message(message.chat.id, "знакомлю с роналдо  за 100 рублей,акция!!!!!")

@bot.message_handler(content_types=['text'])
def first_answer(message):
    text = message.text.lower()

    if 'программисту' in text:
        bot.send_message(message.chat.id, "подари мышку")
    elif 'дизайнеру' in text:
        bot.send_message(message.chat.id, "подари попкорница")   
    elif 'трактористу' in text:
        bot.send_message(message.chat.id, "подари ковш")
    elif 'водителю' in text:
        bot.send_message(message.chat.id, "подари время")
    elif  'копателю' in text:
        bot.send_message(message.chat.id, "подари пластиковую картошку")
    else:
        bot.send_message(message.chat.id,"я могу показать подарки программисту,дизайнеру,трактористу,водитеою,копателю")

        
if '__main__'==__name__:
    bot.infinity_polling()