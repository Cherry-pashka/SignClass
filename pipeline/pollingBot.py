import telebot
from telebot import types

from datasets import get_test_transform, get_label_replacers
from models import get_densenet_121
from utils import get_all
from utils import get_class

API_TOKEN = '1901597833:AAFPAW6c3nYXZZ-06cvd7BZ4BEGfe8X4ts0'

bot = telebot.TeleBot(API_TOKEN)

m = get_densenet_121('cpu', '../checkpoints/dense121(64,64).ckpt')


@bot.message_handler(commands=['start'])
def hello(message):
    text = 'Добро пожаловать в наш бот по классификации знаков ПДД\nС помощью данного бота вы можете загрузить ' \
           'фотографию ' \
           "дорожного знака и получить информацию, к какому классу он относится\n\n/help - нужна помощь\n" \
           "/info - получить информацию про бота\n" \
           "/teaminfo - получить информацию про команду\n" \
           "/classlist - вывести все классы\n"
    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=['help'])
def helped(message):
    text = "Чтобы узнать класс знака - просто пришлите фото знака\n\n/help - нужна помощь\n" \
           "/info - получить информацию про бота\n" \
           "/teaminfo - получить информацию про команду\n" \
           "/classlist - вывести все классы\n"
    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=['info'])
def information(message):
    text = 'Мы команда DatenKrieg\nМы участники «Международного конкурса по искусственному интеллекту для детей» ' \
           'AIIJC\n\nДанный бот - продукт представления нашего алгоритма для задачи "ИИ в геосервисах", в который ' \
           'можно загрузить фотографию ' \
           'дорожного знака и получить информацию, к какому классу он относится. '

    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=['teaminfo'])
def print_classes(message):
    text = "Команда DatenKrieg\n\n" \
           "Участники:\n" \
           " ∙ Кирилл Пантелеев\n" \
           " ∙ Лев Черняховский\n" \
           " ∙ Александр Орлов\n" \
           " ∙ Павел Яковлев"
    text += "\n\nЗадача командного этапа - ИИ в геосервисах"

    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=['classlist'])
def print_classes(message):
    text = "Перед вами 183 класса, на которые бот может классифицировать изображения:\n"
    label2int, _ = get_label_replacers()
    text += '\n'.join(label2int.keys())
    bot.send_message(message.chat.id, text)


@bot.callback_query_handler(func=lambda call: not call.data.startswith('~'))
def da(call):
    subtitle, img_url, link, title = get_all(call.data)
    kb = types.InlineKeyboardMarkup(row_width=1)
    kb.add(types.InlineKeyboardButton(callback_data='~' + call.data, text='Назад'))
    kb.add(types.InlineKeyboardButton(url=link, text='Ещё подробнее'))
    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                          text="<" + call.data + ">" + '\n\n' + subtitle, reply_markup=kb)


@bot.callback_query_handler(func=lambda call: call.data.startswith('~'))
def da(call):
    kb = types.InlineKeyboardMarkup(row_width=1)
    kb.add(types.InlineKeyboardButton(callback_data=call.data[1:], text='Подробнее про знак'))
    bot.edit_message_text(chat_id=call.message.chat.id, text=call.data[1:], reply_markup=kb,
                          message_id=call.message.message_id)


@bot.message_handler(content_types=['photo'])
def send_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('imag.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)
    path = 'imag.jpg'
    kb = types.InlineKeyboardMarkup(row_width=1)
    val_transform = get_test_transform()
    text = get_class(path, m, transform=val_transform)
    kb.add(types.InlineKeyboardButton(callback_data=text, text='Подробнее'))
    bot.send_message(message.chat.id, text, reply_markup=kb)


if __name__ == "__main__":
    bot.polling(none_stop=True, interval=1)
