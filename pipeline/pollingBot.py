import telebot
from telebot import types

from datasets import get_test_transform
from models import get_densenet_121
from utils_ import get_class

API_TOKEN = '1724356931:AAFx-6NpgqFb7_X3DHI62H9-JZITm9fDBPg'  # token of our bot

bot = telebot.TeleBot(API_TOKEN)
val_transform = get_test_transform()
m = get_densenet_121('cpu', '../checkpoints/DENSE2(128,128).ckpt')


@bot.message_handler(commands=['start'])
def hello(message):
    """Message after command /start"""
    text = 'Добро пожаловать в наш бот по классификации знаков ПДД\nС помощью данного бота вы можете загрузить ' \
           'фотографию ' \
           "дорожных знаков 5.15.1 и 5.15.2 и получить подробнейшую информацию о нём - нужна помощь\n" \
           "/info - получить информацию про бота\n" \
           "/teaminfo - получить информацию про команду\n" \
           "/directions - вывести направления\n"
    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=['help'])
def helped(message):
    """Message after command /help"""
    text = "Чтобы узнать класс знака - просто пришлите фото знака\n\n/help - нужна помощь\n" \
           "/info - получить информацию про бота\n" \
           "/teaminfo - получить информацию про команду\n" \
           "/directions - вывести направления\n"
    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=['info'])
def information(message):
    """Message after command /info"""
    text = 'Мы команда DatenKrieg\nМы участники «Международного конкурса по искусственному интеллекту для детей» ' \
           'AIIJC\n\nДанный бот - продукт представления нашего алгоритма для задачи "ИИ в геосервисах", в который ' \
           'можно загрузить картинку ' \
           'дорожного знака и получить информацию, в каких направлениях обязывает двигаться этот знак '

    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=['teaminfo'])
def print_classes(message):
    """Message after command /teaminfo"""
    text = "Команда DatenKrieg\n\n" \
           "Участники:\n" \
           " ∙ Кирилл Пантелеев\n" \
           " ∙ Лев Черняховский\n" \
           " ∙ Александр Орлов\n" \
           " ∙ Павел Яковлев"
    text += "\n\nЗадача финального этапа - ИИ в геосервисах"

    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=['directions'])
def print_classes(message):
    """Message after command /directions"""
    text = "Наш бот распознает любые знаки и может завести вам куда угодно)"
    bot.send_message(message.chat.id, text)


@bot.callback_query_handler(func=lambda call: not call.data.startswith('~'))
def da(call):
    """Function shows additional information about predicted sign"""
    # subtitle, img_url, link, title = get_all(call.data)
    lines = call.data.lower()
    line_count = call.data.count(',') + 1
    text = f'Данный знак обязывает вас продолжить движение по ' + (
        f'одной из {line_count} полос' if line_count > 1 else 'единственной полосе') + '\n'
    # if line_count==1:
    # if call.data=='Прямо'
    lst = []
    dirs = lines.split(',')
    flag_raz = False
    if 'лево' not in dirs[0]:
        flag_raz = True
    flag_st = False
    flag_l = False
    flag_r = False
    for i in dirs:
        if 'прямо' in i:
            flag_st = True
        if 'право' in i:
            flag_r = True
        if 'лево' in i:
            flag_l = True
    if not flag_st:
        lst.append('прямо')
    if not flag_l:
        lst.append('налево')
    if not flag_r:
        lst.append('направо')
    if flag_raz or lst:
        text += 'Вы _не_ сможете '
        if lst:
            text += "проехать " + ', '.join(lst)
        if flag_raz:
            text += " и выполнить разворот"
        text+='\n'
    if 'лево' in lines or 'право' in lines:
        text += 'При поворотах на перекрестке - уступайте дорогу пешеходам\n'

    kb = types.InlineKeyboardMarkup(row_width=1)
    kb.add(types.InlineKeyboardButton(callback_data='~' + call.data, text='Назад'))
    # kb.add(types.InlineKeyboardButton(url=link, text='Ещё подробнее'))
    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                          text=f'_{call.data}_' + '\n' + text, reply_markup=kb, parse_mode='markdown')


@bot.callback_query_handler(func=lambda call: call.data.startswith('~'))
def da(call):
    """Function shows predicted label"""
    kb = types.InlineKeyboardMarkup(row_width=1)
    kb.add(types.InlineKeyboardButton(callback_data=call.data[1:], text='Подробнее про знак'))
    bot.edit_message_text(chat_id=call.message.chat.id, text=call.data[1:], reply_markup=kb,
                          message_id=call.message.message_id)


@bot.message_handler(content_types=['photo'])
def send_photo(message):
    """Function shows predicted label"""
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('imag.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)
    path = 'imag.jpg'
    kb = types.InlineKeyboardMarkup(row_width=1)

    text = get_class(path, m, transform=val_transform)
    kb.add(types.InlineKeyboardButton(callback_data=text, text='Подробнее'))
    bot.send_message(message.chat.id, text, reply_markup=kb)


if __name__ == "__main__":
    bot.polling(none_stop=True, interval=1)
