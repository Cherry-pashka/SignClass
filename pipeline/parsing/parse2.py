import csv
import os
import re

import requests
from bs4 import BeautifulSoup

URL = 'https://www.drom.ru/pdd/pdd/signs/'
HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'accept': '*/*'}

d = {'1': 'Предупреждающие знаки', '2': 'Знаки приоритета', '3': 'Запрещающие знаки', '4': 'Предписывающие знаки',
     '5': 'Знаки особых предписаний', '6': 'Информационные знаки', '7': 'Знаки сервиса'}


def get_html(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    return r


def get_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('div', class_='pub')
    cars = []

    for item in items:
        title = item.find('h4', class_='b-title')
        if title:
            title = title.get_text(strip=True)
            signs = re.findall(r'\d[\d+\.?]+', title)
            img = item.find('img').get('src')  # для всех кроме 8.
            describe = ' '.join(re.findall(r'[А-Яа-я]+', title)[1:])
            for j in range(len(signs)):
                if signs[j][-1] != '.':
                    signs[j] += '.'
                cars.append({
                    'sign': signs[j],
                    'title': describe,
                    'link': URL + '#' + item.get('id'),
                    'img': img


                })
    for i in cars:
        print(i)
    return cars


def save_file(items, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['sign', 'title', 'link', 'img'])
        writer.writerows([[i['sign'], i['title'], i['link'], i['img']] for i in items])


def parse():
    html = get_html(URL)
    if html.status_code:
        signs = get_content(html.text)
        print(len(signs))
        save_file(signs, '.../data/signs.csv')
        os.startfile('../data/signs.csv')

    else:
        print('error')


parse()
