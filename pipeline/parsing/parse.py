import requests
from bs4 import BeautifulSoup
import csv
import os

URL = 'https://pdd.am.ru/road-signs/'

signs_types = ['preduprezhdajushhie-znaki/', 'znaki-prioriteta/', 'zapreshhajushhie-znaki/', 'predpisyvajushhie-znaki/',
               'znaki-osobyh-predpisanij/', 'informacionnye-znaki/', 'znaki-servisa/',
               'znaki-dopolnitelnoj-informacii/']

HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'accept': '*/*'}

d = {'1': 'Предупреждающие знаки', '2': 'Знаки приоритета', '3': 'Запрещающие знаки', '4': 'Предписывающие знаки',
     '5': 'Знаки особых предписаний', '6': 'Информационные знаки', '7': 'Знаки сервиса'}

trans_img = {'5.15.7': 'https://petroznak.ru/image/cache/catalog/znak/dorozhnyj-znak-5.15.7-700x700.png',
             '5.15.2':'https://www.signal-doroga.ru/upload/shop_1/2/0/2/item_2025/shop_items_catalog_image2025.png'}


def get_l(sign):
    if ',' in sign:
        l = sign.split(',')
    elif '-' in sign:
        first, last = sign.split('-')
        first, last = first.strip(), last.strip()
        *_, ft = first.split('.')
        *_, lt = last.split('.')
        l = ['.'.join(_) + '.' + str(h) for h in range(int(ft), int(lt) + 1)]
    else:
        l = [sign]
    return l


def get_html(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    return r


def get_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('tr', class_='au-accordion-trigger')
    signs = []

    for item in items:
        sign = item.find_next().get_text(strip=True)
        l = get_l(sign)
        imgs = item.find_all('img')
        for i in range(len(l)):
            if l[i] in ('5.15.7', '5.15.2'):
                img_url = trans_img[l[i]]
            else:
                img_url = imgs[i].get('src')
            signs.append({
                'sign': l[i].strip()+'.',
                'img_url': img_url,
                'title': ("Знак "+sign+': \n'+item.find('h3', class_='au-accordion__header').get_text(strip=True))
            })

    return signs


def save_file(items, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['sign', 'title', 'img'])
        writer.writerows([[i['sign'], i['title'], i['img_url']] for i in items])


def parse():
    signs = []
    for type in signs_types:
        html = get_html(URL + type)
        if html.status_code:
            content = get_content(html.text)
            signs.extend(content)
            print(f'Parsing {type[:-1]}')
        else:
            print('error')
    print(len(signs))
    for i in signs:
        print(i)
    save_file(signs, '../../data/signs2.csv')
    os.startfile('../../data/signs2.csv')


parse()
