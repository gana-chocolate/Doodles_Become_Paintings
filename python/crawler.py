import requests
import urllib.request

from scrapy.selector import Selector
from icrawler.builtin import GoogleImageCrawler

#https://www.google.com/search?q=[검색어]&source=lnms&tbm=isch&sa=X&dpr=2&sourch=Int&tbs=sur:fc
#재사용가능 라이센스 구글 이미지 스크립트.

""" google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': 'car3'})
google_crawler.crawl(keyword='kia car', max_num=500,
                     min_size=(200,200), max_size=None) # keyword:검색어 / min_size 70x70이상 논문 권장 """

count = 2000
inputSearch = "car"
base_url = "https://www.google.com/search?q=[검색어]&source=lnms" \
           "tbm=isch&sa=1&btnG=%EA%B2%80%EC%83%89&q=" + inputSearch

def img_url_from_page(url):
    html = requests.get(url).text  # r = requests.get(url); html = r.text

    sel = Selector(text=html)

    img_names = sel.css('td a img::attr(src)').extract()

    img_names = [img_name for img_name in img_names]

    return img_names


def img_from_url(image_names):
    global count
    count += 1
    name = count

    full_name = "D:\images\img\profile_" + str(name) + ".jpg"

    urllib.request.urlretrieve(image_names, full_name)


for i in img_url_from_page(base_url):
    img_from_url(i)
