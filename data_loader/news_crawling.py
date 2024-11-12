import requests
from bs4 import BeautifulSoup

def get_naver_finance_news():
    url = 'https://finance.naver.com/news/'  # 네이버 증권 뉴스 페이지 URL
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}

    # HTTP GET 요청
    response = requests.get(url, headers=headers)
    response.encoding = 'euc-kr'  # 네이버 금융 사이트는 euc-kr 인코딩 사용

    # HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 뉴스 제목과 링크 추출
    news_list = soup.select('div.newsList > ul > li')
    news_data = []

    for news in news_list:
        title = news.select_one('a').get_text(strip=True)
        link = 'https://finance.naver.com' + news.select_one('a')['href']
        news_data.append({"title": title, "link": link})

    return news_data

# 실행
news_data = get_naver_finance_news()
for news in news_data:
    print("제목:", news["title"])
    print("링크:", news["link"])
    print('-' * 50)
