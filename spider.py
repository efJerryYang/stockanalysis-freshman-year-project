import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from random import random

StockName_DataSource_DownloadPath = ['PBR',
                                     'https://finance.yahoo.com/',
                                     os.path.join(os.getcwd(), 'ProjectStorage', 'StockData', 'Full')
                                     ]  # 其实我觉得这个列表改成字典更好，不过不影响效果


# 10-19 s
def Yahoofinance_spider_download(StockName_DataSource_DownloadPath):
    print('.', end='')
    desired_capabilities = DesiredCapabilities.CHROME  # 修改页面加载策略，不等待页面加载完成
    desired_capabilities["pageLoadStrategy"] = "none"  # 修改策略从normal为none
    company, url, download_path = StockName_DataSource_DownloadPath

    if not os.path.exists(download_path):  # 如果路径不存在就创建
        os.mkdir(download_path)
    options = webdriver.ChromeOptions()
    prefs = {'profile.default_content_settings.popups': 0,
             'download.default_directory': download_path}
    options.add_experimental_option('prefs', prefs)
    options.headless = True  # 开启无界面模式
    browser = webdriver.Chrome(options=options)
    browser.implicitly_wait(5)
    browser.get(url)
    # 开始查找
    print('.', end='')

    input = browser.find_element(By.XPATH, '//input[@id="yfin-usr-qry"]')
    # time.sleep(random() / 2)
    print('.', end='')

    input.send_keys(company + '\n')
    WebDriverWait(browser, 20).until(EC.element_to_be_clickable(
        (By.XPATH, '//li[@data-test="HISTORICAL_DATA"]/a/span'))).click()
    print('.', end='')
    WebDriverWait(browser, 20).until(EC.element_to_be_clickable(
        (By.XPATH, "//section[@data-test='qsp-historical']//div[@data-test='dropdown']/div/span"))).click()
    print('.', end='')

    # time.sleep(random() / 2)
    button2 = browser.find_element(By.XPATH, '//button[@data-value="MAX"]')
    browser.execute_script("arguments[0].click();", button2)
    WebDriverWait(browser, 20).until(EC.element_to_be_clickable(
        (By.XPATH,
         '//button[@class=" Bgc($linkColor) Bdrs(3px) Px(20px) Miw(100px) Whs(nw) Fz(s) Fw(500) C(white) Bgc($linkActiveColor):h Bd(0) D(ib) Cur(p) Td(n)  Py(9px) Fl(end)"]')))
    button3 = browser.find_element(
        By.XPATH,
        '//button[@class=" Bgc($linkColor) Bdrs(3px) Px(20px) Miw(100px) Whs(nw) Fz(s) Fw(500) C(white) Bgc($linkActiveColor):h Bd(0) D(ib) Cur(p) Td(n)  Py(9px) Fl(end)"]')
    # time.sleep(random() / 2)
    browser.execute_script("arguments[0].click();", button3)
    # 通过xpath找到download对应的网页对象
    WebDriverWait(browser, 180).until(EC.presence_of_element_located(
        (By.XPATH, '//a[@class="Fl(end) Mt(3px) Cur(p)"]')))
    url_download = browser.find_element(
        By.XPATH, '//a[@class="Fl(end) Mt(3px) Cur(p)"]')
    # 打印url_download对象中href的属性值
    # print(url_download.get_attribute('href'))
    title = browser.find_element(By.XPATH, "//h1[@class='D(ib) Fz(18px)']")
    company, abbreviation = title.text.strip(')').split('(')
    StockName_DataSource_DownloadPath[0] = company.strip()  # 将公司名称修改为列表的第0项

    filename = abbreviation + '.csv'  # or extract it dynamically from the link
    filepath = os.path.join(download_path, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    browser.execute_script("arguments[0].click();", url_download)
    while not os.path.exists(filepath):
        time.sleep(1)
    print('.', end='')
    browser.quit()  # 只有当下载完成才quit
    return abbreviation

if __name__ == '__main__':
    start = time.time()
    Yahoofinance_spider_download(StockName_DataSource_DownloadPath)
    print(StockName_DataSource_DownloadPath)
    print(f'run time:{time.time() - start} s')
