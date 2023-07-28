import threading
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import concurrent.futures


global_driver = webdriver.Chrome()
all_pages = []
file_lock = threading.Lock()


def get_data(url):
    driver = webdriver.Chrome()

    driver.get(url)
    time.sleep(2)
    to_write = ""
    tables = driver.find_elements(By.CSS_SELECTOR, ".wrapped.fixed-table.confluenceTable")
    if len(tables) == 1:
        table = tables[0]
        rows = table.find_elements(By.CSS_SELECTOR, "tr")
        for x in rows:
            data = x.find_elements(By.CSS_SELECTOR, "td")
            if len(data) == 2:
                to_write += f"Header: {data[0].text}\n"
                if data[1].text != "":
                    to_write += f"Data: {data[1].text}\n"
                else:
                    to_write += "Data: EMPTY\n"
            to_write += "------------------------------------------------\n"

    write_to_file(to_write)
    driver.quit()


def expand_all():
    global_driver.get("https://wiki.logo.com.tr/display/PEOP/Peoplise")
    time.sleep(5)
    all_clickables = global_driver.find_elements(By.CLASS_NAME,
                                                 'plugin_pagetree_childtoggle.aui-icon.aui-icon-small.aui-iconfont-chevron-right')
    not_expanded = [elements for elements in all_clickables if elements.get_attribute('aria-expanded') == 'false']
    while len(not_expanded) > 0:
        for x in not_expanded:
            x.click()
            time.sleep(1)
        time.sleep(2)
        all_clickables = global_driver.find_elements(By.CLASS_NAME,
                                                     'plugin_pagetree_childtoggle.aui-icon.aui-icon-small.aui-iconfont-chevron-right')
        not_expanded = [elements for elements in all_clickables if elements.get_attribute('aria-expanded') == 'false']
    pages = global_driver.find_elements(By.CLASS_NAME, "plugin_pagetree_children_span")
    for x in pages:
        link = x.find_elements(By.CSS_SELECTOR, "a")
        if len(link) > 0:
            all_pages.append(link[0].get_attribute("href"))
            print(link[0].get_attribute("href"))


def write_to_file(data):
    with file_lock:
        with open("output5.txt", "w", encoding='utf-8') as file:
            file.write(data)


if __name__ == '__main__':
    #expand_all()
    #with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
     #   while len(all_pages) > 0:
      #      executor.submit(get_data, all_pages.pop())
    get_data('https://wiki.logo.com.tr/pages/viewpage.action?pageId=143361433')
    global_driver.quit()
