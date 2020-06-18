import os
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
driver.get("https://coursebook.utdallas.edu/search")
time.sleep(1)

driver.find_element_by_id("srch").send_keys("govt 2305\n")
driver.find_element_by_id("srch").send_keys(Keys.RETURN)
# time.sleep(3)
table = 0
while True:
    try:
        table = driver.find_element_by_xpath("//table/tbody")
        break;
    except Exception as e:
        print('wait...')
        time.sleep(0.3)
rows = table.find_elements_by_tag_name("tr")

for row in rows:
    col = row.find_elements_by_tag_name("td")[3]
    print(col.text)
