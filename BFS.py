

import requests
from bs4 import BeautifulSoup
PATH = 'https://google.com'

def Crawl(path, level):
	queue = [path]
	while (queue) and (level):
		currPath = queue.pop(0)
		level -= 1
		try:
			if (str(currPath).startswith('http')):
				text = requests.get(currPath).text
				s = BeautifulSoup(text, "html.parser")
				for link in s.findAll('a'):
				   new_url = link.get('href')
				   print(new_url)
				   queue.append(new_url);
				print("-----------------------------------------")
		except Exception as e:
			print("Unexpected error:", e)

Crawl(PATH, 3)