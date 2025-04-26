"""Script to fetch all pages content."""
import requests

wiki_url = "https://awoiaf.westeros.org/api.php?action=query&format=json&list=allpages"
response = requests.request("GET", wiki_url)

print(response.text)
