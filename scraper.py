import requests
from bs4 import BeautifulSoup
import csv

# List of news websites to scrape
urls = ['https://www.bbc.com/news', 'https://www.nytimes.com/', 'https://www.reuters.com/']

# Create an empty list to store the data
articles = []

# Loop through each URL
for url in urls:
    # Send a GET request to the URL and get the response
    response = requests.get(url)
    
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'lxml')
    
    # Find all the article elements on the page
    articles_on_page = soup.find_all('article')
    
    # Loop through each article and extract the data
    for article in articles_on_page:
        # Get the headline
        headline = article.find('h1').get_text().strip()
        
        # Get the summary
        summary = article.find('p').get_text().strip()
        
        # Get the date
        date = article.find('time')['datetime']
        
        # Get the source URL
        source_url = url
        
        # Add the data to the list
        articles.append([headline, summary, date, source_url])

# Create a CSV file to store the data
with open('news.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['Headline', 'Summary', 'Date', 'Source URL'])
    
    # Write each article to a new row
    for article in articles:
        writer.writerow(article)
