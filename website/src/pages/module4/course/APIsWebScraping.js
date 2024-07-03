import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const APIsWebScraping = () => {
  return (
    <Container fluid>
      <h1 className="my-4">APIs and Web Scraping</h1>

      <section>
        <h2 id="restful-apis">RESTful APIs</h2>
        <p>
          RESTful APIs (Representational State Transfer) are a standard way for
          systems to communicate over the internet. They allow you to retrieve
          and manipulate data from web services.
        </p>

        <h3>Authentication Methods</h3>
        <p>
          Most APIs require authentication to ensure secure access to data.
          Common authentication methods include:
        </p>
        <ul>
          <li>API Keys</li>
          <li>OAuth</li>
          <li>JSON Web Tokens (JWT)</li>
          <li>Basic Authentication</li>
        </ul>

        <h3>Making API Requests with Python (requests library)</h3>
        <CodeBlock
          language="python"
          code={`
import requests

# GET request
response = requests.get('https://api.example.com/data')
print(response.json())

# POST request with authentication
api_key = 'your_api_key'
headers = {'Authorization': f'Bearer {api_key}'}
data = {'key': 'value'}
response = requests.post('https://api.example.com/create', json=data, headers=headers)
print(response.status_code)

# Handling errors
try:
    response = requests.get('https://api.example.com/data')
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as err:
    print(f"HTTP error occurred: {err}")
except requests.exceptions.RequestException as err:
    print(f"An error occurred: {err}")
          `}
        />

        <h3>Parsing JSON Responses</h3>
        <CodeBlock
          language="python"
          code={`
import json

# Parse JSON response
response = requests.get('https://api.example.com/data')
data = response.json()

# Access nested data
name = data['user']['name']
print(name)

# Handle potential KeyError
age = data['user'].get('age', 'Unknown')
print(age)

# Convert JSON to pandas DataFrame
import pandas as pd
df = pd.DataFrame(data['results'])
print(df.head())
          `}
        />
      </section>

      <section>
        <h2 id="web-scraping">Web Scraping Basics</h2>
        <p>
          Web scraping is the process of extracting data from websites. It's
          useful when data is not available through an API.
        </p>

        <h3>HTML Structure and CSS Selectors</h3>
        <p>
          Understanding HTML structure and CSS selectors is crucial for
          effective web scraping.
        </p>
        <CodeBlock
          language="html"
          code={`
<div class="content">
  <h1 id="title">Page Title</h1>
  <p class="text">Some text here</p>
  <ul>
    <li>Item 1</li>
    <li>Item 2</li>
  </ul>
</div>
          `}
        />
        <p>CSS Selectors:</p>
        <ul>
          <li>
            <code>.content</code>: selects elements with class "content"
          </li>
          <li>
            <code>#title</code>: selects element with id "title"
          </li>
          <li>
            <code>.content p</code>: selects all p elements inside elements with
            class "content"
          </li>
          <li>
            <code>ul li</code>: selects all li elements inside ul elements
          </li>
        </ul>

        <h3>Using BeautifulSoup</h3>
        <CodeBlock
          language="python"
          code={`
import requests
from bs4 import BeautifulSoup

# Fetch the webpage
url = 'https://example.com'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find elements
title = soup.find('h1', id='title').text
paragraphs = soup.find_all('p', class_='text')

# Extract data from a table
table = soup.find('table', class_='data-table')
for row in table.find_all('tr'):
    columns = row.find_all('td')
    if columns:
        data = [column.text.strip() for column in columns]
        print(data)
          `}
        />

        <h3>Using Scrapy</h3>
        <p>Scrapy is a more powerful framework for larger scraping projects.</p>
        <CodeBlock
          language="python"
          code={`
import scrapy

class ExampleSpider(scrapy.Spider):
    name = 'example'
    start_urls = ['https://example.com']

    def parse(self, response):
        for article in response.css('article.post'):
            yield {
                'title': article.css('h2::text').get(),
                'content': article.css('div.content::text').get(),
                'date': article.css('span.date::text').get(),
            }

        next_page = response.css('a.next-page::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)

# Run with: scrapy runspider example_spider.py
          `}
        />

        <h3>Ethical Considerations and Legality</h3>
        <p>
          When web scraping, it's important to consider ethical and legal
          implications:
        </p>
        <ul>
          <li>Respect robots.txt files and website terms of service</li>
          <li>Don't overload servers with too many requests</li>
          <li>Be aware of copyright and data protection laws</li>
          <li>Consider using APIs if available instead of scraping</li>
          <li>Identify your scraper in the user-agent string</li>
        </ul>
        <CodeBlock
          language="python"
          code={`
import requests

headers = {
    'User-Agent': 'YourCompany DataScienceBot 1.0',
}

response = requests.get('https://example.com', headers=headers)
          `}
        />
      </section>

      <section>
        <h2>Conclusion</h2>
        <p>
          APIs and web scraping are powerful tools for collecting data from the
          internet. APIs provide a structured way to access data from web
          services, while web scraping allows you to extract data from websites
          that don't offer an API. Both methods require careful consideration of
          authentication, error handling, and ethical practices. As a data
          scientist, mastering these techniques will greatly expand your ability
          to gather diverse datasets for analysis and modeling.
        </p>
      </section>
    </Container>
  );
};

export default APIsWebScraping;
