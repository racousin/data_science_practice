import React from 'react';
import { Container, Title, Text, Accordion, Stack, List, Paper, Alert } from '@mantine/core';
import { Globe, Shield, Code, FileSearch } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const SectionIcon = ({ type }) => {
  const iconProps = { size: 24, strokeWidth: 1.5 };
  switch (type) {
    case 'overview': return <Globe {...iconProps} />;
    case 'ethics': return <Shield {...iconProps} />;
    case 'extracting': return <Code {...iconProps} />;
    case 'examples': return <FileSearch {...iconProps} />;
    default: return null;
  }
};

const ScrapingSection = ({ type, title, content }) => (
  <Accordion.Item value={type}>
    <Accordion.Control icon={<SectionIcon type={type} />}>
      {title}
    </Accordion.Control>
    <Accordion.Panel>
      <Stack spacing="md">
        {content}
      </Stack>
    </Accordion.Panel>
  </Accordion.Item>
);

const WebScraping = () => {
  const sections = [
    {
      type: 'overview',
      title: 'High-level Overview',
      content: (
        <>
          <Text>
            Web scraping is the process of automatically extracting data from websites. It's a powerful technique for collecting data that's not available through APIs or other structured formats.
          </Text>
          <Title order={4}>Key Concepts:</Title>
          <List>
            <List.Item>HTML parsing: Understanding and navigating the structure of web pages</List.Item>
            <List.Item>HTTP requests: Fetching web pages programmatically</List.Item>
            <List.Item>Data extraction: Identifying and extracting relevant information from HTML</List.Item>
            <List.Item>Automation: Scraping multiple pages or websites efficiently</List.Item>
          </List>
          <Text mt="md">
            Common libraries for web scraping in Python include:
          </Text>
          <List>
            <List.Item>Beautiful Soup: For parsing HTML and XML documents</List.Item>
            <List.Item>Scrapy: A comprehensive framework for large-scale web scraping</List.Item>
            <List.Item>Selenium: For scraping dynamic websites that require JavaScript rendering</List.Item>
          </List>
        </>
      )
    },
    {
      type: 'ethics',
      title: 'Ethical Considerations',
      content: (
        <>
          <Text>
            While web scraping can be a powerful tool for data collection, it's crucial to consider the ethical and legal implications of your scraping activities.
          </Text>
          <Title order={4}>Ethical Guidelines:</Title>
          <List>
            <List.Item>Respect robots.txt: Check and adhere to the website's robots.txt file, which specifies which parts of the site can be crawled</List.Item>
            <List.Item>Rate limiting: Implement reasonable rate limits to avoid overloading the server</List.Item>
            <List.Item>Identify yourself: Use a custom User-Agent string to identify your bot</List.Item>
            <List.Item>Respect copyright: Be aware of and comply with copyright laws regarding the scraped content</List.Item>
            <List.Item>Personal data: Be cautious when scraping personal information and comply with data protection regulations (e.g., GDPR)</List.Item>
          </List>
          <Alert icon={<Shield size={16} />} title="Legal Considerations" color="yellow" mt="md">
            Always review the website's Terms of Service before scraping. Some websites explicitly prohibit scraping, which could lead to legal issues if violated.
          </Alert>
        </>
      )
    },
    {
      type: 'extracting',
      title: 'Extracting Data from HTML Pages',
      content: (
        <>
          <Text>
            Extracting data from HTML pages involves parsing the HTML structure and using selectors to locate and extract the desired information.
          </Text>
          <Title order={4}>Common Techniques:</Title>
          <List>
            <List.Item>CSS Selectors: Target elements based on their CSS classes or IDs</List.Item>
            <List.Item>XPath: Use XML Path Language to navigate through the HTML structure</List.Item>
            <List.Item>Regular Expressions: For extracting patterns from text content</List.Item>
          </List>
          <CodeBlock
            language="python"
            code={`import requests
from bs4 import BeautifulSoup

# Fetch the webpage
url = 'https://example.com'
response = requests.get(url)
html_content = response.text

# Parse the HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Extract data using CSS selectors
title = soup.select_one('h1').text
paragraphs = [p.text for p in soup.select('p')]

# Extract data using XPath
from lxml import html
tree = html.fromstring(html_content)
links = tree.xpath('//a/@href')

# Print extracted data
print(f"Title: {title}")
print(f"Paragraphs: {paragraphs}")
print(f"Links: {links}")
`}
          />
          <Text mt="md">
            This example demonstrates how to use Beautiful Soup for CSS selector-based extraction and lxml for XPath-based extraction.
          </Text>
        </>
      )
    },
    {
      type: 'examples',
      title: 'Simple Scraping Examples',
      content: (
        <>
          <Text>
            Let's look at a couple of simple scraping examples to illustrate common use cases in data science.
          </Text>
          <Title order={4}>Example 1: Scraping a Weather Website</Title>
          <CodeBlock
            language="python"
            code={`import requests
from bs4 import BeautifulSoup

def scrape_weather(city):
    url = f'https://www.example-weather-site.com/{city}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    temperature = soup.select_one('.temperature').text
    condition = soup.select_one('.condition').text
    
    return {
        'city': city,
        'temperature': temperature,
        'condition': condition
    }

# Usage
weather_data = scrape_weather('new-york')
print(weather_data)
`}
          />
          <Text mt="md">
            This example scrapes basic weather information for a given city.
          </Text>
          <Title order={4}>Example 2: Scraping a News Website</Title>
          <CodeBlock
            language="python"
            code={`import requests
from bs4 import BeautifulSoup

def scrape_news_headlines():
    url = 'https://www.example-news-site.com'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    headlines = []
    for article in soup.select('article'):
        headline = article.select_one('h2').text
        link = article.select_one('a')['href']
        headlines.append({
            'title': headline,
            'link': link
        })
    
    return headlines

# Usage
news_headlines = scrape_news_headlines()
for headline in news_headlines:
    print(f"Title: {headline['title']}")
    print(f"Link: {headline['link']}")
    print('---')
`}
          />
          <Text mt="md">
            This example scrapes news headlines and their corresponding links from a news website's homepage.
          </Text>
          <Alert icon={<Shield size={16} />} title="Remember" color="blue" mt="md">
            These are simplified examples. In real-world scenarios, you'd need to handle errors, implement rate limiting, and possibly deal with more complex HTML structures or dynamic content.
          </Alert>
        </>
      )
    }
  ];

  return (
    <Container>
      <Title order={1}>Web Scraping</Title>
      <Text mt="md">
        Web scraping is a powerful technique for collecting data from websites, especially when the data is not available through APIs or other structured formats. It's an essential skill for data scientists, enabling them to gather diverse datasets for analysis and model training.
      </Text>
      
      <Accordion mt="xl">
        {sections.map(section => (
          <ScrapingSection key={section.type} {...section} />
        ))}
      </Accordion>

      <Text mt="xl">
        Web scraping can significantly expand your data collection capabilities, but it's important to approach it responsibly and ethically. Always consider the legal and ethical implications of your scraping activities, and strive to minimize the impact on the websites you're scraping. With the right techniques and considerations, web scraping can be a valuable tool in your data science toolkit.
      </Text>
    </Container>
  );
};

export default WebScraping;