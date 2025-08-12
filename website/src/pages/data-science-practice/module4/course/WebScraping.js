import React from 'react';
import { Container, Title, Text, Accordion, Stack, List, Paper, Alert } from '@mantine/core';
import { Globe, Shield, Code, FileSearch } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const SectionIcon = ({ type }) => {
  const iconProps = { size: 24, strokeWidth: 1.5 };
  switch (type) {
    case 'overview': return <Globe {...iconProps} />;
    case 'ethics': return <Shield {...iconProps} />;
    case 'implementation': return <Code {...iconProps} />;
    default: return null;
  }
};

const ScrapingSection = ({ type, title, content }) => (
  <Accordion.Item value={type}>
    <Accordion.Control icon={<SectionIcon type={type} />}>
      {title}
    </Accordion.Control>
    <Accordion.Panel>
      <Stack gap="md">
        {content}
      </Stack>
    </Accordion.Panel>
  </Accordion.Item>
);

const WebScraping = () => {
  const sections = [
    {
      type: 'overview',
      title: 'Web Scraping Fundamentals',
      content: (
        <>
          <Text>
            Web scraping extracts data from websites when APIs aren't available. It involves parsing HTML content to collect structured data for analysis.
          </Text>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Paper p="md" className="bg-slate-50">
              <Stack gap="sm">
                <Title order={4}>Key Components</Title>
                <List>
                  <List.Item><strong>HTML Parser:</strong> BeautifulSoup or lxml</List.Item>
                  <List.Item><strong>HTTP Client:</strong> requests library</List.Item>
                  <List.Item><strong>Selectors:</strong> CSS or XPath queries</List.Item>
                </List>
              </Stack>
            </Paper>
            <Paper p="md" className="bg-slate-50">
              <Stack gap="sm">
                <Title order={4}>Common Use Cases</Title>
                <List>
                  <List.Item>Product price monitoring</List.Item>
                  <List.Item>Research data collection</List.Item>
                  <List.Item>Content aggregation</List.Item>
                </List>
              </Stack>
            </Paper>
          </div>
        </>
      )
    },
    {
      type: 'ethics',
      title: 'Legal & Ethical Guidelines',
      content: (
        <>
          <Alert icon={<Shield size={16} />} color="blue">
            Before scraping any website, check the robots.txt file and Terms of Service for scraping policies.
          </Alert>
          <Paper p="md" className="bg-slate-50">
            <Stack gap="sm">
              <Title order={4}>Best Practices</Title>
              <List>
                <List.Item><strong>Rate Limiting:</strong> Space out requests (1-2 seconds between calls)</List.Item>
                <List.Item><strong>User Agent:</strong> Identify your scraper in request headers</List.Item>
                <List.Item><strong>Data Usage:</strong> Respect copyright and personal data protection</List.Item>
                <List.Item><strong>Server Load:</strong> Minimize impact on the website's resources</List.Item>
              </List>
            </Stack>
          </Paper>
        </>
      )
    },
    {
      type: 'implementation',
      title: 'Practical Example: Product Price Tracker',
      content: (
        <>
          <Text>
            Let's build a practical price tracker that monitors product prices from an e-commerce website. This example demonstrates proper scraping techniques and data handling.
          </Text>
          <CodeBlock
            language="python"
            code={`import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

class PriceTracker:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Price Tracker Bot 1.0 (Educational Project)',
            'Accept': 'text/html,application/xhtml+xml'
        }
        self.data = []

    def fetch_product_data(self, url):
        """Fetch and parse product data from URL."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            return {
                'title': soup.select_one('h1.product-title').text.strip(),
                'price': self._extract_price(soup.select_one('span.price')),
                'rating': soup.select_one('div.rating').text.strip(),
                'stock': soup.select_one('span.stock-status').text.strip(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def _extract_price(self, element):
        """Extract and clean price value."""
        if not element:
            return None
        # Remove currency symbol and convert to float
        return float(element.text.strip().replace('$', '').replace(',', ''))

    def track_products(self, urls, interval_seconds=3600):
        """Track multiple products over time."""
        while True:
            for url in urls:
                data = self.fetch_product_data(url)
                if data:
                    self.data.append(data)
                time.sleep(2)  # Respect rate limiting
            
            # Save current data
            self._save_data()
            
            # Wait for next check
            time.sleep(interval_seconds)

    def _save_data(self):
        """Save tracked data to CSV."""
        df = pd.DataFrame(self.data)
        df.to_csv('price_history.csv', index=False)
        print(f"Data saved: {len(self.data)} records")

# Usage Example
if __name__ == "__main__":
    tracker = PriceTracker()
    
    # List of products to track
    products = [
        'https://example.com/product/1',
        'https://example.com/product/2'
    ]
    
    # Start tracking (check every hour)
    tracker.track_products(products, interval_seconds=3600)`}
          />
          <Title order={4} mt="md">Key Features</Title>
          <List>
            <List.Item><strong>Proper Headers:</strong> Identifies the bot and sets appropriate request headers</List.Item>
            <List.Item><strong>Error Handling:</strong> Gracefully handles failed requests and parsing errors</List.Item>
            <List.Item><strong>Rate Limiting:</strong> Implements delays between requests</List.Item>
            <List.Item><strong>Data Storage:</strong> Saves historical data in CSV format</List.Item>
            <List.Item><strong>Modular Design:</strong> Easy to extend for different websites or data types</List.Item>
          </List>
        </>
      )
    }
  ];

  return (
    <Container fluid>
      <Stack gap="xl">
        <div>
          <Title order={1}>Web Scraping</Title>
          <Text mt="md">
            Web scraping enables automated data collection from websites when APIs aren't available. This guide covers the fundamentals and best practices, with a practical example of building a price tracker.
          </Text>
        </div>

        <Accordion>
          {sections.map(section => (
            <ScrapingSection key={section.type} {...section} />
          ))}
        </Accordion>
      </Stack>
    </Container>
  );
};

export default WebScraping;