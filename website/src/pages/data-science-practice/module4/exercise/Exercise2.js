import React from 'react';
import { Container, Title, Text, List, Stack, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const Exercise2 = () => {
  return (
    <Container fluid py="xl">
      <Title order={1} mb="md">Exercise 2: Model Deployment with Flask API</Title>

      <Stack spacing="xl">
        <Text>
          In this exercise, you will deploy the model you created in Exercise 1 by building a simple Flask API.
          You'll learn how to serve your model locally and interact with it through HTTP requests.
        </Text>

        <Stack spacing="md">
          <Title order={2} size="h3">Learning Objectives</Title>
          <List spacing="sm">
            <List.Item>Create a REST API endpoint using Flask</List.Item>
            <List.Item>Load and serve your trained model</List.Item>
            <List.Item>Handle HTTP POST requests with JSON data</List.Item>
            <List.Item>Test your API using a web browser interface</List.Item>
          </List>
        </Stack>

        <Stack spacing="md">
          <Title order={2} size="h3">Part 1: Setting Up Your Flask Application</Title>

          <Text>Create a new file called <Code>app.py</Code> in your module4 directory with the following structure:</Text>

          <CodeBlock
            language="python"
            code={`from flask import Flask, request, jsonify
import pickle
import pandas as pd`}
          />

          <Text>Initialize your Flask application:</Text>

          <CodeBlock
            language="python"
            code={`app = Flask(__name__)
model = None  # Will load your model here`}
          />
        </Stack>

        <Stack spacing="md">
          <Title order={2} size="h3">Part 2: Loading Your Model</Title>

          <Text>Save your trained model from Exercise 1 using pickle:</Text>

          <CodeBlock
            language="python"
            code={`# In your Exercise 1 notebook
import pickle
pickle.dump(model, open('model.pkl', 'wb'))`}
          />

          <Text>Load the model in your Flask app:</Text>

          <CodeBlock
            language="python"
            code={`def load_model():
    global model
    model = pickle.load(open('model.pkl', 'rb'))`}
          />
        </Stack>

        <Stack spacing="md">
          <Title order={2} size="h3">Part 3: Creating the Prediction Endpoint</Title>

          <Text>Implement a POST endpoint that accepts item features and returns predictions:</Text>

          <CodeBlock
            language="python"
            code={`@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    # Convert to DataFrame
    # Make prediction
    # Return result as JSON`}
          />

          <Text>Expected input format (JSON):</Text>

          <CodeBlock
            language="json"
            code={`{
  "item_code": "A001",
  "mass": 250,
  "dimension_length": 15.5,
  "dimension_width": 10.2,
  "dimension_height": 5.0,
  "customer_score": 4.2,
  "total_reviews": 125,
  "unit_cost": 9.99,
  "package_volume": 790.5,
  "stock_age": 30,
  "days_since_last_purchase": 5
}`}
          />
        </Stack>

        <Stack spacing="md">
          <Title order={2} size="h3">Part 4: Creating a Simple Web Interface</Title>

          <Text>Create an HTML form to interact with your API. Create <Code>templates/index.html</Code>:</Text>

          <CodeBlock
            language="html"
            code={`<!DOCTYPE html>
<html>
<head>
    <title>Sales Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; }
        input { width: 100%; padding: 8px; margin: 5px 0 15px; box-sizing: border-box; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        #result { margin-top: 20px; padding: 10px; background-color: #f0f0f0; display: none; }
    </style>
</head>
<body>
    <h1>Predict Quantity Sold</h1>
    <form id="prediction-form">
        <label>Item Code:</label>
        <input type="text" id="item_code" required>

        <label>Mass (g):</label>
        <input type="number" id="mass" step="0.1" required>

        <label>Length (cm):</label>
        <input type="number" id="dimension_length" step="0.1" required>

        <label>Width (cm):</label>
        <input type="number" id="dimension_width" step="0.1" required>

        <label>Height (cm):</label>
        <input type="number" id="dimension_height" step="0.1" required>

        <label>Customer Score:</label>
        <input type="number" id="customer_score" step="0.1" min="0" max="5" required>

        <label>Total Reviews:</label>
        <input type="number" id="total_reviews" min="0" required>

        <label>Unit Cost ($):</label>
        <input type="number" id="unit_cost" step="0.01" required>

        <label>Package Volume (cm³):</label>
        <input type="number" id="package_volume" step="0.1" required>

        <label>Stock Age (days):</label>
        <input type="number" id="stock_age" min="0" required>

        <label>Days Since Last Purchase:</label>
        <input type="number" id="days_since_last_purchase" min="0" required>

        <button type="submit">Predict</button>
    </form>
    <div id="result"></div>

    <script>
        document.getElementById('prediction-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const data = {
                item_code: document.getElementById('item_code').value,
                mass: parseFloat(document.getElementById('mass').value),
                dimension_length: parseFloat(document.getElementById('dimension_length').value),
                dimension_width: parseFloat(document.getElementById('dimension_width').value),
                dimension_height: parseFloat(document.getElementById('dimension_height').value),
                customer_score: parseFloat(document.getElementById('customer_score').value),
                total_reviews: parseInt(document.getElementById('total_reviews').value),
                unit_cost: parseFloat(document.getElementById('unit_cost').value),
                package_volume: parseFloat(document.getElementById('package_volume').value),
                stock_age: parseInt(document.getElementById('stock_age').value),
                days_since_last_purchase: parseInt(document.getElementById('days_since_last_purchase').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').innerHTML =
                    '<h3>Prediction Result</h3>' +
                    '<p>Predicted Quantity: ' + result.quantity_sold + '</p>';
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>`}
          />

          <Text>Add a route to serve the HTML page:</Text>

          <CodeBlock
            language="python"
            code={`@app.route('/')
def home():
    return render_template('index.html')`}
          />
        </Stack>

        <Stack spacing="md">
          <Title order={2} size="h3">Part 5: Running and Testing Your API</Title>

          <Text>Start your Flask server:</Text>

          <CodeBlock
            language="bash"
            code={`python app.py`}
          />

          <Text>Test your API using curl:</Text>

          <CodeBlock
            language="bash"
            code={`curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "item_code": "A001",
    "mass": 250.5,
    "dimension_length": 15.5,
    "dimension_width": 10.2,
    "dimension_height": 5.0,
    "customer_score": 4.2,
    "total_reviews": 125,
    "unit_cost": 9.99,
    "package_volume": 790.5,
    "stock_age": 30,
    "days_since_last_purchase": 5
  }'`}
          />

          <Text>Or open your browser and navigate to <Code>http://localhost:5000</Code> to use the web interface.</Text>
        </Stack>


      </Stack>
    </Container>
  );
};

export default Exercise2;