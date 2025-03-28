import React, { useState, useEffect } from 'react';
import { Card, Table, Title, ScrollArea, Divider, Text } from '@mantine/core';
import {scrapableDataCourse} from './ScrapableDataCourse'
import {scrapableDataExercise} from './ScrapableDataExercise'


  const ScrapableData = () => {
    const [courseData, setCourseData] = useState([]);
    const [exerciseData, setExerciseData] = useState([]);
  
    // Process data on component mount
    useEffect(() => {
      const processedDataCourse = scrapableDataCourse.map(item => ({
        id: item.product_id,
        rating: item.rating,
        num_reviews: item.num_reviews,
        ts: 1728482 + Math.floor(Math.random() * 100),
      }));

      const processedDataExercise = scrapableDataExercise.map(item => ({
        id: item.item_code,
        rating: item.customer_score,
        num_reviews: item.total_reviews,
        ts: 1728482 + Math.floor(Math.random() * 100),
      }));
  
      // Split data into course and exercise
      setCourseData(processedDataCourse);
      setExerciseData(processedDataExercise);
    }, []);
  
    return (
      <div style={{ padding: '2rem' }}>
        <Title order={2} align="center" mb="lg">
          Scrapable Data
        </Title>
  
        {/* Course Section */}
        <Card shadow="sm" p="lg" radius="md" withBorder>
          <Text size="lg" weight={500} mb="md">
            Course Data
          </Text>
          <ScrollArea style={{ height: 300 }}>
            <Table highlightOnHover>
              <thead>
                <tr>
                  <th>Product ID</th>
                  <th>Rating</th>
                  <th>Number of Reviews</th>
                  <th>Updated Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {courseData.map((item) => (
                  <tr key={item.id}>
                    <td>{item.id}</td>
                    <td>{item.rating}</td>
                    <td>{item.num_reviews}</td>
                    <td>{item.ts}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </ScrollArea>
        </Card>
  
        {/* Divider between sections */}
        <Divider my="xl" />
  
        {/* Exercise Section */}
        <Card shadow="sm" p="lg" radius="md" withBorder>
          <Text size="lg" weight={500} mb="md">
            Exercise Data
          </Text>
          <ScrollArea style={{ height: 300 }}>
            <Table highlightOnHover>
              <thead>
                <tr>
                  <th>Item Code</th>
                  <th>Customer Score</th>
                  <th>Total Reviews</th>
                  <th>Updated Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {exerciseData.map((item) => (
                  <tr key={item.id}>
                    <td>{item.id}</td>
                    <td>{item.rating}</td>
                    <td>{item.num_reviews}</td>
                    <td>{item.ts}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </ScrollArea>
        </Card>
      </div>
    );
  };
  
  export default ScrapableData;