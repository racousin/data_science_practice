import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';

const mockProductDataCourse = {

};

// Mock API call function
const fakeApiCall = (id, login, password, version) => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      // Simulate authentication
      if (login !== 'admin' || password !== 'password123') {
        // Simulate 401 Unauthorized error
        reject({
          status: 401,
          message: 'Unauthorized: Invalid login or password.',
        });
      } else {
        // Simulate different data based on version
        let responseData = {};
        if (version === 'course') {
          if (mockProductDataCourse.hasOwnProperty(id)) {
            responseData = {
              id: id,
              volume: mockProductDataCourse[id],
              version: version,
              description: `This data is for the ${version} version.`,
            };
          } else {
            reject({
              status: 404,
              message: 'Not Found: No course data available for the given ID.',
            });
          }
        } else if (version === 'exercise') {
          responseData = {
            feature_1: Math.floor(Math.random() * 50),
            feature_2: Math.floor(Math.random() * 50),
            feature_3: Math.floor(Math.random() * 50),
            id: id,
            version: 'exercise',
            description: 'This data is for the exercise version.',
          };
        } else {
          reject({
            status: 400,
            message: 'Bad Request: Invalid version parameter.',
          });
        }
        resolve(responseData);
      }
    }, 500);
  });
};

const MockApi = () => {
  const { id, version } = useParams(); // Get the id and version from the route parameters
  const [login, setLogin] = useState(''); // Login state
  const [password, setPassword] = useState(''); // Password state
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    setError(null); // Reset error state before making a new request

    // Ensure id is a number between 0 and 1000
    const numericId = Math.min(Math.max(parseInt(id, 10), 0), 1000);
    
    // Fetch the mock data based on the id, credentials, and version
    fakeApiCall(numericId, login, password, version)
      .then((response) => {
        setData(response);
        setError(null);
      })
      .catch((err) => {
        setError(err);
        setData(null);
      });
  };

  return (
    <div>
      <h1>API Data for ID: {id} and Version: {version}</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Login:
          <input
            type="text"
            value={login}
            onChange={(e) => setLogin(e.target.value)}
            required
          />
        </label>
        <br />
        <label>
          Password:
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </label>
        <br />
        <button type="submit">Fetch Data</button>
      </form>

      {error && <p style={{ color: 'red' }}>Error: {error.message}</p>}

      {data && (
        <div>
          <p>Feature 1: {data.feature_1}</p>
          <p>Feature 2: {data.feature_2}</p>
          <p>Feature 3: {data.feature_3}</p>
          <p>Description: {data.description}</p>
        </div>
      )}
    </div>
  );
};

export default MockApi;
