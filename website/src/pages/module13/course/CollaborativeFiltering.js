import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const CollaborativeFiltering = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Collaborative Filtering</h1>
      <p>
        In this section, you will master collaborative filtering techniques for
        recommendation systems.
      </p>
      <Row>
        <Col>
          <h2>User-Item Interactions Matrix</h2>
          <p>
            We'll start by discussing the concept of a User-Item interactions
            matrix.
          </p>
          <CodeBlock
            code={`
// Example code for creating a User-Item interactions matrix
function createInteractionsMatrix(data) {
  const users = data.map(entry => entry.userId);
  const items = data.map(entry => entry.itemId);

  // Create an empty matrix
  const matrix = users.map(() => items.map(() => 0));

  // Fill in the matrix with the interactions data
  data.forEach(entry => {
    const userId = entry.userId;
    const itemId = entry.itemId;
    const rating = entry.rating;

    matrix[userId][itemId] = rating;
  });

  return matrix;
}
`}
          />
          <h2>Memory-Based Approaches</h2>
          <p>
            We'll cover memory-based approaches to collaborative filtering,
            including user-user and item-item methods.
          </p>
          <CodeBlock
            code={`
// Example code for user-user collaborative filtering
function userSimilarity(userA, userB, data) {
  const commonItems = userA.items.filter(item => userB.items.includes(item));

  if (commonItems.length === 0) {
    return 0;
  }

  const userARatings = commonItems.map(item => data[userA.id][item.id]);
  const userBRatings = commonItems.map(item => data[userB.id][item.id]);

  return cosineSimilarity(userARatings, userBRatings);
}

function recommendItemsUsingUserSimilarity(user, data, users) {
  const userSimilarities = users.map(otherUser => userSimilarity(user, otherUser, data));

  const weightedRatings = data.map((row, userId) => {
    const similarity = userSimilarities[userId];
    return row.map(rating => rating * similarity);
  });

  const summedRatings = weightedRatings.reduce((acc, row) => acc.map((val, i) => val + row[i]), []);

  const recommendedItems = summedRatings.map((rating, itemId) => ({ itemId, rating })).sort((a, b) => b.rating - a.rating);

  return recommendedItems.slice(0, N);
}
`}
          />
          <h2>Model-Based Approaches</h2>
          <p>
            Finally, we'll explore model-based approaches to collaborative
            filtering, such as matrix factorization techniques like SVD and ALS.
          </p>
          <CodeBlock
            code={`
// Example code for matrix factorization using SVD
function recommendItemsUsingSVD(user, data, k) {
  const U = [];
  const S = [];
  const Vt = [];

  // Perform SVD on the data matrix
  const result = svd(data, k);
  U = result.U;
  S = result.S;
  Vt = result.Vt;

  // Create a reduced user matrix
  const userMatrix = U.map(row => row.slice(0, k));
  const reducedUserMatrix = userMatrix.map(row => row.map(val => val * S[row.indexOf(val)]));

  // Calculate the predicted ratings for the user
  const predictedRatings = Vt.map(row => row.slice(0, k)).map(row => {
    let rating = 0;
    row.forEach((val, i) => {
      rating += val * reducedUserMatrix[user.id][i];
    });
    return rating;
  });

  const recommendedItems = predictedRatings.map((rating, itemId) => ({ itemId, rating })).sort((a, b) => b.rating - a.rating);

  return recommendedItems.slice(0, N);
}
`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default CollaborativeFiltering;
