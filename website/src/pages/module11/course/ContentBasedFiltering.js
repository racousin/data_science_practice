// ContentBasedFiltering.js
import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ContentBasedFiltering = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Content-Based Filtering</h1>
      <p>
        In this section, you will learn how to build recommendation systems
        based on item features.
      </p>
      <Row>
        <Col>
          <h2>Building Item Profiles</h2>
          <p>
            Here, we discuss how to create profiles for items based on their
            features.
          </p>
          <CodeBlock
            code={`
// Example code for building item profiles
function buildItemProfile(item) {
  // Extract features from the item
  const features = extractFeatures(item);

  // Create a profile object with the features
  const profile = {
    itemId: item.id,
    features: features,
  };

  return profile;
}
            `}
          />
          <h2>Measuring Item Similarity</h2>
          <p>
            We'll cover how to measure the similarity between items using
            metrics like cosine similarity.
          </p>
          <CodeBlock
            code={`
// Example code for measuring item similarity using cosine similarity
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((acc, val, i) => acc + (val * vecB[i]), 0);
  const magnitudes = Math.sqrt(vecA.reduce((acc, val) => acc + (val * val), 0)) *
                       Math.sqrt(vecB.reduce((acc, val) => acc + (val * val), 0)));
  return dotProduct / magnitudes;
}
            `}
          />
          <h2>Personalizing Recommendations</h2>
          <p>
            Finally, we'll explore how to personalize recommendations for users
            based on their preferences.
          </p>
          <CodeBlock
            code={`
// Example code for personalizing recommendations
function recommendItems(user, itemProfiles) {
  // Extract the user's preferences
  const preferences = extractPreferences(user);

  // Calculate the similarity between the user's preferences and each item profile
  const similarities = itemProfiles.map(profile => cosineSimilarity(preferences, profile.features));

  // Sort the item profiles by similarity and return the top N items
  const sortedProfiles = itemProfiles.sort((a, b) => similarities[b.id] - similarities[a.id]);
  return sortedProfiles.slice(0, N);
}
            `}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default ContentBasedFiltering;
