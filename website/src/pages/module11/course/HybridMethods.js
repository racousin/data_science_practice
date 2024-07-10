import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const HybridMethods = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Hybrid Methods in Recommendation Systems</h1>
      <p>
        This section covers hybrid methods that combine different recommendation
        techniques to improve overall performance and overcome limitations of
        individual approaches.
      </p>
      <Row>
        <Col>
          <h2 id="weighted-hybrid">Weighted Hybrid</h2>
          <p>
            Weighted hybrid methods combine the scores of different
            recommendation techniques using a weighted sum.
          </p>
          <CodeBlock
            code={`
function weightedHybrid(contentBasedScore, collaborativeScore, weight) {
  return weight * contentBasedScore + (1 - weight) * collaborativeScore;
}

function getHybridRecommendations(user, items, weight) {
  const recommendations = items.map(item => ({
    item: item,
    score: weightedHybrid(
      getContentBasedScore(user, item),
      getCollaborativeScore(user, item),
      weight
    )
  }));
  
  return recommendations.sort((a, b) => b.score - a.score);
}
`}
          />

          <h2 id="switching-hybrid">Switching Hybrid</h2>
          <p>
            Switching hybrid methods choose between different recommendation
            techniques based on certain criteria.
          </p>
          <CodeBlock
            code={`
function switchingHybrid(user, item, threshold) {
  const collaborativeScore = getCollaborativeScore(user, item);
  
  if (collaborativeScore > threshold) {
    return collaborativeScore;
  } else {
    return getContentBasedScore(user, item);
  }
}

function getSwitchingHybridRecommendations(user, items, threshold) {
  return items.map(item => ({
    item: item,
    score: switchingHybrid(user, item, threshold)
  })).sort((a, b) => b.score - a.score);
}
`}
          />

          <h2 id="feature-combination">Feature Combination</h2>
          <p>
            Feature combination methods merge features from different
            recommendation approaches into a single recommendation algorithm.
          </p>
          <CodeBlock
            code={`
function combineFeatures(contentFeatures, collaborativeFeatures) {
  return [...contentFeatures, ...collaborativeFeatures];
}

function getFeatureCombinationRecommendations(user, items) {
  const combinedFeatures = items.map(item => ({
    item: item,
    features: combineFeatures(
      getContentFeatures(item),
      getCollaborativeFeatures(user, item)
    )
  }));
  
  return runRecommendationAlgorithm(user, combinedFeatures);
}
`}
          />

          <h2 id="implementing-hybrid">Implementing Hybrid Methods</h2>
          <p>
            Here's an example of implementing a hybrid recommendation system
            that combines content-based and collaborative filtering approaches.
          </p>
          <CodeBlock
            code={`
class HybridRecommender {
  constructor(contentBasedRecommender, collaborativeRecommender, weight) {
    this.contentBasedRecommender = contentBasedRecommender;
    this.collaborativeRecommender = collaborativeRecommender;
    this.weight = weight;
  }

  recommend(user, items) {
    const contentBasedScores = this.contentBasedRecommender.getScores(user, items);
    const collaborativeScores = this.collaborativeRecommender.getScores(user, items);

    const hybridScores = items.map(item => ({
      item: item,
      score: this.weight * contentBasedScores[item.id] + 
             (1 - this.weight) * collaborativeScores[item.id]
    }));

    return hybridScores.sort((a, b) => b.score - a.score);
  }
}

// Usage
const hybridRecommender = new HybridRecommender(
  new ContentBasedRecommender(),
  new CollaborativeRecommender(),
  0.6  // 60% weight on content-based, 40% on collaborative
);

const recommendations = hybridRecommender.recommend(user, items);
`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default HybridMethods;
