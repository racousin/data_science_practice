import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const EvaluationMetrics = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Evaluation Metrics for Recommendation Systems</h1>
      <p>
        Evaluating recommendation systems is crucial to understand their
        performance and effectiveness. This section covers various metrics used
        to evaluate recommendation systems.
      </p>
      <Row>
        <Col>
          <h2 id="accuracy-metrics">Accuracy Metrics</h2>
          <p>
            Accuracy metrics measure how well the recommendations match the
            user's actual preferences.
          </p>

          <h3>Mean Absolute Error (MAE)</h3>
          <p>
            MAE measures the average absolute difference between predicted and
            actual ratings.
          </p>
          <BlockMath math="MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|" />

          <h3>Root Mean Square Error (RMSE)</h3>
          <p>RMSE is similar to MAE but gives more weight to larger errors.</p>
          <BlockMath math="RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}" />

          <CodeBlock
            code={`
import numpy as np

def calculate_mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted))**2))

# Usage
actual_ratings = [4, 3, 5, 2, 1]
predicted_ratings = [3.8, 3.2, 4.7, 2.3, 1.5]

mae = calculate_mae(actual_ratings, predicted_ratings)
rmse = calculate_rmse(actual_ratings, predicted_ratings)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
`}
          />

          <h2 id="ranking-metrics">Ranking Metrics</h2>
          <p>
            Ranking metrics evaluate the quality of the ordered list of
            recommendations.
          </p>

          <h3>Precision and Recall</h3>
          <p>
            Precision measures the proportion of relevant items among the
            recommended items, while recall measures the proportion of relevant
            items that were recommended.
          </p>
          <BlockMath math="Precision@k = \frac{\text{# of relevant items in top-k recommendations}}{\text{k}}" />
          <BlockMath math="Recall@k = \frac{\text{# of relevant items in top-k recommendations}}{\text{total # of relevant items}}" />

          <h3>Mean Average Precision (MAP)</h3>
          <p>
            MAP provides a single-figure measure of quality across recall
            levels.
          </p>
          <BlockMath math="MAP = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{|R_u|} \sum_{k=1}^n P(k) \times rel(k)" />

          <CodeBlock
            code={`
def precision_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    return len(act_set & pred_set) / float(k)

def average_precision(actual, predicted):
    precisions = [precision_at_k(actual, predicted, k+1) for k in range(len(predicted))]
    return sum(precisions) / len(actual)

def mean_average_precision(actual, predicted):
    return sum(average_precision(a, p) for a, p in zip(actual, predicted)) / len(actual)

# Usage
actual_items = [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]]
predicted_items = [[1, 3, 2, 6, 4], [2, 6, 4, 8, 10]]

map_score = mean_average_precision(actual_items, predicted_items)
print(f"MAP: {map_score}")
`}
          />

          <h2 id="diversity-and-novelty">Diversity and Novelty</h2>
          <p>
            Diversity measures how different the recommended items are from each
            other, while novelty assesses how different the recommended items
            are from what the user has interacted with in the past.
          </p>

          <h3>Intra-List Diversity</h3>
          <p>
            Intra-List Diversity measures the diversity within a list of
            recommendations.
          </p>
          <BlockMath math="ILD = \frac{1}{|L|(|L|-1)} \sum_{i \in L} \sum_{j \in L, j \neq i} d(i, j)" />

          <CodeBlock
            code={`
import numpy as np

def intra_list_diversity(recommendations, similarity_matrix):
    n = len(recommendations)
    diversity_sum = 0
    for i in range(n):
        for j in range(i+1, n):
            diversity_sum += 1 - similarity_matrix[recommendations[i]][recommendations[j]]
    return (2 * diversity_sum) / (n * (n - 1))

# Usage
similarity_matrix = np.random.rand(100, 100)
recommendations = [5, 23, 41, 12, 9]
ild = intra_list_diversity(recommendations, similarity_matrix)
print(f"Intra-List Diversity: {ild}")
`}
          />

          <h2 id="user-studies">User Studies</h2>
          <p>
            User studies involve collecting feedback directly from users to
            evaluate the recommendation system.
          </p>

          <h3>A/B Testing</h3>
          <p>
            A/B testing compares two versions of the recommendation system to
            see which performs better with real users.
          </p>

          <CodeBlock
            code={`
import scipy.stats as stats

def ab_test(control_conversions, control_size, treatment_conversions, treatment_size):
    control_rate = control_conversions / control_size
    treatment_rate = treatment_conversions / treatment_size
    
    z_score, p_value = stats.proportions_ztest(
        [control_conversions, treatment_conversions],
        [control_size, treatment_size],
        alternative='two-sided'
    )
    
    return control_rate, treatment_rate, p_value

# Usage
control_conversions = 200
control_size = 10000
treatment_conversions = 250
treatment_size = 10000

control_rate, treatment_rate, p_value = ab_test(
    control_conversions, control_size, treatment_conversions, treatment_size
)

print(f"Control conversion rate: {control_rate:.2%}")
print(f"Treatment conversion rate: {treatment_rate:.2%}")
print(f"P-value: {p_value:.4f}")
`}
          />

          <p>
            These metrics provide a comprehensive view of a recommendation
            system's performance, covering accuracy, ranking quality, diversity,
            and user satisfaction. It's important to choose the right
            combination of metrics based on the specific goals and context of
            your recommendation system.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default EvaluationMetrics;
