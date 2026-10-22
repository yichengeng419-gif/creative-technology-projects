import {
  featuresForRow,
} from './features.js';

/**
 * Load the supplied Random Forest model.
 *
 * @returns {Promise<object>}
 */
export async function loadForest() {
  const response =
    await fetch(
      'assets/seed-forest.json'
    );

  return response.json();
}
/**
 * Run one feature array through a trained forest.
 *
 * Each tree returns class weights from one leaf.
 * The forest prediction is the class with the
 * largest average vote.
 *
 * @param {object} forest
 * @param {number[]} features
 * @returns {{label: string, confidence: number}}
 */
export function predictForest(
  forest,
  features
) {
  const votes =
    new Array(forest.classes.length)
      .fill(0);

  for (const tree of forest.trees) {
    let node = 0;
    let safety = 0;

    while (
      tree.l[node] !== -1
      && safety < 64
    ) {
      const feature =
        tree.f[node];

      const threshold =
        tree.t[node];

      node =
        features[feature] <= threshold
          ? tree.l[node]
          : tree.r[node];

      safety++;
    }

    const leaf =
      tree.v[node];

    if (!leaf) {
      throw new Error(
        'Invalid forest leaf'
      );
    }

    const total =
      leaf.reduce(
        (sum, value) =>
          sum + value,
        0
      ) || 1;

    leaf.forEach((value, index) => {
      votes[index] +=
        value / total;
    });
  }

  let prediction = 0;

  for (
    let index = 1;
    index < votes.length;
    index++
  ) {
    if (
      votes[index]
      > votes[prediction]
    ) {
      prediction = index;
    }
  }

  return {
    label:
      forest.classes[prediction],

    confidence:
      votes[prediction]
      / forest.trees.length,
  };
}

/**
 * Convert a Session record and predict its class.
 *
 * @param {object} forest
 * @param {object} row
 * @returns {{label: string, confidence: number}}
 */
export function predictSession(
  forest,
  row
) {
  return predictForest(
    forest,
    featuresForRow(row)
  );
}