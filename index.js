(async () => {
  tf.util.shuffle(irisData);
  const data = tf.tensor2d(
    irisData.map(({ sepalLength, sepalWidth, petalLength, petalWidth }) => [
      sepalLength,
      sepalWidth,
      petalLength,
      petalWidth
    ])
  );
  const normalisedData = getNormalisedTensors(data);
  const labels = tf.tensor2d(
    irisData.map(({ species }) => [
      species === "setosa" ? 1 : 0,
      species === "virginica" ? 1 : 0,
      species === "versicolor" ? 1 : 0
    ])
  );
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [4],
      activation: "relu",
      units: 15
    })
  );

  model.add(
    tf.layers.dense({
      activation: "relu",
      units: 15
    })
  );

  model.add(
    tf.layers.dense({
      activation: "softmax",
      units: 3
    })
  );

  model.summary();

  model.compile({
    optimizer: "sgd",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  const results = await model.fit(normalisedData, labels, {
    epochs: 300,
    validationSplit: 0.2
  });
})();

function getNormalisedTensors(tensor) {
  const max = tensor.max(0);
  const min = tensor.min(0);
  return tensor.sub(min).div(max.sub(min));
}
