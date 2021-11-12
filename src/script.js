import * as tf from "@tensorflow/tfjs"; 
import * as tfvis from "@tensorflow/tfjs-vis"
/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
 async function getData() {
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    cylinders: car.Cylinders,
    weight: car.Weight_in_lbs,
    horsepower: car.Horsepower,
  }))
  .filter(car => (car.mpg != null && car.horsepower != null && car.weight != null && car.cylinders != null));

  return cleaned;
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [63], units: 50, activation: 'relu'}));

  model.add(tf.layers.dense({inputShape: [1], units: 50, activation: 'relu'}));
  model.add(tf.layers.dense({inputShape: [1], units: 50, activation: 'relu'}));
  model.add(tf.layers.dense({inputShape: [1], units: 50, activation: 'relu'}));

  // Add an output layer
  model.add(tf.layers.dense({units: 2, activation: 'softmax'}));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
 function convertToTensor(data, normalizationData) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {

    if (!normalizationData)
      tf.util.shuffle(data)

    const pad_array = function(arr,len,fill) {
      return arr.concat(Array(len).fill(fill)).slice(0,len);
    }

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => pad_array(d.inputs, 63, 0))
    const labels = data.map(d => d.labels);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 63]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 2]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    
    let inputMax;
    let inputMin;
    let labelMax;
    let labelMin;

    if (normalizationData) {
      inputMax = normalizationData.inputMax;
      inputMin = normalizationData.inputMin;
      labelMax = normalizationData.labelMax;
      labelMin = normalizationData.labelMin;  
    } else {
      inputMax = inputTensor.max();
      inputMin = inputTensor.min();
      labelMax = labelTensor.max();
      labelMin = labelTensor.min();  
    }

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    // debugger

    return [{
      inputs: normalizedInputs,
      labels: normalizedLabels,
    }, {
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }]
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 20;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  tf.tidy(() => {

    // let xs = tf.linspace(0, 1, 100);
    const [{inputs}] = convertToTensor([
      {horsepower: 165, cylinders: 8, weight: 3693},        // 15
      {horsepower: 130, cylinders: 8, weight: 3504},        // 18
      {horsepower: 88, cylinders: 4, weight: 2130},         // 27
      {horsepower: 100, cylinders: 6, weight: 3329},        // 17,
    ], normalizationData)
    console.log(inputs)
    
    const preds = model.predict(inputs);;

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    unNormPreds.print()
  });
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();

  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);

  console.log(data)

  // Convert the data to a form we can use for training.
  const [{inputs, labels}, normalizationData] = convertToTensor(data);

  inputs.print()

  // Train the model
  await trainModel(model, inputs, labels);
  console.log('Done Training');

  // Make some predictions using the model and compare them to the
  // original data
  // const [{inputsTest}] = convertToTensor(dataTest, normalizationData);
  testModel(model, inputs, normalizationData);

}

document.addEventListener('DOMContentLoaded', run);