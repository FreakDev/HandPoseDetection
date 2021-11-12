import * as tf from "@tensorflow/tfjs"

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [63], units: 50, activation: 'relu'}));

  model.add(tf.layers.dense({units: 50, activation: 'relu'}));
  model.add(tf.layers.dense({units: 50, activation: 'relu'}));
  model.add(tf.layers.dense({units: 50, activation: 'relu'}));

  // Add an output layer
  model.add(tf.layers.dense({units: 2, activation: 'softmax'}));

  return model;
}

function convertToTensor(data, normalizationData) {
  return tf.tidy(() => {
    if (!normalizationData)
      tf.util.shuffle(data)

    const inputs = data.inputs
    const labels = data.labels

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

export default {
  load() {
    const model = createModel()
    let trainningData
    let normalizationData

    return {
      setTrainningDatasets (datasets) {
        return [trainningData, normalizationData] = convertToTensor(datasets);
      },
    
      async train(datasets, trainningParameter = { batchSize: 32, epochs: 50 }) {
        model.compile({
          optimizer: tf.train.adam(),
          loss: tf.losses.meanSquaredError,
          metrics: ['mse'],
        });
      
        const {batchSize, epochs} = trainningParameter;
  
        const [{ inputs, labels}] = this.setTrainningDatasets(datasets)
    
        return await model.fit(inputs, labels, {
          batchSize,
          epochs,
          shuffle: true
        });
      },
    
      estimatePoses(dataset) {
        if (normalizationData) {
          const {labelMin, labelMax} = normalizationData;

          tf.tidy(() => {
        
            // let xs = tf.linspace(0, 1, 100);
            const [{inputs}] = convertToTensor(dataset, normalizationData)
            
            const preds = model.predict(inputs);;
            const unNormPreds = preds
              .mul(labelMax.sub(labelMin))
              .add(labelMin);
  
            unNormPreds.print()
          })  
        }
      }
    }
  }
}