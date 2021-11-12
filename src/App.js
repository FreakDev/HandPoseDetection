import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as handpose from "@tensorflow-models/handpose";

import Webcam from "react-webcam";
import { drawHand } from "./utilities";
import { default as poseDetectorNet} from "./PoseDetectorNet";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // MODE
  const MODE_LOADING = 0
  const MODE_IDLE = 1
  const MODE_COLLECTING = 2
  const MODE_TRAINNING = 3
  const [mode, setMode] = useState(MODE_LOADING)
  const [label, setRawLabel] = useState([0,0,0])
  const [detected, setDetected] = useState('')

  const [labelInput, setLabelInput] = useState(JSON.stringify(label))

  const [dataCollector, setDataCollector] = useState([])
 
  const net = useRef(null);
  const poseDetector = useRef(null);

  useEffect(() => {
    (async () => {
      net.current = await handpose.load();
      console.log("Handpose model loaded.");
      poseDetector.current = poseDetectorNet.load();
      setMode(MODE_IDLE)
    })()
  }, [])

  useEffect(() => {
    //  Loop and detect hands
    const int = setInterval(async () => {
      if (
        typeof webcamRef.current !== "undefined" &&
        webcamRef.current !== null &&
        webcamRef.current.video.readyState === 4
      ) {
        // Get Video Properties
        const video = webcamRef.current.video;
        const videoWidth = webcamRef.current.video.videoWidth;
        const videoHeight = webcamRef.current.video.videoHeight;
  
        // Set video width
        webcamRef.current.video.width = videoWidth;
        webcamRef.current.video.height = videoHeight;
  
        // Set canvas height and width
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        // Make Detections
        const hands = await net.current.estimateHands(video);

        if (hands.length) {
          if (mode === MODE_COLLECTING) {
            setDataCollector ([...dataCollector, { inputs: hands[0].landmarks, labels: label}])
          }
          if (mode === MODE_IDLE) {
            const result = poseDetector.current.estimatePoses(rawHandsToDataset([{ inputs: hands[0].landmarks }]))
            if (result && result.length && result[0].some(score => score >= 0.8)) {
              setDetected(['paper', 'scissors', 'rock'][result[0].findIndex(score => score >= 0.8)])
            }
          }
        } 

        // Draw mesh
        const ctx = canvasRef.current.getContext("2d");
        drawHand(hands, ctx);
      }
    }, 50);
    return () => clearInterval(int)
  }, [dataCollector, label, mode])

  const toogleCollectState = () => {
    if (mode === MODE_IDLE)
      setMode(MODE_COLLECTING)
    else if (mode === MODE_COLLECTING) {
      setMode(MODE_IDLE)
      console.log(dataCollector)
      localStorage.setItem('trainningData', JSON.stringify(dataCollector))
    }
  }

  const rawHandsToDataset = (data) => {
    const labels = [], inputs = []
    data.forEach(entry => { 
      inputs.push(entry.inputs.map(i => [i[0] - entry.inputs[0][0], i[1] - entry.inputs[0][1], i[2] - entry.inputs[0][2]]).flat())
      if (entry.labels)
        labels.push(entry.labels)
    })
    return {
      inputs,
      labels
    }
  }

  const train = async () => {
    setMode(MODE_TRAINNING)
    if (dataCollector.length) {
      const data = rawHandsToDataset(dataCollector)
      await poseDetector.current.train(data)
    }
    setMode(MODE_IDLE)
  }

  const setLabel = (str) => {
    setRawLabel(JSON.parse(str))
  }

  const loadTrainningData = async () => {
    const data = await import ('./trainningData.json')
    setDataCollector(data.default)
  }

  return (
    <div className="App">
      <Webcam
        ref={webcamRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          top: 150,
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: 640,
          height: 480,
        }}
      />

      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          top: 150,
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: 640,
          height: 480,
        }}
      />
      { mode !== MODE_LOADING && (
        <fieldset>
          <legend>Trainning</legend>
          { mode === MODE_COLLECTING && (<span>collecting... {dataCollector.length ? (<React.Fragment>{dataCollector.length} frame(s)</React.Fragment>) : null} </span>)}
          { mode === MODE_TRAINNING && (<span>trainning...</span>)}
          { mode === MODE_IDLE && detected && (<span>{detected}</span>)}
          <fieldset>
            <legend>current label : {JSON.stringify(label)}</legend>
            <label htmlFor="new_label">update to</label><input id="new_label" value={labelInput} onChange={e => setLabelInput(e.target.value)} />
            <button onClick={ () => setLabel(labelInput) } >update !</button>
          </fieldset>
          <button onClick={ loadTrainningData } >Load trainning data !</button>
          <button onClick={ toogleCollectState } disabled={[MODE_TRAINNING].includes(mode)} >collect !</button>
          <button onClick={ train } disabled={[MODE_COLLECTING, MODE_TRAINNING].includes(mode) || !dataCollector.length} >train !</button>
        </fieldset>
      )}
    </div>
  );
}

export default App;
