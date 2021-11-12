import React, { useEffect, useRef } from "react";
// import logo from './logo.svg';
import * as tf from "@tensorflow/tfjs";
import * as handpose from "@tensorflow-models/handpose";

import Webcam from "react-webcam";
import "./App.css";
import { drawHand } from "./utilities";
import { default as poseDetectorNet} from "./PoseDetectorNet";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    (async () => {
      const net = await handpose.load();
      const poseDetector = poseDetectorNet.load();
      const video = webcamRef.current.video;
  
      console.log("Handpose model loaded.");
      //  Loop and detect hands
      setInterval(async () => {
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
          const hands = await net.estimateHands(video);
          console.log(hands);

          // hands[0].landmarks
          // if () 

          // Draw mesh
          const ctx = canvasRef.current.getContext("2d");
          drawHand(hands, ctx);
        }        
      }, 100);        
    })()
  }, [])

  return (
    <div className="App">
      <Webcam
        ref={webcamRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
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
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: 640,
          height: 480,
        }}
      />


    </div>
  );
}

export default App;
