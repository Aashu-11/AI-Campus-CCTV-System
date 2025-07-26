import React, { useEffect, useRef, useState } from "react";

export default function App() {
  const videoRef = useRef(null);
  const [minContour, setMinContour] = useState(500);
  const [scanDuration, setScanDuration] = useState(30);
  const [frameThreshold, setFrameThreshold] = useState(100);
  const [bgHist, setBgHist] = useState(150);
  const [objectDetectionOn, setObjectDetectionOn] = useState(false);
  const [alertStart, setAlertStart] = useState("20:00");
  const [alertEnd, setAlertEnd] = useState("08:00");
  const [alertStatus, setAlertStatus] = useState("default (20:00 - 08:00)");

  useEffect(() => {
    videoRef.current.src = "http://localhost:8000/video_feed";
  }, []);

  const triggerAlert = async () => {
    const res = await fetch("http://localhost:8000/trigger_email", { method: "POST" });
    const data = await res.json();
    alert(data.status);
  };

  const updateAlertTimes = async () => {
    const res = await fetch("http://localhost:8000/update_alert_times", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start: alertStart, end: alertEnd })
    });
    const data = await res.json();
    alert("Alert times updated: " + data.status);
    setAlertStatus(`custom (${alertStart} - ${alertEnd})`);
  };

  return (
    <div style={{ display: "flex", padding: "20px", fontFamily: "Arial" }}>
      <div style={{ flex: 3, paddingRight: "20px" }}>
        <h1 style={{ textAlign: "center" }}>2A2S: AI Surveillance</h1>
        <video
          ref={videoRef}
          autoPlay
          style={{ width: "100%", border: "2px solid black" }}
        ></video>
        <button onClick={triggerAlert} style={{ width: "100%", marginTop: 10 }}>
          🚨 Trigger Manual Alert
        </button>
      </div>

      <div style={{ flex: 1, borderLeft: "1px solid #ccc", paddingLeft: "20px" }}>
        <h2>Control Panel</h2>

        <label>Min Contour Size: {minContour}</label>
        <input
          type="range"
          min="10"
          max="10000"
          value={minContour}
          onChange={(e) => setMinContour(Number(e.target.value))}
        />

        <label>Scan Duration (sec): {scanDuration}</label>
        <input
          type="range"
          min="30"
          max="300"
          value={scanDuration}
          onChange={(e) => setScanDuration(Number(e.target.value))}
        />

        <label>Frame Diff Threshold: {frameThreshold}</label>
        <input
          type="range"
          min="10"
          max="1000"
          value={frameThreshold}
          onChange={(e) => setFrameThreshold(Number(e.target.value))}
        />

        <label>BG Subtract History: {bgHist}</label>
        <input
          type="range"
          min="5"
          max="1000"
          value={bgHist}
          onChange={(e) => setBgHist(Number(e.target.value))}
        />

        <div>
          <button
            onClick={() => setObjectDetectionOn(!objectDetectionOn)}
            style={{ marginTop: 10 }}
          >
            {objectDetectionOn ? "Object Detection: ON" : "Object Detection: OFF"}
          </button>
        </div>

        <hr />
        <h3>[Configure Alert Time]</h3>
        <div>
          <label>Start (HH:MM)</label>
          <input
            type="time"
            value={alertStart}
            onChange={(e) => setAlertStart(e.target.value)}
          />
        </div>
        <div>
          <label>End (HH:MM)</label>
          <input
            type="time"
            value={alertEnd}
            onChange={(e) => setAlertEnd(e.target.value)}
          />
        </div>
        <button onClick={updateAlertTimes} style={{ marginTop: 10 }}>
          ✅ Update Alert Times
        </button>
        <div style={{ marginTop: 10 }}>Status: {alertStatus}</div>
      </div>
    </div>
  );
}
