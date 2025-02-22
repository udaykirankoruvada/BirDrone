import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { FaExpand } from 'react-icons/fa';
import { io } from 'socket.io-client';
import './Detection.css';

const socket = io('http://localhost:5000');  // Ensure the correct port is used

function Detection() {
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [detections, setDetections] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [imageWithBoxes, setImageWithBoxes] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fileType, setFileType] = useState(null);
  const [videoData, setVideoData] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [target, setTarget] = useState("");
  const [isCamera, setIsCamera] = useState(false);
  const [stream, setStream] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [capturedVideo, setCapturedVideo] = useState(null);

  const containerRef = useRef(null);
  const resultsRef = useRef(null);
  const originalVideoRef = useRef(null);
  const detectionVideoRef = useRef(null);
  const imageRef = useRef(null);
  const cameraVideoRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  useEffect(() => {
    if (uploaded && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [uploaded]);

  useEffect(() => {
    if (isCamera) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((mediaStream) => {
          setStream(mediaStream);
          if (cameraVideoRef.current) {
            cameraVideoRef.current.srcObject = mediaStream;
          }
        })
        .catch((err) => {
          alert("Error accessing the camera: " + err.message);
        });
    } else if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  }, [isCamera]);

  useEffect(() => {
    const handleBeforeUnload = () => {
      if (taskId) {
        socket.emit('cancel_task', { task_id: taskId });
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [taskId]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) {
      setFile(null);
      setFilePreview(null);
      setUploaded(false);
      setTarget(""); // Reset target
      return;
    }
    const fileCategory = selectedFile.type.split('/')[0];
    setFile(selectedFile);
    setFileType(fileCategory);
    setUploaded(false);
    setFilePreview(URL.createObjectURL(selectedFile));
    setCapturedImage(null);
    setCapturedVideo(null);
    setIsCamera(false); // Ensure camera is turned off when a file is selected
    setTarget(""); // Reset target
    setDetections(null); // Clear detections
    setMetrics(null); // Clear metrics
    setImageWithBoxes(null); // Clear image with boxes
    setVideoData(null); // Clear video data
  };

  const handleTargetChange = (e) => {
    setTarget(e.target.value);
    setDetections(null); // Clear detections
    setMetrics(null); // Clear metrics
    setImageWithBoxes(null); // Clear image with boxes
    setVideoData(null); // Clear video data
    setUploaded(false); // Reset uploaded state
  };

  const handleUpload = async () => {
    if ((!file && !capturedImage && !capturedVideo) || !target) {
      alert('Please select a file or capture an image/video, and select a detection type');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    } else if (capturedImage) {
      const blob = dataURLtoBlob(capturedImage);
      formData.append('file', blob, 'captured_image.jpg'); // Ensure the file has a name
    } else if (capturedVideo) {
      formData.append('file', capturedVideo, 'captured_video.webm'); // Ensure the file has a name
    }
    formData.append('target', target);

    // Generate a unique task ID
    const newTaskId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setTaskId(newTaskId);
    formData.append('task_id', newTaskId);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: file ? (file.type.startsWith('video') ? 'blob' : 'json') : 'json'
      });

      if (file && file.type.startsWith('video') || capturedVideo) {
        const videoBlob = new Blob([response.data], { type: 'video/mp4' });
        const videoUrl = URL.createObjectURL(videoBlob);
        setVideoData(videoUrl);
        setFileType('video');
        setImageWithBoxes(null);

        if (response.data.metrics) {
          setMetrics({
            precision: response.data.metrics.precision || 0,
            accuracy: response.data.metrics.accuracy || 0,
          });
        } else {
          setMetrics(null);
        }
      } else {
        setDetections(response.data.detections);
        setMetrics({
          precision: response.data.precision || 0,
          accuracy: response.data.accuracy || 0,
        });
        setImageWithBoxes(response.data.image_with_boxes);
        setFileType('image');
        setVideoData(null);
      }
      setUploaded(true);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert(`An error occurred during file upload: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const toggleFullscreen = () => {
    if (!imageRef.current) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      if (imageRef.current.requestFullscreen) {
        imageRef.current.requestFullscreen();
      } else if (imageRef.current.mozRequestFullScreen) {
        imageRef.current.mozRequestFullScreen();
      } else if (imageRef.current.webkitRequestFullscreen) {
        imageRef.current.webkitRequestFullscreen();
      } else if (imageRef.current.msRequestFullscreen) {
        imageRef.current.msRequestFullscreen();
      }
    }
    setIsFullscreen(!isFullscreen);
  };

  const dataURLtoBlob = (dataURL) => {
    const byteString = atob(dataURL.split(',')[1]);
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const uintArray = new Uint8Array(arrayBuffer);
    for (let i = 0; i < byteString.length; i++) {
      uintArray[i] = byteString.charCodeAt(i);
    }
    return new Blob([uintArray], { type: 'image/jpeg' });
  };

  const handleCapture = () => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const video = cameraVideoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const capturedDataUrl = canvas.toDataURL('image/jpeg');
    setCapturedImage(capturedDataUrl);

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setIsCamera(false);
    }
    setTarget(""); // Reset target
    setDetections(null); // Clear detections
    setMetrics(null); // Clear metrics
    setImageWithBoxes(null); // Clear image with boxes
    setVideoData(null); // Clear video data
  };

  const handleStartRecording = () => {
    if (stream) {
      const options = { mimeType: 'video/webm; codecs=vp9' };
      mediaRecorderRef.current = new MediaRecorder(stream, options);
      mediaRecorderRef.current.ondataavailable = handleDataAvailable;
      mediaRecorderRef.current.onstop = handleStop;
      mediaRecorderRef.current.start();
      setIsRecording(true);
    }
  };

  const handleDataAvailable = (event) => {
    if (event.data.size > 0) {
      setRecordedChunks((prev) => prev.concat(event.data));
    }
  };

  const handleStop = () => {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    setCapturedVideo(blob);
    setRecordedChunks([]);
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setIsCamera(false);
    }
  };

  const handleStopRecording = () => {
    mediaRecorderRef.current.stop();
    setIsRecording(false);
  };

  return (
    <div ref={containerRef} className={`container ${uploaded ? 'expanded' : ''}`}>
      <h1 className="heading">BirDrone: A Solution for Airspace and Wildlife Security</h1>
      <p className="subheading">Upload an Image or Video for Detection</p>

      <div className="upload-controls">
        <div className='file-upload'>
          <input
            type="file"
            accept="image/*,video/*"
            onChange={handleFileChange}
            className="file-input"
            disabled={isCamera}
          />
          <div className='separator'></div>
          {!file && (
            <button 
              onClick={() => {
                setIsCamera(!isCamera);
                setFile(null);
                setFilePreview(null);
                setCapturedImage(null);
                setCapturedVideo(null);
                setTarget(""); // Reset target
                setDetections(null); // Clear detections
                setMetrics(null); // Clear metrics
                setImageWithBoxes(null); // Clear image with boxes
                setVideoData(null); // Clear video data
              }} 
              className="camera-btn"
            >
              {isCamera ? 'Stop Camera' : 'Use Camera'}
            </button>
          )}
        </div>
        
        {(file || capturedImage || capturedVideo) && (
          <>
            <select
              value={target}
              onChange={handleTargetChange}
              className="target-select"
            >
              <option value="" disabled hidden>
                Select detection type
              </option>
              <option value="bird">Bird</option>
              <option value="drone">Drone</option>
              <option value="both">Both</option>
            </select>
          </>
        )}
        {((file || capturedImage || capturedVideo) && target) && (
          <button onClick={handleUpload} className="upload-btn" disabled={loading}>
          {loading ? 'Uploading...' : 'Upload & Detect'}
        </button>)}
      </div>

      {(file || capturedImage || capturedVideo) && (
        <div className="content-wrapper">
          <div className="original-image-section">
            <div className="preview-container">
              <h3>{capturedImage ? 'Captured Image:' : capturedVideo ? 'Captured Video:' : 'Original File:'}</h3>
              {fileType === 'video' || capturedVideo ? (
                <video
                  ref={originalVideoRef}
                  src={filePreview || URL.createObjectURL(capturedVideo)}
                  controls
                  playsInline
                  className="preview-video"
                />
              ) : (
                <img src={filePreview || capturedImage} alt="preview" className="preview-image" />
              )}
            </div>
          </div>

          {uploaded && (
            <div className="results-section" ref={resultsRef}>
              <div className="detection-results">
                <h3>Detection Results:</h3>

                {/* 5) Conditionally show "No DETECTIONS Found" if no detections */}
                   {fileType === 'video' || capturedVideo ? (
                  // For video
                    <video
                      ref={detectionVideoRef}
                      src={videoData}
                      controls
                      playsInline
                      className="detection-video"
                      onPlay={() => {
                        if (originalVideoRef.current && originalVideoRef.current.paused) {
                          originalVideoRef.current.play();
                        }
                      }}
                      onPause={() => {
                        if (originalVideoRef.current && !originalVideoRef.current.paused) {
                          originalVideoRef.current.pause();
                        }
                      }}
                      onSeeking={() => {
                        if (originalVideoRef.current) {
                          originalVideoRef.current.currentTime = detectionVideoRef.current.currentTime;
                        }
                      }}
                    />
                  ) : (
                  // For image
                  detections && detections.length > 0  ? (
                    <div style={{ position: 'relative' }}>
                      <img
                        ref={imageRef} // Attach ref here
                        src={`data:image/jpeg;base64,${imageWithBoxes}`}
                        alt="Detection with Bounding Boxes"
                        className="detection-image"
                      />
                      {/* Fullscreen icon with fixed click handler */}
                      <button onClick={toggleFullscreen} className="fullscreen-btn">
                    <FaExpand />
                  </button>
                    </div>
                  ) : (
                    <p className="no-detections">No DETECTIONS Found</p>
                  )
                )}


                {/* Show metrics if available */}
                {detections && detections.length > 0 && metrics && metrics.accuracy !== undefined && metrics.precision !== undefined && (
                  <div className="metrics">
                    <p>
                      <strong>Accuracy:</strong> {(metrics.accuracy * 100).toFixed(2)}%
                    </p>
                    <p>
                      <strong>Precision:</strong> {(metrics.precision * 100).toFixed(2)}%
                    </p>
                  </div>
                )}

                {/* List out detections if there are any */}
                {fileType === 'image' && detections && detections.length > 0 && (
                  <ul>
                    {detections.map((det, index) => (
                      <li key={index}>
                        <span>{det.label}</span> at coordinates {det.bbox.join(', ')} with confidence {(det.confidence * 100).toFixed(2)}%
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {isCamera && !capturedImage && !capturedVideo && (
        <div className="camera-video-section">
          <h3>Camera Feed:</h3>
          <video
            ref={cameraVideoRef}
            autoPlay
            playsInline
            className="camera-video"
          />
          <button onClick={handleCapture} className="capture-btn">Capture Image</button>
          <button onClick={isRecording ? handleStopRecording : handleStartRecording} className="capture-btn">
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </button>
        </div>
      )}
      
    </div>
  );
}

export default Detection;
