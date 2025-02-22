import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';
import image1 from '../images/image2.jpg';
import image2 from '../images/image5.jpg';
import image3 from '../images/image3.jpg';
import introVideo from '../videos/intro_video1.mp4';

function LandingPage() {
  const [progress, setProgress] = useState(0);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [showButton, setShowButton] = useState(false);
  const [remainingTime, setRemainingTime] = useState(20);
  const [videoPlayed, setVideoPlayed] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          window.location.href = 'detection';
        }
        return prev + 5;
      });
      setRemainingTime((prev) => prev - 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const slides = [
    {
      image: image1,
      title: 'Accurate Bird and Drone Identification',
      description: 'Our cutting-edge system provides real-time detection...',
    },
    {
      image: image2,
      title: 'Advanced Technology',
      description: 'We utilize state-of-the-art computer vision...',
    },
    {
      image: image3,
      title: 'Versatile Applications',
      description: 'Our solution is ideal for a wide range of applications...',
    },
  ];

  useEffect(() => {
    if (videoPlayed) {
      const intervalId = setInterval(() => {
        setCurrentSlide((prevSlide) => (prevSlide + 1) % slides.length);
      }, 5000);

      const buttonTimeout = setTimeout(() => {
        setShowButton(true);
      }, 10000);

      return () => {
        clearInterval(intervalId);
        clearTimeout(buttonTimeout);
      };
    }
  }, [slides.length, videoPlayed]);

  const handleVideoEnd = () => {
    setVideoPlayed(true);
  };

  return (
    <div className="landing-page">
      {!videoPlayed ? (
        <video
          src={introVideo}
          autoPlay
          muted
          playsInline
          onEnded={handleVideoEnd}
          className="intro-video"
        />
      ) : (
        <>
          <div
            className="background-image"
            style={{ backgroundImage: `url(${slides[currentSlide].image})` }}
          ></div>
          <header>
            <div className="header-content">
              <h1>Bird vs. Drone Detection</h1>
            </div>
          </header>
          <main>
            <div className="main-content">
              <section id="hero">
                <h2>{slides[currentSlide].title}</h2>
                <p>{slides[currentSlide].description}</p>
                {showButton && (
                  <button 
                    className="learn-more" 
                    onClick={() => navigate('/detection')}
                  >
                    Try Model
                  </button>
                )}
              </section>
            </div>
          </main>
          {remainingTime > 10 ? (
            <p>Redirecting to Try Model in {remainingTime} seconds...</p>
          ) : (
            <p>Redirecting to Try Model in {remainingTime} seconds... or click on Try Model to access model</p>
          )}
        </>
      )}
      <div className="progress-bar">
            <div className="progress" style={{ width: `${progress}%` }}></div>
      </div>
      <footer className="footer">
        <p>&copy; {new Date().getFullYear()} Bird vs. Drone Detection</p>
      </footer>
    </div>
  );
}

export default LandingPage;
