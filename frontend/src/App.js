import React, { useState } from 'react';
import { Upload, Wand2, Loader2 } from 'lucide-react';
import './App.css';

export default function StyleTransferApp() {
  const [contentImage, setContentImage] = useState(null);
  const [styleImage, setStyleImage] = useState(null);
  const [contentPreview, setContentPreview] = useState(null);
  const [stylePreview, setStylePreview] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Parameters
  const [contentWeight, setContentWeight] = useState(1);
  const [styleWeight, setStyleWeight] = useState(1000000);
  const [numSteps, setNumSteps] = useState(300);
  
  const API_URL = 'http://localhost:5000';

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e, type) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file, type);
    }
  };

  const processFile = (file, type) => {
    if (file && file.type.startsWith('image/')) {
      if (type === 'content') {
        setContentImage(file);
        setContentPreview(URL.createObjectURL(file));
      } else {
        setStyleImage(file);
        setStylePreview(URL.createObjectURL(file));
      }
      setError(null);
    }
  };

  const handleImageUpload = (e, type) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file, type);
    }
  };

  const handleSubmit = async () => {
    if (!contentImage || !styleImage) {
      setError('Please upload both content and style images');
      return;
    }

    setLoading(true);
    setError(null);
    setResultImage(null);

    const formData = new FormData();
    formData.append('content_image', contentImage);
    formData.append('style_image', styleImage);
    formData.append('num_steps', numSteps);
    formData.append('content_weight', contentWeight);
    formData.append('style_weight', styleWeight);

    try {
      const response = await fetch(`${API_URL}/style-transfer`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResultImage(data.image);
      } else {
        setError(data.error || 'Style transfer failed');
      }
    } catch (err) {
      setError(`Connection error: ${err.message}. Make sure the backend is running on ${API_URL}`);
    } finally {
      setLoading(false);
    }
  };

  const ImageUploadBox = ({ type, preview, onChange }) => (
    <div className="upload-container">
      <label 
        className="upload-label"
        onDragOver={handleDragOver}
        onDrop={(e) => handleDrop(e, type)}
      >
        <div className="upload-box">
          {preview ? (
            <img src={preview} alt={type} className="preview-image" />
          ) : (
            <div className="upload-placeholder">
              <Upload size={48} />
              <p>Upload or drag & drop {type} image</p>
            </div>
          )}
        </div>
        <input
          type="file"
          accept="image/*"
          onChange={onChange}
          className="file-input"
        />
      </label>
      <p className="image-label">{type} Image</p>
    </div>
  );

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <div className="header">
          <h1>Neural Style Transfer</h1>
          <p className="subtitle">
            Transform your images with artistic styles using deep learning
          </p>
        </div>

        <div className="card main-card">
          <h2>Upload Images</h2>
          
          <div className="image-grid">
            <ImageUploadBox
              type="content"
              preview={contentPreview}
              onChange={(e) => handleImageUpload(e, 'content')}
            />
            <ImageUploadBox
              type="style"
              preview={stylePreview}
              onChange={(e) => handleImageUpload(e, 'style')}
            />
          </div>

          <div className="parameters-section">
            <h3>Parameters</h3>
            
            <div className="parameters-grid">
              <div className="parameter-group">
                <label>Content Weight</label>
                <input
                  type="number"
                  value={contentWeight}
                  onChange={(e) => setContentWeight(parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                />
                <p className="parameter-hint">Higher = more content preserved</p>
              </div>

              <div className="parameter-group">
                <label>Style Weight</label>
                <input
                  type="number"
                  value={styleWeight}
                  onChange={(e) => setStyleWeight(parseFloat(e.target.value))}
                  step="100000"
                  min="0"
                />
                <p className="parameter-hint">Higher = more style applied</p>
              </div>

              <div className="parameter-group">
                <label>Optimization Steps</label>
                <input
                  type="number"
                  value={numSteps}
                  onChange={(e) => setNumSteps(parseInt(e.target.value))}
                  step="50"
                  min="50"
                  max="1000"
                />
                <p className="parameter-hint">More steps = better quality (slower)</p>
              </div>
            </div>
          </div>

          <button
            onClick={handleSubmit}
            disabled={loading || !contentImage || !styleImage}
            className="submit-button"
          >
            {loading ? (
              <>
                <Loader2 className="spinner" size={20} />
                Processing... (this may take a few minutes)
              </>
            ) : (
              <>
                <Wand2 size={20} />
                Apply Style Transfer
              </>
            )}
          </button>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        {resultImage && (
          <div className="card result-card">
            <h2>Result</h2>
            <div className="result-container">
              <img 
                src={resultImage} 
                alt="Style transfer result" 
                className="result-image"
              />
            </div>
            <div className="download-container">
              <a
                href={resultImage}
                download="style-transfer-result.png"
                className="download-button"
              >
                Download Result
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}