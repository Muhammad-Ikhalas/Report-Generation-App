// File: ImageUploadForm.jsx
import React, { useState } from 'react';
import axios from 'axios';
import './Generate_report.css';

const ImageUploadForm = () => {
  const [image, setImage] = useState(null);
  const [responseMessage, setResponseMessage] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
  };

  const handleImageUpload = async () => {
    try {
      const formData = new FormData();
      formData.append('image', image);

      // Change the URL to your Node.js backend endpoint
      const response = await axios.post('http://localhost:3001/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Log the response data to the console
      console.log('Upload successful! Response:', response.data);

      // Update the state with the response message
      setResponseMessage(response.data);
    } catch (error) {
      console.error('Error uploading image:', error);
      // If you want to set an error message in the state, you can do it here.
    }
  };

  return (
    <div>
      <h2>Image Upload</h2>
      <label htmlFor="fileInput">Choose File</label>
      <input
        type="file"
        id="fileInput"
        onChange={handleImageChange}
      />
      {image && <span className="file-name">Selected File: {image.name}</span>}
      <button onClick={handleImageUpload}>Upload Image</button>

      {/* Display the response message in a styled div */}
      {responseMessage !== null && (
        <div className="response-container">
          <h3>Report</h3>
          <div className="response-message">{responseMessage}</div>
        </div>
      )}
    </div>
  );
};

export default ImageUploadForm;
