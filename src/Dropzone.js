import React, { useState } from "react";
import { AiFillFile } from "react-icons/ai";
import { RxCross2 } from "react-icons/rx";

function Dropzone(props) {
  const [files, setFiles] = useState([]);

  const handleBrowseFiles = () => {
    document.getElementById('fileInput').click();
  };

  const handleFileChange = (event) => {
    const newFiles = event.target.files;
    setFiles([...files, ...newFiles]);
    props.setFiles([...files, ...newFiles])
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const newFiles = event.dataTransfer.files;
    props.setFiles([...files, ...newFiles]);
    setFiles([...files, ...newFiles]);
  };

  const removeFile = (fileName) => {
    setFiles(files.filter(file => file.name !== fileName));
    props.setFiles(files.filter(file => file.name !== fileName));
  };

  return (
    <div style={{ width: '306px', position: 'relative',overflow:'auto'}}>
      <p style={{ fontSize: '14px', color: 'rgb(49, 51, 63)', marginBottom: '0.75rem', height: 'auto', }}>
        Upload your PDFs here and click on 'Process'
      </p>
      <section 
        onDragOver={handleDragOver} 
        onDrop={handleDrop} 
        className='' 
        style={{ gap: '1rem', marginRight: '20px', padding: '1rem', backgroundColor: 'rgb(255, 255, 255)', borderRadius: '0.5rem', color: 'rgb(49, 51, 63)' }}
      >
        <input 
          accept="application/pdf"
          type='file' 
          id='fileInput' 
          multiple 
          style={{ display: 'none', overflow: 'visible' }}
          onChange={handleFileChange}
        />
        <div className='d-flex align-items-center ' style={{ marginBottom: '1rem', marginRight: '20px' }}>
          <div className='d-flex flex-column' style={{ gap: '0.5rem', marginBottom: '1rem', marginRight: '20px' }}>
            <span style={{ marginBottom: '0.25rem' }}>Drag and drop files here</span>
            <small style={{ color: 'rgba(49, 51, 63, 0.6)', fontSize: '14px', lineHeight: '1.25' }}>Limit 200MB per file</small>
          </div>
        </div>
        <button className='button-style' onClick={handleBrowseFiles}>Browse files</button>
      </section>
      {files.map((file, index) => (
        <div key={index} className='d-flex align-items-center align-self-center justify-content-between'>
          <div className='d-flex align-items-center align-self-center p-1'>
            <AiFillFile color='#97A6C3' size={32} />
            <div className='m-1 d-flex flex-column'>
              <p style={{ margin: 0 }}>{file.name}</p>
              <small>{(file.size / 1024 / 1024).toFixed(2)}MB</small>
            </div>
          </div>
          <div className='remove-pdf' onClick={() => removeFile(file.name)}>
            <RxCross2 size={24} />
          </div>
        </div>
      ))}
    </div>
  );
}

export default Dropzone;