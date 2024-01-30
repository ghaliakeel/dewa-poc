import React from 'react';
import './App.css';
import { useState } from 'react';
import DropZone from "./Dropzone";
import {RxCross2} from 'react-icons/rx';
import {BsArrowRightShort} from 'react-icons/bs';
import CircularIndeterminate from './CircularProgress';
import axios from 'axios';

const sideBarStyle = {
  //position:'fixed',
  //left:0,
  height:'100vh',
  width:'349px',
  backgroundColor:'#f0f2f6',
  minWidth:'224px',
  maxWidth:'349px'
}

function SideBar(props) {


    const [sidebarVisible, setSidebarVisible] = useState(true); // SIDEBAR VISIBILITY
    const [files, setFiles] = useState([]); // CONTAINTS PDF FILES
    const [circularProgress,updateCircularProgress] = useState(false); // CIRCULAR PROGRESS FOR UPLOADING PDF FILES
    const [info,setInfo] = useState(false);

    window.addEventListener('resize', () => {
        setSidebarVisible(window.innerWidth >= 920);
      });

    const sleep = (time) => {
        return new Promise((resolve) => {
          setTimeout(resolve, time);
          updateCircularProgress(!circularProgress)
        });
        
      };

      // /api/client/upload
    
      async function sendEmbeddings() {

        if(info === true) setInfo(false)
       
        updateCircularProgress(true);
        const formData = new FormData();
      
        for (const file of files) {
          formData.append('file', file);
        }

      
        try {
        
          var url = 'http://localhost:5000/api/client/upload'
          await axios.post(url, formData, {
            headers: {
              'Content-Type': 'multipart/form-data', // Important for file uploads
            },
          }).then(resp => {
            if(resp.status === 200){
                updateCircularProgress(false);
            }
          });
      
           
        } catch (error) {
            
        
        setInfo(true)
        updateCircularProgress(false);
        }
      }

    return (
        <section className='d-flex' style={{flexShrink:0}}>
        <div className=' RxCrossHover' style={{position:'absolute',left:10,top:10,zIndex:0}}>
          <BsArrowRightShort size={32} onClick={() => {setSidebarVisible(true);props.setSidebarVisible(true)}} />
        </div>
        <aside  className={sidebarVisible ? 'sidebar-open' : 'sidebar-closed'}>
          <div style={sideBarStyle}>
            <div className='d-flex flex-row-reverse p-3'>
              <div className='p-1 RxCrossHover' onClick={() => {setSidebarVisible(false);props.setSidebarVisible(false)}}>
                <RxCross2 size={26} />
              </div>
                
            </div>
  
            <div>
              <div style={{padding:'6rem 1.5rem'}}>
                <div style={{width:'100%',padding:'0px',minWidth:'auto',maxWidth:'initial'}}>
                  <div className='d-flex flex-column'>
                    <div style={{width:'306px',position:'relative',display:'flex',flex:'1 1 0%',flexDirection:'column',gap:'1rem'}}>
                      <div style={{width:'306px',position:'relative'}}>
                        <h3 style={{fontSize:'1.125rem',fontWeight:'600'}}>Your documents</h3>
                      </div>
                    </div>
                  </div>
                </div>
  
                <DropZone setFiles={setFiles}/>
                <button disabled={files.length === 0 ? true : false} className='mt-3 button-style' onClick={() => {sendEmbeddings();}} style={{cursor: files.length === 0 ? 'not-allowed' : 'pointer'}}>Process</button>
                <br />
                {circularProgress?<><span style={{display:'inline'}}><CircularIndeterminate  color={'secondary'} margin={'20px'} /> Processing...</span></>: null}
                
  
              </div>
                {info?                <div class="alert alert-info" role="alert">
                    <strong>Something wrong happened</strong> <br />please try again
                </div>:null}
            </div>
            
  
      </div>
      
        </aside>
      </section>
    );
}

export default SideBar;