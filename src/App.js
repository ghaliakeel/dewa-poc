import './App.css';
import { Col,Row,Container } from 'react-bootstrap';
import React,{ useState,useRef,useEffect } from 'react';
import DropZone from "./Dropzone";
import {RxCross2} from 'react-icons/rx';
import {BsArrowRightShort} from 'react-icons/bs';
import DXCLOGO from './images/DXC_Logo.png';
import {AiFillFile} from 'react-icons/ai';
import CircularIndeterminate from './CircularProgress';
import SideBar from './SideBar';
import ChatUserMessage from './ChatUserMessage';
import ChatBotMessage from './ChatBotMessage';
import LoadingChatMessage from './LoadingChatMessage';
import axios from 'axios';


export const textChatMargin = {marginLeft:'35px',marginTop:'15px'} // this is the style for the text inside the component, i margined it left and top to make it look better

export const avatarChatStyle = {
  maxWidth:'78px',
  minWidth:'78px',
  height:'78px',
  width:'78px',
  borderRadius:'50%',
  objectFit:'cover'
} // avatar style image 

// #f0f2f6


function App() {

  const [sidebarVisible, setSidebarVisible] = useState(true); // SIDEBAR VISIBILITY
  const [userChat,updateUserChat] = useState([]); // CONTAINTS USER CHAT HISTORY
  const [botChat,updateBotChat] = useState([]); // CONTAINTS BOT CHAT HISTORY
  const [files, setFiles] = useState([]); // CONTAINTS PDF FILES
  const [circularProgress,updateCircularProgress] = useState(false); // CIRCULAR PROGRESS FOR UPLOADING PDF FILES
  const [text,updateText] = useState(''); // QUESTION IS HERE
  const [loading,setLoading] = useState(false); // The purpose of this is to set the loading img chat untill a message arrives
  const [loadingMessages, setLoadingMessages] = useState([]); // The purpose of this is to set the loading img chat untill a message arrives
  const [disableSending,setDisableSending] = useState(false); // This is to disable sending message when a message is sent 

  const chatContainerRef = useRef(null); // this is used to set the window scroll to the latest image

  useEffect(() => {
    // Scroll to the bottom of the chat container when user or bot messages are added or loading state changes.
    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
  }, [userChat, botChat, loading]);

// this function is responsible for sending api request ot the localhost to ask langchain llm
// input: question
  async function sendAPIRequest(question) {
    try {
      setDisableSending(true) // disable sending message to prevent user from sending something when a message is already sent
      updateUserChat([...userChat, text]); // update userChat state, it contains all users messages
      setLoadingMessages([...loadingMessages, true]); // set the loading img
      // CHANGE URL HERE
      var url = 'http://localhost:5000/api/client/LoadAndRetrieve' // <--------- URL CHANGE HERE
      
      // sending logic:
      // 1. Send the query
      // 2. if response status is 200 extract message and add it to response messages
      // 3. else, set error msg
      await axios.post(url, {
        // no need to check if text is null, already done in the handleKeyPressed
        query: question || text,
      }).then(
        (resp) => {
          // 2. 
          if (resp.status == 200) {
          
            setLoadingMessages((prevLoadingMessages) => {
              const updatedLoadingMessages = [...prevLoadingMessages]; // remove loading
              updatedLoadingMessages[updatedLoadingMessages.length - 1] = false; // remove loading
              return updatedLoadingMessages;
            });

            updateBotChat([...botChat, resp.data.message]); // put the response in bot chat

          }
        }
      );
    // 3.
    }//axios post end
    
    catch (error) {
      setLoadingMessages((prevLoadingMessages) => {
        const updatedLoadingMessages = [...prevLoadingMessages];
        updatedLoadingMessages[updatedLoadingMessages.length - 1] = false;
        return updatedLoadingMessages;
      });
      updateBotChat([...botChat,"I'm sorry, something wrong happened! try again later please"])
      console.error('API request failed:', error);
      
    }//end of catch

    setDisableSending(false) // disabling the sending, user can send now since response arrived
  
  }// end of function
  
 // handle when user press enter -> sendAPIRequest is toggled
  function handleKeyPress(event) {
    if (event.key === 'Enter' && text.trim() !== '') {
      updateUserChat([...userChat, text]);
      sendAPIRequest();
    }
  }



  return (

  //<BsArrowRightShort size={26} /> 
 <div className='d-flex' style={{inset:'0px'}}>
 

    <SideBar setSidebarVisible={setSidebarVisible} /> {/* SideBar.js -> this is the sidebar */}

    
    <section className='d-flex flex-column align-items-center w-100 white-section-animation' style={{overflow:'auto',flexGrow:1,justifyContent:'space-between'}}>
      <div className='w-100 header' style={{padding:'6rem 1rem 0rem',width:'100%',minWidth:'auto',maxWidth:'initial'}}>
        <div className='d-flex flex-column'>
          <div className='m-5 d-flex flex-column ' style={{gap:'1rem'}}>
            {/* DXC LOGO IMG HERE*/}
            <div>
              <img className='dxc-logo' src={DXCLOGO} alt='dxc logo'/>
            </div>
            {/* END DXC LOGO IMG */}
            <h1>DEWA's Virtual Engineer 👷‍♂️</h1>
          </div>
        </div>
    </div>


    <div
          ref={chatContainerRef} {/* Helps keeping the window scrolled to the latest messages*/}
          className="d-flex flex-column w-100"
          style={{ overflowY: 'scroll', maxHeight: '400px' }}
        >
          {/* MESSAGES DISPLAY LOGIC */}
          {/* 1. display user chat 
              2. display bot chat 
              3. if the user send a msg display loadingcomponent
          at */}

          {userChat.map((message, index) => (
            <React.Fragment key={`user_${index}`}>
              <ChatUserMessage message={message} />
              {loadingMessages[index] ? <LoadingChatMessage /> : <ChatBotMessage message={botChat[index]} />}
            </React.Fragment>
          ))}
        </div>
 
    <div className='w-100' style={{width:'100%',minWidth:'auto',maxWidth:'initial',marginTop:'25px'}}>
      <div className='d-flex flex-column'>
            <div className='m-5 d-flex flex-column ' style={{gap:'1rem'}}>
              <div className='w-100'>
              <p style={{fontSize:'14px',marginBottom:'5px'}}>Ask a question about your documents (then press Enter key):</p>
                <input 
                disabled={disableSending}
                onChange={(e)=>{updateText(e.target.value);}} // Change Text State here
                className='user-input' 
                type='text' 
                style={{backgroundColor:'#f0f2f6',borderRadius:'6px',border:'none',width:'100%',padding:'15px'}}
                onKeyDown={handleKeyPress}/> {/* When Enter is pressed, send API CALL */} 
              </div>
            </div>
          </div>
        </div>
        
   


    </section>


 
 </div>
   
  );
}

export default App;
