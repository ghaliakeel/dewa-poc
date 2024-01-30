import React from 'react';
import { textChatMargin } from './App';
import { avatarChatStyle } from './App';
import userIcon from './images/user.png';
import LinearIndeterminate from './LinearProgress';

function LoadingChatMessage(props) {
    return (
        <div className='w-100' style={{width:'100%',minWidth:'auto',maxWidth:'initial'}}>
        <div className='d-flex flex-column ' style={{margin:'1rem 3rem 1rem 3rem',backgroundColor:'#f0f2f6',borderRadius:'6px', height:'110px'}}>
          <div className='d-flex align-self-center align-items-center w-100 ' style={{padding:'1rem'}}>
            <img className='avatar' src='https://www.projexme.com/wp-content/uploads/2017/03/dewa-logo.jpg' style={avatarChatStyle}/>
            <div className='d-flex align-self-center align-items-center p-3' style={{flexGrow:'1'}}>
                <LinearIndeterminate />
            </div>
          </div>
        </div>
      </div>
    );
}

export default LoadingChatMessage;