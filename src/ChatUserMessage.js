import React from 'react';
import { textChatMargin } from './App';
import {avatarChatStyle} from './App';
import userIcon from './images/user.png';

// Chat message component 
// input in props:  props.message -> should be the message you want to render
// you can change the img source if you want

function ChatUserMessage(props) {
    return (
<div className='w-100' style={{width:'100%',minWidth:'auto',maxWidth:'initial'}}>
  <div className='d-flex flex-column chat-margin' style={{margin:'1rem 3rem 1rem 3rem',backgroundColor:'#f0f2f6',borderRadius:'6px'}}>
    <div className='d-flex align-self-center align-items-center w-100' style={{padding:'1rem',wordBreak:'break-all'}}>
      <img className='avatar' src={userIcon} style={avatarChatStyle} />
      <div className='d-flex align-self-center align-items-center p-tag-container' style={{flexGrow:'1'}}>
        <p style={{...textChatMargin, wordWrap: 'break-word', maxWidth: '100%'}}>{props.message}</p>
      </div>
    </div>
  </div>
</div>
    );
}

export default ChatUserMessage;
