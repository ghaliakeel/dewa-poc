import * as React from 'react';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';

export default function CircularIndeterminate(props) {
  return (
    <Box sx={{ display: 'flex' }} style={{margin:props.margin}}>
      <CircularProgress color={props.color} />
    </Box>
  );
}
