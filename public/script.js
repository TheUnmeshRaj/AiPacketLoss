const socket = io('/');
const videoGrid = document.getElementById('video-grid');
const myPeer = new Peer(); // Use PeerJS cloud server
const myVideo = document.createElement('video');
myVideo.muted = true;
const peers = {};
let myStream; // Store the local stream for toggle controls

navigator.mediaDevices.getUserMedia({
  video: true,
  audio: true
}).then(stream => {
  myStream = stream;
  addVideoStream(myVideo, stream);

  myPeer.on('call', call => {
    call.answer(stream);
    const video = document.createElement('video');
    call.on('stream', userVideoStream => {
      addVideoStream(video, userVideoStream);
    });
  });

  socket.on('user-connected', userId => {
    connectToNewUser(userId, stream);
  });

  socket.on('existing-users', userIds => {
    userIds.forEach(userId => connectToNewUser(userId, stream));
  });
}).catch(err => {
  console.error('Failed to access media devices:', err);
  alert('Could not access camera or microphone.');
});

socket.on('user-disconnected', userId => {
  if (peers[userId]) peers[userId].close();
});

socket.on('room-full', () => {
  alert('Room is full! Please try another room.');
});

myPeer.on('open', id => {
  socket.emit('join-room', ROOM_ID, id);
});

function connectToNewUser(userId, stream) {
  const call = myPeer.call(userId, stream);
  const video = document.createElement('video');
  call.on('stream', userVideoStream => {
    addVideoStream(video, userVideoStream);
  });
  call.on('close', () => {
    video.remove();
  });

  peers[userId] = call;
}

function addVideoStream(video, stream) {
  video.srcObject = stream;
  video.addEventListener('loadedmetadata', () => {
    video.play();
  });
  videoGrid.append(video);
}

function toggleAudio() {
  if (myStream) {
    const audioTrack = myStream.getAudioTracks()[0];
    audioTrack.enabled = !audioTrack.enabled;
    return audioTrack.enabled ? 'Unmute' : 'Mute';
  }
}

function toggleVideo() {
  if (myStream) {
    const videoTrack = myStream.getVideoTracks()[0];
    videoTrack.enabled = !videoTrack.enabled;
    return videoTrack.enabled ? 'Turn Off Camera' : 'Turn On Camera';
  }
}