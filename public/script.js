const socket = io('/');
const videoGrid = document.getElementById('video-grid');
const myPeer = new Peer();
const peers = {};
let myStream;

navigator.mediaDevices.getUserMedia({ video: true, audio: true })
  .then(stream => {
    myStream = stream;

    addVideoStreamToSlot(stream, 'me', 'You');

    myPeer.on('call', call => {
      call.answer(stream);
      call.on('stream', userVideoStream => {
        addVideoStreamToSlot(userVideoStream, call.peer, `User ${call.peer}`);
      });
    });

    socket.on('user-connected', userId => {
      connectToNewUser(userId, myStream);
      updateParticipantCount();
    });

    socket.on('existing-users', userIds => {
      userIds.forEach(userId => connectToNewUser(userId, stream));
      updateParticipantCount();
    });
  })
  .catch(err => {
    console.error('Failed to access media devices:', err);
    alert('Could not access camera or microphone.');
  });

socket.on('user-disconnected', userId => {
  if (peers[userId]) peers[userId].close();
  removeUserSlot(userId);
  delete peers[userId];
  updateParticipantCount();
});

socket.on('user-left', userId => {
  showToast(`User ${userId} has left the chat.`);
  if (peers[userId]) peers[userId].close();
  removeUserSlot(userId);
});

socket.on('room-full', () => {
  alert('Room is full! Please try another room.');
});

myPeer.on('open', id => {
  socket.emit('join-room', ROOM_ID, id);
});

function connectToNewUser(userId, stream) {
  const call = myPeer.call(userId, stream);
  call.on('stream', userVideoStream => {
    addVideoStreamToSlot(userVideoStream, userId, `User ${userId}`);
  });
  call.on('close', () => {
    removeUserSlot(userId);
  });
  peers[userId] = call;
}

function addVideoStreamToSlot(stream, userId, name = 'User') {
  const existingSlots = document.querySelectorAll('.video-container');

  for (let slot of existingSlots) {
    if (slot.dataset.userId === userId) return;
  }

  if (existingSlots.length >= 5) {
    console.warn('Max 5 participants reached');
    return;
  }

  const slot = document.createElement('div');
  slot.className = 'video-container';
  slot.dataset.userId = userId;

  const video = document.createElement('video');
  video.srcObject = stream;
  video.autoplay = true;
  video.playsInline = true;
  if (userId === 'me') video.muted = true;

  const overlay = document.createElement('div');
  overlay.className = 'video-overlay';

  const nameSpan = document.createElement('span');
  nameSpan.className = 'participant-name';
  nameSpan.textContent = name;

  overlay.appendChild(nameSpan);
  slot.appendChild(video);
  slot.appendChild(overlay);

  document.getElementById('video-grid').appendChild(slot);
}

function removeUserSlot(userId) {
  const slot = Array.from(document.querySelectorAll('.video-container'))
    .find(s => s.dataset.userId === userId);
  if (slot) {
    slot.innerHTML = '';
    slot.removeAttribute('data-user-id');
  }
}

function updateParticipantCount() {
  const count = Object.keys(peers).length + 1;
  const el = document.getElementById('participants-count');
  if (el) el.textContent = count;
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