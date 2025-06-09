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

  // 🧠 Add dynamic quality monitoring here
  // const pc = call.peerConnection;
  const pc = myPeer._connections?.[userId]?.[0]?.peerConnection || null;

  // const pc = myPeer._connections ? Object.values(myPeer._connections)[0]?.[0]?.peerConnection : null;

  if (pc) {
    setInterval(() => {
      pc.getStats(null).then(stats => {
        stats.forEach(report => {
          if (report.type === 'outbound-rtp' && report.kind === 'video') {
            const loss = report.packetsLost || 0;
            const bitrate = report.bitrateMean || 0;

            console.log('Bitrate:', bitrate, 'Packet Loss:', loss);
            console.log('PC:', pc);
            console.log('Report:', report);

            if (loss > 50 || bitrate < 100000) {
              stream.getVideoTracks()[0].applyConstraints({
                width: { ideal: 320 },
                height: { ideal: 240 },
                frameRate: { ideal: 10 }
              }).then(() => {
                console.log('Constraints applied');
              }).catch(err => {
                console.error('Constraint error:', err);
              });
            }
          }
        });
      });
    }, 5000);
  }
}


function addVideoStreamToSlot(stream, userId, name = 'User') {
  if (document.querySelector(`.video-container[data-user-id="${userId}"]`)) return;

  const existingSlots = document.querySelectorAll('.video-container');
  if (existingSlots.length >= 6) {
    console.warn('Max 6 participants reached');
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
  video.addEventListener('loadedmetadata', () => {
    video.play().catch(() => {});
  });

  const overlay = document.createElement('div');
  overlay.className = 'video-overlay';

  const nameSpan = document.createElement('span');
  nameSpan.className = 'participant-name';
  nameSpan.textContent = name;

  overlay.appendChild(nameSpan);

  const audioIndicator = document.createElement('div');
  audioIndicator.className = 'audio-indicator';

  const wave1 = document.createElement('div');
  wave1.className = 'audio-wave';
  const wave2 = wave1.cloneNode();
  const wave3 = wave1.cloneNode();

  audioIndicator.appendChild(wave1);
  audioIndicator.appendChild(wave2);
  audioIndicator.appendChild(wave3);

  overlay.appendChild(audioIndicator);

  // Show/hide based on mute status AFTER video and overlay created
  const audioTracks = stream.getAudioTracks();
  if (audioTracks.length === 0 || !audioTracks[0].enabled) {
    audioIndicator.style.display = 'none';
  }

  slot.appendChild(video);
  slot.appendChild(overlay);

  videoGrid.appendChild(slot);
  updateVideoLayout();
}

function updateVideoLayout() {
  const containers = document.querySelectorAll('.video-container');

  containers.forEach(container => container.style.flex = '');

  if (containers.length === 1) {
    containers[0].style.flex = '1 1 100%';
  } else if (containers.length === 2) {
    containers.forEach(c => c.style.flex = '1 1 calc(50% - 20px)');
  } else if (containers.length === 3) {
    containers.forEach(c => c.style.flex = '1 1 calc(33.333% - 20px)');
  } else {
    containers.forEach(c => c.style.flex = '1 1 calc((100% - 40px) / 3)');
  }
}

const observer = new MutationObserver(updateVideoLayout);
observer.observe(videoGrid, { childList: true, subtree: false });

function removeUserSlot(userId) {
  const slot = document.querySelector(`.video-container[data-user-id="${userId}"]`);
  if (slot) slot.remove();
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

    document.querySelectorAll('.video-container').forEach(container => {
      if (container.dataset.userId === 'me') {
        const indicator = container.querySelector('.audio-indicator');
        if (indicator) {
          indicator.style.display = audioTrack.enabled ? 'flex' : 'none';
        }
      }
    });

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
