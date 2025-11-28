const { randomInt } = require('crypto')
const express = require('express')
const app = express()
const server = require('http').Server(app)
const io = require('socket.io')(server)
const { v4: uuidV4 } = require('uuid')
// roomParticipants: { [roomId]: Map(userId -> name) }
const roomParticipants = {};
// Map socket.id -> peer userId (PeerJS id) so we can cleanup on disconnect
const socketToPeer = {};
// Names pool used for display names
const NAMES = ['Tony Stark', 'Bruce Banner', 'Natasha Romanoff', 'Steve Rogers', 'Thor Odinson', 'Clint Barton', 'Wanda Maximoff', 'Vision', 'Sam Wilson', 'Bucky Barnes'];

function hashStringToIndex(str, mod) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h) + str.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h) % mod;
}

app.set('view engine', 'ejs')
app.use(express.static('public'))

app.get('/', (req, res) => {
  res.redirect(`/${uuidV4().slice(10,22)}`)
})

app.get('/left', (req, res) => {
  res.render('left');
});

app.get('/:room', (req, res) => {
  res.render('room', { roomId: req.params.room })
})

app.get('/api/participants/:roomId', (req, res) => {
  const roomId = req.params.roomId;
  const participants = roomParticipants[roomId]
    ? Array.from(roomParticipants[roomId].entries()).map(([id, name]) => ({ id, name }))
    : [];
  res.json({ participants });
});


io.on('connection', socket => {
  console.log('User connected:', socket.id)
  
  // Accepts: (roomId, userId, requestedName)
  socket.on('join-room', (roomId, userId, requestedName) => {
  socket.join(roomId);
  if (!roomParticipants[roomId]) roomParticipants[roomId] = new Map();

  // Determine taken names in the room
  const taken = new Set(Array.from(roomParticipants[roomId].values()));

  // Choose assignedName respecting requestedName and avoiding duplicates until exhausted
  let assignedName = null;
  if (requestedName) {
    if (!taken.has(requestedName)) {
      assignedName = requestedName;
    }
  }

  if (!assignedName) {
    // Try deterministic pick based on userId
    const start = hashStringToIndex(userId || socket.id, NAMES.length);
    for (let i = 0; i < NAMES.length; i++) {
      const idx = (start + i) % NAMES.length;
      if (!taken.has(NAMES[idx])) {
        assignedName = NAMES[idx];
        break;
      }
    }
    // If all names are taken, allow reuse (pick deterministic slot)
    if (!assignedName) assignedName = NAMES[start];
  }

  roomParticipants[roomId].set(userId, assignedName);
  socketToPeer[socket.id] = userId;

  console.log(`User ${userId} (requested: ${requestedName}) assigned name '${assignedName}' in room ${roomId}`);

  // Notify others in the room about the new user (id + name)
  socket.to(roomId).emit('user-connected', { userId, name: assignedName });

  // Send existing users (id + name) to this socket, excluding the joining user
  const existingUsers = Array.from(roomParticipants[roomId].entries())
    .filter(([id]) => id !== userId)
    .map(([id, name]) => ({ id, name }));
  // Inform the joining socket of its assigned name
  socket.emit('joined', { userId, name: assignedName });
  socket.emit('existing-users', existingUsers);
});
 
  socket.on('left-call', (userId) => {
  socket.rooms.forEach(roomId => {
    if (roomId === socket.id) return;
    roomParticipants[roomId]?.delete(userId);
    socket.to(roomId).emit('user-left', { userId, name: null });
  });
  // cleanup mapping for this socket (if any)
  delete socketToPeer[socket.id];
});

socket.on('disconnect', () => {
  const peerId = socketToPeer[socket.id];
  socket.rooms.forEach(roomId => {
    if (roomId === socket.id) return;
    if (peerId) {
      const name = roomParticipants[roomId]?.get(peerId) || null;
      roomParticipants[roomId]?.delete(peerId);
      socket.to(roomId).emit('user-disconnected', { userId: peerId, name });
    }
  });
  delete socketToPeer[socket.id];
});

})

const PORT = process.env.PORT || 3000
server.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`)
})