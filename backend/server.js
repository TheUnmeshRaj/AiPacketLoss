const express = require('express')
const app = express()
const server = require('http').Server(app)
const io = require('socket.io')(server)
const { v4: uuidV4 } = require('uuid')
const roomParticipants = {};

app.set('view engine', 'ejs')
app.use(express.static('public'))

app.get('/', (req, res) => {
  res.redirect(`/${uuidV4()}`)
})

app.get('/left', (req, res) => {
  res.render('left');
});

app.get('/:room', (req, res) => {
  res.render('room', { roomId: req.params.room })
})

io.on('connection', socket => {
  console.log('User connected:', socket.id)
  
  socket.on('join-room', (roomId, userId) => {
  socket.join(roomId);
  console.log(`User ${userId} joined room ${roomId}`);

  if (!roomParticipants[roomId]) roomParticipants[roomId] = new Set();
  roomParticipants[roomId].add(userId);

  socket.to(roomId).emit('user-connected', userId);
  const room = io.sockets.adapter.rooms.get(roomId);
  if (room) {
    const existingUsers = Array.from(room).filter(id => id !== socket.id);
    socket.emit('existing-users', existingUsers);
  }
});
 
  socket.on('left-call', (userId) => {
  socket.rooms.forEach(roomId => {
    roomParticipants[roomId]?.delete(userId);
    socket.to(roomId).emit('user-left', userId);
  });
});

socket.on('disconnect', () => {
  socket.rooms.forEach(roomId => {
    roomParticipants[roomId]?.delete(socket.id);
    socket.to(roomId).emit('user-disconnected', socket.id);
  });
});

})

const PORT = process.env.PORT || 3000
server.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`)
})