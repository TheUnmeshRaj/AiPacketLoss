const express = require('express')
const app = express()
const server = require('http').Server(app)
const io = require('socket.io')(server)
const { v4: uuidV4 } = require('uuid')

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
    socket.join(roomId)
    console.log(`User ${userId} joined room ${roomId}`)
    socket.to(roomId).emit('user-connected', userId)
    const room = io.sockets.adapter.rooms.get(roomId)
    if (room) {
      const existingUsers = Array.from(room).filter(id => id !== socket.id)
      socket.emit('existing-users', existingUsers)
    }
  })

 
  socket.on('left-call', (userId) => {
  console.log(`User ${userId} left manually`);
  socket.rooms.forEach(roomId => {
    socket.to(roomId).emit('user-left', userId);
  });
});

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id)
    socket.rooms.forEach(roomId => {
      socket.to(roomId).emit('user-disconnected', socket.id)
    })
  })
})

const PORT = process.env.PORT || 3000
server.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`)
})