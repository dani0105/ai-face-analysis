const personasCheckBox = document.getElementById('personas');
const emocionesCheckBox = document.getElementById('emociones');
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext('2d');
const video = document.getElementById("vid");

let connected = false;
let processing = false;
let socket = null;

function openCamera(){
  const mediaDevices = navigator.mediaDevices;
  vid.muted = true;
  // Accessing the user camera and video.
  mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {

      // Changing the source of video to current stream.
      video.srcObject = stream;
      video.addEventListener("loadedmetadata", () => {
        video.play();
      });

      video.addEventListener('timeupdate', () => {
        // don't do if the socket is disconnected
        if(connected && !processing){
          processing = true;
          capture();
        }
      });
    })
    .catch(alert);
}

function  openSocket(){
  socket = io(
    'http://localhost:5000',
    {
      autoConnect: true, //  like this, could be found in manager piece
    }
  );

  socket.connect();
  openCamera();

  // error events
  socket.on('connect_error', (error) => {
    console.log(error)
    connected = false;
    processing = false;
  })

  socket.on('disconnect', (error) => {
    console.log("Disconnected")
    connected = false;
    processing = false;
  })

  // connection event
  socket.on("connect", () => {
    connected = true;
    processing = false;
    console.log("Connected");
  });

  socket.on("result", (data) => {
    connected = true;
    processing = false;
    console.log("Result",data);

    // draw response here
  });
}

function capture() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  // RGBA Array
  let image = canvas.toDataURL();

  const data = {
    image:image,
    analyse_person: personasCheckBox.checked,
    analyse_emotion: emocionesCheckBox.checked
  };

  socket.emit("analyse_image", data);
}
