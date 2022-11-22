const personasCheckBox = document.getElementById('personas');
const emocionesCheckBox = document.getElementById('emociones');

const canvas = document.getElementById("canvas");
const canvasRect = document.getElementById("canvas-rect");

const ctx = canvas.getContext('2d');
const ctxRect = canvasRect.getContext('2d');

const video = document.getElementById("vid");

let connected = false;
let processing = false;
let socket = null;
let groups = null;

function groupDetected(group) {
  fetch('http://localhost:3000/detected_group', {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ group })
  });
};

async function getGroups() {
  const groups = await fetch('http://localhost:3000/groups', {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
  });

  return groups.json()
}

function personDetected(name) {
  fetch('http://localhost:3000/detected_person', {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ name })
  });
}

function getPeopleNames(people) {
  return people.map((person) => {
    return person.recognition.class_name;
  });
}

function detectGroupByPeople(people) {
  const peopleName = getPeopleNames(people);

  for (let index = 0; index < groups.length; index++) {
    const group = groups[index];

    if(JSON.stringify(peopleName) == JSON.stringify(group.person) ||
      JSON.stringify(peopleName[0]) == JSON.stringify(group.person)) {
      groupDetected(group.group);
    };

  }
}

function peopleDetected(people) {
  if(personasCheckBox.checked) detectGroupByPeople(people);

  people.forEach(person => {
    if(personasCheckBox.checked) personDetected(person.recognition.class_name);
  });
}

async function openCamera() {
  groups = await getGroups();

  const mediaDevices = navigator.mediaDevices;
  video.muted = true;
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
        console.log(video.videoWidth)
        console.log(video.videoHeight)
      });

      video.addEventListener('timeupdate', () => {
        // don't do if the socket is disconnected
        if (connected && !processing) {
          processing = true;
          capture();
        }
      });
    })
    .catch(alert);
}

function drawFaceRectangle({ x1, x2, y1, y2 }) {
  ctxRect.lineWidth = 5;
  ctxRect.strokeStyle = 'green';
  ctxRect.beginPath();
  ctxRect.rect(x1, y1, x2 - x1, y2 - y1);
  ctxRect.stroke();
}

function drawText({ x1, y1, name }) {
  ctxRect.fillStyle = 'red';
  ctxRect.font = "20px Arial";
  ctxRect.fillText(name, x1 - 50, y1 - 20);
}

function drawRecognizedFeatures(facesRecognized) {
  ctxRect.clearRect(0, 0, canvasRect.width, canvasRect.height);

  facesRecognized.forEach(face => {
    const x1 = face.face.box.x1;
    const x2 = face.face.box.x2;

    const y1 = face.face.box.y1;
    const y2 = face.face.box.y2;

    drawFaceRectangle({ x1, x2, y1, y2 });

    if (face.recognition) {
      const name = face.recognition.class_name;
      drawText({ x1, y1: y1 - 20, name });
    }

    if (face.emotion) {
      const emotion = face.emotion.class;
      drawText({ x1, y1: y1, name: emotion });
    }

  });
}

function openSocket() {
  socket = io(
    'http://201.191.35.215:5000/',
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
    console.log("Result", data);

    drawRecognizedFeatures(data);

    peopleDetected(data);
  });
}

function capture() {
  ctx.drawImage(video, 0, 0, 400, 225);

  // RGBA Array
  let image = canvas.toDataURL();

  const data = {
    image: image,
    analyse_person: personasCheckBox.checked,
    analyse_emotion: emocionesCheckBox.checked
  };

  socket.emit("analyse_image", data);
}

