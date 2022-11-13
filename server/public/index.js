const personasCheckBox = document.getElementById('personas');
const emocionesCheckBox = document.getElementById('emociones');

function capture() {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext('2d');
  const video = document.getElementById("vid");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  // RGBA Array
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  
  const data = {
    imgData,
    personas: personasCheckBox.checked,
    emociones: emocionesCheckBox.checked
  };
  console.log(data);
}

document.addEventListener("DOMContentLoaded", () => {
  const openCameraButton = document.getElementById("openCameraButton");
  const video = document.getElementById("vid");
  const mediaDevices = navigator.mediaDevices;
  vid.muted = true;
  openCameraButton.addEventListener("click", () => {

    // Accessing the user camera and video.
    mediaDevices
      .getUserMedia({
        video: true,
        audio: true,
      })
      .then((stream) => {

        // Changing the source of video to current stream.
        video.srcObject = stream;
        video.addEventListener("loadedmetadata", () => {
          video.play();
        });

        video.addEventListener('timeupdate', () => {
          capture();
        });
      })
      .catch(alert);
  });
});