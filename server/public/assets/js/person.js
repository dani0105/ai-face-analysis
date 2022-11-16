//class_name
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

async function getPeople() {
  const people = await fetch('http://localhost:3000/people', {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
  });

  return people.json()
}

function getGroupAttributeArray(people, attribute) {
  const attributes = [];

  people.forEach(person => {
    attributes.push(person[attribute]);
  });

  return attributes;
}

async function showChart() {
  const people = await getPeople();

  const peopleNames = getGroupAttributeArray(people, 'fname');
  const peopleAmount = getGroupAttributeArray(people, 'amount');

  console.log(peopleNames);

  const ctx = document.getElementById('myChart').getContext('2d');;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: peopleNames,
      datasets: [{
        label: '# de detecciones',
        backgroundColor: ["#32a852", "#327da8", "#7532a8", "#87a832", "#e3af12", "#ab2e71", "#2E2D88", '#9534eb'], 
        data: peopleAmount,
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });
};
