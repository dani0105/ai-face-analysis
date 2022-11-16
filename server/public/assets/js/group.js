const form = document.getElementById('groupForm');
const personList = document.getElementById('personList');

function addPerson() {
  const newPerson = document.getElementById('newPerson').value;
  const input = document.createElement('input');
  const newBullet = document.createElement('li');

  newBullet.textContent = newPerson

  input.value = newPerson;
  input.name = 'person';
  input.style.display = 'none';

  form.appendChild(input);
  form.append(newBullet);
}

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

function getGroupAttributeArray(groups, attribute) {
  const attributes = [];

  groups.forEach(group => {
    attributes.push(group[attribute]);
  });

  return attributes;
}

async function showChart() {
  const groups = await getGroups();
  console.log(groups);

  const groupNames = getGroupAttributeArray(groups, 'group');
  const groupAmount = getGroupAttributeArray(groups, 'amount');

  console.log({groupNames, groupAmount});

  const ctx = document.getElementById('myChart');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: groupNames,
      datasets: [{
        label: '# de detecciones',
        backgroundColor: ["#32a852", "#327da8", "#7532a8", "#87a832", "#e3af12", "#ab2e71", "#2E2D88", '#9534eb'], 
        data: groupAmount,
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
