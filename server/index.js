require('dotenv').config();

const EXPRESS = require('express');
const APP = EXPRESS();

const people = [];
const group = [];

APP.use(EXPRESS.json({ limit: '10mb' }));
APP.use(EXPRESS.urlencoded({ extended: true }));

APP.set('port', process.env.PORT || process.env.APP_PORT);

APP.use(EXPRESS.static(__dirname + '/public'));

APP.use((req, res, next) => {
  res.header("Access-Control-Allow-Headers", "x-requested-with, content-type,Authorization");
  next();
});


APP.post('/add_person', (req, res) => {    
  req.body.amount = 0;
  people.push(req.body);
  res.writeHead(301, {
    Location: 'http://localhost:3000/pages/person.html'
  }).end();
})

APP.get('/groups', (req, res) => {    
  res.json(group);
})

APP.get('/people', (req, res) => {    
  res.json(people);
})

APP.post('/detected_person', (req, res) => {    
  const personName = req.body.name;

  people.forEach(person => {
    if(person.fname === personName) { 
      person.amount += 1;
      return;
    }
  });

  res.writeHead(301, {
    Location: 'http://localhost:3000/pages/person.html'
  }).end();
})

APP.post('/detected_group', (req, res) => {    
  const groupName = req.body.group;

  group.forEach(team => {
    if(team.group === groupName) { 
      team.amount += 1;      
      return;
    }
  });

  res.writeHead(301, {
    Location: 'http://localhost:3000/pages/person.html'
  }).end();
})

APP.post('/add_group', (req, res) => {    
  req.body.amount = 0;
  group.push(req.body);
  res.writeHead(301, {
    Location: 'http://localhost:3000/pages/group.html'
  }).end();
})

APP.listen(APP.get('port'),() => {
  console.log(`server running in http://localhost:${APP.get('port')}`);
});