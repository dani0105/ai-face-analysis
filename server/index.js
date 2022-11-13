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
  people.push(req.body);
  console.log(people);
  res.writeHead(301, {
    Location: 'http://localhost:3000'
  }).end();
})

APP.post('/add_group', (req, res) => {    
  group.push(req.body);
  console.log(group);
  res.writeHead(301, {
    Location: 'http://localhost:3000'
  }).end();
})



APP.listen(APP.get('port'),() => {
  console.log(`server running in http://localhost:${APP.get('port')}`);
});