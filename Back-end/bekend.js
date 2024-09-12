// app.js
const express = require('express');
const mysql = require('mysql');
const moment = require('moment');
const cors = require('cors');

const app = express();
const port = 3500;

// Use CORS middleware
app.use(cors());

// MySQL Connection
const connection = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: '',
  database: 'eyettention'
});

connection.connect((err) => {
  if (err) {
    console.error('Error connecting to MySQL: ', err);
    return;
  }
  console.log('Connected to MySQL database');
});


app.get('/', (req, res) => {
  res.send('Eye Contact App');
});


app.get('/daily', (req, res) => {
  const today = moment().format('YYYY-MM-DD');
  const query = `
    SELECT COUNT(*) AS total_users, SUM(IF(looking = true, 1, 0)) AS looking_users
    FROM face_detections
    WHERE DATE(timestamp) = '${today}'
  `;

  connection.query(query, (err, results) => {
    if (err) {
      console.error('Error executing MySQL query: ', err);
      res.status(500).send('Error fetching data');
      return;
    }

    if (results.length > 0) {
      const { total_users, looking_users } = results[0];
      const percentage = (looking_users / total_users) * 100;
      res.json({
        date: today,
        total_users,
        looking_users,
        percentage: percentage.toFixed(2)
      });
    } else {
      res.json({
        date: today,
        total_users: 0,
        looking_users: 0,
        percentage: 0
      });
    }
  });
});


app.get('/weekly', (req, res) => {
  const startOfWeek = moment().startOf('week').format('YYYY-MM-DD');
  const endOfWeek = moment().endOf('week').format('YYYY-MM-DD');
  const query = `
    SELECT COUNT(*) AS total_users, SUM(IF(looking = true, 1, 0)) AS looking_users
    FROM face_detections
    WHERE DATE(timestamp) BETWEEN '${startOfWeek}' AND '${endOfWeek}'
  `;

  connection.query(query, (err, results) => {
    if (err) {
      console.error('Error executing MySQL query: ', err);
      res.status(500).send('Error fetching data');
      return;
    }

    if (results.length > 0) {
      const { total_users, looking_users } = results[0];
      const percentage = (looking_users / total_users) * 100;
      res.json({
        start_date: startOfWeek,
        end_date: endOfWeek,
        total_users,
        looking_users,
        percentage: percentage.toFixed(2)
      });
    } else {
      res.json({
        start_date: startOfWeek,
        end_date: endOfWeek,
        total_users: 0,
        looking_users: 0,
        percentage: 0
      });
    }
  });
});


app.get('/monthly', (req, res) => {
  const startOfMonth = moment().startOf('month').format('YYYY-MM-DD');
  const endOfMonth = moment().endOf('month').format('YYYY-MM-DD');
  const query = `
    SELECT COUNT(*) AS total_users, SUM(IF(looking = true, 1, 0)) AS looking_users
    FROM face_detections
    WHERE DATE(timestamp) BETWEEN '${startOfMonth}' AND '${endOfMonth}'
  `;

  connection.query(query, (err, results) => {
    if (err) {
      console.error('Error executing MySQL query: ', err);
      res.status(500).send('Error fetching data');
      return;
    }

    if (results.length > 0) {
      const { total_users, looking_users } = results[0];
      const percentage = (looking_users / total_users) * 100;
      res.json({
        start_date: startOfMonth,
        end_date: endOfMonth,
        total_users,
        looking_users,
        percentage: percentage.toFixed(2)
      });
    } else {
      res.json({
        start_date: startOfMonth,
        end_date: endOfMonth,
        total_users: 0,
        looking_users: 0,
        percentage: 0
      });
    }
  });
});

app.get('/monthly_chart', (req, res) => {
  const startOfMonthTes = moment().startOf('month').format('YYYY-MM');
  const startOfMonth = moment().startOf('month').format('YYYY-MM-DD');
  const endOfMonth = moment().endOf('month').format('YYYY-MM-DD');
  const query = `WITH weeks AS (
    SELECT
        1 AS week_number,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 0 WEEK) AS start_of_week,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 1 WEEK) AS end_of_week
    UNION
    SELECT
        2 AS week_number,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 1 WEEK) AS start_of_week,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 2 WEEK) AS end_of_week
    UNION
    SELECT
        3 AS week_number,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 2 WEEK) AS start_of_week,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 3 WEEK) AS end_of_week
    UNION
    SELECT
        4 AS week_number,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 3 WEEK) AS start_of_week,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 4 WEEK) AS end_of_week
    UNION
    SELECT
        5 AS week_number,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 4 WEEK) AS start_of_week,
        DATE_ADD(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 5 WEEK) AS end_of_week
)
SELECT
    weeks.week_number,
    COALESCE(COUNT(fd.timestamp), 0) AS total_users,
    COALESCE(SUM(IF(fd.looking = true, 1, 0)), 0) AS looking_users
FROM
    weeks
LEFT JOIN
    face_detections fd ON DATE(fd.timestamp) BETWEEN weeks.start_of_week AND weeks.end_of_week
GROUP BY
    weeks.week_number
ORDER BY
    weeks.week_number;
`;

  connection.query(query, (err, results) => {
    if (err) {
      console.error('Error executing MySQL query: ', err);
      res.status(500).send('Error fetching data');
      return;
    }

    if (results.length > 0) {
      const { total_users, looking_users } = results[0];
      const percentage = (looking_users / total_users) * 100;
      const data = results.map(row => ({
        week_number: row.week_number,
        total_users: row.total_users,
        looking_users: row.looking_users
      }));
      res.json({
        start_date_tes: startOfMonthTes,
        start_date: startOfMonth,
        end_date: endOfMonth,
        data,
        total_users,
        looking_users,
        percentage: percentage.toFixed(2),
      });
    } else {
      res.json({
        start_date: startOfMonth,
        end_date: endOfMonth,
        total_users: 0,
        looking_users: 0,
        percentage: 0
      });
    }
  });
});


app.listen(port, () => {
  console.log(`App listening at http://localhost:${port}`);
});
