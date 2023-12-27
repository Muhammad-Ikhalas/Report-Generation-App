const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();

app.use(cors());
app.use(express.json());

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadPath = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadPath)) {
      fs.mkdirSync(uploadPath);
    }
    cb(null, uploadPath);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  },
});

const upload = multer({ storage: storage });

function generateReport(imageData) {
  return new Promise((resolve, reject) => {
    const pythonScript = './model.py';
    const pythonProcess = spawn('python', [pythonScript]);

    pythonProcess.stdin.write(imageData);
    pythonProcess.stdin.end();

    let predictions = '';

    pythonProcess.stdout.on('data', (data) => {
      predictions = data.toString().trim();
      console.log(`Predictions: ${predictions}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Error (stderr): ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        reject(`Python process exited with code ${code}`);
        return;
      }

      try {
        const fileContent = fs.readFileSync('./generated_report.txt', 'utf-8');
        resolve(fileContent);
      } catch (error) {
        reject(`Error reading report file: ${error.message}`);
      }
    });
  });
}

app.post('/upload', upload.single('image'), async (req, res) => {
  try {
    const imageBuffer = fs.readFileSync(req.file.path);
    const report = await generateReport(imageBuffer);

    res.send(report);
  } catch (error) {
    console.error('Error handling image:', error);
    res.status(500).send('Internal Server Error');
  }
});

const port = 3001;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
