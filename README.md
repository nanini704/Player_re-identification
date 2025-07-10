This project detects and re-identifies soccer players across frames in a short video using the YOLOv11 object detection model and simple appearance-based tracking.


Project Structure

player_re-identification/
├── main.py # Main script to run the video tracking
├── tracker.py # PlayerTracker class with detection + re-ID
├── best.pt # Trained YOLOv11 model for person detection
├── 15sec_input_720p.mp4 # Sample soccer video (~15 seconds)
├── output/
│ └── output_tracked.mp4 # Output video with tracked players
├── requirements.txt # Python dependencies
└── README.md # Project overview


Install dependencies using pip
pip install -r requirements.txt

To run:
python main.py

Output video will be saved at:
output/output_tracked.mp4

Download the Object Detection Model from here:
https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view
This model is basic fine tuned version of Ultralytics YOLLOv11, trained for palyerss, not the ball.
