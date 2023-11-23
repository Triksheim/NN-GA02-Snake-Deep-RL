# Running Graded Assignment 2
## 1. Training
- Run file training.py to start training
    - Line 33 & 34 to change num of episodes and logging frequency
    - Line 18 & 19 to load pretrained model (choose iteration from models/v17.1)
    - Optional: Run supervised_training.py first to generate a buffer (avoids NaN errors)
## 2. Logging
- Logs saved in model_logs/v17.1 and model weights in models/v17.1
- Edit logging frequency in training.py
## 3. Visualization
- Run file game_visualization.py to generate a video file of snake playthrough
- Line 21 to change which model to visualize (choose iteration from models/v17.1)
- Mp4 file saved in images/
