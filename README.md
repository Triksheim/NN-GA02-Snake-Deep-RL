# Running Graded Assignment 2
## 1. Training
**To avoid NaN value error at beginning of training load the supplied buffer that is generated from BFS
or generate a new buffer using supervised_learning.py.**

**models/17.1/buffer_0001 automatically gets loaded if it exists.**

- Run file training.py to start training
    - Line 33 & 34 to change num of episodes and logging frequency
    - Line 18 & 19 to load pretrained model (choose iteration from models/v17.1)
    
## 2. Logging
- Logs saved in model_logs/v17.1 and model weights in models/v17.1
- Edit logging frequency in training.py
  
## 3. Visualization
- Run file game_visualization.py to generate a video file of snake playthrough
- Line 21 to change which model to visualize (choose iteration from models/v17.1)
- Mp4 file saved in folder /images/


# Results
I have saved the best test results, model weights and visualizations in folder /best_results/

## Game visualization of best performing agent trained with DQN

| ![500k](https://github.com/Triksheim/NN-GA02-Snake-Deep-RL/blob/main/best_results/500k_episodes_score32.gif) | ![1000k](https://github.com/Triksheim/NN-GA02-Snake-Deep-RL/blob/main/best_results/1000k_episodes_score39.gif) |
|:--:|:--:|
| 500k episodes: Score: 32 | 1000k episodes: Score: 39 |

| ![1500k](https://github.com/Triksheim/NN-GA02-Snake-Deep-RL/blob/main/best_results/1500k_episodes_score52.gif) | ![1500k_length_reward](https://github.com/Triksheim/NN-GA02-Snake-Deep-RL/blob/main/best_results/1500k_score54_length_reward.gif) |
|:--:|:--:|
| 1500k episodes: Score: 52 | 1500k episodes with length based reward system: Score: 54 |
