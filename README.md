


MuZero - Lite
======

This repository is a Python implementation of a heavily modified MuZero algorithm. The current configuration learns to play Super Mario Bros, but with modification it could be used to learn a number of OpenAI gym environments.

Examples
======

Below are examples of gameplay after 6 days of training on a gaming PC. This AI was trained on random overworld levels.

![](1-1.gif) ![](2-1.gif) ![](3-1.gif) ![](3-2.gif) ![](4-1.gif) ![](5-1.gif) ![](5-2.gif) ![](6-1.gif) ![](7-1.gif) ![](8-1.gif) ![](8-2.gif) ![](8-3.gif) 


Neural Networks
======
 - Representation network: observation --> hidden state
 - Policy network: hidden state --> relative predicted value of each move
 - Value network: hidden state --> predicted value
 - Reward network: hidden state --> predicted reward
 - Dynamics network: hidden state + action --> hidden state (future)
 - Exploration network: hidden state --> predicted policy

![](NetworkDiagram.png)


Self Play
======
- Instead of performing a Monte-Carlo tree search, actions are selected through a truncated search. The outputs of the Policy network are ranked and the highest-ranking moves are evaluated through a look-ahead search. 
- Games are saved to file instead of loaded directed into the replay_buffer
- Gifs of each game are saved during training because they are fun to watch


Training
======
- Games are loaded from file to fill the replay_buffer
- Experiences are sampled using Prioritized Experience Replay
- Policy network is trained to the normalized value of each move from the truncated search during self play
- Value network is trained to the time-discounted future value + discounted accumulated future reward
- Reward network is trained to the reward values
- Exploration network is trained to the predicted policy from the Policy network. The loss value for the policy network is reduced by a fraction of the exploration network's loss.



**DISCLAIMER**: this code is early research code. What this means is:

- Silent bugs may exist.
- It may not work reliably on other environments or with other hyper-parameters.
- The code quality and documentation are quite lacking, and much of the code might still feel "in-progress".
- The training and testing pipeline is not very advanced.
