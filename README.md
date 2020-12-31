# Human-Object Interaction(HOI) For Basketball Videos #
This repository contains my dissertation. This project exploit features of the detected players like their gaze-orientation, pose and outfit using state-of-the-art machine learning methods to build a HOI pipeline made out multiple components that included **player and ball localization and a player interaction classification module** was built. For each component multiple approaches were considered and evaluated, the best performing methods were integrated into the HOI pipeline. The **functionality to track individual players and their shot attempt was also implemented successfully**.

![](./images/demo.gif)

### Project Objectives 
- [x] Develop an simple HOI detection pipeline that is capable of localizing the player and the ball and the interaction between them(analogous to player action) when in not-occluded scenarios. 

- [x] Enable HOI pipeline to do the classification of action and localization of players and the ball that takes into account occlusion. 

- [x] Improve the pipeline such that it can classify the action of all the players and have the ability to track specific player’s shot attempts.


## INSTALLATION ##
- Install all the python librabies in the requirement text
- [Download the models for Action Classification, Ball Detection, Team Detection, Vptrees](https://drive.google.com/file/d/1BCw4_bjZXpVlE51HuPyCETCUbRt1N6nd/view?usp=sharing)
    - Put the downloaded folders inside the **models** folder
- Install Node.js
    - https://nodejs.org/en/download/
    - After Node.js installation:
        - Go inside "posenetSever" folder
        - run the following command: npm install
            - This will install all the javascript libraries used that is saved in the **package.json** file
    - Run the TeamDectection.py file
        - **python TeamDectection**


## Dataset 
[SpaceJam: a Dataset for Basketball Action Recognition](https://github.com/simonefrancia/SpaceJam) is used for used for the project. A varition of this dataset which consist of player poses (L2-Normalized) was generated for action-classification, [download here](https://drive.google.com/file/d/1lrju16Xz0XWliCGTf5TlZPKfjGCnQSJV/view?usp=sharing).



## Link to dissertation #
- [Disseration pdf](https://drive.google.com/file/d/1DCNEAR5iROFBwBRPVpynTdWAMRniDQ65/view?usp=sharing)

