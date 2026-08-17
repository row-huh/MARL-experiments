## Multi Agent Reinforcement Learning

**MARL Approaches**
De-Centralized Architecture - Each agent is trained independently, no information is shared between the agents


2 agents -> simpler puzzles -> must solve by cooperating

Rewards -> hardcoded 'good' moves for example if the agents (in order to jump from one obstacle need a higher height and the agents stand on top of each other) then they will get a reward. 
If end-gate reached then also a reward

Penalty if obstacle is hit, also if fallen into a ditch or something. 

Start with one map: only one obstacle that needs agents to get on top of each other to get to this ledge which has the key. grabbing the key also gives rewards and going to the door also gives a reward. door is unlocked when you have the key and walking into the door with the key is 'success' - terminal state so to speak. 