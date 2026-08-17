## Multi Agent Reinforcement Learning


Experience Buffer - storing all experiences
Parameter Sharing - exchanging weights with each other (is that parans? that wont make sense would it- sharing weights would make all agents learn similar patterns no? )


**MARL Approaches**
De-Centralized Architecture - Each agent is trained independently, no information is shared between the agents

2 agents -> simpler puzzles -> must solve by cooperating

Rewards -> hardcoded 'good' moves for example if the agents (in order to jump to a ledge, the agents need a higher height and the agents stand on top of each other) to jump on the ledge to get the reward then they will get a reward. 
If end-gate reached then also a reward

Penalty if obstacle is hit, there will be a random object like a pteranodon or something that will 
Start with one map: only one obstacle that needs agents to get on top of each other to get to this ledge which has the key. grabbing the key also gives rewards and going to the door also gives a reward. door is unlocked when you have the key and walking into the door with the key is 'success' - terminal state so to speak. 


the interface is kind of like: 

                                                                                            key
                                                                                        ___________(too high, need the agents to stand on each other then one jumps up to get the key)

__________obstacle here(jumping over it is possible, no coordination needded)__________

agent can do - forward, backward, jump (3 actions)

to keep it simple first let's have the agents learn to jump over obstacles; 

________obstacle_________key______door

simple architecture first, only 2 agents getting to the door