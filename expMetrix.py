import numpy as np
from numpy import array
class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
            self.max_memory = max_memory
            self.discount = discount
            self.memory = list()
        
    def remember(self, states, game_over):
        #print("rembmer")
        #memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states,game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
    
    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        inputs = []
        dim = len(self.memory[0][0][0])
       
        targets = np.zeros((min(len_memory, batch_size), num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]
            data = state_t.reshape(-1)
            inputs.append(data)
            datatp1 = state_tp1.reshape(-1)
            targets[i] = model.predict(np.array([data]))[0]
            Q_sa = np.max(model.predict(np.array([datatp1]))[0])
            if game_over :
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.discount * Q_sa
            
            
        inputs = array(inputs)
        #print(len(inputs))
        #print(len(targets))
        #print("---")
        return inputs, targets