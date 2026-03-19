import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    """
    Policy network for REINFORCE algorithm. 
    Returns action probabilities as opposed to Q-values.
    """

    def __init__(self, input_size, output_size, hidden_size=128):

        super(PolicyNet, self).__init__()

        #Defining my neural network
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        #Standard ReLU activation on input and hidden layers
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        
        #Softmax since we have 2 output neurons, it is multi-class classification task
        action_probabilities = F.softmax(self.output_layer(x), dim=-1)

        return action_probabilities
    
    def act(self, state):
        """
        Return an action based on network output
        """
        
        #Do a forward pass for given state
        action_probabilities = self.forward(state)

        #Create a distribution for action chances
        prob_distribution = Categorical(action_probabilities)

        #Sample action from distribution
        chosen_action = prob_distribution.sample()

        #Find log(prob()) as this is used for training in REINFORCE algorithm
        log_prob = prob_distribution.log_prob(chosen_action) #log policy(a,s)

        return chosen_action, log_prob
    

#Test if correct output
if __name__ == "__main__":
    
    input_size = 12
    output_size = 2

    network = PolicyNet(input_size, output_size, hidden_size=128)

    #Generate one random state based on input size
    state = torch.rand(1, input_size)

    action, log_prob = network.act(state)
    
    print(f"Action = {action}, Log Probability = {log_prob}")


