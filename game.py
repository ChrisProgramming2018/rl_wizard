import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
from torch.utils.tensorboard import SummaryWriter
import argparse
from collections import deque


from agent import Agent




def createDeck():
    """ Creates a default deck which contains all 52 cards and returns it. """
    deck = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13']
    deck1 = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13']
    deck2 = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13']
    deck3 = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12', 'Y13']
    values = range(1,14)
    deck_dict = {} 
    deck_dict2 = {} 
    deck_dict3 = {} 
    deck_dict4 = {} 
    for d, v in zip(deck, values):
        deck_dict.update({d:v})

    for d, v in zip(deck1, values):
        deck_dict2.update({d:v})
    for d, v in zip(deck2, values):
        deck_dict3.update({d:v})
    for d, v in zip(deck3, values):
        deck_dict4.update({d:v})
    deck_dict.update(deck_dict2)
    deck_dict.update(deck_dict3)
    deck_dict.update(deck_dict4)
    return deck_dict


class Player:
    def __init__(self, token):
        self.score = 0
        self.current_wins = 0
        self.estimate_wins = 0
        self.current_cards = []
        self.token = token
    
    def set_cards(self, cards):
        self.current_cards = cards
        # print(self.current_cards)
        
    def set_estimate(self, action):
        self.estimate_wins = action

    def set_token(self, token):
        self.token = token

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]


def write_into_file(text, file_name='document.csv'):
    """
    """
    with open(file_name,'a', newline='\n') as fd:
        fd.write(str(text)+"\n")


class Game(object):
    def __init__(self, args, deck):
        self.player1 = Player(True)
        self.player2 = Player(False)
        self.current_trumph = None
        self.deck = deck 
        self.play_game = True
        self.size = args.state_size
        if args.seed == 0:
            np.random.seed(args.seed)

    def reset_round(self, round_idx):
        """ round_idx indicates the amount of cards each player gets 
            and which player starts to tell how many wins he expect to make
            it also the constrant for the last player is (not allowed) to say how many
        
        """
        self.round_idx = round_idx
        self.differnce_estimate = 0
        key_list = []
        for key in self.deck.keys():
            key_list.append(key)
        cards1 =  np.random.choice(key_list, size=round_idx, replace=False)
        [key_list.remove(card) for card in cards1]
        cards1 = list(cards1)
        cards1.sort(key=natural_keys)
        cards2 =  np.random.choice(key_list, size=round_idx, replace=False)
        [key_list.remove(card) for card in cards2]
        cards2 = list(cards2)
        cards2.sort(key=natural_keys)
        self.player1.set_cards(cards1)
        self.player2.set_cards(cards2)
        self.current_trumph = str(np.random.choice(key_list, size=1))
        if round_idx % 2 ==  0:
            self.player2.token = True
        else:
            self.player2.token = False
        self.done = 0

        text = 'card1 {} cards2 {} trumph {}'
        text = text.format(cards1, cards2, self.current_trumph)
        write_into_file(text)

    def evalute_game(self):
        """ If game is over add score to the player """
        
        if self.player1.current_wins == self.player1.estimate_wins:
            self.player1.score += 20 + (10 * self.player1.current_wins)
        
        if self.player2.current_wins == self.player2.estimate_wins:
            self.player2.score += 20 + (10 * self.player2.current_wins)
        
        # print("Player 1 score {}".format(self.player1.score))
        # print("Player 2 score {}".format(self.player2.score))
        # print("Player 1 est {}".format(self.player1.estimate_wins))
            
        #print("Player 2 est {}".format(self.player2.estimate_wins))
    def set_buffer(self, buffer_1, buffer_2, writer, step):
        writer.add_scalar('Reward player 1', self.player1.score , step)
        writer.add_scalar('Reward player 2', self.player2.score , step)
        text = 'player1 score {} est {} startsgame {} p2 est {}'
        text = text.format(self.player1.score, self.player1.estimate_wins, self.player1.token, self.player2.estimate_wins)
        write_into_file(text)
        buffer_1_updated = []
        for traj in buffer_1:
            traj.append(self.player1.score)
            buffer_1_updated.append(traj)
        
        
        buffer_2_updated = []
        for traj in buffer_2:
            traj.append(self.player2.score)
            buffer_2_updated.append(traj)

        return buffer_1_updated, buffer_2_updated
    
    def set_estimate(self, action1, action2):
        """ 
        
        """
        self.player1.estimate_wins = action1
        self.player2.estimate_wins = action2
        self.differnce_estimate = self.round_idx - action1 - action2
        
    def create_state_vector(self, estimate):
        # create a state vector # first 52 are the cards(1 has card 0 does not)
        # 51 # set trumph card to 2 
        # 52 plays the first card
        # 53 current estimate 
        # 54 current wins
        # 55 to many wins estimate 1, equal 0 to less -1
        # 56 estimate 1 else 0
        key_list = []
        for key in self.deck.keys():
            key_list.append(key)
        state_vector_1 =  np.zeros(self.size)
        state_vector_2 =  np.zeros(self.size)
        state_vector_1 -= 1
        state_vector_2 -= 1

        for idx, card in enumerate(self.player1.current_cards):
            state_vector_1[key_list.index(card)] = idx
        
        state_vector_1[key_list.index(self.current_trumph[2:-2])] = -2
        state_vector_1[52] = 1 - self.player2.token
        state_vector_1[53] = self.player1.estimate_wins
        state_vector_1[54] = self.player1.current_wins
        state_vector_1[55] = self.differnce_estimate
        state_vector_1[56] = estimate
        
        for idx, card in enumerate(self.player2.current_cards):
            state_vector_2[key_list.index(card)] = idx
        state_vector_2[key_list.index(self.current_trumph[2:-2])] = -2
        state_vector_2[52] = 0 + self.player2.token
        state_vector_2[53] = self.player2.estimate_wins
        state_vector_2[54] = self.player2.current_wins
        state_vector_2[55] = self.differnce_estimate
        state_vector_2[56] = estimate

        return torch.Tensor(state_vector_1), torch.Tensor(state_vector_2)
    
    def play_cards(self, player1_card, player2_card):
        """ compare cards and set winner of the current round
            if round is mod 2 the player 2 starts else player 1
        """
        # remove cards form playershand
         
        player1_card  = self.player1.current_cards[player1_card]
        player2_card  = self.player2.current_cards[player2_card]
        self.player1.current_cards.remove(player1_card)
        self.player2.current_cards.remove(player2_card)
        if len(self.player1.current_cards) == 0:
            self.play_game = False
        
        self.done = 0 + self.play_game
        current_trumph = self.current_trumph[2]
        player1_card = str(player1_card)
        player2_card = str(player2_card)
        

        P1_wins = self.player1.current_wins
        P2_wins = self.player2.current_wins
        
        #print("player 2 stars")
        #print("current trump", current_trumph)
        #print("play1 card ", player1_card)
        #print("play2 card ", player2_card)
        if self.player2.token:
            played_color = player2_card[0]
            # case played_card is trumph
            #print("played color", played_color)
            #print("1 number",int(player1_card[1:]))
            #print("2 number",int(player2_card[1:]))
            if played_color == current_trumph:
                # case 1 player one has also trump
                if player1_card[0] == current_trumph:
                    # check which is higher and set winner
                    if int(player1_card[1:]) > int(player2_card[1:]):
                        
                        self.player1.current_wins += 1
                        self.player2.token = False
                    else:
                        self.player2.token = True
                        self.player2.current_wins +=1
                else:
                    # only player 2 has trumph
                    self.player2.token = True
                    self.player2.current_wins +=1
            # case no tumph
            else:
                # check if same color
                if played_color == player1_card[0]:
                    # check how is higher
                    if int(player1_card[1:]) > int(player2_card[1:]):
                        
                        self.player1.current_wins += 1
                        self.player2.token = False
                    else:
                        self.player2.token = True
                        self.player2.current_wins +=1
                else:
                    # different cards player2 wins
                    self.player2.token = True
                    self.player2.current_wins +=1
                    
        else:
            played_color = player1_card[0]
            if played_color == current_trumph:
                if player2_card[0] == current_trumph:
                    # check which is higher and set winner
                    if int(player1_card[1:]) > int(player2_card[1:]):
                        self.player1.current_wins += 1
                        self.player2.token = False
                    else:
                        self.player2.token = True
                        self.player2.current_wins +=1
                else:
                    # only player 2 has trumph
                    self.player1.current_wins +=1
                    self.player2.token = False
            else:
                # case no tumph of player 1
                # check if same color
                if played_color == player2_card[0]:
                    # check how is higher
                    if int(player1_card[1:]) > int(player2_card[1:]):
                        self.player2.token = False
                        self.player1.current_wins += 1
                    else:
                        self.player2.token = True
                        self.player2.current_wins +=1
                else:
                    # check if player 2 has trump
                    if player2_card[0] == current_trumph:
                        # print("won")
                        self.player2.token = True
                        self.player2.current_wins +=1
                    else:
                        # print("p1 starts with different color and p2 no trumph")
                        self.player2.token = False
                        self.player1.current_wins +=1

        text = "no one won"
        if P1_wins < self.player1.current_wins:
            text = "P1 won"
        if P2_wins < self.player2.current_wins:
            text = "P2 won"
        
        write_into_file(text)
        
        #print("player 1 wins {} player2 wins {}".format(self.player1.current_wins, self.player2.current_wins))
                    
                
       



def main(args):
    epochs = args.epochs
    #agent = Agent(state_size=8, action_size=4, seed=0)
    #agent1_est =[Agent(args.state_size, 2), Agent(args.state_size, 3)]
    #agent2_est =[Agent(args.state_size, 2), Agent(args.state_size, 3)]
    agent1_est =[Agent(args.state_size, 2, args)]
    agent2_est =[Agent(args.state_size, 2, args)]
    agent1_act =[Agent(args.state_size, 2, args)]
    agent2_act =[Agent(args.state_size, 2, args)]
    #agent1_buffer = [RandomMemory(args.buffer_size, args.batch_size), RandomMemory(args.buffer_size, args.batch_size)]
    #agent2_buffer = [RandomMemory(args.buffer_size, args.batch_size), RandomMemory(args.buffer_size, args.batch_size)]
    deck = createDeck()
    total_time_steps = 0
    pathname = "wizard"
    tensorboard_name = str(args.locexp) + '/runs/' + pathname
    writer = SummaryWriter(tensorboard_name)
    scores_window1 = deque(maxlen=100) 
    scores_window2 = deque(maxlen=100) 
    eps_end = 0.01
    eps = 1.
    eps_decay = 0.999995
    agent1_actions = []
    agent2_actions = []
    agent1_action_mean = []
    agent2_action_mean = []
    
    #writer = SummaryWriter()
    for ep in range(epochs):
        if ep % 1000 == 0:
            print("epoch {} ".format(ep))
        for i in range(1,2):
            game = Game(args, deck)
            # print("start game with {} cards".format(i))
            buffer_1 = []
            buffer_2 = []
            buffer_2_est = []
            game.reset_round(i)
            state_agent1, state_agent2 = game.create_state_vector(1)
            estimate1 =  agent1_est[i-1].act(state_agent1, eps) 
            estimate2 = agent2_est[i-1].act(state_agent2, eps)
            game.set_estimate(estimate1, estimate2)
            agent1_actions.append(estimate1)
            agent2_actions.append(estimate2)
            while game.play_game:
                total_time_steps += 1
                # action_1  = np.random.randint(len(game.player1.current_cards))
                # if only on card to play
                if len(game.player1.current_cards) == 1:
                    action_1 = 0
                    action_2 = 0
                else:
                    action_1  = agent1_act[i - 2].act(state_agent1)
                    action_2  = agent2_act[i - 2].act(state_agent2)
                game.play_cards(action_1, action_2)
                next_state_agent1, next_state_agent2 = game.create_state_vector(0)
                buffer_1.append([state_agent1, action_1, next_state_agent1, 0])
                buffer_2.append([state_agent2, action_2, next_state_agent2, 0])
                buffer_2_est.append([state_agent2, estimate2, next_state_agent2, 0])
                state_agent1 = next_state_agent1
                state_agent2 = next_state_agent2
            game.evalute_game()
            buffer_1, buffer_2 = game.set_buffer(buffer_1, buffer_2_est, writer, total_time_steps)
            eps = max(eps_end, eps_decay*eps)
            scores_window1.append(game.player1.score)
            scores_window2.append(game.player2.score)
            mean1 = np.mean(scores_window1)
            mean2 = np.mean(scores_window2)
            if total_time_steps % 1000 == 0:
                agent1_action_mean.append(np.mean(agent1_actions))
                agent1_actions = []
                print("agent 1 action mean ", agent1_action_mean[-1])
                agent2_action_mean.append(np.mean(agent2_actions))
                agent2_actions = []
                print("agent 2 action mean ", agent2_action_mean[-1])
                #plt.plot(agent1_actions)
                #plt.show()
            text = 'Episode = {}:  P1  score {} P2 score {}'
            text = text.format(total_time_steps, game.player1.score, game.player2.score)
            #print(text)
            
            text = 'Episode = {}:  P1  Avg Return = {:.2f} P2 Avg Reward {:.2f} eps {:.2f}'
            text = text.format(total_time_steps, mean1, mean2, eps)
            print(text, end='\r')
            sys.stdout.flush()
            writer.add_scalar('Reward mean player 1', mean1 , total_time_steps)
            writer.add_scalar('Reward mean player 2', mean2 , total_time_steps)
            i = 0
            for traj1, traj2 in zip(buffer_1, buffer_2_est):
                agent1_est[i].step(traj1[0], traj1[1], traj1[4], traj1[2], traj1[3])  # state action reward newstate done
                agent2_est[i].step(traj2[0], traj2[1], traj2[4], traj2[2], traj2[3])  # state action reward newstate done
                # agent1_act[i].step(traj1[0], traj1[1], traj1[4], traj1[2], traj1[3])  # state action reward newstate done
                #agent2_act[i].step(traj2[0], traj2[1], traj2[4], traj2[2], traj2[3])  # state action reward newstate done
                # print("reward player 2 {} action {} ".format(traj2[4], traj2[1]))
                break
                i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--state_size', type=int, default=58, help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=int(1e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--hidden-size-1', type=int, default=128, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--hidden-size-2', type=int, default=64, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='sigma', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=10, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--multi-step', type=int, default=7, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.999, metavar='gamma', help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(4), metavar='tau', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='mue', help='Learning rate')
    parser.add_argument('--tau', type=float, default=1e-3, metavar='eps', help='Adam epsilon')
    parser.add_argument('--batch_size', type=int, default=64, metavar='SIZE', help='Batch size')
    parser.add_argument('--learn-start', type=int, default=int(5000), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=1000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
    parser.add_argument('--evaluation-size', type=int, default=50000, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--update-every', default=10000)
    parser.add_argument('--eps_decay', type=float, default=0.99)
    parser.add_argument('--n_episodes', default=2000)
    parser.add_argument('--locexp', type=str , default = 'sep')
    parser.add_argument('--device', default='cuda')
    arg = parser.parse_args() 
    main(arg)
