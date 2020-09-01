import numpy as np
import re

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
        print(self.current_cards)
        
    def set_estimate(self, action):
        self.estimate_wins = action


def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]


class Game(object):
    def __init__(self, deck):
        self.player1 = Player(True)
        self.player2 = Player(False)
        self.current_trumph = None
        self.deck = deck 
        self.play_game = True
     
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
        cards2 = list(cards1)
        cards2.sort(key=natural_keys)
        self.player1.set_cards(cards1)
        self.player2.set_cards(cards2)
        self.current_trumph = str(np.random.choice(key_list, size=1))
        if round_idx % 2 ==  0:
            self.player2.token = True
        else:
            self.player2.token = False
            
    def evalute_game(self):
        """ If game is over add score to the player """
        
        if self.player1.current_wins == self.player1.estimate_wins:
            self.player1.score += 20 + (10 * self.player1.current_wins)
        
        if self.player2.current_wins == self.player2.estimate_wins:
            self.player2.score += 20 + (10 * self.player2.current_wins)
        
        print("Player 1 score {}".format(self.player1.score))
        print("Player 2 score {}".format(self.player2.score))
    
    def set_estimate(self, action1, action2):
        """ 
        
        """
        self.player1.estimate_wins = action1
        self.player2.estimate_wins = action2
        self.differnce_estimate = self.round_idx - action1 - action2
        
    def create_state_vector(self):
        # create a state vector # first 52 are the cards(1 has card 0 does not)
        # 51 # set trumph card to 2 
        # 52 plays the first card
        # 53 current estimate 
        # 54 current wins
        # 55 to many wins estimate 1, equal 0 to less -1
        key_list = []
        for key in self.deck.keys():
            key_list.append(key)
        print("create vector")
        state_vector_1 =  np.zeros(56)
        state_vector_2 =  np.zeros(56)
        for card in self.player1.current_cards:
            state_vector_1[key_list.index(card)] = 1
        state_vector_1[key_list.index(self.current_trumph[2:-2])] = 2
        state_vector_1[52] = 1 - self.player2.token
        state_vector_1[53] = self.player1.estimate_wins
        state_vector_1[54] = self.player1.current_wins
        state_vector_1[55] = self.differnce_estimate
        for card in self.player2.current_cards:
            state_vector_2[key_list.index(card)] = 1
        state_vector_2[key_list.index(self.current_trumph[2:-2])] = 2
        state_vector_2[52] = 0 + self.player2.token
        state_vector_2[53] = self.player2.estimate_wins
        state_vector_2[54] = self.player2.current_wins
        state_vector_2[55] = self.differnce_estimate
        return state_vector_1, state_vector_2
    
    def play_cards(self, player1_card, player2_card, round_idx):
        """ compare cards and set winner of the current round
            if round is mod 2 the player 2 starts else player 1
        """
        # remove cards form playershand
        
        self.player1.current_cards.remove(player1_card)
        self.player2.current_cards.remove(player2_card)
        if len(self.player1.current_cards) == 0:
            self.play_game = False
        current_trumph = self.current_trumph[2]
        player1_card = str(player1_card)
        player2_card = str(player2_card)
        print("player 2 stars", )
        print("current trump", current_trumph)
        print("play1 card ", player1_card)
        print("play2 card ", player2_card)
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
             # case no tumph
            else:
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
                    # different cards player2 wins
                    self.player2.token = False
                    self.player1.current_wins +=1
        print("player 1 wins {} player2 wins {}".format(self.player1.current_wins, self.player2.current_wins))
                    
                
       



def main():
    epochs = 100
    replay_buffer = []
    deck = createDeck()
    for ep in range(epochs):
        for i in range(1,3):
            game = Game(deck)
            print("start game with {} cards".format(i))
            game.reset_round(i)
            game.set_estimate(np.random.randint(i), np.random.randint(i))
            while game.play_game:
                action_1  = game.player1.current_cards[np.random.randint(len(game.player1.current_cards))]
                action_2  = game.player2.current_cards[np.random.randint(len(game.player1.current_cards))]
                game.play_cards(action_1, action_2, i)
                state_agent1, state_agent2 = game.create_state_vector()
                replay_buffer.append((state_agent1, state_agent2))
            game.evalute_game()


if __name__ == "__main__":
    main()
