import random

def Game(player_choice, player_score, computer_score):
    # 1: Rock, 2: Paper, 3: Scissor
    computer_choice = random.randint(1, 3)
    status = ""
    
    if player_choice == computer_choice:
        status = "Draw"
    elif player_choice == 1: # Rock
        if computer_choice == 2: # Paper
            status = "Computer Wins"
            computer_score += 1
        else: # Scissor
            status = "Player Wins"
            player_score += 1
    elif player_choice == 2: # Paper
        if computer_choice == 1: # Rock
            status = "Player Wins"
            player_score += 1
        else: # Scissor
            status = "Computer Wins"
            computer_score += 1
    elif player_choice == 3: # Scissor
        if computer_choice == 1: # Rock
            status = "Computer Wins"
            computer_score += 1
        else: # Paper
            status = "Player Wins"
            player_score += 1
            
    return status, player_score, computer_score, computer_choice