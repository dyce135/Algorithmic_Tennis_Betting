import numpy as np
import pandas as pd

# Rules (Based on wimbledon men's singles):
# Best of 5 sets
# 7-point tiebreaker in 1st to 4th set
# 10-point tiebreaker in 5th set


# Point level simulation
def simulate_point(transition_matrix, server):
    # Generate a random number between 0 and 1
    random_num = np.random.rand()

    # Use the random number and the current state to determine the outcome
    # of the point (i.e. which player wins the point)
    if server == 1:  # Player A serving, Player B receiving
        if random_num <= transition_matrix[0, 0]:  # Player A wins the point
            current_state = 0
        else:  # Player B wins the point
            current_state = 1
    else:  # Player B serving, Player A receiving
        if random_num <= transition_matrix[1, 1]:  # Player B wins the point
            current_state = 1
        else:  # Player A wins the point
            current_state = 0
    
    # Return the updated current state
    return current_state


# Game level simmulation
def simulate_game(transition_matrix, server):

    # Initialize the scores for each player
    player_a_score = 0
    player_b_score = 0

    # Play out each point until one player wins the game by two points
    while 1:
        # Simulate a point being played
        current_state = simulate_point(transition_matrix, server)

        # Update the scores based on the outcome of the point
        if current_state == 0:  # Player A won the point
            player_a_score += 1
        else:  # Player B won the point
            player_b_score += 1

        if (player_a_score >= 4 or player_b_score >= 4) and abs(player_a_score - player_b_score) >= 2:
            break

    # Return the winner of the match
    if player_a_score > player_b_score:
        return 0
    else:
        return 1


# Set level simulation
def simulate_set(transition_matrix, server, set_count):

    # Initialize the scores for each player
    player_a_score = 0
    player_b_score = 0

    # Play out each point until one player wins the game by two points
    while 1:
        # Simulate a point being played
        current_state = simulate_game(transition_matrix, server)

        # Serving player switches for the next game
        if server == 0:
            server = 1
        elif server == 1:
            server = 0

        # Update the scores based on the outcome of the point
        if current_state == 0:  # Player A won the point
            player_a_score += 1
        else:  # Player B won the point
            player_b_score += 1

        # Play until a player with 6 or more points wins by 2 points
        if (player_a_score >= 6 or player_b_score >= 6) and abs(player_a_score - player_b_score) >= 2:
            break
        elif player_a_score == 6 and player_b_score == 6:
            # Play a tiebreak when both players reach 6 points

            if set_count == 5:
                # Special tiebreak in the final deciding set
                current_state = simulate_fifthTiebreak(transition_matrix, server)

                # Switch servers
                if server == 0:
                    server = 1
                elif server == 1:
                    server = 0
            else:
                # Play tiebreak
                current_state = simulate_tiebreak(transition_matrix, server)

                # Switch servers
                if server == 0:
                    server = 1
                elif server == 1:
                    server = 0

            # Return the winner of the set and the next serving player
            if current_state == 0:
                # Player a wins
                if server == 0:
                    # Player b serves next
                    return 0
                else:
                    # Player a serves next
                    return 2
            else:
                if server == 0:
                    return 1
                else:
                    return 3

    # Return the winner of the set and the next serving player
    if player_a_score > player_b_score:
        # Player a wins
        if server == 0:
            # Player b serves next
            return 0
        else:
            # Player a serves next
            return 2
    else:
        if server == 0:
            return 1
        else:
            return 3


def simulate_tiebreak(transition_matrix, server):

    # Initialize the scores for each player
    player_a_score = 0
    player_b_score = 0

    # Simulate a point being played (First player begins serving for one point only)
    current_state = simulate_point(transition_matrix, server)

    # Update the scores based on the outcome of the point
    if current_state == 0:  # Player A won the point
        player_a_score += 1
    else:  # Player B won the point
        player_b_score += 1

    while 1:

        # Players serve for two points for the rest of the game
        for i in range(2):
            # Simulate a point being played
            current_state = simulate_point(transition_matrix, server)

            # Update the scores based on the outcome of the point
            if current_state == 0:  # Player A won the point
                player_a_score += 1
            else:  # Player B won the point
                player_b_score += 1

            # Play until a player reaches 7 points win a 2 point margin
            if (player_a_score >= 7 or player_b_score >= 7) and abs(player_a_score - player_b_score) >= 2:

                # Return the winner of the match
                if player_a_score > player_b_score:
                    return 0
                else:
                    return 1

        # Switch serving player
        if server == 0:
            server = 1
        else:
            server = 0


def simulate_fifthTiebreak(transition_matrix, server):

    # Initialize the scores for each player
    player_a_score = 0
    player_b_score = 0

    # Simulate a point being played (First player begins serving for one point only)
    current_state = simulate_point(transition_matrix, server)

    # Update the scores based on the outcome of the point
    if current_state == 0:  # Player A won the point
        player_a_score += 1
    else:  # Player B won the point
        player_b_score += 1

    while 1:

        # Players serve for two points for the rest of the game
        for i in range(2):
            # Simulate a point being played
            current_state = simulate_point(transition_matrix, server)

            # Update the scores based on the outcome of the point
            if current_state == 0:  # Player A won the point
                player_a_score += 1
            else:  # Player B won the point
                player_b_score += 1

            # Play until a player reaches 10 points win a 2 point margin
            if (player_a_score >= 10 or player_b_score >= 10) and abs(player_a_score - player_b_score) >= 2:

                # Return the winner of the match
                if player_a_score > player_b_score:
                    return 0
                else:
                    return 1

        # Switch serving player
        if server == 0:
            server = 1
        else:
            server = 0



def simulate_match(transition_matrix):
    # First serve
    server = np.random.randint(0,1)

    # Initialize the scores for each player
    player_a_score = 0
    player_b_score = 0

    # Set counter for a 10-point tiebreak
    set_count = 1

    # Play out each set until one player wins the match (Best of 5)
    while player_a_score < 3 and player_b_score < 3:
        # Simulate a match being played
        set_winner = simulate_set(transition_matrix, server, set_count)

        set_count += 1

        # Update the scores and next serve based on the outcome of the match
        if set_winner == 0:
            player_a_score += 1
            server = 0
        elif set_winner == 2:
            player_a_score += 1
            server = 1
        elif set_winner == 1:
            player_b_score += 1
            server = 0
        elif set_winner == 3:
            player_b_score += 1
            server = 1

    # Return the winner of the match
    if player_a_score > player_b_score:
        return 0
    else:
        return 1

