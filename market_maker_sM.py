class MarketMaker:
    def __init__(self, simulated_market, spread=0.02):
        self.simulated_market = simulated_market
        self.spread = spread

    def calc_implied_prob(self, odds):
        return 1 / odds

    def calc_implied_odds(self, prob):
        return 1 / prob

    def calc_kelly_bet_size(self, implied_prob, actual_prob, odds):
        kelly_f = (implied_prob - actual_prob) / (odds - 1)
        return kelly_f

    def get_best_orders(self):
        best_bid = self.simulated_market[-1]["bets"][0]["stake"]
        best_ask = self.simulated_market[-1]["bets"][-1]["stake"]
        for bet in self.simulated_market[-1]["bets"]:
            if bet["player"]["odds"] > best_bid:
                best_bid = bet["player"]["odds"]
            if bet["player"]["odds"] < best_ask:
                best_ask = bet["player"]["odds"]
        return best_bid, best_ask

    def make_market(self, num_ticks=5):
        best_bid, best_ask = self.get_best_orders()
        mid_price = (best_bid + best_ask) / 2
        tick_size = (best_ask - best_bid) / (num_ticks - 1)

        # calculate implied probabilities for each tick
        probs = []
        for i in range(num_ticks):
            price = best_bid + i * tick_size
            prob = self.calc_implied_prob(price)
            probs.append(prob)

        # use the latest market snapshot to predict match outcome probabilities
        predicted_probs = [player["odds"] for player in self.simulated_market[-1]["bets"]]

        # calculate optimal bet sizes using Kelly Criterion
        optimal_bets = []
        for i in range(num_ticks):
            actual_prob = predicted_probs[i]
            kelly_f = self.calc_kelly_bet_size(probs[i], actual_prob, mid_price)
            optimal_bets.append(kelly_f)

        # print optimal bet sizes
        for i in range(num_ticks):
            print(f"Optimal bet {i + 1}: {optimal_bets[i]}")
