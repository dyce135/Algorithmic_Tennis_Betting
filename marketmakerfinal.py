import numpy as np

class MarketMaker:
    def __init__(self, simulated_market, win_probabilities):
        self.simulated_market = simulated_market
        self.positions = {runner["id"]: 0 for runner in simulated_market.runners}
        self.mid_price_ema = None
        self.ema_alpha = 0.1
        self.inventory_limit = 1000
        self.win_probabilities = win_probabilities
        self.profits = []
    
    def kelly_criterion(self, odds, probability):
        return (odds * probability - (1 - probability)) / odds
    
    def get_runner_traded_volume(self, runner_id):
        traded_volume = 0
        for market_data in self.simulated_market.market_data:
            for runner_data in market_data["bets"]:
                if runner_data["id"] == runner_id:
                    traded_volume += runner_data["tv"]

        return traded_volume
    
    def calculate_vwap(self, runner_id):
        total_volume = self.get_runner_traded_volume(runner_id)
        if total_volume == 0:
            return None

        weighted_sum = 0
        for market_data in self.simulated_market.market_data:
            for runner in market_data["bets"]:
                if runner["id"] == runner_id:
                    weighted_sum += runner["ltp"] * runner["tv"]

        return weighted_sum / total_volume

    def get_market_momentum(self, runner_id):
        ltp_list = []
        for market_data in self.simulated_market.market_data:
            for runner in market_data["bets"]:
                if runner["id"] == runner_id and runner["ltp"]:
                    ltp_list.append(runner["ltp"])

        if len(ltp_list) < 2:
            return 0

        return ltp_list[-1] - ltp_list[-2]

    def get_optimal_prices(self, best_back, best_lay, runner_id):
        # Calculate the VWAP and use it as the reference price
        vwap = self.calculate_vwap(runner_id)
        if vwap is None:
            vwap = (best_back + best_lay) / 2

        # (Existing logic for calculating optimal prices)
        position = self.positions[runner_id]
        position_risk = abs(position) / self.inventory_limit

        bid_adjustment = (1 - self.ema_alpha) * (1 + position_risk)
        ask_adjustment = (1 + self.ema_alpha) * (1 - position_risk)

        bid_price = vwap * bid_adjustment
        ask_price = vwap * ask_adjustment

        # Incorporate market momentum
        momentum = self.get_market_momentum(runner_id)
        bid_price += momentum
        ask_price += momentum

        return bid_price, ask_price

    def place_bets(self):
        for runner_data in self.simulated_market.market_data[-1]["bets"]:
            runner_id = runner_data["id"]
            best_back, best_lay = runner_data["odds"]["back"][0][0], runner_data["odds"]["lay"][0][0]

            bid_price, ask_price = self.get_optimal_prices(best_back, best_lay, runner_id)

            win_probability = self.win_probabilities[runner_id]

            kelly_back = self.kelly_criterion(bid_price, win_probability)
            kelly_lay = self.kelly_criterion(ask_price, 1 - win_probability)

            back_order_size = max(0, self.inventory_limit * kelly_back)
            lay_order_size = max(0, self.inventory_limit * kelly_lay)

            order_back = self.simulated_market.place_order(runner_id, bid_price, "back", back_order_size)
            order_lay = self.simulated_market.place_order(runner_id, ask_price, "lay", lay_order_size)

            profit = self.calculate_profit(2374763)
            self.profits.append(profit)

            if order_back is not None:
                self.positions[runner_id] += order_back["size"]
            if order_lay is not None:
                self.positions[runner_id] -= order_lay["size"]
                
    def calculate_profit(self, actual_outcome_id):
        profit = 0

        for runner_id in self.positions:
            position = self.positions[runner_id]
            last_odds = None

            for market_data in reversed(self.simulated_market.market_data):
                for runner_data in market_data["bets"]:
                    if runner_data["id"] == runner_id:
                        last_odds = runner_data["odds"]
                        break
                if last_odds is not None:
                    break

            if last_odds is not None:
                if runner_id == actual_outcome_id:
                    if position > 0 and last_odds["back"]:
                        profit += position * last_odds["back"][0][0]
                else:
                    if position < 0 and last_odds["lay"]:
                        profit -= position * last_odds["lay"][0][0]

        return profit
