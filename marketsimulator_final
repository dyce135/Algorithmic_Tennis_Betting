import json
import time
from marketmaker2 import MarketMaker

def json_file_generator(json_file):
    with open(json_file, 'r') as file:
        for line in file:
            yield json.loads(line)

def load_and_process_json_data(json_file):
    json_data_gen = json_file_generator(json_file)

    market_definitions = None
    runners = None

    for entry in json_data_gen:
        if 'marketDefinition' in entry['mc'][0]:
            market_definitions = entry['mc'][0]['marketDefinition']
            runners = market_definitions['runners']
            break

    return market_definitions, runners



class SimulatedMarket:
    def __init__(self, market_definition, runners):
        self.market_definition = market_definition
        self.runners = runners
        self.odds = {runner["id"]: {"back": [], "lay": []} for runner in runners}
        self.market_data = []

    def update_odds(self, runner_id, odds, side, tv=None, ltp=None):
        self.odds[runner_id][side] = odds
        if tv is not None:
            self.odds[runner_id]["tv"] = tv
        if ltp is not None:
            self.odds[runner_id]["ltp"] = ltp

    def display_market(self):
        print("Market:", self.market_definition["name"])
        for runner in self.runners:
            print(runner["name"])
            print("Back:", self.odds[runner["id"]]["back"])
            print("Lay:", self.odds[runner["id"]]["lay"])
            print()

    def process_odds_data(self, odds_update):
        snapshot = {"bets": []}
        for item in odds_update:
            runner_id = item["id"]
            if "atb" in item:
                self.update_odds(runner_id, item["atb"], side="back")
            if "atl" in item:
                self.update_odds(runner_id, item["atl"], side="lay")
            snapshot["bets"].append({
                "id": runner_id, 
                "odds": self.odds[runner_id],
                "tv": item.get("tv", 0),  # Adding the 'tv' value
                "ltp": item.get("ltp", 0)  # Adding the 'ltp' value
            })
        self.market_data.append(snapshot)


    def place_order(self, runner_id, price, side, order_size):
        if side == "back":
            current_best_back = self.odds[runner_id]["back"][0][0] if self.odds[runner_id]["back"] else 0
            if price > current_best_back:
                self.odds[runner_id]["back"].insert(0, [price, order_size])
            else:
                self.odds[runner_id]["back"].append([price, order_size])
        elif side == "lay":
            current_best_lay = self.odds[runner_id]["lay"][0][0] if self.odds[runner_id]["lay"] else float("inf")
            if price < current_best_lay:
                self.odds[runner_id]["lay"].insert(0, [price, order_size])
            else:
                self.odds[runner_id]["lay"].append([price, order_size])



    def simulate_real_time_market(self, market_maker, json_file, delay=1):
        json_data_gen = json_file_generator(json_file)

        for entry in json_data_gen:
            if 'marketDefinition' in entry['mc'][0]:
                self.market_definition = entry['mc'][0]['marketDefinition']
                self.runners = self.market_definition['runners']

            if 'rc' in entry['mc'][0]:
                self.process_odds_data(entry['mc'][0]['rc'])
                market_maker.place_bets()
                self.display_market()
                print("\n---\n")
                time.sleep(delay)

# Use the function to load and process the JSON data
market_definitions, runners = load_and_process_json_data("1.145385390.json")

# Create an instance of the SimulatedMarket class using the loaded data
market = SimulatedMarket(market_definitions, runners)


# Create an instance of the MarketMaker class
win_probabilities = {
    4105819: 0.4,  # Runner 1 has a 60% chance of winning
    2474763: 0.6   # Runner 2 has a 40% chance of winning
}
market_maker = MarketMaker(market,win_probabilities)

# Simulate real-time market updates with a 1-second delay between each update
market.simulate_real_time_market(market_maker, "1.145385390.json", delay=0)

# Calculate the market maker's profit
actual_outcome_id = 2474763  # Change this to the actual player ID of the match outcome
profit = market_maker.calculate_profit(actual_outcome_id)
print(profit)
