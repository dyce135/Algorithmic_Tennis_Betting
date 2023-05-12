import json
import time
import pandas as pd
from marketmaker2 import MarketMaker
# from simple_market_maker import SimpleMarketMaker

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
    def __init__(self, market_definition, runners,market_data):
        self.market_definition = market_definition
        self.runners = runners
        self.odds = {runner["id"]: {"back": [], "lay": []} for runner in runners}
        self.market_data = []
        self.match_data = match_data

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


    def get_market_depth(self, runner_id):
        market_depth = 0
        for market_data in self.market_data:
            for runner in market_data["bets"]:
                if runner["id"] == runner_id:
                    if runner["odds"]["back"]:
                        market_depth += sum([price_data[1] for price_data in runner["odds"]["back"]])
                    if runner["odds"]["lay"]:
                        market_depth += sum([price_data[1] for price_data in runner["odds"]["lay"]])
        return market_depth

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
        return order_size  # Add this line to return the order size




    def simulate_real_time_market(self, market_maker,score_update_interval, json_file, delay=0,sampling_interval = 20):
        json_data_gen = json_file_generator(json_file)
        current_point = 0
        iteration = 1  # Add this line to keep track of the number of iterations

        for i, entry in enumerate(json_data_gen):
            if i % sampling_interval != 0:
                continue
            if 'marketDefinition' in entry['mc'][0]:
                self.market_definition = entry['mc'][0]['marketDefinition']
                self.runners = self.market_definition['runners']

            if 'rc' in entry['mc'][0]:
                self.process_odds_data(entry['mc'][0]['rc'])
                if iteration % score_update_interval == 0:
                    if current_point < len(self.match_data):
                        market_maker.update_scores_from_row(self.match_data.iloc[current_point])
                        current_point += 1
                market_maker.place_bets()
                # simple_mm.place_bets()
                print(market_maker.positions)
                print(iteration)
                self.display_market()
                print("\n---\n")
                print(delay)
                time.sleep(delay)
                iteration +=1

# Use the function to load and process the JSON data
market_definitions, runners = load_and_process_json_data("1.145392229.json")

match_data = pd.read_csv("match_points.csv")  # Replace with the path to your Excel file
market = SimulatedMarket(market_definitions, runners, match_data)


# Create an instance of the MarketMaker class

market_maker = MarketMaker(market)

# simple_mm = SimpleMarketMaker(market)


# Simulate real-time market updates with a 1-second delay between each update
market.simulate_real_time_market(market_maker, 48, "1.145392229.json", delay=0)


# Calculate the market maker's profit
actual_outcome_id = 2519549  # Change this to the actual player ID of the match outcome
metrics = market_maker.calculate_metrics()
# metrics2 = simple_mm.calculate_metrics()
print(metrics)
# print(metrics2)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(range(len(market_maker.cumulative_profit)), market_maker.cumulative_profits)
plt.xlabel('Number of Trades')
plt.ylabel('Cumulative Profit')
plt.title('Cumulative Profit Over Time')
plt.show()
plt.savefig("figure1.png")

plt.figure(figsize=(10,6))
plt.hist(market_maker.profits_per_trade, bins=50)
plt.xlabel('Profit')
plt.ylabel('Number of Trades')
plt.title('Histogram of Profits Per Trade')
plt.show()
plt.savefig("figure2.png")

plt.figure(figsize=(10,6))
plt.plot(range(len(market_maker.win_rates)), market_maker.win_rates)
plt.xlabel('Number of Trades')
plt.ylabel('Win Rate')
plt.title('Win Rate Over Time')
plt.show()
plt.savefig("figure3.png")
