from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import string
import jsonpickle
import collections
import json
import math
import numpy as np


N_cdf = lambda x: 0.5 * (1 + math.erf(x / np.sqrt(2)))

def BS_CALL(S, sigma):
    T = 1
    K = 10000
    if sigma * np.sqrt(T) != 0.0:
        d1 = (np.log(S/K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N_cdf(d1) - K * N_cdf(d2)
    else:
        return S

def BS_vol(S, price):
    error = 0.0001
    left = 0.0
    right = 20.0
    iterations = 100
    cnt_t = 0
    m = (left + right) / 2.0
    diff = BS_CALL(S, m) - price
    while abs(diff) > error and cnt_t < iterations:
        if diff > 0:
            right = m
        else:
            left = m
        m = (left + right) / 2.0
        diff = BS_CALL(S, m) - price
        cnt_t += 1
    return m

ALPHA = 0.000125
SIGMA_DEFAULT = 0.16

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()

class Trader:
    
    def run(self, state: TradingState):
        time = state.timestamp // 100 + 1

        if state.traderData:
            cum_upper, cum_lower, b_flag, s_flag, cum_udiff, cum_ldiff, sigma_ema = jsonpickle.decode(state.traderData)
        else:
            cum_upper, cum_lower, b_flag, s_flag, cum_udiff, cum_ldiff, sigma_ema = 410, 385, 0, 0, 60, 60, SIGMA_DEFAULT

        conditions = state.observations.conversionObservations["ORCHIDS"]
        storageFee = 0.1
        limit_prices = {"AMETHYSTS": 10000, "STARFRUIT": 0}
        limit_position = {"CHOCOLATE": 250, "STRAWBERRIES": 350, "ROSES": 60, "GIFT_BASKET": 60}
        trade_max = 10
        result = {}
        conversions = 0

        curr_prices = {}
        curr_ask_amount = {}
        curr_bid_amount = {}
        curr_position = {}

        for product in state.order_depths:
            position = state.position.get(product, 0)
            curr_position[product] = position
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            # print("Acceptable price : " + str(acceptable_price))
            # print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            osell = sorted(order_depth.sell_orders.items())
            obuy = sorted(order_depth.buy_orders.items(), reverse=True)
            if osell:
                best_ask, best_ask_amount = osell[0]
            else:
                best_ask, best_ask_amount = obuy[0]
            if obuy:
                best_bid, best_bid_amount = obuy[0]
            else:
                best_bid, best_bid_amount = osell[0]
            buy_flag , sell_flag = 0, 0

            if product == "AMETHYSTS":
                buy_flag , sell_flag = 0, 0
                if best_ask < limit_prices[product]:
                    buy_flag = 1
                elif best_bid + 1 < limit_prices[product]:
                    buy_flag = 2
                if best_bid > limit_prices[product]:
                    sell_flag = 1
                elif best_ask - 1 > limit_prices[product]:
                    sell_flag = 2
                if buy_flag and position < 20:
                    buy_amount = 20 - position
                    if buy_flag == 1:
                        orders.append(Order(product, best_ask, buy_amount))
                    else:
                        buy_amount = min(buy_amount, (limit_prices[product] - best_bid) * 3)
                        orders.append(Order(product, best_bid + 1, buy_amount))
                if sell_flag and position > -20:
                    sell_amount = position + 20
                    if sell_flag == 1:
                        orders.append(Order(product, best_bid, -sell_amount))
                    else:
                        sell_amount = min(sell_amount, (best_ask - limit_prices[product]) * 3)
                        orders.append(Order(product, best_ask - 1, -sell_amount))
                
            elif product == "STARFRUIT":
                ask_price = 0
                ask_vol = 0
                for price, volume in osell:
                    ask_price += price * volume
                    ask_vol += volume
                ask_price /= ask_vol
                bid_price = 0
                bid_vol = 0
                for price, volume in obuy:
                    bid_price += price * volume
                    bid_vol += volume
                bid_price /= bid_vol
                mid_price = (ask_price + bid_price) / 2
                urgency = (best_ask - best_bid) * (best_bid_amount + best_ask_amount) / (best_bid_amount - best_ask_amount)
                buy_flag , sell_flag = 0, 0
                fair_price = mid_price + 0.25 * urgency
                if best_ask < fair_price:
                    buy_flag = 1
                elif best_bid + 1 < fair_price:
                    buy_flag = 2
                if best_bid > fair_price:
                    sell_flag = 1
                elif best_ask - 1 > fair_price:
                    sell_flag = 2

                if buy_flag and position < 20:
                    buy_amount = min(20 - position, -best_ask_amount, trade_max)
                    if buy_flag == 2:
                        orders.append(Order(product, best_bid + 1, 20 - position))
                    else:
                        orders.append(Order(product, best_ask, buy_amount))
                if sell_flag and position > -20:
                    sell_amount = min(position + 20, best_bid_amount, trade_max)
                    if sell_flag == 2:
                        orders.append(Order(product, best_ask - 1, -(position + 20)))
                    else:
                        orders.append(Order(product, best_bid, -sell_amount))

            elif product == 'ORCHIDS':
                south_bid = conditions.bidPrice
                south_ask = conditions.askPrice
                transportFee = conditions.transportFees
                exportTariff = conditions.exportTariff
                importTariff = conditions.importTariff
                if importTariff >= 0 and exportTariff >= 0:
                    continue
                sunlight = conditions.sunlight
                humidity = conditions.humidity
                real_bid = south_bid - exportTariff - transportFee - storageFee
                real_ask = south_ask + importTariff + transportFee + storageFee
                if importTariff < 0:
                    gap = round(-importTariff - transportFee - storageFee)
                    if gap <= 1:
                        continue
                    else:
                        gap = 1
                else:
                    gap = round(-exportTariff - transportFee - storageFee)
                    if gap <= 1:
                        continue
                    else:
                        gap = 1

                if best_ask < real_bid:
                    buy_flag = 1
                elif exportTariff < 0:
                    buy_flag = 2
                else:
                    buy_flag = 0

                if best_bid > round(real_ask) + gap:
                    sell_flag = 1
                elif importTariff < 0:
                    sell_flag = 2
                else:
                    sell_flag = 0

                if position == 0:
                    if buy_flag:
                        if buy_flag == 1:
                            orders.append(Order(product, best_ask, -best_ask_amount))
                            buy_amount = 100 + best_ask_amount
                            orders.append(Order(product, round(real_bid) - gap, buy_amount))
                        else:
                            buy_amount = 100
                            orders.append(Order(product, round(real_bid) - gap, buy_amount))
                            orders.append(Order(product, round(real_bid) - gap + 2, -buy_amount))

                    
                    if sell_flag:
                        if sell_flag == 1:
                            orders.append(Order(product, best_bid, -best_bid_amount))
                            sell_amount = 100 - best_bid_amount
                            orders.append(Order(product, round(real_ask) + gap, -sell_amount))
                        else:
                            sell_amount = 100
                            orders.append(Order(product, round(real_ask) + gap, -sell_amount))
                            orders.append(Order(product, round(real_ask) + gap - 2, sell_amount))
                else:
                    conversions = -position
                    if buy_flag:
                        if buy_flag == 1:
                            orders.append(Order(product, best_ask, -best_ask_amount))
                            buy_amount = 100 + best_ask_amount
                            orders.append(Order(product, round(real_bid) - gap, buy_amount))
                        else:
                            buy_amount = 100
                            orders.append(Order(product, round(real_bid) - gap, buy_amount))
                            orders.append(Order(product, round(real_bid) - gap + 2, -buy_amount))

                    if sell_flag:
                        if sell_flag == 1:
                            orders.append(Order(product, best_bid, -best_bid_amount))
                            sell_amount = 100 - best_bid_amount
                            orders.append(Order(product, round(real_ask) + gap, -sell_amount))
                        else:
                            sell_amount = 100
                            orders.append(Order(product, round(real_ask) + gap, -sell_amount))
                            orders.append(Order(product, round(real_ask) + gap - 2, sell_amount))

                logger.print('conversions: '+ str(conversions) + ' position: '+ str(position))
            elif product in ["CHOCOLATE","STRAWBERRIES","ROSES","GIFT_BASKET"]:
                curr_prices[product] = [best_ask, best_bid]
                curr_ask_amount[product] = min(-best_ask_amount, limit_position[product] - position)
                curr_bid_amount[product] = min(best_bid_amount, limit_position[product] + position)
            elif product == "COCONUT":
                curr_prices[product] = [best_ask, best_bid]
                curr_bid_amount[product] = min(best_bid_amount, 300 + position)
                curr_ask_amount[product] = min(-best_ask_amount, 300 - position)

            elif product == "COCONUT_COUPON":
                curr_prices[product] = [best_ask, best_bid]
                curr_bid_amount[product] = min(best_bid_amount, 600 + position)
                curr_ask_amount[product] = min(-best_ask_amount, 600 - position)

            result[product] = orders

        factors = {'CHOCOLATE':4, 'ROSES':1}
        a_upper = curr_prices["GIFT_BASKET"][0]
        b_upper = curr_prices['CHOCOLATE'][0]*4+curr_prices['STRAWBERRIES'][0]*6+curr_prices['ROSES'][0]
        a_lower = curr_prices["GIFT_BASKET"][1]
        b_lower = curr_prices['CHOCOLATE'][1]*4+curr_prices['STRAWBERRIES'][1]*6+curr_prices['ROSES'][1]
        z_upper = a_upper - b_lower
        z_lower = a_lower - b_upper
        upper_diff = cum_upper / time - z_upper
        lower_diff = z_lower - cum_lower / time


        curr_mid = sum(curr_prices["COCONUT"]) / 2
        curr_option_mid = sum(curr_prices["COCONUT_COUPON"]) / 2
        sigma = BS_vol(curr_mid, curr_option_mid)
        fair_option_price = BS_CALL(curr_mid, sigma_ema)
        sigma_ema = ALPHA * sigma + (1 - ALPHA) * sigma_ema

        if time >= 100:
            if lower_diff > 0.3 * cum_ldiff / time + 61:
                # Sell A, Buy B
                b_flag = -1
                s_flag = 6
                first_amount = min(curr_ask_amount['CHOCOLATE'] // 4, curr_ask_amount['STRAWBERRIES'] // 6, curr_ask_amount['ROSES'], curr_bid_amount['GIFT_BASKET'])
                amount = min(first_amount, trade_max + 5, limit_position["GIFT_BASKET"] + curr_position["GIFT_BASKET"])
                result['GIFT_BASKET'].append(Order('GIFT_BASKET', curr_prices["GIFT_BASKET"][1], -amount))
                for product in factors:
                    result[product].append(Order(product, curr_prices[product][0], amount * factors[product]))
            elif s_flag > 0:
                first_amount = min(curr_ask_amount['CHOCOLATE'] // 4, curr_ask_amount['STRAWBERRIES'] // 6, curr_ask_amount['ROSES'], curr_bid_amount['GIFT_BASKET'])
                amount = min(first_amount, trade_max + 5, limit_position["GIFT_BASKET"] + curr_position["GIFT_BASKET"])
                result['GIFT_BASKET'].append(Order('GIFT_BASKET', curr_prices["GIFT_BASKET"][1], -amount))
                for product in factors:
                    result[product].append(Order(product, curr_prices[product][0], amount * factors[product]))
                s_flag -= 1
            elif curr_position['GIFT_BASKET'] == -58 and lower_diff < 0 and b_flag == -1:
                b_flag = 0
                first_amount = min(curr_bid_amount['CHOCOLATE'] // 4, curr_bid_amount['STRAWBERRIES'] // 6, curr_bid_amount['ROSES'], curr_ask_amount['GIFT_BASKET'])
                amount = min(first_amount, 58, limit_position["GIFT_BASKET"] - curr_position["GIFT_BASKET"])
                result['GIFT_BASKET'].append(Order('GIFT_BASKET', curr_prices["GIFT_BASKET"][0], amount))
                for product in factors:
                    result[product].append(Order(product, curr_prices[product][1], -amount * factors[product]))
            elif upper_diff > 0.3 * cum_udiff / time + 66:
                # Buy A, Sell B
                s_flag = -1
                b_flag = 6
                first_amount = min(curr_bid_amount['CHOCOLATE'] // 4, curr_bid_amount['STRAWBERRIES'] // 6, curr_bid_amount['ROSES'], curr_ask_amount['GIFT_BASKET'])
                amount = min(first_amount, trade_max + 5, limit_position["GIFT_BASKET"] - curr_position["GIFT_BASKET"])
                result['GIFT_BASKET'].append(Order('GIFT_BASKET', curr_prices["GIFT_BASKET"][0], amount))
                for product in factors:
                    result[product].append(Order(product, curr_prices[product][1], -amount * factors[product]))
            elif b_flag > 0:
                first_amount = min(curr_bid_amount['CHOCOLATE'] // 4, curr_bid_amount['STRAWBERRIES'] // 6, curr_bid_amount['ROSES'], curr_ask_amount['GIFT_BASKET'])
                amount = min(first_amount, trade_max + 5, limit_position["GIFT_BASKET"] - curr_position["GIFT_BASKET"])
                result['GIFT_BASKET'].append(Order('GIFT_BASKET', curr_prices["GIFT_BASKET"][0], amount))
                for product in factors:
                    result[product].append(Order(product, curr_prices[product][1], -amount * factors[product]))
                b_flag -= 1
            elif curr_position['GIFT_BASKET'] == 58 and upper_diff < 0 and s_flag == -1:
                s_flag = 0
                first_amount = min(curr_ask_amount['CHOCOLATE'] // 4, curr_ask_amount['STRAWBERRIES'] // 6, curr_ask_amount['ROSES'], curr_bid_amount['GIFT_BASKET'])
                amount = min(first_amount, 58, limit_position["GIFT_BASKET"] + curr_position["GIFT_BASKET"])
                result['GIFT_BASKET'].append(Order('GIFT_BASKET', curr_prices["GIFT_BASKET"][1], -amount))
                for product in factors:
                    result[product].append(Order(product, curr_prices[product][0], amount * factors[product]))
            
            if curr_prices["COCONUT_COUPON"][1] - fair_option_price > 2:
                amount = min(curr_bid_amount["COCONUT_COUPON"] // 2, curr_ask_amount["COCONUT"])
                if amount != 0:
                    result["COCONUT_COUPON"].append(Order("COCONUT_COUPON", curr_prices["COCONUT_COUPON"][1], -amount * 2))
                    result["COCONUT"].append(Order("COCONUT", curr_prices["COCONUT"][0], amount))
            elif curr_prices["COCONUT_COUPON"][0] - fair_option_price < -2:
                amount = min(curr_ask_amount["COCONUT_COUPON"] // 2, curr_bid_amount["COCONUT"])
                if amount != 0:
                    result["COCONUT_COUPON"].append(Order("COCONUT_COUPON", curr_prices["COCONUT_COUPON"][0], amount * 2))
                    result["COCONUT"].append(Order("COCONUT", curr_prices["COCONUT"][1], -amount))

        else:
            if z_lower > 430:
                first_amount = min(curr_ask_amount['CHOCOLATE'] // 4, curr_ask_amount['STRAWBERRIES'] // 6, curr_ask_amount['ROSES'], curr_bid_amount['GIFT_BASKET'])
                amount = min(first_amount, trade_max, limit_position["GIFT_BASKET"] + curr_position["GIFT_BASKET"])
                result['GIFT_BASKET'].append(Order('GIFT_BASKET', curr_prices["GIFT_BASKET"][1], -amount))
                for product in factors:
                    result[product].append(Order(product, curr_prices[product][0], amount * factors[product]))
            elif z_upper < 366:
                first_amount = min(curr_bid_amount['CHOCOLATE'] // 4, curr_bid_amount['STRAWBERRIES'] // 6, curr_bid_amount['ROSES'], curr_ask_amount['GIFT_BASKET'])
                amount = min(first_amount, trade_max, limit_position["GIFT_BASKET"] - curr_position["GIFT_BASKET"])
                result['GIFT_BASKET'].append(Order('GIFT_BASKET', curr_prices["GIFT_BASKET"][0], amount))
                for product in factors:
                    result[product].append(Order(product, curr_prices[product][1], -amount * factors[product]))

        
        

        traderData = jsonpickle.encode([cum_upper + z_upper, cum_lower + z_lower, b_flag, s_flag, cum_udiff + abs(upper_diff), cum_ldiff + abs(lower_diff), sigma_ema])
        
				# Sample conversion request. Check more details below. 
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData