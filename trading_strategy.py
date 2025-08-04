# -*- coding: utf-8 -*-
"""
@author: A2NG Services
"""

import pandas as pd
import numpy as np
import datetime

class TradingStrategy:
    def __init__(self, rebalancing_frequency="M", initial_capital=10000000, 
                 start_date=datetime.datetime(2010,1,4), end_date=datetime.datetime(2023,12,29)):
        self.rebalancing_frequency = rebalancing_frequency
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        
        self.load_pricing_data()
        
        self.trades = pd.DataFrame(columns=['Ticker', 'Date', 'Price', 'n_shares_traded', 'Signal'])
        self.n_shares_position = pd.DataFrame(0, columns=self.prices.columns, index=self.prices.index)
        self.position_amount = pd.DataFrame(columns=list(self.prices.columns) + ["Cash"], index=self.prices.index)
        self.position_atp = pd.DataFrame(0, columns=self.prices.columns, index=self.prices.index)
        self.position_long_short = pd.DataFrame(0, columns=self.prices.columns, index=self.prices.index)
        
        self.position_amount['Cash'] = 0.0
    
    def load_pricing_data(self):
        self.prices = pd.read_csv("Prices.csv", index_col='Date', parse_dates=True)
        self.prices.sort_index(inplace=True)
    
    def get_trades(self):
        return self.trades
    
    def get_positions(self):
        return self.n_shares_position
    
    def get_position_amount(self):
        return self.position_amount
    
    def get_cash(self):
        return self.position_amount['Cash']
    
    def get_portfolio_value(self):
        return self.position_amount.drop('Cash', axis=1).sum(axis=1) + self.position_amount['Cash']

    def run_backtest(self):
        prices = self.prices
        rebalance_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
        rebalance_dates_actual = [prices.index[prices.index <= date].max() for date in rebalance_dates]

        prev_date = None
        for date in prices.index:
            if date < self.start_date or date > self.end_date:
                continue
            
            if prev_date is None:
                self._initialize_first_day(date)
                prev_date = date
                continue
            
            self._carry_forward_positions(date, prev_date)
            
            if date in rebalance_dates_actual:
                current_cash = self.position_amount.loc[date, 'Cash']
                current_shares = self.n_shares_position.loc[date].copy()
                total_capital = (current_shares * prices.loc[date]).sum() + current_cash
                
                longs, shorts = self._calculate_momentum(date, prices)
                
                target_weights = self._calculate_target_weights(longs, shorts)
                
                new_shares, delta_shares = self._calculate_share_changes(
                    target_weights, current_shares, total_capital, prices.loc[date]
                )
                
                new_shares, new_cash = self._apply_gross_exposure(
                    new_shares, current_cash, prices.loc[date], total_capital
                )
                
                new_shares, new_cash = self._apply_net_exposure(
                    new_shares, new_cash, longs, shorts, prices.loc[date]
                )
                
                new_shares, new_cash = self._handle_negative_cash(
                    new_shares, new_cash, prices.loc[date], current_shares
                )
                
                new_shares, new_cash = self._apply_cash_constraint(
                    new_shares, new_cash, prices.loc[date], total_capital
                )
                
                self._update_positions(date, new_shares, new_cash, prices.loc[date], current_shares)
                
            prev_date = date

    def _initialize_first_day(self, date):
        self.n_shares_position.loc[date] = 0
        self.position_amount.loc[date, self.prices.columns] = 0.0
        self.position_amount.at[date, 'Cash'] = self.initial_capital
        self.position_atp.loc[date] = 0.0
        self.position_long_short.loc[date] = 0

    def _carry_forward_positions(self, date, prev_date):
        current_prices = self.prices.loc[date]
        self.n_shares_position.loc[date] = self.n_shares_position.loc[prev_date]
        self.position_amount.loc[date, self.prices.columns] = self.n_shares_position.loc[date] * current_prices
        self.position_amount.loc[date, 'Cash'] = self.position_amount.loc[prev_date, 'Cash']
        self.position_atp.loc[date] = self.position_atp.loc[prev_date]
        self.position_long_short.loc[date] = self.position_long_short.loc[prev_date]

    def _calculate_momentum(self, date, prices):
        lookback_end = date - pd.DateOffset(months=1)
        valid_ends = prices.index[prices.index <= lookback_end]
        if valid_ends.empty:
            return [], []
        lookback_end_valid = valid_ends.max()

        lookback_start = lookback_end_valid - pd.DateOffset(months=12)
        valid_starts = prices.index[prices.index >= lookback_start]
        if valid_starts.empty:
            return [], []
        lookback_start_valid = valid_starts.min()
        prices_lookback = prices.loc[lookback_start_valid:lookback_end_valid]

        if prices_lookback.shape[0] < 2:
            return [], []
        returns = (prices_lookback.iloc[-1] / prices_lookback.iloc[0]) - 1
        ranked = returns.sort_values(ascending=False)

        n_longs = min(11, len(ranked))
        n_shorts = min(9, len(ranked))
        return ranked.head(n_longs).index.tolist(), ranked.tail(n_shorts).index.tolist()

    def _calculate_target_weights(self, longs, shorts):
        target_weights = pd.Series(0.0, index=self.prices.columns)
        target_weights[longs] = 0.05
        target_weights[shorts] = -0.05
        return target_weights

    def _calculate_share_changes(self, target_weights, current_shares, total_capital, current_prices):
        new_shares = pd.Series(0, index=self.prices.columns)
        delta_shares = pd.Series(0, index=self.prices.columns)
        
        for stock in self.prices.columns:
            price = current_prices[stock]
            target_value = target_weights[stock] * total_capital
            target_shares = target_value / price if price != 0 else 0
            

            if target_weights[stock] > 0:
                min_shares = np.ceil(0.01 * total_capital / price)
                max_shares = np.floor(0.05 * total_capital / price)
                target_shares = np.clip(target_shares, min_shares, max_shares)
            elif target_weights[stock] < 0:
                min_abs = np.ceil(0.01 * total_capital / price)
                max_abs = np.floor(0.05 * total_capital / price)
                target_shares = -np.clip(abs(target_shares), min_abs, max_abs)
            
            new_shares[stock] = int(round(target_shares))
            delta_shares[stock] = new_shares[stock] - current_shares[stock]
        
        return new_shares, delta_shares

    def _apply_gross_exposure(self, new_shares, current_cash, current_prices, total_capital):
        current_values = new_shares * current_prices
        gross_exposure = abs(current_values).sum() / (current_values.sum() + current_cash)
        
        if gross_exposure > 1.0:
            scale_factor = 1.0 / gross_exposure
            scaled_values = current_values * scale_factor
            new_shares = (scaled_values / current_prices).round().astype(int)
            new_cash = total_capital - (new_shares * current_prices).sum()
        else:
            new_cash = current_cash - (new_shares * current_prices).sum()
        
        return new_shares, new_cash

    def _apply_net_exposure(self, new_shares, new_cash, longs, shorts, current_prices):
        current_values = new_shares * current_prices
        total_value = current_values.sum() + new_cash
        net_exposure = current_values.sum() / total_value
        
        if abs(net_exposure) > 0.1:
            target_net = 0.1 if net_exposure > 0 else -0.1
            adjustment = (target_net * total_value) - current_values.sum()
            

            for stock in longs + shorts:
                if adjustment == 0:
                    break
                
                position_value = current_values[stock]
                price = current_prices[stock]
                
                if adjustment > 0 and position_value > 0:
                    max_adjust = adjustment / price
                    shares = min(int(max_adjust), new_shares[stock])
                    new_shares[stock] += shares
                    adjustment -= shares * price
                    new_cash -= shares * price
                elif adjustment < 0 and position_value < 0:
                    max_adjust = abs(adjustment) / price
                    shares = min(int(max_adjust), abs(new_shares[stock]))
                    new_shares[stock] += shares
                    adjustment += shares * price
                    new_cash += shares * price
        
        return new_shares, new_cash

    def _handle_negative_cash(self, new_shares, new_cash, current_prices, current_shares):
        if new_cash < 0:
            required_cash = abs(new_cash)
            for stock in new_shares.index:
                if required_cash <= 0:
                    break
                
                price = current_prices[stock]
                shares_available = new_shares[stock] - current_shares[stock]
                
                if shares_available > 0:   
                    max_shares = min(shares_available, int(required_cash // price))
                    if max_shares > 0:
                        new_shares[stock] -= max_shares
                        required_cash -= max_shares * price
                        new_cash += max_shares * price
                
                elif shares_available < 0:   
                    max_shares = min(abs(shares_available), int(required_cash // price))
                    if max_shares > 0:
                        new_shares[stock] += max_shares
                        required_cash -= max_shares * price
                        new_cash += max_shares * price
            

            if new_cash < 0:
                new_cash = 0
        
        return new_shares, new_cash

    def _apply_cash_constraint(self, new_shares, new_cash, current_prices, total_capital):
        current_values = new_shares * current_prices
        total_value = current_values.sum() + new_cash
        
        if total_value == 0:
            return new_shares, new_cash
        

        max_cash = 0.05 * total_value
        if new_cash > max_cash:
            excess_cash = new_cash - max_cash
            for stock in new_shares.index:
                if excess_cash <= 0:
                    break
                
                price = current_prices[stock]
                shares_to_buy = int(excess_cash // price)
                if shares_to_buy > 0:
                    new_shares[stock] += shares_to_buy
                    excess_cash -= shares_to_buy * price
                    new_cash -= shares_to_buy * price
        
        return new_shares, new_cash

    def _update_positions(self, date, new_shares, new_cash, current_prices, current_shares):
        self.n_shares_position.loc[date] = new_shares
        self.position_amount.loc[date, self.prices.columns] = new_shares * current_prices
        self.position_amount.loc[date, 'Cash'] = new_cash
        

        delta_shares = new_shares - current_shares
        for stock in delta_shares.index:
            if delta_shares[stock] != 0:
                self.trades = pd.concat([
                    self.trades,
                    pd.DataFrame({
                        'Ticker': [stock],
                        'Date': [date],
                        'Price': [current_prices[stock]],
                        'n_shares_traded': [delta_shares[stock]],
                        'Signal': [1 if delta_shares[stock] > 0 else -1]
                    })
                ], ignore_index=True)
        
        self.position_long_short.loc[date] = np.sign(new_shares)


if __name__ == "__main__":
    ts = TradingStrategy(rebalancing_frequency="M", initial_capital=10000000,
                         start_date=datetime.datetime(2010,1,4), end_date=datetime.datetime(2023,12,29))
    ts.run_backtest()
    # print(ts.get_trades())
    # print(ts.get_positions())
    # print(ts.get_position_amount())
    # print(ts.get_cash())
    # print(ts.get_portfolio_value())