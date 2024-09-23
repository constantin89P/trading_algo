import os, sys, requests, io, socket
import pandas as pd
import datetime as dt
import numpy as np
from os import walk
from time import sleep
import pandas_ta as ta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, ReplaceOrderRequest, TrailingStopOrderRequest, StopOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
from alpaca.data.requests import StockLatestBarRequest
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.trading.enums import AssetClass
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce, OrderType, OrderClass, PositionSide
pd.options.mode.chained_assignment = None


SCRIPT_NAME = os.getenv('SCRIPT_NAME')
TOKEN_TELEGRAM = os.getenv('TOKEN_TELEGRAM')
TOKEN_TELEGRAM_ERROR = os.getenv('TOKEN_TELEGRAM_ERROR')
GIT_TOKEN = os.getenv('GIT_TOKEN')
A_KEY = os.getenv('B_KEA_KEYY_L')
A_SECRET = os.getenv('A_SECRET')
SENS = int(os.getenv('SENS'))
CHAT_ID = os.getenv('CHAT_ID')
IS_PAPER = True



hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
is_running_server = True if local_ip not in list(os.getenv('WHITELISTIP')) else False




SCRIPT_NAME = "Live_crosschecking"
ticker = "TSLA"
interval = 1
Strategy = ['supertrend', 'ema', 'non', 'non', 'non', '1', '1', '0', '10', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'] # Enter on supertrend 1,1
signals_combi = ['All', 'EMA_out', 'SL_fixed', '1', '10', '0.2']  # Enter type All, exit on EMA cross 1/10, SL fixed 0.2%
suffix = ""
QUANTITY = 10





# SESSION variables
HOUR_TIMEZONE = 2
CLOSING_GAP = 10 #(min)
SESSION_LENGHT = ((6*60)+30-CLOSING_GAP)  #6 hours 30 min 
REPO_NAME = "Avril_2024"






histo_client = StockHistoricalDataClient(A_KEY, A_SECRET)
trading_client = TradingClient(A_KEY, A_SECRET, paper=IS_PAPER)
if interval <60 : timestamp = TimeFrame(interval, TimeFrameUnit.Minute)
SESSION = "Close"




def telegram_api(*text):
    for x in range(5):
        try : 
            for element in text : 
                url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage?chat_id={CHAT_ID}&text={element}"
                sending = requests.get(url).json()
                return 
        except Exception as e :
            print(f"Error in telegram_api {dt.datetime.now().time().strftime('%H:%M:%S')}")
            print(e) 
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    message = "Error in telegram_api :" + str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
    print(message, "\n")
def telegram_api_erreur(*text):
    for x in range(5):
        try : 
            for element in text : 
                element = SCRIPT_NAME + " - " + element
                url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM_ERROR}/sendMessage?chat_id={CHAT_ID}&text={element}"
                sending = requests.get(url).json()
                print(element)
                return
        except Exception as e : 
            print(f"Error in telegram_api_erreur {dt.datetime.now().time().strftime('%H:%M:%S')}")
            print(e) 
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    message = "Error in telegram_api_erreur :" + str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
    print(message, "\n")
def get_historical_data(symbol, timeframe, start, end):
    for x in range(10) : 
        try : 
            client = StockHistoricalDataClient(A_KEY, A_SECRET)

            request_params = StockBarsRequest(
                                    symbol_or_symbols=[symbol],
                                    timeframe=timeframe,
                                    start=start,
                                    end=end, 
                                    feed='iex')
            
            bars = client.get_stock_bars(request_params)


            # Check if there are prices in response
            try : 
                bars_df = bars.df
                bars_df = bars_df.droplevel(level=0)

            except Exception as e :
                print(f"No prices yet {dt.datetime.now().time().strftime('%H:%M:%S')}")
                return pd.DataFrame()
            bars_df['Time'] = bars_df.index   
            bars_df['Time'] = bars_df['Time'] + dt.timedelta(hours=HOUR_TIMEZONE) 
            bars_df = bars_df.reset_index(drop=True)

            bars_df = bars_df.rename(columns={'open': 'Open',
                                            'low':'Low',
                                            'close':'Close',
                                            'high':'High',
                                            'volume':'Volume'})
            return bars_df

        except Exception as e :
            print(f"Error in getting prices {dt.datetime.now().time().strftime('%H:%M:%S')}")
            print(e) 
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            message = "Error getting prices :" + str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
            print(message, "\n")
            telegram_api_erreur(message) 
            sleep(0.1)
        
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    message = "Error in getting prices :" + str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
    print(message, "\n")
    telegram_api_erreur(message)
    return pd.DataFrame()
        
def get_position():
    for x in range(5): 
        try : 
            account = trading_client.get_account()
            side, holding_qty, unrealized_pnl, avg_entry_price = get_position_asset(ticker)
            # print(account)
            if holding_qty != 0 : 
                if side == PositionSide.LONG : position = 1
                elif side == PositionSide.SHORT : position = -1
            else : position = 0
            print(f"position : {position}")

            return position, holding_qty, account.portfolio_value
        except Exception as e :
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{e} : {dt.datetime.now().time().strftime('%H:%M:%S')}")
            sleep(0.1)

        print(f"Error in get_position {dt.datetime.now().time().strftime('%H:%M:%S')}")
        message = "Error in get_position :" + str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        telegram_api_erreur(message)
        return "", "", ""
def get_portfolio_positions():
    try : 
        positions = {}
        # Get a list of all of our positions.
        portfolio = trading_client.get_all_positions()

        # Print the quantity of shares for each position.
        for pos in portfolio:
            print("{} shares of {}".format(pos.qty, pos.symbol))
            positions.update({pos.symbol : pos.qty})
        return positions
    except Exception as e :
        print(e) 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        telegram_api_erreur(message) ; sleep(30000)
        sys.exit()
def get_position_asset(ticker):
    try : 
        asset_position = trading_client.get_open_position(ticker)

        return asset_position.side, float(asset_position.qty), float(asset_position.unrealized_pl), float(asset_position.avg_entry_price)
    except Exception as e :
        print(e) 
        if str(e.args[0]) == '{"code":40410000,"message":"position does not exist"}' : return 0, 0, 0, 0
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        telegram_api_erreur(message) ; sleep(30000)
        sys.exit()

def close_all_positions():
    
    # cancel_all_orders()
    closing = trading_client.close_all_positions(cancel_orders=True)
    print(f"Closing all position order :\n{closing}")

    while True :
        test = 0
        try : 
            sleep(3)
            position, holding_qty, portfolio_value = get_position()  
            if position == 0 : 
                print(f"Position liquidated, essai n°{test+1}")
                # telegram_api(f"Position liquidated, essai n°{test+1}")
                return
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            if test == 15 : break
            sleep(1)
            test+=1
            print(f"{test} - {e}")

    message = f"Error on close_all_positions, test: {test}, line : {str(exc_tb.tb_lineno)}"
    print(message)
    telegram_api_erreur(message)
    sleep(30000)
    return
def close_asset_positions(ticker):
    try : 
        cancel_orders_asset(ticker)
        closing = trading_client.close_position(ticker)
        for x in range(20):
            try : 
                position, holding_qty, portfolio_value = get_position()  
                if position == 0 : 
                    print(f"Position liquidated, essai n°{x+1}")
                    # telegram_api(f"Position liquidated, essai n°{x+1}")
                    return
                sleep(0.1)
            except Exception as e :
                print(e)
                sleep(1)

        message = f"Error Position close_asset_positions"
        print(message)
        telegram_api_erreur(message)
    except Exception as e :
        print(e) 
        if str(e.args[0]) == '{"code":40410000,"message":"position does not exist"}' or "position not found" in str(e.args[0]) : return 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        telegram_api(message) ; sleep(3000)
        sys.exit()
def cancel_all_orders(ticker=None):
    for x in range(2):
        try : 
            oders = trading_client.get_orders(GetOrdersRequest(symbols=ticker, status="Open"))
            cancel_status = trading_client.cancel_orders()
            message = f"Order canceled {cancel_status}"
            print(message)
            telegram_api(message)
            return
        except Exception as e :
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(e)
    message = f"Error cancel orders failed : {e} line : {str(exc_tb.tb_lineno)}"
    telegram_api_erreur(message)
    sleep(30000)
def cancel_orders_asset(ticker):
    try : 
        orders_id = ["init"]
        orders = ["init"]
        id = "init"

        orders = trading_client.get_orders(GetOrdersRequest(symbols=[ticker], status="open"))
        orders = [{key: value for (key, value) in row} for row in orders]
        orders_id = [d['id'] for d in orders]

        for id in orders_id :
            try : 
                canceled_orders = trading_client.cancel_order_by_id(id)
            except Exception as e : 
                if """{"code":42210000""" in str(e.args[0]) : continue
                # if str(e.args[0]) == """{"code":42210000,"message":"order is already in \"filled\" state"}""" : continue
                else : raise

    except Exception as e :
        print(e) 
        print(f"orders_id : {orders_id}")
        print(f"orders : {orders}")
        print(f"id : {id}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        telegram_api_erreur(message) ; sleep(30000)
        sys.exit()
def get_all_orders(status, side):
    try : 
        # params to filter orders by
        request_params = GetOrdersRequest(
                            status=status,
                            side=side
                        )

        # orders that satisfy params
        orders = trading_client.get_orders(filter=request_params)
        for order in orders : 
            print(f"order {order}")
    except Exception as e :
        print(f"Error in Send_Order {dt.datetime.now().time().strftime('%H:%M:%S')}")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = "Error Send_Order :" + str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        telegram_api_erreur(message) ; sleep(30000)
def get_signal(suffix, df, ticker, name, param1, param2, param3):
    try :
        dataframe = df.copy()

        if name not in ["delta", "supertrend"] :
            param1 = int(param1)
            param2 = int(param2)
            param3 = int(param3)

        if name == "non" : return 3
        elif name == "supertrend" : 

            length = int(param1) 
            multiplier = float(param2)  

            dataframe[['1', '2', '3', '4']] = ta.supertrend(close=dataframe[f'Close{suffix}'], high=dataframe[f'High{suffix}'], low=dataframe[f'Low{suffix}'], length=length, multiplier=multiplier)

            return dataframe['2']    
        elif name == "tsi_OB_OS" : 
            fast = param1  
            slow = param2 
            signal = param3
            ref_name = f"TSI_{fast}_{slow}_{signal}"
            dataframe[[ref_name, 'lag_name']] = ta.tsi(close=dataframe[f'Close{suffix}'], fast=fast, slow=slow)
            dataframe['prev'] = dataframe[ref_name].shift(1).fillna(0)
            dataframe['position'] = 0
            dataframe['position'] = np.where((dataframe['prev'] <= -50) & (dataframe[ref_name] > -50), 1, dataframe['position'])
            dataframe['position'] = np.where((dataframe['prev'] >= 50) & (dataframe[ref_name] < 50), -1, dataframe['position'])
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,45)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()

            return dataframe['position']     
        elif name == "tsi_lag" : 
            fast = param1  
            slow = param2 
            signal = param3 
            ref_name = f"TSI_{fast}_{slow}_{signal}"
            lag_name = f"TSIs_{fast}_{slow}_{signal}" 
            dataframe[[ref_name, lag_name]]  = ta.tsi(close=dataframe[f'Close{suffix}'], fast=fast, slow=slow, signal=signal)
            dataframe['position'] = np.where(dataframe[ref_name]>dataframe[lag_name], 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,30)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "stoch_OB_OS" : 
            k = param1  
            d = param2 
            smooth_k = param3  
            ref_name = f"STOCHk_{k}_{d}_{smooth_k}"
            dataframe[[ref_name, 'lag_name']] = ta.stoch(close=dataframe[f'Close{suffix}'], high=dataframe[f'High{suffix}'], low=dataframe[f'Low{suffix}'], k=k, d=smooth_k, smooth_k=d)
            dataframe['prev'] = dataframe[ref_name].shift(1).fillna(0)
            dataframe['position'] = np.nan
            dataframe['position'] = np.where((dataframe['prev'] <= 20) & (dataframe[ref_name] > 20), 1, dataframe['position'])
            dataframe['position'] = np.where((dataframe['prev'] >= 80) & (dataframe[ref_name] < 80), -1, dataframe['position'])
            dataframe['position'] = dataframe['position'].ffill()
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,30)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "stoch_lag" : 
            k = param1  
            d = param2 
            smooth_k = param3  
            ref_name = f"STOCHk_{k}_{d}_{smooth_k}"
            lag_name = f"STOCHkd_{k}_{d}_{smooth_k}"
            dataframe[[ref_name, lag_name]] = ta.stoch(close=dataframe[f'Close{suffix}'], high=dataframe[f'High{suffix}'], low=dataframe[f'Low{suffix}'], k=k, d=smooth_k, smooth_k=d)
            dataframe['position'] = np.where(dataframe[ref_name]>dataframe[lag_name], 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "stoch_rsi_lag" : 
            length = param1  
            rsi_length = param2 
            smooth_k = param3  
            ref_name = f"ref"
            lag_name = f"lag"
            dataframe[[ref_name, lag_name]] = ta.stochrsi(close=dataframe[f'Close{suffix}'], high=dataframe[f'High{suffix}'], length=length, rsi_length=rsi_length, k=smooth_k, d=5)
            dataframe['position'] = np.where(dataframe[ref_name]>dataframe[lag_name], 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "stoch_rsi_OB_OS" : 
            length = param1  
            rsi_length = param2 
            smooth_k = param3  
            ref_name = f"ref"
            lag_name = f"lag"
            dataframe[[ref_name, lag_name]] = ta.stochrsi(close=dataframe[f'Close{suffix}'], high=dataframe[f'High{suffix}'], length=length, rsi_length=rsi_length, k=smooth_k)
            dataframe['prev'] = dataframe[ref_name].shift(1).fillna(0)
            dataframe['position'] = np.nan
            dataframe['position'] = np.where((dataframe['prev'] <= 20) & (dataframe[ref_name] > 20), 1, dataframe['position'])
            dataframe['position'] = np.where((dataframe['prev'] >= 80) & (dataframe[ref_name] < 80), -1, dataframe['position'])
            dataframe['position'] = dataframe['position'].ffill()
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "rsi_OB_OS" : 
            length = param1 
            ref_name = f"RSI_{length}"
            dataframe[ref_name] = ta.rsi(close=dataframe[f'Close{suffix}'], length=length)
            dataframe['prev'] = dataframe[ref_name].shift(1).fillna(0)

            dataframe['position'] = np.where((dataframe['prev'] <= 30) & (dataframe[ref_name] > 30), 1, np.nan)
            dataframe['position'] = np.where((dataframe['prev'] >= 70) & (dataframe[ref_name] < 70), -1, dataframe['position'])
            dataframe['position'] = dataframe['position'].ffill()
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "rsi_lag_ema" : 
            length = param1 
            d = param2 
            ref_name = f"RSI_{length}_{d}"
            lag_name = f"RSI_lag"
            dataframe[ref_name] = ta.rsi(close=dataframe[f'Close{suffix}'], length=length)
            dataframe[lag_name] = ta.ema(close=dataframe[ref_name], length=d)

            dataframe['position'] = np.where(dataframe[ref_name]>dataframe[lag_name], 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "vwma" : 
            length = param1 
            ref_name = f"vwma_{length}"
            dataframe[ref_name] = ta.vwma(close=dataframe[f'Close{suffix}'], volume=dataframe['Volume'], length=length)

            dataframe['position'] = np.where(dataframe[ref_name]>dataframe[f'Close{suffix}'], -1, 1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "vwma_hma" : 
            length_v = param1 
            length_h = param2
            name_vwma = f"vwma_{length_v}"
            name_hma = f"hma_{length_h}"
            dataframe[name_vwma] = ta.vwma(close=dataframe[f'Close{suffix}'], volume=dataframe['Volume'], length=length_v)

            dataframe[name_hma] = ta.hma(close=dataframe[f'Close{suffix}'], length=length_h)

            dataframe['position'] = np.where(dataframe[name_hma]>dataframe[name_vwma], 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "macd" : 
            fast = param1 
            slow = param2
            signal = param3
            macd = f"macd"
            macdh = f"macdh"
            macds = f"macds"
            dataframe[[macd, macdh, macds]] = ta.macd(close=dataframe[f'Close{suffix}'], fast=fast, slow=slow, signal=signal)

            dataframe['position'] = np.where(dataframe[macdh]>0, 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "hma" : 
            length = param1 
            dataframe['hma'] = ta.hma(close=dataframe[f'Close{suffix}'], length=length)

            dataframe['position'] = np.where(dataframe['hma']>dataframe[f'Close{suffix}'], -1, 1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # sys.exit()
            return dataframe['position']
        elif name == "ema" : 
            length = param1 
            
            dataframe['ema'] = ta.ema(close=dataframe[f'Close{suffix}'], length=length)

            dataframe['position'] = np.where(dataframe['ema']>dataframe[f'Close{suffix}'], -1, 1)

            return dataframe['position']
        elif name == "Low_High_cross" : 
            length = param1 

            dataframe['MA_low'] = ta.ema(dataframe["Low"], length=length)
            dataframe['MA_high'] = ta.ema(dataframe["High"], length=length)

            dataframe['position'] = np.where((dataframe["Open"] < dataframe["MA_low"]) & (dataframe["Close"] > dataframe["MA_low"]), 1, 0)   
            dataframe['position'] = np.where((dataframe["Open"] > dataframe["MA_high"]) & (dataframe["Close"] < dataframe["MA_high"]), -1,dataframe['position'])

            return dataframe['position']
        elif name == "ema_n" : 
            length = param1 
            dataframe['ema'] = ta.ema(close=dataframe[f'Close{suffix}'], length=length)

            dataframe['position'] = np.where(dataframe['ema']<dataframe[f'Close{suffix}'], -1, 1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # sys.exit()
            return dataframe['position']
        elif name == "dema" : 
            length = param1 
            dataframe['dema'] = ta.dema(close=dataframe[f'Close{suffix}'], length=length)

            dataframe['position'] = np.where(dataframe['dema']>dataframe[f'Close{suffix}'], -1, 1)
            return dataframe['position']
        elif name == "tema" : 
            length = param1 
            dataframe['tema'] = ta.tema(close=dataframe[f'Close{suffix}'], length=length)

            dataframe['position'] = np.where(dataframe['tema']>dataframe[f'Close{suffix}'], -1, 1)
            return dataframe['position']
        elif name == "dema_ema" : 
            length_dema = param1 
            length_ema = param2
            dataframe['dema'] = ta.dema(close=dataframe[f'Close{suffix}'], length=length_dema)
            dataframe['ema'] = ta.ema(close=dataframe[f'Close{suffix}'], length=length_ema)

            dataframe['position'] = np.where(dataframe['dema']>dataframe['ema'], 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # sys.exit()
            return dataframe['position']
        elif name == "vortex" : 
            length = param1 
            dataframe[['blue', 'red']] = ta.vortex(close=dataframe[f'Close{suffix}'], high=dataframe[f'High{suffix}'], low=dataframe[f'Low{suffix}'], length=length)

            dataframe['position'] = np.where(dataframe['blue']>dataframe['red'], 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # sys.exit()
            return dataframe['position']
        elif name == "cmf" : 
            length = param1 
            dataframe['cmf'] = ta.cmf(close=dataframe[f'Close{suffix}'], high=dataframe[f'High{suffix}'], low=dataframe[f'Low{suffix}'], volume=dataframe['Volume'], length=length)

            dataframe['position'] = np.where(dataframe['cmf']>0, 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # sys.exit()
            return dataframe['position']
        elif name == "kvo" : 
            fast = param1
            slow = param2
            dataframe[['ref', 'lag']] = ta.kvo(close=dataframe[f'Close{suffix}'], high=dataframe[f'High{suffix}'], low=dataframe[f'Low{suffix}'], volume=dataframe['Volume'], fast=fast, slow=slow)

            dataframe['position'] = np.where(dataframe['ref']>0, 1, -1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # sys.exit()
            return dataframe['position']
        elif name == "delta" :     
            param1 = float(param1)
            pourc = param1

            dataframe['delta'] = ((dataframe[f'Close{suffix}'] / dataframe[f'Open{suffix}'] ) -1 ) * 100
            dataframe['position'] = np.where(dataframe['delta']<=pourc, 3, 0)
            return dataframe['position']
        elif name == "change_pos" :     

            dataframe['position'] = np.where((dataframe[f'Open{suffix}']<=dataframe[f'Close{suffix}']) & (dataframe[f'Open{suffix}'].shift()>=dataframe[f'Close{suffix}'].shift()), 1, 0)
            dataframe['position'] = np.where((dataframe[f'Open{suffix}']>=dataframe[f'Close{suffix}']) & (dataframe[f'Open{suffix}'].shift()<=dataframe[f'Close{suffix}'].shift()), -1, dataframe['position'])
            return dataframe['position']
        elif name == "volume_high" : 
            length = param1 
            dataframe['vol_ema'] = ta.ema(close=dataframe[f'Volume'], length=length)
            dataframe['position'] = np.where(dataframe['Volume']>dataframe['vol_ema'], 3, 0)
            return dataframe['position']
        elif name == "volume_low" : 
            length = param1 
            dataframe['vol_ema'] = ta.ema(close=dataframe[f'Volume'], length=length)
            dataframe['position'] = np.where(dataframe['Volume']<dataframe['vol_ema'], 3, 0)
            return dataframe['position']
        else : 
            message = f"Signal name not recognized : {name}"
            print(message)    
            if is_running_server :  telegram_api_erreur(message) ; sleep(300)
            sys.exit()
            return 0
    except Exception as e :
        print(e) 
        print(f"param1 : {param1}")
        print(f"param2 : {param2}")
        print(f"param3 : {param3}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        if is_running_server :  telegram_api_erreur(message) ; sleep(300)
        sys.exit()
def get_df_github(url, sep):
    try : 
        github_session = requests.Session()
        github_session.auth = ('constantin89P', GIT_TOKEN)

        REQUEST = github_session.get(url)
        if REQUEST.status_code == 200 :
            df = pd.read_csv(io.StringIO(REQUEST.content.decode('utf-8')), sep=sep)
            return df
        else : 
            message = f"Error fetching {url}, status code : {REQUEST.status_code}"
            print(message)
            telegram_api(message) ; sleep(30000)
            return pd.DataFrame()
    except Exception as e:
        print(e)
        return pd.DataFrame()
def Send_Order(ticker, side, qty, last_price, SL_type, SL_value):
    try :

        print(f"Try {side} order {ticker}, quantity : {qty}")
        if dt.datetime.now().second > 5 : return None, ""

        market_order_data = MarketOrderRequest(
                                symbol=ticker,
                                qty=qty,
                                side= side,
                                time_in_force=TimeInForce.DAY)
        market_order = trading_client.submit_order(order_data=market_order_data)
        print(market_order)

        
        SL_side = OrderSide.BUY if side == OrderSide.SELL else OrderSide.SELL

        test=0
        SLEEP_TIME = 0.1

        while True : 
            order = trading_client.get_order_by_id(order_id= market_order.id)
            if order.qty == order.filled_qty :
                filled_avg_price = order.filled_avg_price
                break
            else : 
                sleep(SLEEP_TIME)
                if test == 30 : return market_order, ""
                test+=1
                print(f"filled_avg_price n°{test} ")
                if test == 35 : SLEEP_TIME +=0.1

        while True : 
            try : 
                if SL_type == "Trailing_SL_fixed": 
                    SL_order_data = TrailingStopOrderRequest(
                                            symbol=ticker,
                                            qty=qty,
                                            side=SL_side,
                                            time_in_force=TimeInForce.DAY,
                                            trail_percent = float(SL_value),
                        )
                elif SL_type == "SL_fixed" : 
                    if SL_side == OrderSide.BUY : stop_price = np.round(float(filled_avg_price)*(1+(float(SL_value)/100)),2)
                    else : stop_price = np.round(float(filled_avg_price)*(1-(float(SL_value)/100)),2)
                    print(f"filled_avg_price : {filled_avg_price}")
                    print(f"stop_price : {stop_price}")
                    SL_order_data = StopOrderRequest(
                                            symbol=ticker,
                                            qty=qty,
                                            side=SL_side,
                                            time_in_force=TimeInForce.DAY,
                                            stop_price = stop_price,
                        )
                else : 
                    message = f"Unrecognized Stop Loss {SL_type}"
                    print(message)
                    telegram_api_erreur(message) ; sleep(30000)
                    
                SL_order = trading_client.submit_order(order_data=SL_order_data)
                print(SL_order)
                return market_order, SL_order

            except Exception as e:
                sleep(SLEEP_TIME)
                if test == 30 : return market_order, ""
                test+=1
                print(f"SL order n°{test} - {e}")
                if test == 25 : SLEEP_TIME +=0.1

    except Exception as e :
        print(f"Error in Send_Order {dt.datetime.now().time().strftime('%H:%M:%S')}")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = "Error Send_Order :" + str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        close_asset_positions(ticker)
        telegram_api_erreur(message) ; sleep(30000)
        return "0"




(SIGNAL_in, C_in_2, C_in_3, C_in_4, C_in_5,  
signal_in_param1, signal_in_param2, signal_in_param3, 
cond_2_param1, cond_2_param2, cond_2_param3,
cond_3_param1, cond_3_param2, cond_3_param3,
cond_4_param1, cond_4_param2, cond_4_param3,
cond_5_param1, cond_5_param2, cond_5_param3) = Strategy

(Signal_IN_type, Signal_OUT_type, SL_type, param1, param2, SL_value) = signals_combi



MEAN_SPREAD_L = []
MEAN_SPREAD_S = []
filled_p = None
previous_time = dt.datetime.now()
telegram_api_erreur(f"Début {SCRIPT_NAME}")



df_prices = get_historical_data(ticker, timestamp, (dt.datetime.now() - dt.timedelta(hours=7)), dt.datetime.now())
if not df_prices.empty : 
    SESSION = 'Open'
    END_SESSION = dt.time(21,50)
    # ending_session = (dt.datetime.combine(dt.datetime.min, END_SESSION) - dt.timedelta(minutes=10)).time()
    message = f"Session OPEN, will close all positions at {END_SESSION}" #- dt.timedelta(minutes=CLOSING_GAP)
    print(message)
    telegram_api_erreur(message)



while True:

    try:

        # 1 - Get prices 
        while True :

            # Get basics
            if SESSION == "Open" : SLEEP_TIME = 0.1
            elif (dt.datetime.now().minute in [29,30]) and (dt.datetime.today().weekday()<5) and (dt.datetime.now().hour in [13]): SLEEP_TIME = 0.1
            else : SLEEP_TIME = 30


            if SLEEP_TIME <= 1 :

                if (SESSION == "Close"):
                    df_prices = get_historical_data(ticker, timestamp, (dt.datetime.now() - dt.timedelta(hours=3)), dt.datetime.now())
                    df_prices = df_prices[(df_prices['Time'].dt.time >= pd.to_datetime('15:30').time()) & (df_prices['Time'].dt.time <= pd.to_datetime('22:00').time())]
                    if not df_prices.empty : 
                        SESSION = "Open" 
                        END_SESSION = dt.time(21,50)
                        df_prices = get_historical_data(ticker, timestamp, (dt.datetime.now() - dt.timedelta(days=4)), dt.datetime.now())
                        telegram_api_erreur(f"Opening session {dt.datetime.now().time()}, will close all positions on {END_SESSION}")
                        break
                
                elif (dt.datetime.now().second <= 2) or (dt.datetime.now().second >= 59)  : 
                    df_prices = get_historical_data(ticker, timestamp, (dt.datetime.now() - dt.timedelta(days=4)), dt.datetime.now())

                    if not df_prices.empty : 
                        print(f"Checking prices : {dt.datetime.now().time()}, previous_time : {previous_time.time().strftime('%H:%M:%S')}")
                        
                        # Break if it is new price
                        if df_prices['Time'].iloc[-1] != previous_time : previous_time = df_prices['Time'].iloc[-1] ; break

            sleep(SLEEP_TIME)
            

        start_analyse = dt.datetime.now()


        # 2 - If (heure de cloture- 10 min), je close ma position 
        print(f"END_SESSION : {END_SESSION} vs {df_prices['Time'].iloc[-1].time()}")
        if df_prices['Time'].iloc[-1].time() >= END_SESSION: 
            message = f"End session, closing all positions {dt.datetime.now().time()}"
            print(message)
            telegram_api_erreur(message)
            close_all_positions() 
            # cancel_all_orders() 
            SESSION = "Close"
            telegram_api_erreur("All positions closed, wait 30 min now")
            sleep(30*60) # wait 30 minutes to make sure I don't take anymore position


        # 3 - Get position
        position, holding_qty, portfolio_value = get_position()  


        # 4 - Check if need to exit position
        if (position!=0) and (Signal_OUT_type != "trailing"): 

            seuil = float(param1) if Signal_OUT_type == "target_ema" else 0
            
            # Check if pnl >0
            if filled_p : 
                if ((position > 0) and (df_prices['Close'].iloc[-1]>(filled_p*(1+seuil)))) or ((position < 0) and (df_prices['Close'].iloc[-1]<(filled_p*(1-seuil)))) :  

                    # Check if need to exit based on EMA
                    try : 
                        if Signal_OUT_type == "EMA_out" : 
                            df_prices['inf'] = ta.ema(df_prices["Close"], length=int(param1))
                            df_prices['sup'] = ta.ema(df_prices["Close"], length=int(param2))
                            df_prices['OUT_signal'] = np.where(df_prices['inf']>df_prices['sup'], 1, -1)
                            df_prices['OUT_signal'] = df_prices['OUT_signal'].diff() /2 
                        if Signal_OUT_type == "target_ema" : 
                            df_prices['sup'] = ta.ema(df_prices["Close"], length=int(param2))
                            df_prices['OUT_signal'] = np.where(df_prices['Close']>df_prices['sup'], 1, -1)
                            df_prices['OUT_signal'] = df_prices['OUT_signal'].diff() /2 
                    except Exception as e : 
                        message = f"{e}"
                        print(message)
                        telegram_api_erreur(message)  
                        print(df_prices.tail(10))     
                        sleep(900)   
                
                    # Exit position if signal_OUT 
                    print(f"Checking if position need to be exited, position = {position}")
                    print(df_prices.tail(10))
                    if (df_prices['OUT_signal'].iloc[-1]*-1) == position : close_asset_positions(ticker)

                else : 
                    print(f"Exit annulé, position :{position}, filled_p : {filled_p}, last price : {df_prices['Close'].iloc[-1]}")

        # 5 - If no position, check need to enter position
        if (SESSION == "Open"): 
            print(f"Check IN signal")

            # Cancel previous orders, dans le doute
            # cancel_all_orders()

            df_prices["COND_1"] = get_signal(suffix, df_prices, ticker, SIGNAL_in, signal_in_param1, signal_in_param2, signal_in_param3)
            df_prices["COND_2"] = get_signal(suffix, df_prices, ticker, C_in_2, cond_2_param1, cond_2_param2, cond_2_param3)
            df_prices["COND_3"] = get_signal(suffix, df_prices, ticker, C_in_3, cond_3_param1, cond_3_param2, cond_3_param3)
            df_prices["COND_4"] = get_signal(suffix, df_prices, ticker, C_in_4, cond_4_param1, cond_4_param2, cond_4_param3)
            df_prices["COND_5"] = get_signal(suffix, df_prices, ticker, C_in_5, cond_5_param1, cond_5_param2, cond_5_param3)

            bool_signal_in = df_prices["COND_1"].diff()!=0 # Only on change
            bool_cond_2 = (df_prices["COND_2"] == df_prices["COND_1"]) | (df_prices["COND_2"] == 3)
            bool_cond_3 = (df_prices["COND_3"] == df_prices["COND_1"]) | (df_prices["COND_3"] == 3)
            bool_cond_4 = (df_prices["COND_4"] == df_prices["COND_1"]) | (df_prices["COND_4"] == 3)
            bool_cond_5 = (df_prices["COND_5"] == df_prices["COND_1"]) | (df_prices["COND_5"] == 3)


            # Signal IN
            if Signal_IN_type == "force_pos" :
                df_prices['IN'] = np.where(bool_signal_in & bool_cond_2 & bool_cond_3 & bool_cond_4 & bool_cond_5, df_prices["COND_1"], np.nan)
            elif Signal_IN_type == "on_change" :
                df_prices['IN'] = np.where(bool_signal_in & bool_cond_2 & bool_cond_3 & bool_cond_4 & bool_cond_5, df_prices["COND_1"], np.nan)
            elif Signal_IN_type == "All" :
                df_prices['IN'] = np.where(bool_cond_2 & bool_cond_3 & bool_cond_4 & bool_cond_5, df_prices["COND_1"], np.nan)
            else :
                message = f"Error with Signal_IN_type not recognized : {Signal_IN_type}"
                print(message)
                if is_running_server :  telegram_api_erreur(message) ; sleep(300)
                sys.exit()



            print(df_prices[['Close', 'Time', 'IN']].tail(5))

            if (df_prices['IN'].iloc[-1] == 1) and (position != 1) and (df_prices['IN'].diff().iloc[-1]!=0): 
                # BUYING
                cancel_status = trading_client.cancel_orders()
                if position == -1 : close_asset_positions(ticker)
                market_order, stop_loss_order = Send_Order(ticker, OrderSide.BUY, QUANTITY, df_prices['Close'].iloc[-1], SL_type, SL_value) 
                orders = trading_client.get_orders(GetOrdersRequest(status="closed",side="buy",limit=1))
                filled_p = float(orders[0].filled_avg_price)
                spread = np.round(((filled_p - df_prices['Close'].iloc[-1])/df_prices['Close'].iloc[-1])*100,4)
                MEAN_SPREAD_L.append(spread)
                print(MEAN_SPREAD_L)
                print(f"BOUGHT {filled_p} vs expected : {df_prices['Close'].iloc[-1]}")
                mean_L = np.round(sum(MEAN_SPREAD_L) / len(MEAN_SPREAD_L),4)
                message_L = f"BUY orders history: {len(MEAN_SPREAD_L)}, mean spread : {mean_L}%"
                print(message_L)
            elif (df_prices['IN'].iloc[-1] == -1) and (position != -1): 
                # SHORT SELLING
                cancel_status = trading_client.cancel_orders()
                if position == 1 : close_asset_positions(ticker)
                market_order, stop_loss_order = Send_Order(ticker, OrderSide.SELL, QUANTITY, df_prices['Close'].iloc[-1], SL_type, SL_value) 
                
                orders = trading_client.get_orders(GetOrdersRequest(status="closed",side="sell",limit=1))
                filled_p = float(orders[0].filled_avg_price)
                spread = np.round(((df_prices['Close'].iloc[-1] - filled_p)/df_prices['Close'].iloc[-1])*100,4)
                MEAN_SPREAD_S.append(spread)
                print(MEAN_SPREAD_S)
                print(f"SOLD {filled_p} vs expected : {df_prices['Close'].iloc[-1]}")
                mean_S = np.round(sum(MEAN_SPREAD_S) / len(MEAN_SPREAD_S),4)
                message_S = f"SELL orders history: {len(MEAN_SPREAD_S)}, mean spread : {mean_S}%"
                print(message_S)

        

        print(f"Durée d'analyse : {(dt.datetime.now()- start_analyse).total_seconds()}s")
        print('\n\n- - - - - - - - - \n\n') ; sleep(3)
 	
    except IndexError as e :
        print(e) 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message)
    except KeyboardInterrupt: 
        print("KeyboardInterrupt")
        sys.exit()
    except Exception as e :
        print(e) 
        print(df_prices)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        telegram_api_erreur(message)
        close_asset_positions(ticker)
        sleep(300)
        sys.exit()
