import pandas as pd
import numpy as np
import sys, socket, os, requests, io, random, tempfile, ast
from tqdm import tqdm
import datetime as dt
import pandas_ta as ta
from github import Github
from time import sleep
from itertools import product
from pathlib import Path
from binance.client import Client
from binance.enums import *
import matplotlib.pyplot as plt

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
pd.options.mode.chained_assignment = None


hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

is_running_server = True if local_ip not in list(os.getenv('WHITELISTIP')) else False

# Ajouter un paramètre mettre à jour les prix ? True False
GIT_TOKEN = os.getenv('GIT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
TOKEN_TELEGRAM = os.getenv('TOKEN_TELEGRAM')
TOKEN_TELEGRAM_ERROR = os.getenv('TOKEN_TELEGRAM_ERROR')
B_KEY_C = os.getenv('B_KEY_C')
B_SECRET_C = os.getenv('B_SECRET_C')
A_KEY = os.getenv('A_KEY')
A_SECRET = os.getenv('A_SECRET')
ALPHA_KEY = os.getenv('ALPHA_KEY')

frequence_dic = {'1m': '1min', '3m': '3min', '5m': '5min', '15m' : '15min', '30m' : '30min',
                '1h': '1H', '2h': '2H', '4h': '4H', '6h' : '6H', '8h' : '8H',
                '12h': '12H', '1d': '1D', '3d': '3D', '1w' : '1W', '1M' : '1M'}
KLINE = {
    '1m': Client.KLINE_INTERVAL_1MINUTE,
    '3m'  :Client.KLINE_INTERVAL_3MINUTE,
    '5m': Client.KLINE_INTERVAL_5MINUTE,
    '15m': Client.KLINE_INTERVAL_15MINUTE,
    '30m': Client.KLINE_INTERVAL_30MINUTE,
    '1h':  Client.KLINE_INTERVAL_1HOUR,  
    '2h': Client.KLINE_INTERVAL_2HOUR,
    '4h'  :Client.KLINE_INTERVAL_4HOUR,
    '6h': Client.KLINE_INTERVAL_6HOUR,
    '8h': Client.KLINE_INTERVAL_8HOUR,
    '12h': Client.KLINE_INTERVAL_12HOUR,    
    '1d':  Client.KLINE_INTERVAL_1DAY
}



def telegram_api(SCRIPT_NAME, *text):
    for x in range(5):
        try : 
            for element in text : 
                element = SCRIPT_NAME + " - " + element
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
def telegram_api_erreur(SCRIPT_NAME, *text):
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
    sys.exit()
def get_signal(suffix, df, ticker, name, param1, param2, param3):
    try :
        dataframe = df.copy()

        try : param1 = int(param1)
        except : pass
        try : param2 = int(param2)
        except : pass
        try : param3 = int(param3)
        except : pass


        tolerance = 0.025 /100  # Tolerance level for Open and Close being almost equal
        shadow_ratio = 3  # Ratio of body length to lower shadow length
        is_Bull_Doji = (((abs(dataframe['Open'] / dataframe['Close'])-1) <= tolerance) &   
            ((dataframe['Open'] - dataframe['Low']) > shadow_ratio * (dataframe['High'] - dataframe['Open'])) &  
            (dataframe['High'] - dataframe['Close'] <= tolerance)) 
        is_Bear_Doji = (((abs(dataframe['Open'] / dataframe['Close'])-1) <= tolerance) &  
            ((dataframe['High'] - dataframe['Open']) > shadow_ratio * (dataframe['Open'] - dataframe['Low'])) &  
            (dataframe['Low'] - dataframe['Close'] <= tolerance)) 
        dataframe['is_bull_doji'] = np.where(is_Bull_Doji, 1, np.nan)
        dataframe['is_bear_doji'] = np.where(is_Bear_Doji, 1, np.nan)
        
        
        dataframe['is_red'] = np.where(dataframe['Close'] < dataframe['Open'], 1, np.nan)
        dataframe['is_green'] = np.where(dataframe['Close'] > dataframe['Open'], 1, np.nan)
        dataframe['bar_lenght'] = dataframe['High'] - dataframe['Low']
        dataframe['body_lenght'] = abs(dataframe['Open'] - dataframe['Close'])
        dataframe['upper_shadow'] = np.where((dataframe['is_green'] == 1), ((dataframe['High']-dataframe['Close'])/dataframe['bar_lenght']), ((dataframe['High']-dataframe['Open'])/dataframe['bar_lenght'])) 
        dataframe['lower_shadow'] = np.where((dataframe['is_green'] == 1), ((dataframe['Open']-dataframe['Low'])/dataframe['bar_lenght']), ((dataframe['Close']-dataframe['Low'])/dataframe['bar_lenght'])) 
        dataframe['body_pourc'] = np.round(abs(dataframe['Open']-dataframe['Close']) / dataframe['bar_lenght'], 3)

        dataframe['body_pourc_Q80'] = dataframe['body_pourc'].rolling(200).quantile(0.8)

        
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

            dataframe['position'] = np.where((dataframe['prev'] <= -50) & (dataframe[ref_name] > -50), 1, 0)
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

            dataframe['position'] = np.where((dataframe['prev'] <= 20) & (dataframe[ref_name] > 20), 1, 0)
            dataframe['position'] = np.where((dataframe['prev'] >= 80) & (dataframe[ref_name] < 80), -1, dataframe['position'])
            # dataframe['position'] = dataframe['position'].ffill()
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
            dataframe['position'] = np.where((dataframe['prev'] <= 20) & (dataframe[ref_name] > 20), 1, 0)
            dataframe['position'] = np.where((dataframe['prev'] >= 80) & (dataframe[ref_name] < 80), -1, dataframe['position'])
            # dataframe['position'] = dataframe['position'].ffill()
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # print(dataframe.tail(10))
            # sys.exit()
            return dataframe['position']
        elif name == "rsi_OB_OS" : 
            length = param1 
            maximum = int(param2)
            ref_name = f"RSI_{length}"
            dataframe[ref_name] = ta.rsi(close=dataframe[f'Close{suffix}'], length=length)
            dataframe['prev'] = dataframe[ref_name].shift(1).fillna(0)

            dataframe['position'] = np.where((dataframe['prev'] <= maximum) & (dataframe[ref_name] > maximum), 1, 0)
            dataframe['position'] = np.where((dataframe['prev'] >= (100-maximum)) & (dataframe[ref_name] < (100-maximum)), -1, dataframe['position'])
            # dataframe['position'] = dataframe['position'].ffill()
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
        elif name == "rsi_50" : 
            length = int(param1) 
            ref_name = f"RSI_{length}"
            dataframe[ref_name] = ta.rsi(close=dataframe[f'Close{suffix}'], length=length)

            dataframe['position'] = np.where(dataframe[ref_name]>50, -1, 1)
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
        elif name == "hma_hma" : 
            length = param1 
            dataframe['hma1'] = ta.hma(close=dataframe[f'Close{suffix}'], length=int(param1))
            dataframe['hma2'] = ta.hma(close=dataframe[f'Close{suffix}'], length=int(param2))

            dataframe['position'] = np.where(dataframe['hma2']>dataframe['hma2'], -1, 1)
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
        elif name == "Dojis" : 


            bool_long = ((dataframe['is_bull_doji'] == 1) &
                        (dataframe['is_red'].shift(1) == 1) &
                        (dataframe['is_red'].shift(2) == 1)) 
            
            bool_short = ((dataframe['is_bear_doji'] == 1) &
                        (dataframe['is_green'].shift(1) == 1) &
                        (dataframe['is_green'].shift(2) == 1)) 
            

            dataframe['position'] = np.where(bool_long, 1, 0)
            dataframe['position'] = np.where(bool_short,-1, dataframe['position'])

            return dataframe['position']
        elif name == "5_soldiers" : 
            bool_long = ((dataframe['is_green'].shift(4) ==1)
                          & (dataframe['is_green'].shift(3) ==1)
                          & (dataframe['is_green'].shift(2) ==1)
                          & (dataframe['is_green'].shift(1) ==1)
                          & (dataframe['is_green'] ==1)
                          & (abs((df['Close']/df['Open'].shift(4))-1) > 0.0092))
            
            bool_short = ((dataframe['is_red'].shift(4) ==1)
                          & (dataframe['is_red'].shift(3) ==1)
                          & (dataframe['is_red'].shift(2) ==1)
                          & (dataframe['is_red'].shift(1) ==1)
                          & (dataframe['is_red'] ==1)
                          & (abs((df['Open']/df['Close'].shift(4))-1) > 0.0092))
            
            dataframe['position'] = np.where(bool_long,   1, 0)   
            dataframe['position'] = np.where(bool_short, -1,dataframe['position'])


            return dataframe['position']
        elif name == "3_soldiers" : 
            bool_short = ((dataframe["Close"].shift(2) < dataframe["Open"].shift(2))    # Red
                        & (dataframe["Close"].shift(1) < dataframe["Open"].shift(1))    # Red
                        & (dataframe["Close"] < dataframe["Open"]))                     # Red
            
            bool_long = ((dataframe["Close"].shift(2) > dataframe["Open"].shift(2))     # Green
                        & (dataframe["Close"].shift(1) > dataframe["Open"].shift(1))    # Green
                        & (dataframe["Close"] > dataframe["Open"]))                     # Green
            
            dataframe['position'] = np.where(bool_long,   1, 0)   
            dataframe['position'] = np.where(bool_short, -1,dataframe['position'])

            return dataframe['position']
        elif name == "strong_bar" : 
            pourc_max = float(param1)
            
            dataframe['position'] = np.where((dataframe['is_green'] == 1) & (dataframe['upper_shadow'] < pourc_max), 1, 0)   
            dataframe['position'] = np.where((dataframe['is_red'] == 1) & (dataframe['lower_shadow'] < pourc_max), -1,dataframe['position'])

            return dataframe['position']
        elif name == "ema_n" : 
            length = param1 
            dataframe['ema'] = ta.ema(close=dataframe[f'Close{suffix}'], length=length)

            dataframe['position'] = np.where(dataframe['ema']<dataframe[f'Close{suffix}'], -1, 1)
            # print(dataframe[dataframe['Time']>= dt.datetime(2024,5,31,19,28)].head(20))
            # sys.exit()
            return dataframe['position']
        elif name == "eam_ema" : 

            dataframe['ema1'] = ta.ema(close=dataframe[f'Close{suffix}'], length=int(param1))
            dataframe['ema2'] = ta.ema(close=dataframe[f'Close{suffix}'], length=int(param2))

            dataframe['position'] = np.where(dataframe['ema1']>dataframe['ema2'], 1, -1)


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
def save_to_csv(dataframe, filepath, sep=';', index = False):
    try : 
        if "Open" in dataframe.columns : dataframe['Open'] = dataframe['Open'].astype(float)
        if "High" in dataframe.columns : dataframe['High'] = dataframe['High'].astype(float)
        if "Low" in dataframe.columns : dataframe['Low'] = dataframe['Low'].astype(float)
        if "Close" in dataframe.columns : dataframe['Close'] = dataframe['Close'].astype(float)
        if "Volume" in dataframe.columns : dataframe['Volume'] = dataframe['Volume'].astype(float)
        dataframe.to_csv(filepath, sep=sep, index=index, encoding='utf-8-sig')
        print("Dataframe saved corretly")
    except Exception as e:
        print(e)
        backup_filepath = f"{filepath[0: -4]}_backup.csv"
        dataframe.to_csv(backup_filepath, sep=sep, index = index, encoding='utf-8-sig')
def get_df_github(url, sep, MODE):

    if MODE != "PROD" : 
        filepath = url.split('main/')[-1]
        df_filepath = f"D:\Documents\Python\Bot\Avril_2024\{filepath}"
        df = pd.read_csv(df_filepath, sep=sep)
        return df

    github_session = requests.Session()
    github_session.auth = ('constantin89P', GIT_TOKEN)

    # Try version NON incremental 
    REQUEST = github_session.get(url)
    if REQUEST.status_code == 200 :
        df = pd.read_csv(io.StringIO(REQUEST.content.decode('utf-8')), sep=sep, low_memory=False)
        df = df.astype(str).drop_duplicates()
        return df

    # Else try version WITH incremental 
    df = pd.DataFrame()
    new_url = url
    num = 0
    while True : 
        try : 
            base, ext = os.path.splitext(url)
            new_url = f"{base}_{num}{ext}"
        
            REQUEST = github_session.get(new_url)
            if REQUEST.status_code == 200 :
                df_int = pd.read_csv(io.StringIO(REQUEST.content.decode('utf-8')), sep=sep, low_memory=False)
                df = pd.concat([df, df_int])
                num += 1
            elif int(REQUEST.status_code) == 404 :
                df = df.astype(str).drop_duplicates()
                return df
            else : 
                print(f"Unable to get {new_url}, status code : {REQUEST.status_code}")
                df = df.astype(str).drop_duplicates()
                return df
        except Exception as e:
            print(f"No dataframe github for: {new_url.split('/')[-1]} // {e}")
            return df
def push_df_github(df, sep, repo_name, git_file, commit_text, SCRIPT_NAME, MODE):

    if MODE != "PROD" : 
        df_filepath = f"D:\Documents\Python\Bot\Avril_2024\{git_file}"
        df = save_to_csv(df, df_filepath, sep)
        return df


    MAXIMUM_SIZE = 30 # MB


    time_sleep = 60
    for x in range(5):
        try : 

            g = Github(GIT_TOKEN)
            repo = g.get_user().get_repo(repo_name)

            # 1 - Try to update version SANS incremental 
            try : 
                contents = repo.get_contents(git_file)
                repo.update_file(contents.path, commit_text, content, contents.sha)
                message = f"UPDATED original, {git_file}"
                print(message)
                telegram_api(SCRIPT_NAME, message)
                return 
            except : pass
            

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                # Save the DataFrame to the temporary file
                df.to_csv(tmp_file.name, index=False)
                temp_filepath = tmp_file.name

            # Get the size of the temporary file
            file_size_mb = (os.path.getsize(temp_filepath)) / (1024**2)
            # print(f"Estimated file size: {file_size_mb:.2f} MB")
            os.remove(temp_filepath)

            num_chunks = int((file_size_mb + MAXIMUM_SIZE - 1) // MAXIMUM_SIZE) +1  # Ceiling division to determine the number of chunks
            # print(f"num_chunks : {num_chunks}")


            chunk_size = int(len(df) / max(((file_size_mb) / MAXIMUM_SIZE), 1)) + 1
            # num_chunks = (len(df) + chunk_size - 1) // chunk_size  # Ceiling division to determine the number of chunks
            # print(f"chunk_size : {chunk_size}")
            # sys.exit()
            # sys.exit()
            new_git_filename = git_file
            for i in range(num_chunks):
                # print(f"chunk i : {i}")

                start_row = i * chunk_size
                end_row = start_row + chunk_size
                chunk = df.iloc[start_row:end_row]
                content = chunk.to_csv(sep=sep, index=False)
                directory, filename = os.path.split(git_file)
                name, ext = os.path.splitext(filename)
                new_git_filename = directory + f"/{name}_{i}{ext}"
                # print(f"new_git_filename : {new_git_filename}, len : {len(chunk)}")

                if len(chunk) == 0 : continue


                # 2 - Try to update version AVEC incremental 
                try:
                    contents = repo.get_contents(new_git_filename)
                    repo.update_file(contents.path, commit_text, content, contents.sha)
                    message = f"UPDATED, {new_git_filename}, taille :{len(chunk)}"
                    print(message)
                    telegram_api(SCRIPT_NAME, message)
                # 3 - Create file with incremental 
                except :
                    repo.create_file(new_git_filename, commit_text, content)
                    message = f"CREATED, {new_git_filename}, taille :{len(chunk)}"
                    print(message)
                    telegram_api(SCRIPT_NAME, message)

            return
        except KeyboardInterrupt : 
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"KeyboardInterrupt line : {exc_tb.tb_lineno}")
            sys.exit()
        except Exception as e : 
            exc_type, exc_obj, exc_tb = sys.exc_info()
            message = f"{dt.datetime.now()} - Error {x} while uploading {new_git_filename} to Github : \n{e}\nWill now sleep for : {time_sleep}, line {exc_tb.tb_lineno}"
            print(message)
            if is_running_server :  telegram_api(SCRIPT_NAME, message) 
            sleep(time_sleep)
            time_sleep *= 2
    sys.exit()
def update_and_push_df_github(df, sep, repo, git_file, commit_text, full_url, SCRIPT_NAME, MODE, incremental=False):
    time_sleep = 60
    for x in range(5):
        try : 
            print(f"\n{x} - Trying to save {git_file}")
            old_df = get_df_github(full_url, sep, MODE)
            df_new = pd.concat([old_df, df])
            df_new = df_new.astype(str).drop_duplicates()
            print(f"len df_new : {len(df_new)} vs old :{len(old_df)}")
            if len(old_df) >= len(df_new) : print(f"No new rows to save, ending") ; return
            push_df_github(df_new, sep, repo, git_file, commit_text, SCRIPT_NAME, MODE)
            return
        except KeyboardInterrupt : 
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"KeyboardInterrupt line : {exc_tb.tb_lineno}")
            sys.exit()
        except Exception as e : 
            message = f"{dt.datetime.now()} - Error while update_and_push Github : \n{e}\nWill now sleep for : {time_sleep}"
            print(message)
            if is_running_server :  telegram_api(SCRIPT_NAME, message) 
            sleep(time_sleep)
            time_sleep *= 2
    sys.exit()
def add_HA_OHLC(df):
    try : 
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
        ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

        # Create a new DataFrame to store Heikin Ashi values
        ha_df = pd.DataFrame(index=df.index, columns=['HA_Open', 'HA_High', 'HA_Low', 'HA_Close'])
        ha_df['HA_Close'] = ha_close

        # Calculate Heikin Ashi Open using shift
        ha_df['HA_Open'] = (ha_df['HA_Close'].shift(1) + ha_df['HA_Close'].shift(2)) / 2
        ha_df['HA_Open'].iloc[0] = df['Open'].iloc[0]  # Initialize the first HA_Open

        # Forward fill the HA_Open for the second value (manual adjustment for the first calculation)
        ha_df['HA_Open'].iloc[1] = (df['Open'].iloc[0] + ha_close.iloc[0]) / 2

        # Calculate Heikin Ashi High and Low
        ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close']].join(df['High']).max(axis=1)
        ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close']].join(df['Low']).min(axis=1)

        # Adding signals for bullish or bearish candles
        # ha_df['sens'] = np.where(ha_df['HA_Close'] > ha_df['HA_Open'], 1, -1)

        return ha_df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]
    except Exception as e :
        print(f"Error add_HA_OHLC : {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        sys.exit()


def epoch2human(epoch):
    #return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(epoch)/1000.0))
    return dt.datetime.fromtimestamp(epoch)
def get_prices_binance(ticker, interval, debut, fin, REPO, SCRIPT_NAME, MODE):

    if MODE != "PROD" : 
        df_prices_filepath = f"D:\Documents\Python\Bot\Avril_2024\Prices\\{ticker}_{str(interval)}.csv"
        df_prices = pd.read_csv(df_prices_filepath)
        df_prices['Time'] = pd.to_datetime(df_prices['Time'])
        df_prices = df_prices.sort_values(by='Time')
        df_prices = df_prices.reset_index(drop=True)
        df_prices = df_prices.drop_duplicates()
        return df_prices

        


    client = Client(B_KEY_C, B_SECRET_C)

    df_prices_filepath = f"Prices/{ticker}_{interval}.csv"
    copy_filename = df_prices_filepath.replace('.csv', '_copy.csv')
    full_url = f"https://raw.githubusercontent.com/constantin89P/{REPO}/main/{df_prices_filepath}"
    

    
    heures = int(1000 * (interval / 60))
    df_newly_created = False

    start_to_do = 0
    end_to_do = 0

    if ticker == "BTCFDUSD" : debut = dt.datetime(2023,8,5)

    try : 
        # raise
        df = get_df_github(full_url, ";", MODE)
        if df.empty : raise
        df['Time'] = pd.to_datetime(df['Time'])
        df_prices = df.sort_values(by='Time')
        df_prices[['Open', 'High', 'Low', 'Close']] = df_prices[['Open', 'High', 'Low', 'Close']].astype(float)
        
        # print(df_prices)


        if df_prices['Time'].min() > debut : start_to_do = 1
        if df_prices['Time'].max() < fin : end_to_do = 1
        if (start_to_do + end_to_do) == 0 : print("Prices recuperation OK, no need to get any more price\n\n") ; return df_prices

    except Exception as e:
        # print(e)
        time_to_delete = dt.datetime.now() + dt.timedelta(minutes=10)
        df_prices = pd.DataFrame({'Time': [time_to_delete], 'Close': [0]})
        start_to_do = 1
        end = fin
        start = end - dt.timedelta(hours=heures)
        df_newly_created = True

    
    # Create copy of the prices file
    # print("Saving copy")
    # push_df_github(df_prices, ";", REPO, copy_filename, "Saving backup", SCRIPT_NAME, MODE)


    if interval in [1,3,5,15,28] : interval_prices = f"{interval}m"
    elif interval == 60 : interval_prices = '1h'
    else : print(f'interval {interval} not recognized or not coded yet') ; sys.exit()
    

    # print("Writing in csv")

    try : 

        if start_to_do == 1 : 
            if df_newly_created : end = fin
            else : end = df_prices['Time'].min() + dt.timedelta(minutes=5)
            start = end - dt.timedelta(hours=heures)
            
            while df_prices['Time'].min() > debut : 

                print(start)
                print(end, "\n")

                # Make the request
                url = client.get_historical_klines(symbol= ticker, interval=KLINE[interval_prices], start_str =str(start), end_str = str(end))

                Open_time, Open, High, Low, Close, Volume, Close_time, Quote_asset_volume, Number_of_trades, Taker_buy_base_asset_volume, Taker_buy_quote_asset_volume, Ignore = map(list, zip(*url))
                timing = [int(str(t)[:-3]) for t in Close_time]
                humanTimes = [epoch2human(t) for t in timing]

                df_prices_int = pd.DataFrame({'Time': humanTimes, 'Low': Low, 'High': High, 'Open': Open, 'Close': Close, 'Volume': Volume})
                df_prices = pd.concat([df_prices_int, df_prices])
                df_prices = df_prices.drop_duplicates()
                
                end = start
                start = end - dt.timedelta(hours=heures)
        
        if end_to_do == 1 :
            print(f"maximum available {df_prices['Time'].max()}")
            print(f"maximum requested {fin}")

            start = df_prices['Time'].max() - dt.timedelta(minutes=5)
            end = start + dt.timedelta(hours=heures)
            end = min(end, dt.datetime.now())
            while df_prices['Time'].max() < fin : 

                print(start)
                print(end, "\n")

                # Make the request
                url = client.get_historical_klines(symbol= ticker, interval=KLINE[interval_prices], start_str =str(start), end_str = str(end))

                Open_time, Open, High, Low, Close, Volume, Close_time, Quote_asset_volume, Number_of_trades, Taker_buy_base_asset_volume, Taker_buy_quote_asset_volume, Ignore = map(list, zip(*url))
                timing = [int(str(t)[:-3]) for t in Close_time]
                humanTimes = [epoch2human(t) for t in timing]

                df_prices_int = pd.DataFrame({'Time': humanTimes, 'Low': Low, 'High': High, 'Open': Open, 'Close': Close, 'Volume': Volume})
                df_prices = pd.concat([df_prices_int, df_prices])
                df_prices = df_prices.drop_duplicates()
                
                start = end
                end = end + dt.timedelta(hours=heures)

    except KeyboardInterrupt : 
        pass
    except Exception as e :
        print(f"Error requesting prices : {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")

    if df_newly_created : df_prices = df_prices[df_prices['Time'] != time_to_delete]
    
    if df_prices.empty : print("df_prices empty") ; sys.exit()
    # df_prices[['Low', 'High', 'Open', 'Close']] = np.round(df_prices[['Low', 'High', 'Open', 'Close']], decimals=2)
    df_prices = df_prices.drop_duplicates()
    df_prices = df_prices.sort_values(by='Time')
    print(df_prices)
    push_df_github(df_prices, ";", REPO, df_prices_filepath, "Updated prices", SCRIPT_NAME, MODE)

    df_prices[['Open', 'High', 'Low', 'Close']] = df_prices[['Open', 'High', 'Low', 'Close']].astype(float)

    return df_prices
def get_prices_alpaca(ticker, interval, debut, fin, REPO, SCRIPT_NAME, MODE):

    if MODE != "PROD" : 
        df_prices_filepath = f"D:\Documents\Python\Bot\Avril_2024\Prices\\{ticker}_{str(interval)}.csv"
        df_prices = pd.read_csv(df_prices_filepath)
        df_prices['Time'] = pd.to_datetime(df_prices['Time'])
        df_prices = df_prices.sort_values(by='Time')
        df_prices = df_prices.reset_index(drop=True)
        df_prices = df_prices.drop_duplicates()
        return df_prices

    df_prices_filepath = f"Prices/{ticker}_{interval}.csv"
    copy_filename = df_prices_filepath.replace('.csv', '_copy.csv')
    full_url = f"https://raw.githubusercontent.com/constantin89P/{REPO}/main/{df_prices_filepath}"

    heures = int(1000 * (interval / 60))
    df_newly_created = False

    start_to_do = 0
    end_to_do = 0

    # Read current dataframe
    df = get_df_github(full_url, ";", MODE)
    if not df.empty : 
        df['Time'] = pd.to_datetime(df['Time'])
        df_prices = df.sort_values(by='Time')
        # print(df_prices)

        if debut.time() < dt.time(9, 0) : debut = debut.replace(hour=9, minute=1, second=0, microsecond=0)

        if df_prices['Time'].min() > debut : start_to_do = 1
        if (df_prices['Time'].max() + dt.timedelta(days=1)) < fin : end_to_do = 1
        if (start_to_do + end_to_do) == 0 : print("Prices recuperation OK, no need to get any more price\n\n") ; return df_prices

        debut = debut - dt.timedelta(days=5)
        fin = min((fin + dt.timedelta(days=5)), (dt.datetime.now()-dt.timedelta(days=1)))

        print(f"start_to_do {start_to_do} with debut : {debut}, min : {df_prices['Time'].min()}")
        print(f"end_to_do {end_to_do} with fin : {fin}, max : {df_prices['Time'].max()}")
    else :
        time_to_delete = dt.datetime.now() + dt.timedelta(minutes=10)
        df_prices = pd.DataFrame({'Time': [time_to_delete], 'Close': [0]})
        start_to_do = 1
        end = fin
        start = end - dt.timedelta(hours=heures)
        df_newly_created = True


    try : 

        if start_to_do == 1 : 
            # print("Fetching start")
            if df_newly_created : end = fin
            else : end = df_prices['Time'].min() 
            start = debut - dt.timedelta(minutes=2)
            
            bars_df = get_historical_data_alpaca(ticker, interval, start, end) 
        elif end_to_do == 1 :
            print("Fetching end")
            start = df_prices['Time'].max() - dt.timedelta(minutes=5)
            end = min((fin + dt.timedelta(days=1)), dt.datetime.now())

            bars_df = get_historical_data_alpaca(ticker, interval, start, end) 

        if bars_df.empty : print("bars_df empty") ; sys.exit()

        df_prices = pd.concat([df_prices, bars_df])
        df_prices = df_prices.drop_duplicates()
        if df_newly_created : df_prices = df_prices[df_prices['Time'] != time_to_delete]
        df_prices = df_prices.sort_values(by='Time')
        df_prices = df_prices.reset_index(drop=True)

        push_df_github(df_prices, ";", REPO, df_prices_filepath, "Updated prices", SCRIPT_NAME, MODE)

        return df_prices

    except KeyboardInterrupt : 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"KeyboardInterrupt line : {exc_tb.tb_lineno}")
        sys.exit()
    except Exception as e :
        print(f"Error requesting prices : {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        sys.exit()

def get_historical_data_alpaca(symbol, interval, start, end):
    if interval <60 : 
        timeframe = TimeFrame(interval, TimeFrameUnit.Minute) 
        start = start - dt.timedelta(days=4)
    elif interval == 60 : 
        timeframe = TimeFrame(1, TimeFrameUnit.Minute) 
        start = start - dt.timedelta(days=4)
    elif interval == 1440 : 
        timeframe = TimeFrame(1, TimeFrameUnit.Day)
        start = start - dt.timedelta(days=50)
    else : print(f'interval {interval} not recognized or not coded yet') ; sys.exit()
    end = min(end + dt.timedelta(days=1), dt.datetime.now())

    

    for x in range(10) : 
        try : 
            client = StockHistoricalDataClient(A_KEY, A_SECRET)

            request_params = StockBarsRequest(
                                    symbol_or_symbols=[symbol],
                                    timeframe=timeframe,
                                    start=start,
                                    end=end, 
                                    feed='iex',
                                    adjustment='all') # 'all', 'split', 'dividend'
            
            bars = client.get_stock_bars(request_params)

            # Check if there are prices in response
            if bars.df.empty : 
                print(f"No prices yet {dt.datetime.now().time().strftime('%H:%M:%S')}")
                return pd.DataFrame()
            
            bars = bars.df.droplevel(level=0) 
            bars_df = bars.tz_convert('Europe/Paris')

            if interval == 60 :
                agg_functions = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'trade_count': 'sum'}
                bars_df = bars_df.resample('1H', offset='30T').agg(agg_functions).between_time('15:30', '22:00')


            bars_df['Time'] = bars_df.index   
            bars_df = bars_df.reset_index(drop=True)
            bars_df['Time'] = bars_df['Time'].dt.tz_localize(None)
            bars_df = bars_df[['Time', 'open', 'high', 'low', 'close', 'volume']]

            bars_df = bars_df.rename(columns={'open': 'Open',
                                            'low':'Low',
                                            'close':'Close',
                                            'high':'High',
                                            'volume':'Volume'})
            bars_df = bars_df.dropna()
            bars_df['Open'] = bars_df['Open'].astype(float)
            bars_df['High'] = bars_df['High'].astype(float)
            bars_df['Low'] = bars_df['Low'].astype(float)
            bars_df['Close'] = bars_df['Close'].astype(float)
            bars_df = bars_df.reset_index(drop=True)
            bars_df = bars_df.drop_duplicates()

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


def get_nb_trades(df, SCRIPT_NAME):
    try : 

        df['trade_price'] = np.where((df['IN'].notna()) & ((df['IN']!=df['position'].shift()) | ((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1)))), df['Close'], np.nan)
        df['trade_price'] = np.where(df['position']==0, 0, df['trade_price'])
        df['trade_price'] = df['trade_price'].ffill()

        df['nb_trades'] = np.where((df['IN'].notna()) & ((df['IN']!=df['position'].shift()) | ((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1)))), 1, 0)
        df['nb_trades'] = df['nb_trades'].cumsum()

        return df
    except Exception as e :
        print(e) 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        if is_running_server :  telegram_api(SCRIPT_NAME,message) ; sleep(300)
        sys.exit()
def get_position(df, Signal_OUT_type, param1, SCRIPT_NAME, OUT_mode):
    try : 

        # Create position 
        df['position'] = np.where((df['OUT'] == 2)| (df['OUT'] == 3), 0, np.nan)
        df['position'] = np.where((df['IN'].notna())  & ((df['IN']!=df['position'].shift()) | ((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1)))), df['IN'], df['position']) # bool_in
        bool_exit_ema = ((df['OUT_signal']*-1) == df['position'].ffill()) & (df['position'].ffill()!=0)
        df['position'] = np.where(bool_exit_ema, 0, df['position'])       # bool_ema_exit
        df['exit_ema'] = np.where(bool_exit_ema, 1, 0)                    # bool_ema_exit
        df['position'] = np.where((df['IN'].notna()) & ((df['IN']!=df['position'].shift()) | ((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1)))), df['IN'], df['position']) # bool_in
        df['position'] = df['position'].ffill()
        df['position_diff'] = df['position'].diff()
        df['exit_ema'] = np.where(df['position'].shift() == 0, 0, df['exit_ema'])
        if (OUT_mode == "positif") & ("target" not in Signal_OUT_type): # Then if trade_return < 0, annule le exit et refait la position
            # Annule les exit si pnl < 0
            df['trade_price'] = np.where((df['IN'].notna()) & ((df['IN']!=df['position'].shift()) | ((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1)))), df['Close'], np.nan)
            df['trade_price'] = df['trade_price'].ffill()
            df['trade_return'] = np.where(df['position'].shift()>0, ((df['Close'] - df['trade_price']) / df['trade_price'].shift())*100, 0)
            df['trade_return'] = np.where(df['position'].shift()<0, ((df['trade_price'].shift() - df['Close']) / df['trade_price'].shift())*100, df['trade_return'].fillna(0))
            seuil = float(param1) if Signal_OUT_type == "target_ema" else 0
            # seuil = 0
            df['OUT_signal'] = np.where((df['trade_return']<seuil) & (df['exit_ema']==1), 0, df['OUT_signal'])
            # Recreate position again 
            df['position'] = np.where((df['OUT'] == 2)| (df['OUT'] == 3), 0, np.nan)
            df['position'] = np.where((df['IN'].notna()) & ((df['IN']!=df['position'].shift()) | ((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1)))), df['IN'], df['position']) # bool_in
            bool_exit_ema = ((df['OUT_signal']*-1) == df['position'].ffill()) & (df['position'].ffill()!=0)
            df['position'] = np.where(bool_exit_ema, 0, df['position'])       # bool_ema_exit
            df['exit_ema'] = np.where(bool_exit_ema, 1, 0)                    # bool_ema_exit
            df['position'] = np.where((df['IN'].notna()) & ((df['IN']!=df['position'].shift()) | ((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1)))), df['IN'], df['position']) # bool_in
            df['position'] = df['position'].ffill()
            df['position_diff'] = df['position'].diff()
            df['exit_ema'] = np.where(df['position'].shift() == 0, 0, df['exit_ema'])

        df = get_nb_trades(df, SCRIPT_NAME)
        
        return df
    except Exception as e :
        print(e) 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        if is_running_server :  telegram_api(SCRIPT_NAME,message) ; sleep(300)
        sys.exit()
def get_final_return(df, SPREAD):
    try : 

        df = get_nb_trades(df, "SCRIPT_NAME")

        df['trade_return'] = np.where(df['position']>0, ((df['Close'] - df['trade_price']) / df['trade_price'])*100, 0)
        df['trade_return'] = np.where(df['position']<0, ((df['trade_price'] - df['Close']) / df['trade_price'])*100, df['trade_return'].fillna(0))

        df['Return'] = np.where((df['exit_price'].notna()) & (df['position'].shift()==1), (((df['exit_price']*(1-SPREAD)) - (df['trade_price'].shift()*(1+SPREAD))) / (df['trade_price'].shift()*(1+SPREAD)))*100, np.nan)
        df['Return'] = np.where((df['exit_price'].notna()) & (df['position'].shift()==-1), (((df['trade_price'].shift()*(1-SPREAD)) - (df['exit_price']*(1+SPREAD))) / (df['trade_price'].shift()*(1-SPREAD)))*100, df['Return'].fillna(0))

        df['trade_return'] = np.where(df['Return'] != 0, df['Return'], df['trade_return'])
      
        return df
    except Exception as e :
        print(e) 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        if is_running_server :  telegram_api_erreur(message) ; sleep(300)
        sys.exit()
def get_full_process(df, suffix, ticker, signals_combi, start, end, SCRIPT_NAME, SPREAD, moment, OUT_mode, overnight, earnings_dates_list, interval, inverse):
    try : 

        (Signal_IN_type, Signal_OUT_type, SL_type, param1, param2, param3) = signals_combi


        df['IN'] = get_signal_in(df, moment, overnight, Signal_IN_type)
        df['OUT_signal'] = get_signal_out(df, suffix, Signal_OUT_type, param1, param2, param3)

        # Exit position 10 min before end session (force IN / OUT)
        if (not 'BTC' in ticker) and (moment != 'daily') and (overnight == "N"):
            df = set_moment_boundaries(df, moment, overnight, interval) 


        #  Exit on bar juste before any earnings_dates
        if overnight != "N" : # Alors exit position before earnings anouncment
            df = set_earning_exit(df, earnings_dates_list)
 

        
        df = get_position(df, Signal_OUT_type, param1, SCRIPT_NAME, OUT_mode)
        

        
        new_SL = 0
        while True :

            if new_SL != 0 : print(f"new_SL : {new_SL}")

            df = get_nb_trades(df, SCRIPT_NAME)

            if "target" in Signal_OUT_type : 
                df = get_target_exit_col(df, Signal_OUT_type, param1, param2)

            # Stop Losses
            df = get_SL(df, SL_type, param3)


            # Keep only first SL per trades
            df['nb_trades_sh'] = df['nb_trades'].shift()
            df['cumsum'] = df.groupby('nb_trades_sh')['SL'].cumsum()
            df.loc[df['cumsum'] > 1, 'SL'] = 0
                        
            # Update position according to SL
            df['OUT'] = np.where(df['SL']!=0, 3, df['OUT'])
            df = get_position(df, Signal_OUT_type, param1, SCRIPT_NAME, OUT_mode)


            # Add exit price
            bool_l_to_s = (df["position"].shift() == 1) & (df["position"] == -1) & (df["OUT"] != 3) 
            bool_s_to_l = (df["position"].shift() == -1) & (df["position"] == 1) & (df["OUT"] != 3) 

            # elif moment == "daily" :
            if "target" in Signal_OUT_type :
                df['exit_price'] = np.where((bool_l_to_s | bool_s_to_l), df['Close'], np.nan)
                df['exit_price'] = np.where((df['exit_ema']==1), df['target'], df['exit_price'])
            else : 
                df['exit_price'] = np.where(((df['exit_ema']==1) | bool_l_to_s | bool_s_to_l), df['Close'], np.nan)
            df['exit_price'] = np.where((df['OUT']==2) & (df['position_diff']!=0) , df['Close'], df['exit_price'])
            df['exit_price'] = np.where((((df['SL']==1)) | (df['OUT']==3)) & (((df['Open']>df['SL_price'].shift()) & (df["position"].shift() == 1)) | ((df['Open']<df['SL_price'].shift()) & (df["position"].shift() == -1))), df['SL_price'].shift(), df['exit_price'])
            df['exit_price'] = np.where((((df['SL']==1)) | (df['OUT']==3)) & (((df['Open']<df['SL_price'].shift()) & (df["position"].shift() == 1)) | ((df['Open']>df['SL_price'].shift()) & (df["position"].shift() == -1))), df['Open'], df['exit_price'])

            if new_SL == (len(df[df['OUT']!=0]) + len(df[df['OUT_signal']!=0]) + len(df[df['exit_ema']!=0]) + df['nb_trades'].iloc[-1] + len(df[df['SL']!=0])) :  
                break
            new_SL = (len(df[df['OUT']!=0]) + len(df[df['OUT_signal']!=0]) + len(df[df['exit_ema']!=0]) + df['nb_trades'].iloc[-1] + len(df[df['SL']!=0]))

        # Retirer les OUT ==3 inutilisés
        df['OUT'] = np.where((df['OUT'] == 3) & (df['position'].shift() == 0), 0, df['OUT'])
        df['OUT_signal'] = np.where((df['OUT_signal'] == 3) & (df['position'].shift() == 0), np.nan, df['OUT_signal'])

        if inverse == "Y" : 
            df[['IN','OUT','OUT_signal']] = df[['IN','OUT','OUT_signal']] * -1
            df = get_position(df, Signal_OUT_type, param1, SCRIPT_NAME, OUT_mode)

        # Crop to exact study dates
        df = df[(df['Time']>=start) & (df['Time']<=end)]
        col_delete = ["Volume", "Open", "COND_1", "COND_2", "COND_3", "COND_4", "COND_5", "trade_count", "vwap", "max", "min", "SL_value"]
        col_delete = [col for col in col_delete if col in df.columns]  

        df = get_final_return(df, SPREAD)

        df['New_return'] = (df['Return'] / 100) +1
        df['Final_return'] = np.round((df['New_return'].cumprod() -1) *100 , decimals=2)
        df.drop(columns=['nb_trades_sh', 'cumsum', 'New_return'], inplace=True) 

        return df
    except Exception as e :
        print(e) 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        if is_running_server :  telegram_api(SCRIPT_NAME, message) ; sleep(300)
        sys.exit()

def get_return_col(df, suffix, ticker, signals_combi, start, end, SCRIPT_NAME, SPREAD, moment, OUT_mode, overnight, earnings_dates_list, interval, interval_min, forbidden_ts):
    try : 

        (Signal_IN_type, Signal_OUT_type, SL_type, param1, param2, param3) = signals_combi

        df['IN'] = get_signal_in(df, moment, overnight, Signal_IN_type)
        df['OUT_signal'] = get_signal_out(df, suffix, Signal_OUT_type, param1, param2, param3)

        # Exit position 10 min before end session (force IN / OUT)
        if (not 'BTC' in ticker) and (moment != 'daily') and (overnight == "N"):
            df = set_moment_boundaries(df, moment, overnight, interval) 

        # Remove forbidden ts
        if len(forbidden_ts) != 0 : 
            df = remove_forbidden_ts(df, forbidden_ts)

        #  Exit on bar juste before any earnings_dates
        if overnight != "N" : # Alors exit position before earnings anouncment
            df = set_earning_exit(df, earnings_dates_list)

        
        col_delete = ["Volume", "Open", "COND_1", "COND_2", "COND_3", "COND_4", "COND_5", "trade_count", "vwap", "max", "min", "SL_value"]
        col_delete = [col for col in col_delete if col in df.columns]  
        
        if interval == 1440 : df['Time'] = df['Time'].apply(lambda x: x.replace(hour=21, minute=55))


        df_min = pd.DataFrame({'Time': pd.date_range(start=start, end=end, freq=f'{interval_min}T')})
        if not 'BTC' in ticker: # Removing extended hours (can be removed later if needed)
            start_time = pd.to_datetime('15:30').time()
            end_time = pd.to_datetime('22:00').time()
            df_min = df_min[(df_min['Time'].dt.time >= start_time) & (df_min['Time'].dt.time < end_time)]
        df_min = df_min.merge(df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'IN', 'OUT', 'OUT_signal']], on='Time', how='left')
        df_min[['Open', 'High', 'Low', 'Close', 'Volume']] = df_min[['Open', 'High', 'Low', 'Close', 'Volume']].ffill()
        df = df_min.copy()

        # Remove forbidden ts
        if len(forbidden_ts) != 0 : 
            df = remove_forbidden_ts(df, forbidden_ts)


        # Crop to exact study dates
        df = df[(df['Time']>=start) & (df['Time']<=end)]
        # Removing extended hours (can be removed later if needed)
        if not 'BTC' in ticker : 
            start_time = pd.to_datetime('15:30').time()
            end_time = pd.to_datetime('22:00').time()
            df = df[(df['Time'].dt.time >= start_time) & (df['Time'].dt.time <= end_time)]

        df = get_position(df, Signal_OUT_type, param1, SCRIPT_NAME, OUT_mode)

        df['OUT'].iloc[-1] = 2
        new_SL = 0
        while True :

            df = get_nb_trades(df, SCRIPT_NAME)


            # target exit
            if "target" in Signal_OUT_type :
                df = get_target_exit_col(df, Signal_OUT_type, param1, param2)

            # Stop Losses
            df = get_SL(df, SL_type, param3)
               

            # Keep only first SL per trades
            df['nb_trades_sh'] = df['nb_trades'].shift()
            df['cumsum'] = df.groupby('nb_trades_sh')['SL'].cumsum()
            df.loc[df['cumsum'] > 1, 'SL'] = 0
            
           
            # Update position according to SL
            df['OUT'] = np.where(df['SL']!=0, 3, df['OUT'])
            df = get_position(df, Signal_OUT_type, param1, SCRIPT_NAME, OUT_mode)


            # Add exit price
            bool_l_to_s = (df["position"].shift() == 1) & (df["position"] == -1) & (df["OUT"] != 3) 
            bool_s_to_l = (df["position"].shift() == -1) & (df["position"] == 1) & (df["OUT"] != 3) 

            # elif moment == "daily" :
            if "target" in Signal_OUT_type :
                df['exit_price'] = np.where((bool_l_to_s | bool_s_to_l), df['Close'], np.nan)
                df['exit_price'] = np.where((df['exit_ema']==1), df['target'], df['exit_price'])
            else : 
                df['exit_price'] = np.where(((df['exit_ema']==1) | bool_l_to_s | bool_s_to_l), df['Close'], np.nan)
            df['exit_price'] = np.where((df['OUT']==2) & (df['position_diff']!=0) , df['Close'], df['exit_price'])
            df['exit_price'] = np.where((((df['SL']==1)) | (df['OUT']==3)) & (((df['Open']>df['SL_price'].shift()) & (df["position"].shift() == 1)) | ((df['Open']<df['SL_price'].shift()) & (df["position"].shift() == -1))), df['SL_price'].shift(), df['exit_price'])
            df['exit_price'] = np.where((((df['SL']==1)) | (df['OUT']==3)) & (((df['Open']<df['SL_price'].shift()) & (df["position"].shift() == 1)) | ((df['Open']>df['SL_price'].shift()) & (df["position"].shift() == -1))), df['Open'], df['exit_price'])

            if new_SL == (len(df[df['OUT']!=0]) + len(df[df['OUT_signal']!=0]) + len(df[df['exit_ema']!=0]) + df['nb_trades'].iloc[-1] + len(df[df['SL']!=0])) :  
                # if df[df['SL_price'].isna() & df['position']!= 0 ].empty : break
                break
            new_SL = (len(df[df['OUT']!=0]) + len(df[df['OUT_signal']!=0]) + len(df[df['exit_ema']!=0]) + df['nb_trades'].iloc[-1] + len(df[df['SL']!=0]))

        
        df = get_final_return(df, SPREAD)

        return df
    except Exception as e :
        print(e) 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        if is_running_server :  telegram_api(SCRIPT_NAME, message) ; sleep(300)
        sys.exit()
def get_signal_out(df, suffix, Signal_OUT_type, param1, param2, param3):
    df['OUT'] = 0
    df['OUT_signal'] = 0
    df['position'] = 0

    if "target" in Signal_OUT_type : return df['OUT_signal']
    elif Signal_OUT_type == "EMA_out": 
        df['inf'] = ta.ema(df[f'Close{suffix}'], length=int(param1))
        df['sup'] = ta.ema(df[f'Close{suffix}'], length=int(param2))
        df['OUT_signal'] = np.where(df['inf']>df['sup'], 1, -1)
        df['OUT_signal'] = df['OUT_signal'].diff() /2 
        df.drop(columns=['inf', 'sup'], inplace=True)
    elif Signal_OUT_type == "supertrend_OUT" : 
        length = int(param1) 
        multiplier = float(param2)  
        df[['1', '2', '3', '4']] = ta.supertrend(close=df[f'Close{suffix}'], high=df[f'High{suffix}'], low=df[f'Low{suffix}'], length=length, multiplier=multiplier)
        df['OUT_signal'] = df['2'].diff() /2 
        df.drop(columns=['1', '2', '3', '4'], inplace=True)
    elif Signal_OUT_type == "rsi_OUT" : 
        length = int(param1) 
        d = int(param2) 
        ref_name = f"RSI_{length}_{d}"
        lag_name = f"RSI_lag"
        df[ref_name] = ta.rsi(close=df[f'Close{suffix}'], length=length)
        df[lag_name] = ta.ema(close=df[ref_name], length=d)

        df['OUT_signal'] = np.where(df[ref_name]>df[lag_name], 1, -1)
        df['OUT_signal'] = df['OUT_signal'].diff() /2 
        df.drop(columns=[ref_name, lag_name], inplace=True)
    elif Signal_OUT_type == "MA_low_high_1": 
        df['MA_low'] = ta.ema(df[f'Low{suffix}'], length=int(param1))
        df['MA_high'] = ta.ema(df[f'High{suffix}'], length=int(param2))
        df['OUT_signal'] = np.where((df[f'Close{suffix}'].shift() > df["MA_high"].shift()) & (df[f'Close{suffix}'] < df["MA_high"]),  -1,0)               # Exit long 
        df['OUT_signal'] = np.where((df[f'Close{suffix}'].shift() < df["MA_low"].shift()) & (df[f'Close{suffix}'] > df["MA_low"]), 1, df['OUT_signal'])   # Exit short
    elif Signal_OUT_type == "2_bars": 
        df['OUT_signal'] = np.where((df[f'Close{suffix}'].shift() < df[f'Open{suffix}'].shift()) & (df[f'Close{suffix}'] < df[f'Open{suffix}']), -1, 0)# Exit long 
        df['OUT_signal'] = np.where((df[f'Close{suffix}'].shift() > df[f'Open{suffix}'].shift()) & (df[f'Close{suffix}'] > df[f'Open{suffix}']),  1, df['OUT_signal'])   # Exit short
    elif Signal_OUT_type == "1_bar":
        df['OUT_signal'] = np.where((df[f'Close{suffix}'] < df[f'Open{suffix}']), -1, 0)                    # Exit long 
        df['OUT_signal'] = np.where((df[f'Close{suffix}'] > df[f'Open{suffix}']) ,  1, df['OUT_signal'])    # Exit short
    elif Signal_OUT_type == "trailing":
        df['OUT_signal'] = 0
    elif Signal_OUT_type == "pos_change":
        df['OUT_signal'] = 0
    elif Signal_OUT_type == "target_ema": 
        # df['inf'] = ta.ema(df[f'Close{suffix}'], length=int(param1))
        df['sup'] = ta.ema(df[f'Close{suffix}'], length=int(param2))
        df['OUT_signal'] = np.where(df['Close']>df['sup'], 1, -1)
        df['OUT_signal'] = df['OUT_signal'].diff() /2 
        df.drop(columns=['sup'], inplace=True)
    else : 
        message = f"Error with Signal_OUT_type not recognized : {Signal_OUT_type}"
        print(message)
        if is_running_server :  telegram_api_erreur(message) ; sleep(300)
        sys.exit()
    return df['OUT_signal']
def get_signal_in(df, moment, overnight, Signal_IN_type):
    opening = pd.to_datetime('05:32').time()
    if moment == "full" and overnight == "N" : opening = pd.to_datetime('15:30').time()
    elif moment in ["debut"] : opening = pd.to_datetime('15:30').time()
    elif moment == "milieu" : opening = pd.to_datetime('16:30').time()
    elif moment == "fin" : opening = pd.to_datetime('21:30').time()
    bool_signal_in = ((df["COND_1"].diff()!=0) & (df["COND_1"]!=0)) | (df['Time'].dt.time == opening) # Only on change or opening session
    bool_cond_2 = (df["COND_2"] == df["COND_1"]) | (df["COND_2"] == 3)
    bool_cond_3 = (df["COND_3"] == df["COND_1"]) | (df["COND_3"] == 3)
    bool_cond_4 = (df["COND_4"] == df["COND_1"]) | (df["COND_4"] == 3)
    bool_cond_5 = (df["COND_5"] == df["COND_1"]) | (df["COND_5"] == 3)


    # Signal IN
    if Signal_IN_type == "force_pos" :
        df['IN'] = np.where(bool_signal_in & bool_cond_2 & bool_cond_3 & bool_cond_4 & bool_cond_5, df["COND_1"], np.nan)
    elif Signal_IN_type == "on_change" :
        df['IN'] = np.where(bool_signal_in & bool_cond_2 & bool_cond_3 & bool_cond_4 & bool_cond_5, df["COND_1"], np.nan)
        if moment == "daily" : df['IN'].iloc[0] = df["COND_1"].iloc[0]
    elif Signal_IN_type == "All" :
        df['IN'] = np.where(bool_cond_2 & bool_cond_3 & bool_cond_4 & bool_cond_5, df["COND_1"], np.nan)
    else :
        message = f"Error with Signal_IN_type not recognized : {Signal_IN_type}"
        print(message)
        if is_running_server :  telegram_api_erreur(message) ; sleep(300)
        sys.exit()
    return df['IN']
def set_moment_boundaries(df, moment, overnight, interval):
    if moment == "debut" : 
        time_to_exit = pd.to_datetime('15:30').time()
        time_to_exit_end = pd.to_datetime('16:30').time()
    elif moment == "milieu" : 
        time_to_exit = pd.to_datetime('16:30').time()
        time_to_exit_end = pd.to_datetime('21:30').time()
    elif moment == "fin" : 
        time_to_exit = pd.to_datetime('21:30').time()
        time_to_exit_end = pd.to_datetime('21:50').time()
    elif moment == "full" : 
        time_to_exit = pd.to_datetime('15:30').time()
        time_to_exit_end = pd.to_datetime('21:50').time()
    else : print(f"Moment {moment} not recognized") ; sys.exit()
    if moment == "full" and interval == 60 : 
        time_to_exit_end = pd.to_datetime('21:30').time()
    # print(f"Moment {moment}, 1 : {time_to_exit}, 2 : {time_to_exit_end}")
    df.loc[(df['Time'].dt.time < time_to_exit) | (df['Time'].dt.time >= time_to_exit_end), 'OUT'] = 2
    df.loc[(df['Time'].dt.time < time_to_exit) | (df['Time'].dt.time >= time_to_exit_end), 'IN'] = np.nan

    if overnight == "N":
        # Backup for non 20:50 available rows to exit on last row
        last_row_indices = df.groupby(df['Time'].dt.date)['Time'].transform('idxmax')
        df['OUT'] = np.where(df.index == last_row_indices, 2, df['OUT'])
        df['IN'] = np.where(df.index == last_row_indices, np.nan, df['IN'])

    
    return df
def set_earning_exit(df, earnings_dates_list):
    for ts in earnings_dates_list:
        # Find the last available time before the current timestamp
        last_index = df[df['Time'] < (ts+dt.timedelta(days=1))].index.max()
        if pd.notna(last_index):  # Check if a valid index is found
            df.at[last_index, 'OUT'] = 2
            df.at[last_index, 'IN'] = np.nan
    return df
def get_target_exit_col(df, Signal_OUT_type, param1, param2):
    try : 

        if float(param1) == 0 : print(f"param1 is equal to 0, please change this value") ; sys.exit()
        elif Signal_OUT_type == "target_fixed" :
            df['target'] = np.where((df['IN']==1) & (df['position']!=0) & ((df['IN']!=df['position'].shift()) | (((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1))))), df['Close'] * (1+float(param1)), np.nan)
            df['target'] = np.where((df['IN']==-1) & (df['position']!=0) & ((df['IN']!=df['position'].shift()) | (((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1))))), df['Close'] * (1-float(param1)), df['target'])
            df['target'] = df['target'].ffill().shift()

            df['OUT_signal'] = np.where((df['position'].shift()==1) & (df['High']>df['target']), -1, np.nan)
            df['OUT_signal'] = np.where((df['position'].shift()==-1) & (df['Low']<df['target']), 1, df['OUT_signal'])
        elif Signal_OUT_type == "target_BAR_reverse" :  # Checked on 20/08/2024
            df['min'] = df['Low'].rolling(window=int(param1)).min()
            df['max'] = df['High'].rolling(window=int(param1)).max()
            df['min_ecart_pourc'] = (df['Close']/df['min']-1) * (1-float(param2))
            df['max_ecart_pourc'] = (df['max']/df['Close']-1) * (1-float(param2))

            df['min'] = np.where(df['min_ecart_pourc'] < (0.1/100), (df['Close'] * 0.999), df['Close'] * (1-df['min_ecart_pourc']))
            df['max'] = np.where(df['max_ecart_pourc'] < (0.1/100), (df['Close'] * 1.001), df['Close'] * (1+df['max_ecart_pourc']))
            df['target'] = np.where((df['IN']==1) & (df['position']!=0) & ((df['IN']!=df['position'].shift()) | (((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1))))), df['max'], np.nan)
            df['target'] = np.where((df['IN']==-1) & (df['position']!=0) & ((df['IN']!=df['position'].shift()) | (((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1))))), df['min'], df['target'])
            df['target'] = df['target'].ffill().shift()

            df['OUT_signal'] = np.where((df['position'].shift()==1) & (df['High']>df['target']), -1, np.nan)
            df['OUT_signal'] = np.where((df['position'].shift()==-1) & (df['Low']<df['target']), 1, df['OUT_signal'])
            
            # Keep only first SL per trades
            df['nb_trades_sh'] = df['nb_trades'].shift()
            df['nb_target'] = df.groupby('nb_trades_sh')['OUT_signal'].cumsum()
            df.loc[df['nb_target'] > 1, 'OUT_signal'] = np.nan
            # print(df[df['Time']>= dt.datetime(2024,8,5,19,35)].drop(columns = drop_col).head(10))
        elif Signal_OUT_type == "target_BODY_reverse" :   # Check on 20/08/2024
            df['min'] = df['Close'].rolling(window=int(param1)).min() 
            df['max'] = df['Open'].rolling(window=int(param1)).max() 
            df['min_ecart_pourc'] = (df['Close']/df['min']-1) * (1-float(param2))
            df['max_ecart_pourc'] = (df['max']/df['Close']-1) * (1-float(param2))

            df['min'] = np.where(df['min_ecart_pourc'] < (0.1/100), (df['Close'] * 0.999), df['Close'] * (1-df['min_ecart_pourc']))
            df['max'] = np.where(df['max_ecart_pourc'] < (0.1/100), (df['Close'] * 1.001), df['Close'] * (1+df['max_ecart_pourc']))

            df['target'] = np.where((df['IN']==1) & (df['position']!=0) & ((df['IN']!=df['position'].shift()) | (((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1))))), df['max'], np.nan)
            df['target'] = np.where((df['IN']==-1) & (df['position']!=0) & ((df['IN']!=df['position'].shift()) | (((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1))))), df['min'], df['target'])
            df['target'] = df['target'].ffill().shift()

            df['OUT_signal'] = np.where((df['position'].shift()==1) & (df['High']>df['target']), -1, np.nan)
            df['OUT_signal'] = np.where((df['position'].shift()==-1) & (df['Low']<df['target']), 1, df['OUT_signal'])
            
            # Keep only first SL per trades
            df['nb_trades_sh'] = df['nb_trades'].shift()
            df['nb_target'] = df.groupby('nb_trades_sh')['OUT_signal'].cumsum()
            df.loc[df['nb_target'] > 1, 'OUT_signal'] = np.nan
        elif Signal_OUT_type == "target_BAR" :          # Check on 20/08/2024
            df['min'] = df['Low'].rolling(window=int(param1)).min()
            df['max'] = df['High'].rolling(window=int(param1)).max()
            df['min_ecart_pourc'] = (df['Close']/df['min']-1) * (1-float(param2))
            df['max_ecart_pourc'] = (df['max']/df['Close']-1) * (1-float(param2))

            df['min_ecart_pourc'] = np.where(df['min_ecart_pourc'] < (0.1/100), (0.1/100), df['min_ecart_pourc'])
            df['max_ecart_pourc'] = np.where(df['max_ecart_pourc'] < (0.1/100), (0.1/100), df['max_ecart_pourc'])

            df['target'] = np.where((df['IN']==1) & (df['position']!=0) & (df['nb_trades'].diff()!=0), df['Close'] * (1+df['min_ecart_pourc']), np.nan)
            df['target'] = np.where((df['IN']==-1) & (df['position']!=0) & (df['nb_trades'].diff()!=0), df['Close'] * (1-df['max_ecart_pourc']), df['target'])

            df['target'] = df['target'].ffill().shift()
            df['OUT_signal'] = np.where((df['position'].shift()==1) & (df['High']>df['target']), -1, np.nan)
            df['OUT_signal'] = np.where((df['position'].shift()==-1) & (df['Low']<df['target']), 1, df['OUT_signal'])
            

            # Keep only target out signal
            df['nb_trades_sh'] = df['nb_trades'].shift()
            df['OUT_signal_abs'] = abs(df['OUT_signal'])
            df['nb_target'] = df.groupby('nb_trades_sh')['OUT_signal_abs'].cumsum()
            df.loc[df['nb_target'] > 1, 'OUT_signal'] = np.nan
        elif Signal_OUT_type == "target_BODY" :         # Check on 20/08/2024
            df['min'] = df['Close'].rolling(window=int(param1)).min()
            df['max'] = df['Open'].rolling(window=int(param1)).max()
            df['min_ecart_pourc'] = (df['Close']/df['min']-1) * (1-float(param2))
            df['max_ecart_pourc'] = (df['max']/df['Close']-1) * (1-float(param2))

            df['min_ecart_pourc'] = np.where(df['min_ecart_pourc'] < (0.1/100), (0.1/100), df['min_ecart_pourc'])
            df['max_ecart_pourc'] = np.where(df['max_ecart_pourc'] < (0.1/100), (0.1/100), df['max_ecart_pourc'])

            df['target'] = np.where((df['IN']==1) & (df['position']!=0) & ((df['IN']!=df['position'].shift()) | (((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1))))), df['Close'] * (1+df['min_ecart_pourc']), np.nan)
            df['target'] = np.where((df['IN']==-1) & (df['position']!=0) & ((df['IN']!=df['position'].shift()) | (((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1))))), df['Close'] * (1-df['max_ecart_pourc']), df['target'])
            df['target'] = df['target'].ffill().shift()
            df['OUT_signal'] = np.where((df['position'].shift()==1) & (df['High']>df['target']), -1, np.nan)
            df['OUT_signal'] = np.where((df['position'].shift()==-1) & (df['Low']<df['target']), 1, df['OUT_signal'])
            

            # Keep only target out signal
            df['nb_trades_sh'] = df['nb_trades'].shift()
            df['OUT_signal_abs'] = abs(df['OUT_signal'])
            df['nb_target'] = df.groupby('nb_trades_sh')['OUT_signal_abs'].cumsum()
            df.loc[df['nb_target'] > 1, 'OUT_signal'] = np.nan

        df = df.drop(columns=[col for col in ['min', 'max', 'nb_target', 'min_ecart_pourc', 'max_ecart_pourc', 'OUT_signal_abs'] if col in df.columns])

        return df
    except Exception as e :
        print(e) 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(exc_type) + ". Line : " + str(exc_tb.tb_lineno)
        print(message, "\n")
        if is_running_server :  telegram_api("SCRIPT_NAME", message) ; sleep(300)
        sys.exit()
def get_SL(df, SL_type, param3):

    if SL_type == "SL_fixed" :                  # Checked on 20/08/2024
        df['SL_value'] = float(param3) 
        df['SL_price'] = np.where(df['position']>0, df['trade_price'] * (1-(df['SL_value']/100)), np.nan)
        df['SL_price'] = np.where(df['position']<0, df['trade_price'] * (1+(df['SL_value']/100)), df['SL_price'])

        (df['IN'].notna()) & ((df['IN']!=df['position'].shift()) | ((df['OUT']==3) | (df['OUT_signal']==(df['IN']*-1))))

        # Normal cases
        df['SL'] = np.where((df['position'].shift() > 0) & (df['Low']<= df['SL_price'].shift()), 1, 0)
        df['SL'] = np.where((df['position'].shift() < 0) & (df['High']>= df['SL_price'].shift()), 1, df['SL'])
    elif SL_type == "SL_fixed_window_BAR" :     # Checked on 20/08/2024

        df['min'] = df['Low'].rolling(window=int(float(param3))).min() 
        df['max'] = df['High'].rolling(window=int(float(param3))).max() 

        df['SL_price'] = np.where((df['position_diff']!=0) & (df['IN']==1), np.minimum((df['min']-0.03), df['trade_price'] * (0.999)), np.nan)
        df['SL_price'] = np.where((df['position_diff']!=0) & (df['IN']==-1), np.maximum((df['max']+0.03), df['trade_price'] * (1.001)), df['SL_price'])
        if (pd.isna(df['SL_price'].iloc[0])) and (df['IN'].iloc[0]==1) : df['SL_price'].iloc[0] = df['Close'].iloc[0] * 0.99
        elif (pd.isna(df['SL_price'].iloc[0])) and (df['IN'].iloc[0]==-1) : df['SL_price'].iloc[0] = df['Close'].iloc[0] * 1.01
        df['SL_price'] = df['SL_price'].ffill()

        # Normal cases
        df['SL'] = np.where((df['position'].shift() > 0) & (df['Low']<= df['SL_price'].shift()), 1, 0)
        df['SL'] = np.where((df['position'].shift() < 0) & (df['High']>= df['SL_price'].shift()), 1, df['SL'])
        # print(df)
    elif SL_type == "SL_fixed_window_BODY" :    # Checked on 20/08/2024

        df['oppen_r'] = df['Open'].rolling(window=int(float(param3))).min() 
        # df['close_r'] = df['Close'].rolling(window=int(float(param3))).max() 

        df['SL_price'] = np.where((df['position_diff']!=0) & (df['IN']==1), np.minimum((df['oppen_r']-0.04), df['trade_price'] * (0.999)), np.nan)
        df['SL_price'] = np.where((df['position_diff']!=0) & (df['IN']==-1), np.maximum((df['oppen_r']+0.04), df['trade_price'] * (1.001)), df['SL_price'])
        if (pd.isna(df['SL_price'].iloc[0])) and (df['IN'].iloc[0]==1) : df['SL_price'].iloc[0] = df['Close'].iloc[0] * 0.99
        elif (pd.isna(df['SL_price'].iloc[0])) and (df['IN'].iloc[0]==-1) : df['SL_price'].iloc[0] = df['Close'].iloc[0] * 1.01
        df['SL_price'] = df['SL_price'].ffill()

        # Normal cases
        df['SL'] = np.where((df['position'].shift() > 0) & (df['Low']<= df['SL_price'].shift()), 1, 0)
        df['SL'] = np.where((df['position'].shift() < 0) & (df['High']>= df['SL_price'].shift()), 1, df['SL'])
    elif SL_type == "1_bar" :
        df['SL'] = np.where((df['position'].shift() > 0) & (df['Close']< df['Open'].shift()), 1, 0)
        df['SL'] = np.where((df['position'].shift() < 0) & (df['Close']> df['Open'].shift()), 1, df['SL'])

        df['SL_price1'] = np.where(df['SL']==1, df['Close'], np.nan)
        df['SL_price2'] = df['SL_price1'].shift(-1)
        df['SL_price'] = np.where(df['SL_price1'].notna(), df['SL_price1'], df['SL_price2'])
    else : 
        print(f"SL_type not recognized : {SL_type}")
        sys.exit()

    df = df.drop(columns=[col for col in ['min', 'max', 'nb_target', 'min_ecart_pourc', 'max_ecart_pourc', 'OUT_signal_abs', 'oppen_r', 'SL_price1', 'SL_price2'] if col in df.columns])
    return df
def remove_forbidden_ts(df, forbidden_ts):
    df.loc[df['Time'].isin(forbidden_ts), 'OUT'] = 2
    df.loc[df['Time'].isin(forbidden_ts), 'IN'] = np.nan
    return df


def update_earnings(from_date=dt.datetime(1970,1,1), to_date=(dt.datetime.now()+dt.timedelta(days=360)), asset_list = []):

    # Read df_earnings on github
    df_earnings_filepath = "1_Backtest/earnings.csv"
    full_url_earnings = f"https://raw.githubusercontent.com/constantin89P/Referentiel/main/{df_earnings_filepath}"
    df_earnings = get_df_github(full_url_earnings, ";", "PROD")
    
    if not df_earnings.empty : 
        df_earnings['Date'] = pd.to_datetime(df_earnings['Date'], format='%Y-%m-%d')
        already_symbols = list(set(df_earnings['Ticker'].to_list()))
    else : already_symbols = []
    symbol_list = [ticker for ticker in asset_list if ticker not in already_symbols]


    # Add more assets in scope if provided
    # symbol_list = list(set(asset_list + already_symbols))
    print(f"symbol_list to update: {symbol_list}") 
    # print(f"Please confirme scope to updtae, remember 25 ticker per day : {symbol_list}")

    # TO DO : revoir cette ligne
    if len(symbol_list) == 0 : return pd.DataFrame()
 
    updated_earnings = pd.DataFrame()
    for symbol in symbol_list : 
        # symbol = 'AAPL'# symbol of stock/company
        print(f"Fetching symbol : {symbol}")

        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={ALPHA_KEY}'

        response = requests.get(url)
        response_json = response.json()

        any_update = False
        if 'Information' in response_json and 'Our standard API rate limit' in response_json['Information']: print(response_json) ; break
        elif response.status_code == 200:

            response = response.json()

            dates_list = []
            try : 
                for announcement in response['quarterlyEarnings']:
                    dates_list.append(announcement['reportedDate'])
                    any_update = True
            except Exception as e:
                print(response)
                print(e)
            
            # TO DO : convert dates_list dates into pd.datetime
            df_int = pd.DataFrame({'Date':dates_list})
            df_int['Ticker'] = symbol
            updated_earnings = pd.concat([updated_earnings, df_int])
        else:
            print(f"Error occurred while fetching earnings announcements. Response status : {response.status_code}") ; sys.exit()

    df_earnings = pd.concat([df_earnings, updated_earnings])
    df_earnings['Date'] = pd.to_datetime(df_earnings['Date']) #, format='%d-%m-%Y'
    df_earnings = df_earnings.drop_duplicates(subset=['Ticker', 'Date'])
    print(df_earnings)

    if any_update : 
        print("Try to save github")
        push_df_github(df_earnings, ";", "Referentiel", df_earnings_filepath, f"Save updated earnings", "Script", "PROD")

    # Sonner une alerte si pour un asset, il y a deux dates à moins de 30j
    df_sorted = df_earnings.sort_values(by=['Ticker', 'Date'])
    df_sorted['Date_Diff'] = df_sorted.groupby('Ticker')['Date'].diff().dt.days
    df_sorted = df_sorted[df_sorted['Date_Diff'] < 30]
    if not df_sorted.empty :
        print(df_sorted)
        ticker_list = df_sorted['Ticker'].to_list()
        telegram_api_erreur("Script", f"Error, double earning dates for {ticker_list}")


    print(df_earnings)


    # TO DO : print/telegram a list of [all the tickers] without a coming announcement date ! 

    return df_earnings

def get_forward_test_ptf(df_ptf, from_date, to_date):

    df_ptf = pd.read_csv(r"D:\Documents\Python\Bot\Avril_2024\2_Confirm_paper_vs_backtest\ptf_test_hybride.csv", sep=";")
    df_ptf['Group_weight'] = df_ptf['Group_weight'].str.rstrip('%').astype('float') / 100


    df_earnings_filepath = "1_Backtest/earnings.csv"
    full_url_earnings = f"https://raw.githubusercontent.com/constantin89P/Referentiel/main/{df_earnings_filepath}"
    df_earnings = get_df_github(full_url_earnings, ";", "PROD")
    df_earnings['Date'] = pd.to_datetime(df_earnings['Date'], format='%Y-%m-%d')


    # Démarrer par le plus petit interval du ptf 
    interval_min = df_ptf["interval"].min()
    groups_to_do = df_ptf["Group"].drop_duplicates().to_list()

    if interval_min == 1440 : print("Interval min daily not parameted correctly I think, must check ") ; sys.exit()

    df_timeframe = pd.DataFrame({'Time': pd.date_range(start=from_date, end=to_date, freq=f'{interval_min}T')})

    # Removing extended hours (can be removed later if needed)
    if not 'BTC' in df_ptf["ticker"].iloc[0] : 
        start_time = pd.to_datetime('15:30').time()
        end_time = pd.to_datetime('22:00').time()
        df_timeframe = df_timeframe[(df_timeframe['Time'].dt.time >= start_time) & (df_timeframe['Time'].dt.time < end_time)]


    is_group_priority = False
    for group in groups_to_do :

        df_group = df_ptf[df_ptf['Group'] == group]
        df_group = df_group.sort_values("Priority", ascending=True)

        # Check if it is a priority or equal group 
        if any(df_group['Priority']==2) : is_group_priority = True
        else : is_group_priority = False


        print(f"\nGroup {group}, priorité: {is_group_priority}\n{df_group}")
        
        forbidden_ts = []
        for x in range(len(df_group)):
            ticker = df_group['ticker'].iloc[x]
            interval = df_group['interval'].iloc[x]
            combinaison = ast.literal_eval(df_group['combinaison'].iloc[x])
            signals_combi = ast.literal_eval(df_group['signals_combi'].iloc[x])
            IN_datatype = df_group['IN_datatype'].iloc[x]
            OUT_datatype = df_group['OUT_datatype'].iloc[x]
            moment = df_group['moment'].iloc[x]
            OUT_mode = df_group['OUT_mode'].iloc[x]
            overnight = df_group['overnight'].iloc[x]

            earnings_dates_list = df_earnings[df_earnings['Ticker']== ticker]['Date'].to_list()
            SPREAD = 0.02/100


            (SIGNAL_in, C_in_2, C_in_3, C_in_4, C_in_5,  
                signal_in_param1, signal_in_param2, signal_in_param3, 
                cond_2_param1, cond_2_param2, cond_2_param3,
                cond_3_param1, cond_3_param2, cond_3_param3,
                cond_4_param1, cond_4_param2, cond_4_param3,
                cond_5_param1, cond_5_param2, cond_5_param3) = combinaison

            # df_prices = get_prices_alpaca(ticker, interval, from_date, to_date, "Referentiel", "Function test", "PROD")
            df = get_historical_data_alpaca(ticker, interval, from_date, to_date)

        
            suffix = ""
            if ((IN_datatype == "HA") or (OUT_datatype == "HA"))and ("Open_HA" not in df.columns): 
                df[['Open_HA', 'High_HA', 'Low_HA', 'Close_HA']] = add_HA_OHLC(df)

            suffix = "_HA" if (IN_datatype == "HA")  else ""
            df["COND_1"] = get_signal(suffix, df, ticker, SIGNAL_in, signal_in_param1, signal_in_param2, signal_in_param3)
            df["COND_2"] = get_signal(suffix, df, ticker, C_in_2, cond_2_param1, cond_2_param2, cond_2_param3)
            df["COND_3"] = get_signal(suffix, df, ticker, C_in_3, cond_3_param1, cond_3_param2, cond_3_param3)
            df["COND_4"] = get_signal(suffix, df, ticker, C_in_4, cond_4_param1, cond_4_param2, cond_4_param3)
            df["COND_5"] = get_signal(suffix, df, ticker, C_in_5, cond_5_param1, cond_5_param2, cond_5_param3)
            df = df.dropna(subset=['COND_1', 'COND_2', 'COND_3', 'COND_4', 'COND_5'])
            # df["COND_1"] = df["COND_1"] * -1


            suffix = "_HA" if (OUT_datatype == "HA")  else ""
            df = get_return_col(df.copy(), suffix, ticker, signals_combi, from_date, to_date, "SCRIPT_NAME", SPREAD, moment, OUT_mode, overnight, earnings_dates_list, interval, interval_min, forbidden_ts)
            
            # print(df)
            # print(df[df['position']!=0])

            # Mulitply by the weight
            if is_group_priority : 
                df[['trade_return', 'Return']] = df[['trade_return', 'Return']] * (df_group["Group_weight"].iloc[x])
            else : 
                df[['trade_return', 'Return']] = df[['trade_return', 'Return']] * (df_group["Group_weight"].iloc[x] / len(df_group))
            df = df.rename(columns={'trade_return': f'TR_{group}_{x}', 'Return': f'Ret_{group}_{x}'})

            df_timeframe = df_timeframe.merge(df[['Time', f'Ret_{group}_{x}', f'TR_{group}_{x}']], on='Time', how='left') # 

            if is_group_priority : # Then update forbidden_ts list
                forbidden_ts = list(set(forbidden_ts + df.loc[df['position'] != 0, 'Time'].to_list()))


    # STRATEGY LEVEL
    df_ptf[['RETURN', 'negativ', 'positiv', 'NB_negativ', 'NB_positiv', 'NB_trades', 'mean_return', 'win_nb', 'win_return', 'nb_SL', 'SL_pourc']] = 0
    group = 1
    group_line = 0
    for line in range(len(df_ptf)):
        current_group = df_ptf['Group'].iloc[line]
        if current_group != group : group_line = 0
        group = current_group

        R_col = f'Ret_{group}_{group_line}'
        TR_col = f'TR_{group}_{group_line}'
        

        # Statistics
        df_timeframe['New_return'] = (df_timeframe[R_col] / 100) +1
        df_timeframe['Final_return'] = np.round((df_timeframe['New_return'].cumprod() -1) *100 , decimals=2)
        RETURN = df_timeframe['Final_return'].iloc[-1] 

        negativ = np.round(df_timeframe[df_timeframe[R_col] < 0][R_col].sum(), decimals=2)
        if negativ == 0 : equilibre = RETURN
        elif abs(negativ) < 1  : equilibre = RETURN
        else : equilibre = np.round(RETURN / abs(negativ), decimals=2)
        
        positiv = np.round(df_timeframe[df_timeframe[R_col] > 0][R_col].sum(), decimals=2)
        NB_negativ = len(df_timeframe[df_timeframe[R_col] < 0])
        NB_positiv = len(df_timeframe[df_timeframe[R_col] > 0])
        nb_trade = len(df_timeframe[df_timeframe[R_col]!=0])
        mean_return = np.round(df_timeframe[df_timeframe[R_col] != 0][R_col].mean(), decimals=2)

        win_nb = np.round((NB_positiv / (NB_negativ+NB_positiv))*100,2)
        win_return = np.round((positiv / (abs(negativ)+positiv))*100,2)
        SL_value = ast.literal_eval(df_ptf['signals_combi'].iloc[line])
        SL_value = float(SL_value[-1]) * df_ptf['Group_weight'].iloc[line]
        nb_SL = len(df_timeframe[df_timeframe[R_col] < -SL_value]) 
        SL_pourc = np.round((nb_SL / nb_trade)*100, 2)

        # ++ Holding Period 
        
        
        # MDD : TO REVIEW
        # df_timeframe['cummax'] = df_timeframe['Final_return'].cummax() 
        # df_timeframe['cummax'] = df_timeframe['cummax'].shift()
        # df_timeframe['New_return'] = (df_timeframe[R_col] / 100) +1
        # df_timeframe['worst_perte'] = np.round((df_timeframe.groupby(['cummax'])['New_return'].cumprod() - 1) * 100, decimals=2)
        # MDD = np.round(df_timeframe['worst_perte'].min(), decimals=2)
        # cumax_MDD = df_timeframe[df_timeframe['worst_perte']==df_timeframe['worst_perte'].min()]['cummax'].iloc[-1]
        # cumax_time = df_timeframe[df_timeframe['worst_perte']==df_timeframe['worst_perte'].min()]['Time'].iloc[-1]
        # df_MDD = df_timeframe[(df_timeframe['cummax']==cumax_MDD) & (df_timeframe['Time']<=cumax_time) & (df_timeframe['OUT']!=2)]
        # MDD_period = (df_MDD['Time'].iloc[-1] - df_MDD['Time'].iloc[0]).days


        df_ptf['RETURN'].iloc[line] = RETURN
        df_ptf['negativ'].iloc[line] = negativ
        df_ptf['positiv'].iloc[line] = positiv
        df_ptf['NB_negativ'].iloc[line] = NB_negativ
        df_ptf['NB_positiv'].iloc[line] = NB_positiv
        df_ptf['NB_trades'].iloc[line] = nb_trade
        df_ptf['mean_return'].iloc[line] = mean_return
        df_ptf['win_nb'].iloc[line] = win_nb
        df_ptf['win_return'].iloc[line] = win_return
        df_ptf['nb_SL'].iloc[line] = nb_SL
        df_ptf['SL_pourc'].iloc[line] = SL_pourc

        group_line +=1 


    # PORTFOLIO LEVEL
    # TO DO : écrire les stats
    # PORTFOLIO level (start data * weight): Final_Return, MDD, positiv, negativ, avr_daily_returns, SL_pourc, NB_trades, NB_negativ, NB_positiv, mean_holding_perc, win_nb, win_return
    R_col = 'ptf_return'
    df_timeframe[R_col] = df_timeframe.filter(like='Ret').sum(axis=1)
    df_timeframe['New_return'] = (df_timeframe[R_col] / 100) +1
    df_timeframe['Final_return'] = np.round((df_timeframe['New_return'].cumprod() -1) *100 , decimals=2)
    RETURN = df_timeframe['Final_return'].iloc[-1] 


    negativ = np.round(df_timeframe[df_timeframe[R_col] < 0][R_col].sum(), decimals=2)
    if negativ == 0 : equilibre = RETURN
    elif abs(negativ) < 1  : equilibre = RETURN
    else : equilibre = np.round(RETURN / abs(negativ), decimals=2)
    
    positiv = np.round(df_timeframe[df_timeframe[R_col] > 0][R_col].sum(), decimals=2)
    NB_negativ = len(df_timeframe[df_timeframe[R_col] < 0])
    NB_positiv = len(df_timeframe[df_timeframe[R_col] > 0])
    nb_trade = len(df_timeframe[df_timeframe[R_col]!=0])
    mean_return = np.round(df_timeframe[df_timeframe[R_col] != 0][R_col].mean(), decimals=2)

    win_nb = np.round((NB_positiv / (NB_negativ+NB_positiv))*100,2)
    win_return = np.round((positiv / (abs(negativ)+positiv))*100,2)

    df_int = pd.DataFrame({'RETURN':[RETURN], 'positiv':[positiv], 'negativ':[negativ], 'mean_return':[mean_return],  
                        'NB_trades':[nb_trade], 'NB_negativ':[NB_negativ], 'NB_positiv':[NB_positiv],
                        'win_nb':[win_nb], 'win_return':[win_return]}) #'MDD':[MDD], 'MDD_period':[MDD_period], 


    df_ptf = pd.concat([df_ptf, df_int])
     
    from_d = from_date.strftime("%Y-%m-%d")
    to_d = to_date.strftime("%Y-%m-%d")
    result_filename = f"D:\\Documents\\Python\\Bot\\Avril_2024\\2_Confirm_paper_vs_backtest\\ptf_forward_{from_d}_{to_d}.csv"
    save_to_csv(df_ptf, result_filename)


    # PLOT portfolio
    # ax = plt.gca()
    # df_timeframe.plot(kind='line',x='Time',y='y',color='red',lw=0.5,ax=ax)
    # df_timeframe.plot(kind='line',x='Time',y='Final_return',color='darkorange',lw=0.5,ax=ax)
    # df_timeframe.plot(kind='line',x='Time',y='rest',color='c',lw=1,ax=ax,secondary_y=True)
    # df_timeframe.plot(kind='line',x='Time',y='short',color='red',lw=1,ax=ax,secondary_y=True)
    # df_timeframe.plot(kind='line',x='Time',y='long',color='green',lw=1,ax=ax,secondary_y=True)
    # plt.show()

    return df_ptf
