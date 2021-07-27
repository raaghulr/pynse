import datetime as dt
import enum
import io
import logging
import os
import pickle
import shutil
import time
import urllib.parse
import zipfile
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from glob import glob
logger = logging.getLogger(__name__)


class IndexSymbol(enum.Enum):
    All = 'ALL'
    FnO = 'FNO'
    Nifty50 = 'NIFTY 50'
    NiftyNext50 = 'NIFTY NEXT 50'
    Nifty100 = 'NIFTY 100'
    Nifty200 = 'NIFTY 200'
    Nifty500 = 'NIFTY 500'
    NiftyMidcap50 = 'NIFTY MIDCAP 50'
    NiftyMidcap100 = 'NIFTY MIDCAP 100'
    NiftySmlcap100 = 'NIFTY SMLCAP 100'
    NiftyMidcap150 = 'NIFTY MIDCAP 150'
    NiftySmlcap50 = 'NIFTY SMLCAP 50'
    NiftySmlcap250 = 'NIFTY SMLCAP 250'
    NiftyMidsml400 = 'NIFTY MIDSML 400'
    NiftyBank = 'NIFTY BANK'
    NiftyAuto = 'NIFTY AUTO'
    NiftyFinService = 'NIFTY FIN SERVICE'
    NiftyFmcg = 'NIFTY FMCG'
    NiftyIt = 'NIFTY IT'
    NiftyMedia = 'NIFTY MEDIA'
    NiftyMetal = 'NIFTY METAL'
    NiftyPharma = 'NIFTY PHARMA'
    NiftyPsuBank = 'NIFTY PSU BANK'
    NiftyPvtBank = 'NIFTY PVT BANK'
    NiftyRealty = 'NIFTY REALTY'
    Nifty50Value20 = 'NIFTY50 VALUE 20'
    NiftyAlpha50 = 'NIFTY ALPHA 50'
    Nifty50EqlWgt = 'NIFTY50 EQL WGT'
    Nifty100EqlWgt = 'NIFTY100 EQL WGT'
    Nifty100Lowvol30 = 'NIFTY100 LOWVOL30'
    Nifty200Qualty30 = 'NIFTY200 QUALTY30'
    NiftyCommodities = 'NIFTY COMMODITIES'
    NiftyConsumption = 'NIFTY CONSUMPTION'
    NiftyEnergy = 'NIFTY ENERGY'
    NiftyInfra = 'NIFTY INFRA'
    NiftyMnc = 'NIFTY MNC'
    NiftyPse = 'NIFTY PSE'
    NiftyServSector = 'NIFTY SERV SECTOR'


class Format(enum.Enum):
    pkl = 'pkl'
    csv = 'csv'


class Segment(enum.Enum):
    EQ = 'EQ'
    FUT = 'FUT'
    OPT = 'OPT'


class OptionType(enum.Enum):
    CE = 'Call'
    PE = 'Put'


class Nse:
    """
    pynse is a library to extract realtime and historical data from NSE website

    Examples
    --------

    >>> from pynse import *
    >>> nse = Nse()
    >>> nse.market_status()

    """

    @staticmethod
    def __read_object(filename, obj_format):
        if obj_format == Format.pkl:
            with open(filename, 'rb')as f:
                obj = pickle.load(f)
            return obj
        elif obj_format == Format.csv:
            with open(filename, 'r')as f:
                obj = f.read()
            return obj
        else:
            raise FileNotFoundError(f'{filename} not found')

    @staticmethod
    def __save_object(obj, filename, obj_format):
        if obj_format == Format.pkl:
            with open(filename, 'wb')as f:
                pickle.dump(obj, f)
        elif obj_format == Format.csv:
            with open(filename, 'w')as f:
                f.write(obj)
        logger.debug(f'saved {filename}')

    def __init__(self):

        self.expiry_list = []
        self.strike_list = []
        self.max_retries = 5
        self.timeout = 10

        self.__urls = dict()
        # home directory for user
        home = os.path.expanduser("~")

        path = os.path.join(home, '.pynse/')
        # root dir for data
        self.dir = {'data_root': path}

        # update dirs
        self.dir.update({d: f'{self.dir["data_root"]}{d}/' for d in
                         ['bhavcopy_eq', 'bhavcopy_fno', 'option_chain', 'symbol_list', 'pre_open', 'hist',
                          'fii_dii', 'temp']})

        # symbol file names for IndexSymbols
        self.__symbol_files = {
            i.name: f"{self.dir['symbol_list']}{i.name}.pkl" for i in IndexSymbol}

        # __file__ store the location of this file
        self.__zero_files = {
            i.name: f"{f'{os.path.split(__file__)[0]}/symbol_list/'}{i.name}.pkl" for i in IndexSymbol}

        # create dir and copy symbol files
        self.__startup()

        # store symbol list for indexes here
        self.symbols = {i.name: self.__read_object(
            self.__symbol_files[i.name], Format.pkl) for i in IndexSymbol}

        logger.info(
            f'pyNse cache size: {self.__data_size()}.\nYou may want to run `nse.clear_data()` if running low on disk space.')

    def __temp(self, new=False):
        temp_file = f"{self.dir['temp']}temp"
        if new or not os.path.exists(temp_file):
            session = requests.Session()
            session.get('https://www.nseindia.com', headers={'Accept': '*/*',
                                                             'Connection': 'keep-alive',
                                                             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626 Safari/537.36',
                                                             'Accept-Encoding': 'gzip, deflate, br',
                                                             'Accept-Language': 'en-US;q=0.5,en;q=0.3',
                                                             'DNT': '1'})
            with open(temp_file, 'wb')as f:
                pickle.dump(session, f)
        else:
            with open(temp_file, 'rb')as f:
                session = pickle.load(f)
        return session

    def __get_resp(self, url, timeout=0):
        headers = {'Accept': '*/*',
                   'Connection': 'keep-alive',
                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626 Safari/537.36',
                   'Accept-Encoding': 'gzip, deflate, br',
                   'Accept-Language': 'en-US;q=0.5,en;q=0.3',
                   'DNT': '1',
                   'referer':'https://www.nseindia.com'}
        # use global timeout if not specified
        timeout = self.timeout if timeout == 0 else timeout

        session = self.__temp(new=True)

        try:
            response = session.get(url, headers=headers, timeout=timeout)

        except Exception as e:
            logger.error(e)

        else:
            with open(f"{self.dir['temp']}session", 'wb')as f:
                pickle.dump(session, f)
            return response

    def __startup(self):
        """
        creates required dirs to store data
        :return:
        """

        self.__urls = self.__read_object(
            f'{os.path.split(__file__)[0]}/config/config', Format.pkl)
        # create folder if doesn't exists
        for _, path in self.dir.items():
            if path != '':
                if not os.path.exists(path):
                    os.mkdir(path)

        # check if first run
        if not os.path.exists(self.__symbol_files['All']):
            logger.debug(
                'First run.\nCreating folders and downloading symbol files')
            for i in IndexSymbol:
                # if file is not found
                if not os.path.exists(self.__symbol_files[i.name]):
                    # copy from the install dir
                    try:
                        shutil.copyfile(
                            self.__zero_files[i.name], self.__symbol_files[i.name])
                        logger.debug(f'copying symbol list for {i.name}')
                    except FileNotFoundError:
                        self.symbol_list(i)

                    except Exception as e:
                        logger.error(e)

    @staticmethod
    def __validate_symbol(symbol, _list):
        '''
        parse symbol, replaces spaces and special characters to amke url compatible
        :param symbol:
        :return: symbol in url format symbol
        '''

        symbol = symbol if isinstance(symbol, IndexSymbol) else symbol.upper()
        if isinstance(symbol, IndexSymbol):
            symbol = urllib.parse.quote(symbol.value)

            return symbol

        elif symbol in _list:
            symbol = urllib.parse.quote(symbol.upper())

            return symbol

        else:
            symbol = None
            raise ValueError('not a vaild symbol')

    def market_status(self) -> dict:
        """
        get market status

        Examples
        --------

        >>> nse.market_status()

        """
        config = self.__urls
        logger.info("downloading market status")
        url = config['host'] + config['path']['marketStatus']

        return self.__get_resp(url=url).json()

    def info(self, symbol: str = 'SBIN') -> dict:
        '''
        Get symbol information from nse

        Examples
        --------

        >>> nse.info('SBIN')

        '''
        config = self.__urls
        symbol = self.__validate_symbol(
            symbol, self.symbols[IndexSymbol.All.name])
        if symbol is not None:
            logger.info(f"downloading symbol info for {symbol}")
            url = config['host'] + config['path']['info'].format(symbol=symbol)

            return self.__get_resp(url=url).json()

    def get_quote(self,
                  symbol: str = 'HDFC',
                  segment: Segment = Segment.EQ,
                  expiry: dt.date = None,
                  optionType: OptionType = OptionType.CE,
                  strike: float = 0.) -> dict:
        """

        Get realtime quote for EQ, FUT and OPT

        if no expiry date is provided for derivatives, returns date for nearest expiry

        Examples
        --------
        for cash
        >>> nse.get_quote('RELIANCE')

        for futures
        >>> nse.get_quote('TCS', segment=Segment.FUT, expiry=dt.date( 2020, 6, 30 ))

        for options
        >>> nse.get_quote('HDFC', segment=Segment.OPT, optionType=OptionType.PE)

        """
        config = self.__urls
        segment = segment.value
        optionType = optionType.value

        if symbol is not None:
            logger.info(f"downloading quote for {symbol} {segment}")
            quote = {}
            if segment == 'EQ':
                symbol = self.__validate_symbol(symbol,
                                                self.symbols[IndexSymbol.All.name] + [idx.value for idx in IndexSymbol])

                url = config['host'] + \
                    config['path']['quote_eq'].format(symbol=symbol)
                url1 = config['host'] + \
                    config['path']['trade_info'].format(symbol=symbol)
                data = self.__get_resp(url).json()
                data.update(self.__get_resp(url1).json())
                quote = data['priceInfo']
                quote['timestamp'] = dt.datetime.strptime(
                    data['metadata']['lastUpdateTime'], '%d-%b-%Y %H:%M:%S')
                quote.update(series=data['metadata']['series'])
                quote.update(symbol=data['metadata']['symbol'])
                quote.update(data['securityWiseDP'])
                quote['low'] = quote['intraDayHighLow']['min']
                quote['high'] = quote['intraDayHighLow']['max']
                # quote.pop('intraDayHighLow')
                # quote.pop('weekHighLow')

            elif segment == 'FUT':
                symbol = self.__validate_symbol(symbol,
                                                self.symbols[IndexSymbol.FnO.name] + ['NIFTY', 'BANKNIFTY'])

                url = config['host'] + \
                    config['path']['quote_derivative'].format(symbol=symbol)

                data = self.__get_resp(url).json()
                quote['timestamp'] = dt.datetime.strptime(
                    data['fut_timestamp'], '%d-%b-%Y %H:%M:%S')

                # filter data for fut segment
                data = [
                    i for i in data['stocks']
                    if segment.lower() in i['metadata']['instrumentType'].lower()
                ]

                # expiry date for futures contract
                # dict is made to remove duplicates as dict cant have duplicates
                self.expiry_list = list(
                    dict.fromkeys([dt.datetime.strptime(i['metadata']['expiryDate'], '%d-%b-%Y').date() for i in data]))
                self.expiry_list = sorted(self.expiry_list)

                # if expiry date is not given select first expiry date
                if expiry is None:
                    expiry = self.expiry_list[0]

                # filter data for expiry
                data = [i for i in data if
                        dt.datetime.strptime(i['metadata']['expiryDate'], '%d-%b-%Y').date() == expiry]

                quote.update(data[0]['marketDeptOrderBook']['tradeInfo'])
                quote.update(data[0]['metadata'])
                quote['expiryDate'] = dt.datetime.strptime(
                    quote['expiryDate'], '%d-%b-%Y').date()

            elif segment == 'OPT':
                url = config['host'] + \
                    config['path']['quote_derivative'].format(symbol=symbol)

                data = self.__get_resp(url).json()

                quote['timestamp'] = dt.datetime.strptime(
                    data['opt_timestamp'], '%d-%b-%Y %H:%M:%S')

                # filter data for segment and option type
                data = [
                    i for i in data['stocks']
                    if segment.lower() in i['metadata']['instrumentType'].lower()
                    and i['metadata']['optionType'] == optionType
                ]

                # get expiry list
                # dict is made to remove duplicates as dict cant have duplicates
                self.expiry_list = list(
                    dict.fromkeys([dt.datetime.strptime(i['metadata']['expiryDate'], '%d-%b-%Y').date() for i in data]))
                self.expiry_list = sorted(self.expiry_list)

                # select expiry
                if expiry is None:
                    expiry = self.expiry_list[0]

                # filter data for expiry
                data = [i for i in data if
                        dt.datetime.strptime(i['metadata']['expiryDate'], '%d-%b-%Y').date() == expiry]

                # get strike list
                self.strike_list = list(dict.fromkeys(
                    [i['metadata']['strikePrice'] for i in data]))
                self.strike_list = sorted([float(s) for s in self.strike_list])

                # select strike
                strike = strike if strike in self.strike_list else self.strike_list[0]

                # filter data for strike price
                data = [i for i in data if i['metadata']
                        ['strikePrice'] == strike]

                quote.update(data[0]['marketDeptOrderBook']['tradeInfo'])
                quote.update(data[0]['marketDeptOrderBook']['otherInfo'])
                quote.update(data[0]['metadata'])
                quote['expiryDate'] = dt.datetime.strptime(
                    quote['expiryDate'], '%d-%b-%Y').date()

            return quote

    def bhavcopy(self, req_date: dt.date = None,
                 series: str = 'eq') -> pd.DataFrame:
        """
        download bhavcopy from nse
        or
        read bhavcopy if already downloaded

        Examples
        --------

        >>> nse.bhavcopy()

        >>> nse.bhavcopy(dt.date(2020,6,17))
        """

        series = series.upper()
        # if date is not given select first date from history
        req_date = self.trading_days(
        )[-1].date() if req_date is None else req_date

        filename = f'{self.dir["bhavcopy_eq"]}bhav_{req_date}.pkl'

        bhavcopy = None
        # if file is present, ie downloaded previously
        # read file
        if os.path.exists(filename):
            bhavcopy = pd.read_pickle(filename)
            logger.debug(f'read {filename} from disk')

        # if file is not present
        # download the file
        else:
            config = self.__urls
            url = config['path']['bhavcopy'].format(
                date=req_date.strftime("%d%m%Y"))
            csv = self.__get_resp(url).content.decode('utf8').replace(" ", "")

            bhavcopy = pd.read_csv(io.StringIO(csv))
            bhavcopy["DATE1"] = bhavcopy["DATE1"].apply(
                lambda x: dt.datetime.strptime(x, '%d-%b-%Y').date())

            # save the downloaded bhavcopy
            bhavcopy.to_pickle(filename)

        if bhavcopy is not None:
            # filter as as required
            if series != 'ALL':
                bhavcopy = bhavcopy.loc[bhavcopy['SERIES'] == series]

            # set symbol series as index
            bhavcopy.set_index(['SYMBOL', 'SERIES'], inplace=True)

        return bhavcopy

    def bhavcopy_fno(self, req_date: dt.date = None) -> pd.DataFrame:
        """
        download bhavcopy from nse
        or
        read bhavcopy if already downloaded

        Examples
        --------

        >>> nse.bhavcopy_fno()

        >>> nse.bhavcopy_fno(dt.date(2020,6,17))

        """
        # if date is not given select first date from history
        req_date = self.trading_days(
        )[-1].date() if req_date is None else req_date

        filename = f'{self.dir["bhavcopy_fno"]}bhav_{req_date}.pkl'

        bhavcopy = None
        if os.path.exists(filename):
            bhavcopy = pd.read_pickle(filename)
            logger.debug(f'read {filename} from disk')

        else:
            config = self.__urls
            url = config['path']['bhavcopy_derivatives'].format(date=req_date.strftime("%d%b%Y").upper(),
                                                                month=req_date.strftime(
                                                                    "%b").upper(),
                                                                year=req_date.strftime("%Y"))

            logger.debug("downloading bhavcopy for {}".format(req_date))
            stream = self.__get_resp(
                url).content

            filebytes = io.BytesIO(stream)
            zf = zipfile.ZipFile(filebytes)

            bhavcopy = pd.read_csv(zf.open(zf.namelist()[0]))

            bhavcopy.set_index('SYMBOL', inplace=True)
            bhavcopy.dropna(axis=1, inplace=True)
            bhavcopy.EXPIRY_DT = bhavcopy.EXPIRY_DT.apply(
                lambda x: dt.datetime.strptime(x, '%d-%b-%Y'))

            bhavcopy.to_pickle(filename)

        return bhavcopy

    def pre_open(self) -> pd.DataFrame:
        """

        get pre open data from nse

        Examples
        --------

        >>> nse.pre_open()

        """

        # check if todays file exists
        filename = f"{self.dir['pre_open']}{dt.date.today()}.pkl"
        if os.path.exists(filename):
            # read if exists
            pre_open_data = pd.read_pickle(filename)
            logger.debug('pre_open data read from file')

        # otherwise download
        else:
            logger.debug("downloading preopen data")
            config = self.__urls
            url = config['host'] + config['path']['preOpen']

            data = self.__get_resp(url).json()

            timestamp = dt.datetime.strptime(
                data['timestamp'], "%d-%b-%Y %H:%M:%S").date()
            pre_open_data = pd.json_normalize(data['data'])

            pre_open_data = pre_open_data.set_index('metadata.symbol')

            pre_open_data["detail.preOpenMarket.lastUpdateTime"] = pre_open_data[
                "detail.preOpenMarket.lastUpdateTime"].apply(
                lambda x: dt.datetime.strptime(x, '%d-%b-%Y %H:%M:%S'))

            filename = f"{self.dir['pre_open']}{timestamp}.pkl"
            pre_open_data.to_pickle(filename)

        return pre_open_data

    def option_chain(self, symbol: str, expiry: dt.date = None) -> pd.DataFrame:
        """
        downloads the latest available option chain from nse website
        if no expiry is None current contract option chain 

        :returns dictonaly containing
            timestamp as str
            option_chain as pd.Dataframe
            expiry_list as list

        Examples
        --------

        >>> nse.option_chain('INFY')

        >>> nse.option_chain('INFY',expiry=dt.date(2020,6,30))

        """

        symbol = self.__validate_symbol(
            symbol, self.symbols[IndexSymbol.FnO.name] + ['NIFTY', 'BANKNIFTY', 'NIFTYIT'])
        logger.debug(f'download option chain')
        config = self.__urls

        url = config['host'] + (config['path']['option_chain_index'] if 'NIFTY' in symbol else config['path'][
            'option_cahin_equities']).format(symbol=symbol)
        data = self.__get_resp(url).json()

        self.expiry_list = sorted([dt.datetime.strptime(
            d, '%d-%b-%Y').date() for d in data['records']['expiryDates']])
        expiry = expiry or self.expiry_list[0]
        option_chain = pd.json_normalize(data['records']['data'])
        option_chain.expiryDate = option_chain.expiryDate.apply(
            lambda x: dt.datetime.strptime(x, '%d-%b-%Y').date())

        option_chain = option_chain[option_chain.expiryDate == expiry]
        self.strike_list = sorted(list(option_chain.strikePrice))

        return option_chain

    def fii_dii(self) -> dict:
        """
        get FII and DII data from nse

        Examples
        --------

        >>> nse.fii_dii()

        """

        # filename to store fii/dii data
        filename = f'{self.dir["fii_dii"]}fii_dii.csv'

        # if fine not found
        if not os.path.exists(filename):
            # set mode to w
            mode = 'w'

            # and time stamp to two days before
            timestamp = dt.date.today() - dt.timedelta(days=2)
        # if file is found
        else:
            # set mode to w
            mode = 'a'
            # read the existing file
            csv_file = pd.read_csv(filename, header=[0, 1], index_col=[0])
            # get the timestamp of last row
            timestamp = dt.datetime.strptime(
                csv_file.tail(1).index[0], '%d-%b-%Y').date()

        # if timestamp is today
        # or yesterday and time now is less than 15,30 , fii dii date available on website will not be updated
        # in that case return previous data
        if timestamp == dt.date.today() or timestamp == dt.date.today() - dt.timedelta(
                days=1) and dt.datetime.now().time() < dt.time(15, 30):
            logger.debug('read fii/dii data from disk')
            return csv_file.tail(1)

        # otherwise get from website
        else:
            config = self.__urls
            url = config['host'] + config['path']['fii_dii']

            resp = self.__get_resp(url).json()

            resp[0].pop('date')
            date = resp[1].pop('date')
            fii = [d for d in resp if d['category'] == 'FII/FPI *'][0]
            dii = [d for d in resp if d['category'] == 'DII **'][0]
            fii_dii = pd.concat(
                [pd.json_normalize(fii),
                 pd.json_normalize(dii)],
                axis=1,
                keys=[fii['category'], dii['category']])
            fii_dii.index = [date]
            if dt.datetime.strptime(date, '%d-%b-%Y').date() != timestamp:
                fii_dii.to_csv(filename, mode=mode,
                               header=True if mode == 'w' else False)
            return fii_dii.tail(1)

    def __get_hist(self, symbol='SBIN', from_date=None, to_date=None):
        config = self.__urls

        # max days that can be requested
        max_date_range = 480

        if from_date == None:
            from_date = dt.date.today() - dt.timedelta(days=30)
        if to_date == None:
            to_date = dt.date.today()

        # initilise  empty dataframe
        hist = pd.DataFrame()

        while True:
            # if required length is more than allowed range
            if (to_date - from_date).days > max_date_range:

                marker = from_date + dt.timedelta(max_date_range)
                url = config['host'] + config['path']['hist'].format(symbol=symbol,
                                                                     from_date=from_date.strftime(
                                                                         '%d-%m-%Y'),
                                                                     to_date=marker.strftime('%d-%m-%Y'))

                from_date = from_date + dt.timedelta(days=(max_date_range + 1))

                csv = self.__get_resp(url).content.decode(
                    'utf8').replace(" ", "")
                is_complete = False

            else:
                url = config['host'] + config['path']['hist'].format(symbol=symbol,
                                                                     from_date=from_date.strftime(
                                                                         '%d-%m-%Y'),
                                                                     to_date=to_date.strftime('%d-%m-%Y'))

                from_date = from_date + dt.timedelta(max_date_range + 1)

                csv = self.__get_resp(url).content.decode(
                    'utf8').replace(" ", "")
                is_complete = True

            hist = pd.concat([hist, pd.read_csv(io.StringIO(csv))[::-1]])
            if is_complete:
                break
            time.sleep(1)

        hist['Date'] = pd.to_datetime(hist['Date'])
        hist.set_index('Date', inplace=True)
        hist.drop(['series', 'PREV.CLOSE', 'ltp', 'vwap', '52WH',
                   '52WL', 'VALUE', 'Nooftrades'], axis=1, inplace=True)
        try:
            hist.columns = ['open', 'high', 'low', 'close', 'volume']
        except Exception as e:
            print(hist.columns, e)
            time.sleep(5)

        for column in hist.columns[:4]:
            hist[column] = hist[column].astype(str).str.replace(
                ',', '').replace('-', '0').astype(float)
        hist['volume'] = hist['volume'].astype(int)
        return hist

    def __get_hist_index(self, symbol='NIFTY 50', from_date=None,
                         to_date=None):
        if from_date == None:
            from_date = dt.date.today() - dt.timedelta(days=30)
        if to_date == None:
            to_date = dt.date.today()

        config = self.__urls
        base_url = config['path']['indices_hist_base']

        urls = []
        max_range_len = 100
        while True:
            if (to_date - from_date).days > max_range_len:
                s = from_date
                e = s + dt.timedelta(max_range_len)
                url = f"{base_url}{symbol}&fromDate={s.strftime('%d-%m-%Y')}&toDate={e.strftime('%d-%m-%Y')}"
                urls.append(url)
                from_date = from_date + dt.timedelta(max_range_len + 1)

            else:
                url = f"{base_url}{symbol}&fromDate={from_date.strftime('%d-%m-%Y')}&toDate={to_date.strftime('%d-%m-%Y')}"
                urls.append(url)

                break

        hist = pd.DataFrame(columns=[
            'Date', 'Open', 'High', 'Low', 'Close', 'SharesTraded',
            'Turnover(Cr)'
        ])
        for url in urls:
            page = self.__get_resp(url).content.decode('utf-8')
            raw_table = BeautifulSoup(page, 'lxml').find_all('table')[0]

            rows = raw_table.find_all('tr')

            for row_no, row in enumerate(rows):

                if row_no > 2:
                    _row = [
                        cell.get_text().replace(" ", "").replace(",", "")
                        for cell in row.find_all('td')
                    ]
                    if len(_row) > 4:
                        hist.loc[len(hist)] = _row

            time.sleep(1)
        hist.Date = hist.Date.apply(
            lambda d: dt.datetime.strptime(d, '%d-%b-%Y'))

        hist.set_index("Date", inplace=True)

        for col in hist.columns:
            hist[col] = hist[col].astype(str).replace(
                ',', '').replace('-', '0').astype(float)

        return hist

    def get_hist(self, symbol: str = 'SBIN', from_date: dt.date = None, to_date: dt.date = None) -> pd.DataFrame:
        """
        get historical data from nse
        symbol index or symbol

        Examples
        --------

        >>> nse.get_hist('SBIN')

        >>> nse.get_hist('NIFTY 50', from_date=dt.date(2020,1,1),to_date=dt.date(2020,6,26))


        """
        symbol = self.__validate_symbol(symbol,
                                        self.symbols[IndexSymbol.All.name] + [idx.value for idx in IndexSymbol])

        if "NIFTY" in symbol:
            return self.__get_hist_index(symbol, from_date, to_date)
        else:
            return self.__get_hist(symbol, from_date, to_date)

    def get_indices(self, index: IndexSymbol = None) -> pd.DataFrame:
        """
        get realtime index value

        Examples
        --------

        >>> nse.get_indices(IndexSymbol.NiftyInfra)
        >>> nse.get_indices(IndexSymbol.Nifty50))

        """
        # need validation only so not assigned
        if index is not None:
            self.__validate_symbol(index, [idx for idx in IndexSymbol])
        config = self.__urls
        url = config['host'] + config['path']['indices']
        data = self.__get_resp(url).json()['data']

        data = pd.json_normalize(data).set_index('indexSymbol')
        if index is not None:
            data = data[data.index == index.value]

        data.drop(['chart365dPath', 'chartTodayPath',
                   'chart30dPath'], inplace=True, axis=1)
        return data

    def __gainers_losers(self, index, advance=False):
        index = self.__validate_symbol(
            index.value, [idx.value for idx in IndexSymbol if idx.value != 'ALL'])
        index = 'SECURITIES%20IN%20F%26O' if index == 'FNO' else index
        config = self.__urls
        url = config['host'] + \
            config['path']['gainer_loser'].format(index=index)
        data = self.__get_resp(url).json()

        # if requested by adv decl
        if advance:
            return data["advance"]

        table = pd.DataFrame(data['data'])
        table.drop([
            'totalTradedValue', 'lastUpdateTime', 'yearHigh', 'yearLow', 'nearWKH',
            'nearWKL', 'perChange365d', 'date365dAgo', 'chart365dPath',
            'date30dAgo', 'perChange30d', 'chart30dPath', 'chartTodayPath', 'meta'],
            axis=1,
            inplace=True)

        table.set_index('symbol', inplace=True)
        return table

    def __symbol_list(self, index: IndexSymbol):
        """

        :param index: index name or fno
        :return: list ig symbols for selected group
        """
        if not isinstance(index, IndexSymbol):
            raise TypeError('index is not of type "Index"')
        # get value of index

        config = self.__urls

        logger.debug(f'downloading symbol list for {index.name}')
        if index == IndexSymbol.All:
            data = list(self.bhavcopy(req_date=dt.date(
                2020, 10, 8)).reset_index().SYMBOL)

        elif index == IndexSymbol.FnO:
            url = config['host'] + config['path']['fnoSymbols']
            data = self.__get_resp(url).json()

            # add nifty and banknifty in group as they will not be in list
            data.extend(['NIFTY', 'BANKNIFTY'])

        else:

            url = config['host'] + config['path']['symbol_list'].format(
                index=self.__validate_symbol(index, IndexSymbol))
            data = self.__get_resp(url).json()['data']

            data = [i['meta']['symbol']
                    for i in data if i['identifier'] != index.value]

        data.sort()
        with open(self.dir['symbol_list'] + index.name + '.pkl', 'wb')as f:
            pickle.dump(data, f)
            logger.info(f'symbol list saved for {index}')
        return data

    def update_symbol_list(self):
        """
        Update list of symbols
        no need to run frequently
        required when constituent of an index is changed
        or
        list of securities in fno are updates
        :return: None

        Examples:
        ```
        nse.update_symbol_list()
        ```
        """
        for i in [a for a in IndexSymbol]:
            self.__symbol_list(i)
            time.sleep(1)

    def trading_days(self):
        # todo make this private
        """
                update trading days
                :return: pandas as column of trading days
                """
        filename = f'{self.dir["data_root"]}/trading_days.csv'

        # if this file exists, load trading days and find previous trading day
        if os.path.exists(filename):
            trading_days = pd.read_csv(filename, header=None)
            trading_days.columns = ['Date']
            trading_days['Date'] = trading_days['Date'].apply(
                lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
            previous_trading_day = list(trading_days.tail(1)['Date'])[0].date()

        # otherwise assume previous trading day available as today -100 days
        else:
            previous_trading_day = dt.date.today() - dt.timedelta(days=100)
            trading_days = pd.DataFrame()

        # if previous trading read from file same as today or yerterday and present time is before 18:30
        # do nothing
        if previous_trading_day == dt.date.today() or previous_trading_day == dt.date.today() - dt.timedelta(
                days=1) and dt.datetime.now().time() <= dt.time(18, 45):
            pass

        # otherwise download hisory  and append to file and drop duplicates
        # write that to csv file
        else:
            _trading_days = self.get_hist(symbol='SBIN', from_date=previous_trading_day - dt.timedelta(7),
                                          to_date=dt.date.today()).reset_index()[['Date']]

            trading_days = pd.concat(
                [trading_days, _trading_days]).drop_duplicates()
            # trading_days.index = trading_days.index.map(lambda x: x.date)
            trading_days.to_csv(filename, mode='w', index=False, header=False)

        # once again read the file and return the index as trading days
        trading_days = pd.read_csv(filename, header=None, index_col=0)
        trading_days.index = trading_days.index.map(
            lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))

        return trading_days.index

    def top_gainers(self, index: IndexSymbol = IndexSymbol.FnO, length: int = 10) -> pd.DataFrame:
        """
        get top gainers in given index
        Examples
        --------

        >>> nse.top_gainers(IndexSymbol.FnO,length=10)

        """
        gainers = self.__gainers_losers(index).sort_values(by=['pChange'],
                                                           axis=0,
                                                           ascending=False).head(length)
        gainers = gainers[gainers.pChange > 0.]

        return gainers

    def top_losers(self, index: IndexSymbol = IndexSymbol.FnO, length: int = 10) -> pd.DataFrame:
        """
        get lop losers in given index
        Examples
        --------

        >>> nse.top_gainers(IndexSymbol.FnO,length=10)

        """
        losers = self.__gainers_losers(index).sort_values(by=['pChange'],
                                                          axis=0,
                                                          ascending=True).head(length)
        losers = losers[losers.pChange < 0.]

        return losers

    def __list_all_files(self):
        root = self.dir['data_root']
        files = glob(f'{root}/*')
        files.extend(glob(f'{root}/*'))
        files.extend(glob(f'{root}/*/*'))
        files.extend(glob(f'{root}/*/*/*'))

        return files

    def __data_size(self):
        """size of home folder .pynse
        """

        files = self.__list_all_files()
        size_bytes = sum(os.path.getsize(f)
                         for f in files if os.path.isfile(f))

        y = 1
        for unit in ['B', 'KB', 'MB', 'GB']:

            if size_bytes/y < 1024 or unit == 'GB':
                size = round(size_bytes/y, 2), unit
                break
            else:
                y = y*1024

        return size

    def clear_data(self):
        """clear data from .pynse folder
        """
        if input(f"Delete '{self.dir['data_root']}' and its content, are you sure?? y/n") == 'y':
            shutil.rmtree(self.dir['data_root'])
            if os.path.exists(self.dir['data_root']):
                print('some files might not be deleted. Trt manually removing')
            else:
                print('data cleared')
            exit()

        else:
            print('skipped')
