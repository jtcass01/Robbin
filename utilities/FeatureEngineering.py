import numpy as np
import pandas as pd
import math
from utilities.Logger import Logger
from MarketInterfaces.MarketDataInterface import retrieve_yahoo_fin_stock_data
from enum import Enum

# Reference https://github.com/DarkKnight1991/stock_cnn_blog_pub/blob/master/src/utils.py


class Indicators(Enum):
    # Momentum Indicators
    RELATIVE_STOCK_INDEX = "rsi"  # Reference: https://www.investopedia.com/terms/r/rsi.asp
    WILLIAMS_PERCENT_RANGE = "WilliamsR"  # Reference: https://www.investopedia.com/terms/w/williamsr.asp
    CHANDE_MOMENTUM_OSCILLATOR = "cmo"  # Reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo
    RATE_OF_CHANGE = "roc"  # Reference: https://www.investopedia.com/terms/p/pricerateofchange.asp ; https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum
    STOCHASTIC_OSCILLATOR = "stoch"
    ULTIMATE_OSCILLATOR = "uo"
    AWESOME_OSCILLATOR = "ao"
    KAUFMAN_ADAPTIVE_MOVING_AVERAGE = "kama"
    TRUE_STRENGTH_INDEX = "tsi"
    # Averages
    WEIGHTED_MOVING_AVERAGE = "wma"  # Reference: https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/weighted-moving-average-wma/#:~:text=The%20weighted%20moving%20average%20(WMA)%20is%20a%20technical%20indicator%20that,weighting%20on%20past%20data%20points.
    EXPONENTIAL_MOVING_AVERAGE = "ema"  # Reference: https://www.investopedia.com/terms/e/ema.asp
    SIMPLE_MOVING_AVERAGE = "sma"  # Reference: https://www.investopedia.com/terms/s/sma.asp
    HULL_MOVING_AVERAGE = "hma"  # Reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/hull-moving-average
    TRIPLE_EXPONENTIALLY_SMOOTHED_MOVING_AVERAGE = "trix"  # Reference: https://www.investopedia.com/terms/t/trix.asp
    # Trend Indicators
    MOVING_AVERAGE_CONVERGENCE_DIVERGENCE = "macd"  # Reference: https://www.investopedia.com/terms/m/macd.asp
    COMMODITY_CHANNEL_INDEX = "cci"  # Reference: https://www.investopedia.com/articles/active-trading/031914/how-traders-can-utilize-cci-commodity-channel-index-trade-stock-trends.asp
    DETRENDED_PRICE_OSCILLATOR = "dpo"  # Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:detrended_price_osci
    AVERAGE_DIRECTIONAL_INDEX = "adx"  # Reference: https://www.investopedia.com/terms/a/adx.asp
    MASS_INDEX = 'mi'
    ICHIMOKU_A = 'ichimoku_a'
    # Volume Indicators
    CHAIKIN_MONEY_FLOW = "cmf"  # Reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmf
    ACCUMULATION_DISTRIBUTION_INDEX = 'adi'
    EASE_OF_MOVEMENT = 'eom'
    FORCE_INDEX = "Force Index"  # Reference: https://www.investopedia.com/terms/f/force-index.asp


def convert_to_one_hot(Y: np.array, C: int) -> np.array:
    return np.eye(C)[Y.reshape(-1, 1)]


def engineer_additional_features(stock_data: pd.DataFrame) -> None:
    intervals = range(6, 21)

    Logger.console_log("Calculating Relative Stock Index over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    RSI(stock_data, 'close', intervals)
    Logger.console_log("Calculating Williams Percent Range over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    WilliamsR(stock_data, intervals)
    Logger.console_log("Calculating Weighted Moving Average over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    WMA(stock_data,'close',intervals)
    Logger.console_log("Calculating Exponential Moving Average over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    EMA(stock_data, 'close', intervals)
    Logger.console_log("Calculating Simple Moving Average over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    SMA(stock_data, 'close', intervals)
    Logger.console_log("Calculating Hull Moving Average over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    HMA(stock_data,'close',intervals)
    Logger.console_log("Calculating Triple Exponentially Smoothed Moving Average over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    TRIX(stock_data, 'close', intervals)
    Logger.console_log("Calculating Commodity Channel Index over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    CCI(stock_data, intervals)
    Logger.console_log("Calculating Chande Momentum Oscillator over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    CMO(stock_data, 'close', intervals)
    Logger.console_log("Calculating Moving Average Convergence/Divergence over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    MACD(stock_data, 'close', intervals)
    Logger.console_log("Calculating Detrended Price Oscillator over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    DPO(stock_data, 'close', intervals)
    Logger.console_log("Calculating Rate of Change over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    ROC(stock_data, 'close', intervals)
    Logger.console_log("Calculating Chaikin Money Flow over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    CMF(stock_data, intervals)
    Logger.console_log("Calculating Average Directional Index over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    ADX(stock_data, intervals)
    Logger.console_log("Calculating Force Index over intervals " + str(intervals),
                       Logger.LogStatus.EMPHASIS)
    ForceIndex(stock_data, intervals)

    stock_data = stock_data.drop(stock_data.index[range(intervals[-1])])


def engineer_data_over_single_interval(df: pd.DataFrame,
                                       indicators: list,
                                       ticker: str = "",
                                       rsi_n: int = 14,
                                       cmo_n: int = 7,
                                       macd_fast: int = 12,
                                       macd_slow: int = 26,
                                       macd_sign: int = 9,
                                       roc_n: int = 12,
                                       cci_n: int = 20,
                                       dpo_n: int = 20,
                                       cmf_n: int = 20,
                                       adx_n: int = 14,
                                       mass_index_low: int = 9,
                                       mass_index_high: int = 25,
                                       trix_n: int = 15,
                                       stochastic_oscillator_n: int = 14,
                                       stochastic_oscillator_sma_n: int = 3,
                                       ultimate_oscillator_short_n: int = 7,
                                       ultimate_oscillator_medium_n: int = 14,
                                       ultimate_oscillator_long_n: int = 28,
                                       ao_short_n: int = 5,
                                       ao_long_n: int = 34,
                                       kama_n: int = 10,
                                       tsi_high_n: int = 25,
                                       tsi_low_n: int = 13,
                                       eom_n: int = 14,
                                       force_index_n: int = 13,
                                       ichimoku_low_n: int = 9,
                                       ichimoku_medium_n: int = 26):
    from ta.momentum import rsi, wr, roc, ao, stoch, uo, kama, tsi
    from ta.trend import macd, macd_signal, cci, dpo, adx, mass_index, trix, ichimoku_a
    from ta.volume import chaikin_money_flow, acc_dist_index, ease_of_movement, force_index

    # Momentum Indicators
    if Indicators.RELATIVE_STOCK_INDEX in indicators:
        Logger.console_log(message="Calculating " + Indicators.RELATIVE_STOCK_INDEX.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.RELATIVE_STOCK_INDEX.value] = rsi(close=df['close'],
                                                        n=rsi_n)

    if Indicators.WILLIAMS_PERCENT_RANGE in indicators:
        Logger.console_log(message="Calculating " + Indicators.WILLIAMS_PERCENT_RANGE.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.WILLIAMS_PERCENT_RANGE.value] = wr(df['high'], df['low'], df['close'])

    if Indicators.CHANDE_MOMENTUM_OSCILLATOR in indicators:
        Logger.console_log(message="Calculating " + Indicators.CHANDE_MOMENTUM_OSCILLATOR.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.CHANDE_MOMENTUM_OSCILLATOR.value] = chande_momentum_oscillator(close_data=df['close'],
                                                                                     period=cmo_n)

    if Indicators.RATE_OF_CHANGE in indicators:
        Logger.console_log(message="Calculating " + Indicators.RATE_OF_CHANGE.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.RATE_OF_CHANGE.value] = roc(close=df['close'], n=roc_n)

    if Indicators.STOCHASTIC_OSCILLATOR in indicators:
        Logger.console_log(message="Calculating " + Indicators.STOCHASTIC_OSCILLATOR.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.STOCHASTIC_OSCILLATOR.value] = stoch(high=df['high'], low=df['low'],
                                                           close=df['close'], n=stochastic_oscillator_n,
                                                           d_n=stochastic_oscillator_sma_n)

    if Indicators.ULTIMATE_OSCILLATOR in indicators:
        Logger.console_log(message="Calculating " + Indicators.ULTIMATE_OSCILLATOR.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.ULTIMATE_OSCILLATOR.value] = uo(high=df['high'], low=df['low'],
                                                      close=df['close'], s=ultimate_oscillator_short_n,
                                                      m=ultimate_oscillator_medium_n, len=ultimate_oscillator_long_n)

    if Indicators.AWESOME_OSCILLATOR in indicators:
        Logger.console_log(message="Calculating " + Indicators.AWESOME_OSCILLATOR.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.AWESOME_OSCILLATOR.value] = ao(high=df['high'], low=df['low'], s=ao_short_n, len=ao_long_n)

    if Indicators.KAUFMAN_ADAPTIVE_MOVING_AVERAGE in indicators:
        Logger.console_log(message="Calculating " + Indicators.KAUFMAN_ADAPTIVE_MOVING_AVERAGE.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.KAUFMAN_ADAPTIVE_MOVING_AVERAGE.value] = kama(close=df['close'], n=kama_n)

    if Indicators.TRUE_STRENGTH_INDEX in indicators:
        Logger.console_log(message="Calculating " + Indicators.TRUE_STRENGTH_INDEX.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.TRUE_STRENGTH_INDEX.value] = tsi(close=df['close'], r=tsi_high_n, s=tsi_low_n)

    # Trend Indicator
    if Indicators.MOVING_AVERAGE_CONVERGENCE_DIVERGENCE in indicators:
        Logger.console_log(message="Calculating " + Indicators.MOVING_AVERAGE_CONVERGENCE_DIVERGENCE.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.MOVING_AVERAGE_CONVERGENCE_DIVERGENCE.value] = macd(close=df['close'],
                                                                          n_slow=macd_slow,
                                                                          n_fast=macd_fast) - \
                                                                     macd_signal(close=df['close'],
                                                                                 n_slow=macd_slow,
                                                                                 n_fast=macd_fast,
                                                                                 n_sign=macd_sign)

    if Indicators.COMMODITY_CHANNEL_INDEX in indicators:
        Logger.console_log(message="Calculating " + Indicators.COMMODITY_CHANNEL_INDEX.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.COMMODITY_CHANNEL_INDEX.value] = cci(high=df['high'], low=df['low'], close=df['close'], n=cci_n)

    if Indicators.DETRENDED_PRICE_OSCILLATOR in indicators:
        Logger.console_log(message="Calculating " + Indicators.DETRENDED_PRICE_OSCILLATOR.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.DETRENDED_PRICE_OSCILLATOR.value] = dpo(close=df['close'], n=dpo_n)

    if Indicators.AVERAGE_DIRECTIONAL_INDEX in indicators:
        Logger.console_log(message="Calculating " + Indicators.AVERAGE_DIRECTIONAL_INDEX.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.AVERAGE_DIRECTIONAL_INDEX.value] = adx(high=df['high'], low=df['low'],
                                                             close=df['close'], n=adx_n)

    if Indicators.MASS_INDEX in indicators:
        Logger.console_log(message="Calculating " + Indicators.MASS_INDEX.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.MASS_INDEX.value] = mass_index(high=df['high'], low=df['low'],
                                                     n=mass_index_low, n2=mass_index_high)

    if Indicators.TRIPLE_EXPONENTIALLY_SMOOTHED_MOVING_AVERAGE in indicators:
        Logger.console_log(message="Calculating " + Indicators.TRIPLE_EXPONENTIALLY_SMOOTHED_MOVING_AVERAGE.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.TRIPLE_EXPONENTIALLY_SMOOTHED_MOVING_AVERAGE.value] = trix(close=df['close'], n=trix_n)

    if Indicators.ICHIMOKU_A in indicators:
        Logger.console_log(message="Calculating " + Indicators.ICHIMOKU_A.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.ICHIMOKU_A.value] = ichimoku_a(high=df['high'], low=df['low'],
                                                     n1=ichimoku_low_n, n2=ichimoku_medium_n)

    # Volume Indicator
    if Indicators.CHAIKIN_MONEY_FLOW in indicators:
        Logger.console_log(message="Calculating " + Indicators.CHAIKIN_MONEY_FLOW.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.CHAIKIN_MONEY_FLOW.value] = chaikin_money_flow(high=df['high'], low=df['low'],
                                                                     close=df['close'], volume=df['volume'], n=cmf_n)

    if Indicators.ACCUMULATION_DISTRIBUTION_INDEX in indicators:
        Logger.console_log(message="Calculating " + Indicators.ACCUMULATION_DISTRIBUTION_INDEX.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.ACCUMULATION_DISTRIBUTION_INDEX.value] = acc_dist_index(high=df['high'], low=df['low'],
                                                                              close=df['close'], volume=df['volume'])

    if Indicators.EASE_OF_MOVEMENT in indicators:
        Logger.console_log(message="Calculating " + Indicators.EASE_OF_MOVEMENT.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.EASE_OF_MOVEMENT.value] = ease_of_movement(high=df['high'], low=df['low'], volume=df['volume'], n=eom_n)

    if Indicators.FORCE_INDEX in indicators:
        Logger.console_log(message="Calculating " + Indicators.FORCE_INDEX.value + " for stock " + ticker,
                           status=Logger.LogStatus.EMPHASIS)
        df[Indicators.FORCE_INDEX.value] = force_index(close=df['close'], volume=df['volume'], n=force_index_n)


def get_offset(x: int) -> int:
    """

    :param x:
    :return:
    """
    return int(x) if x else 0


def verify_series(series: pd.Series) -> pd.Series:
    """

    :param series:
    :return:
    """
    if series is not None and isinstance(series, pd.Series):
        return series


def RSI(df, col_name, intervals):
    """
        Relative Stock index
    """
    from ta.momentum import rsi
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df["rsi_" + str(interval)] = rsi(df[col_name], n=interval)


def WilliamsR(df, intervals):
    """
        Williams Percent Range.  Momentum indicator. Moves between 0 to-100 and measures overbought and oversold levels.
        Used to find entry and exit points in the market.

        Reference: https://www.investopedia.com/terms/w/williamsr.asp
    """
    from ta.momentum import wr
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df["wr_" + str(interval)] = wr(df['high'], df['low'], df['close'], interval)


def MoneyFlowIndex(df, intervals):
    """
        Technical oscillator.  Also called Volume-Weighted RSI
        Reference: https://www.investopedia.com/terms/m/mfi.asp
    """
    from ta.volume import money_flow_index
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df['mfi_' + str(interval)] = money_flow_index(df['high'], df['low'], df['close'], df['volume'], n=interval, fillna=True)


def SMA(df, col_name, intervals):
    """
        Simple Moving Average.
        Reference: https://www.investopedia.com/terms/s/sma.asp
    """
    from tqdm.auto import tqdm

    def sma(close, length=None, offset=None, **kwargs):
        """Indicator: Simple Moving Average (SMA)"""
        # Validate Arguments
        close = verify_series(close)
        length = int(length) if length and length > 0 else 10
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
        offset = get_offset(offset)

        # Calculate Result
        sma = close.rolling(length, min_periods=min_periods).mean()

        # Offset
        if offset != 0:
            sma = sma.shift(offset)

        # Name & Category
        sma.name = f"SMA_{length}"
        sma.category = 'overlap'

        return sma

    for interval in tqdm(intervals):
        df["sma_" + str(interval)] = sma(df[col_name], length=interval)


def EMA(df, col_name, intervals):
    """
        Exponential Moving Average.  Greater weight and significance on most recent data points.
        Reference: https://www.investopedia.com/terms/e/ema.asp
    """
    from tqdm.auto import tqdm

    def ema(close, length=None, offset=None, **kwargs):
        """Indicator: Exponential Moving Average (EMA)"""
        # Validate Arguments
        close = verify_series(close)
        length = int(length) if length and length > 0 else 10
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length#int(0.25 * length)
        adjust = bool(kwargs['adjust']) if 'adjust' in kwargs and kwargs['adjust'] is not None else True
        offset = get_offset(offset)

        # Calculate Result
        if 'presma' in kwargs and kwargs['presma']:
            initial_sma = sma(close=close, length=length)[:length]
            rest = close[length:]
            close = pd.concat([initial_sma, rest])

        ema = close.ewm(span=length, min_periods=min_periods, adjust=adjust).mean()

        # Offset
        if offset != 0:
            ema = ema.shift(offset)

        # Name & Category
        ema.name = f"EMA_{length}"
        ema.category = 'overlap'

        return ema

    for interval in tqdm(intervals):
        df["ema_" + str(interval)] = ema(df[col_name], length=interval)


def wma(close, length=None, asc=None, offset=None, **kwargs):
    """Indicator: Weighted Moving Average (WMA)"""
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    asc = asc if asc else True
    offset = get_offset(offset)

    # Calculate Result
    total_weight = 0.5 * length * (length + 1)
    weights_ = pd.Series(np.arange(1, length + 1))
    weights = weights_ if asc else weights_[::-1]

    def linear_weights(w):
        def _compute(x):
            return (w * x).sum() / total_weight
        return _compute

    close_ = close.rolling(length, min_periods=length)
    wma = close_.apply(linear_weights(weights), raw=True)

    # Offset
    if offset != 0:
        wma = wma.shift(offset)

    # Name & Category
    wma.name = f"WMA_{length}"
    wma.category = 'overlap'

    return wma


def WMA(df, col_name, intervals, hma_step=0):
    """
        Weighted Moving Average.  Greater weight on most recent data points.
        Reference: https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/weighted-moving-average-wma/
    """
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df["wma_" + str(interval)] = wma(df['close'], length=interval)


def HMA(df, col_name, intervals):
    """
        Hull Moving Average. More responsive moving average.  Derivative of TTM Squeeze volatility indicator.
        Reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/hull-moving-average
    """
    from tqdm.auto import tqdm

    def hma(close, length=None, offset=None, **kwargs):
        """Indicator: Hull Moving Average (HMA)

        Use help(df.ta.hma) for specific documentation where 'df' represents
        the DataFrame you are using.
        """
        # Validate Arguments
        close = verify_series(close)
        length = int(length) if length and length > 0 else 10
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
        offset = get_offset(offset)

        # Calculate Result
        half_length = int(length / 2)
        sqrt_length = int(math.sqrt(length))

        wmaf = wma(close=close, length=half_length)
        wmas = wma(close=close, length=length)
        hma = wma(close=2 * wmaf - wmas, length=sqrt_length)

        # Offset
        if offset != 0:
            hma = hma.shift(offset)

        # Name & Category
        hma.name = f"HMA_{length}"
        hma.category = 'overlap'

        return hma

    for interval in tqdm(intervals):
        df["hma_" + str(interval)] = hma(df['close'], length=interval)


def MACD(df, col_name, intervals):
    """
        Moving Average Convergence/divergence
        Reference: https://www.investopedia.com/terms/m/macd.asp
    """
    from ta.trend import macd_signal, macd
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df['macd_' + str(interval)] = macd_signal(df[col_name], n_fast=min(intervals), n_slow=max(intervals), n_sign=interval)


def TRIX(df, col_name, intervals):
    """
        Triple Exponentially Smoothed Moving Average
        Reference: https://www.investopedia.com/terms/t/trix.asp
    """
    from ta.trend import trix
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df['trix_' + str(interval)] = trix(df[col_name], interval, fillna=True)


def ADX(df, intervals):
    """
        Average Directional Index
    """
    from ta.trend import adx
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df['adx_' + str(interval)] = adx(df['high'], df['low'], df['close'], n=interval)


def PPO(df, col_name, intervals):
    """
        PPO: Percentage Price Oscillator
        Reference: https://www.investopedia.com/terms/p/ppo.asp
    """
    from tqdm.auto import tqdm

    def ppo(close, fast=None, slow=None, signal=None, offset=None, **kwargs):
        """Indicator: Percentage Price Oscillator (PPO)"""
        # Validate Arguments
        close = verify_series(close)
        fast = int(fast) if fast and fast > 0 else 12
        slow = int(slow) if slow and slow > 0 else 26
        signal = int(signal) if signal and signal > 0 else 9
        if slow < fast:
            fast, slow = slow, fast
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else fast
        offset = get_offset(offset)

        # Calculate Result
        fastma = close.rolling(fast, min_periods=min_periods).mean()
        slowma = close.rolling(slow, min_periods=min_periods).mean()

        ppo = 100 * (fastma - slowma) / slowma

        # Offset
        if offset != 0:
            ppo = ppo.shift(offset)
            signalma = signalma.shift(offset)
            histogram = histogram.shift(offset)

        # Handle fills
        if 'fillna' in kwargs:
            ppo.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            ppo.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        ppo.name = f"PPO_{fast}_{slow}_{signal}"
        ppo.category = 'momentum'

        return ppo

    for interval in tqdm(intervals):
        df['ppo_' + str(interval)] = ppo(close=df[col_name], fast=min(intervals), slow=max(intervals), signal=interval)


def PSI(df):
    from ta.trend import PSARIndicator

    df['psar'] = PSARIndicator(df['high'], df['low'], df['close']).psar()


def Vortex(df, intervals):
    from ta.trend import vortex_indicator_pos, vortex_indicator_neg
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df['vortex_gt_' + str(interval)] = (vortex_indicator_pos(df['high'], df['low'], df['close'], n=interval) >= vortex_indicator_neg(df['high'], df['low'], df['close'], n=interval))


def CCI(df, intervals):
    """
        Commodity Channel Index: Measures the current price level relative to an average price level over a given period of time.
        CCI is high when prices are far above their average and low when prices are below their average
        Reference: https://www.investopedia.com/articles/active-trading/031914/how-traders-can-utilize-cci-commodity-channel-index-trade-stock-trends.asp
    """
    from ta.trend import cci
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df['cci_' + str(interval)] = cci(df['high'], df['low'], df['close'], interval, fillna=True)


def BB_MAV(df, col_name, intervals):
    """
        Bollinger Bands Moving Average.
        Reference: https://www.investopedia.com/ask/answers/112814/how-do-i-create-trading-strategy-bollinger-bands-and-moving-averages.asp
    """
    from ta.volatility import bollinger_mavg
    from tqdm.auto import tqdm

    for interval in tqdm(intervals):
        df['bb_' + str(interval)] = bollinger_mavg(df[col_name], n=interval, fillna=True)


def CMO(df, col_name, intervals):
    """
        Chande Momentum Oscillator.  Indicated overbought levels when greater than 50, oversold levels when less than -50.
        Reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo
    """
    from tqdm.auto import tqdm

    def calculate_CMO(series):
        sum_gains = series[series>=0].sum()
        sum_losses = np.abs(series<0).sum()
        cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
        return np.round(cmo, 3)

    diff = df[col_name].diff()[1:]

    for interval in tqdm(intervals):
        df['cmo_' + str(interval)] = np.nan
        res = diff.rolling(interval).apply(calculate_CMO, args=(), raw=False)
        df['cmo_' + str(interval)][1:] = res


def chande_momentum_oscillator(close_data, period):
    """
    Chande Momentum Oscillator.
    Formula:
    cmo = 100 * ((sum_up - sum_down) / (sum_up + sum_down))
    # Reference https://github.com/kylejusticemagnuson/pyti/blob/master/pyti/chande_momentum_oscillator.py
    """
    moving_period_diffs = [[(close_data[idx+1-period:idx+1][i] -
                 close_data[idx+1-period:idx+1][i-1]) for i in range(1, len(close_data[idx+1-period:idx+1]))] for idx in range(0, len(close_data))]

    sum_up = []
    sum_down = []
    for period_diffs in moving_period_diffs:
        ups = [val if val > 0 else 0 for val in period_diffs]
        sum_up.append(sum(ups))
        downs = [abs(val) if val < 0 else 0 for val in period_diffs]
        sum_down.append(sum(downs))

    sum_up = np.array(sum_up)
    sum_down = np.array(sum_down)
    # numpy is able to handle dividing by zero and makes those calculations
    return 100 * ((sum_up - sum_down) / (sum_up + sum_down))


def ROC(df, col_name, intervals):
    """
        Rate of Change = ((close_price_n - close_price_(n-1)) / close_price_(n-1)) * 100
        Positive values indicate buying pressure while negative values indicate selling pressure
        Reference: https://www.investopedia.com/terms/p/pricerateofchange.asp
                   https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum
    """
    from tqdm.auto import tqdm
    from ta.momentum import roc

    for interval in tqdm(intervals):
        df['roc_' + str(interval)] = roc(df[col_name], n=interval, fillna=True)


def DPO(df, col_name, intervals):
    """
        Detrended Price Oscillator
        DPO is positive when price is above displaced moving average.
        DPO is negative when price is below displaced moving average.
        Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:detrended_price_osci
    """
    from tqdm.auto import tqdm
    from ta.trend import dpo

    for interval in tqdm(intervals):
        df['dpo_' + str(interval)] = dpo(df[col_name], n=interval)


def KST(df, col_name, intervals):
    """
        Know Sure Thing
        Reference: https://www.investopedia.com/terms/k/know-sure-thing-kst.asp
    """
    from tqdm.auto import tqdm
    from ta.trend import kst

    for interval in tqdm(intervals):
        df['kst_' + str(interval)] = kst(df[col_name], interval)


def CMF(df, intervals):
    """
        Chaikin Money Flow
        A CMF above the money flow is a sign of strength in the market and a value below the zero line is a sign of weakness in the market.
        Reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmf
    """
    from tqdm.auto import tqdm
    from ta.volume import chaikin_money_flow

    for interval in tqdm(intervals):
        df['cmf_' + str(interval)] = chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], n=interval, fillna=True)


def ForceIndex(df, intervals):
    """
        Force Index
        Key Takeaways from Reference: https://www.investopedia.com/terms/f/force-index.asp
            - A rising force index, above zero, helps confirm rising prices.
            - A falling force index, below zero, helps confirm falling prices.
            - A breakout, or a spike, in the force index, helps confirm a breakout in price.
            - If the force index is making lower swing highs while the price is making higher swing highs, this is bearish divergence and warns the price may soon decline.
            - If the force index is making higher swing lows while the price is making lower swing lows, this is bullish divergence and warns the price may soon head higher.
            - The force index is typically 13 periods but this can be adjusted based on preference. The more periods used the smoother the movements of the index, typically preferred by longer-term traders
    """
    from tqdm.auto import tqdm
    from ta.volume import force_index

    for interval in tqdm(intervals):
        df['fi_' + str(interval)] = force_index(df['close'], df['volume'], n=interval, fillna=True)


def EOM(df, intervals):
    """
        Ease of Movement
        Key Takeaways from Reference: https://www.investopedia.com/terms/e/easeofmovement.asp
            - This indicator calculates how easily a price can move up or down.
            - The calculation subtracts yesterday's average price from today's average price and divides the difference by volume.
            - This generates a volume-weighted momentum indicator.
    """
    from tqdm.auto import tqdm
    from ta.volume import ease_of_movement

    for interval in tqdm(intervals):
        df['eom_' + str(interval)] = ease_of_movement(df['high'], df['low'], df['volume'], n=interval, fillna=True)


if __name__ == "__main__":
    test_data = retrieve_yahoo_fin_stock_data(ticker='AMZN')
    engineer_data_over_single_interval(test_data)

    for column in test_data.columns:
        print(column, test_data[column].tail())

