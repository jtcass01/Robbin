import os
from string import ascii_uppercase
from utilities.Logger import Logger
from pandas import DataFrame


def survey_market():
    from yahoo_fin.stock_info import get_data

    Logger.console_log("Beginning survey of all possible stock symbols", 1)

    def concatenate_letters(one, two, three, four, five):
        if letter_four is None:
            return letter_one + letter_two + letter_three
        elif five is None:
            return letter_one + letter_two + letter_three + letter_four
        else:
            return letter_one + letter_two + letter_three + letter_four + letter_five

    def log_valid_stock_symbol(symbol):
        log_location = os.getcwd() + os.path.sep + "Data" + os.path.sep + "valid_stocks.txt"

        with open(log_location, 'a+') as stock_log:
            stock_log.write(symbol + '\n')

    def survey_symbol(symbol):
        try:
            try:
                data = get_data(symbol)
#                Logger.console_log("Stock for symbol {} exists.".format(symbol), 1)
                log_valid_stock_symbol(symbol)
            except KeyError:
#                Logger.console_log("Stock for symbol {} produced KeyError on adjclose.".format(symbol), 4)
                pass
            except ValueError:
                Logger.console_log("ValueError when attempting to retrieve data for stock symbol {}. Retrying...".format(symbol), Logger.LogStatus.FAIL)
                survey_symbol(symbol)
        except AssertionError:
#            Logger.console_log("Stock for symbol {} does not exist.".format(symbol), 2)
            pass

    ascii_uppercase_and_none = list(ascii_uppercase)
    ascii_uppercase_and_none.append(None)

    for letter_one in ascii_uppercase[2:]:
        for letter_two in ascii_uppercase:
            for letter_three in ascii_uppercase:
                for letter_four in ascii_uppercase_and_none:
                    if letter_four is None:
                        three_letter_symbol = concatenate_letters(letter_one, letter_two, letter_three, letter_four, None)
                        survey_symbol(three_letter_symbol)
                    else:
                        for letter_five in ascii_uppercase_and_none:
                            four_or_five_letter_symbol = concatenate_letters(letter_one, letter_two, letter_three, letter_four, letter_five)
                            survey_symbol(four_or_five_letter_symbol)


def retrieve_yahoo_fin_stock_data(ticker: str) -> DataFrame:
    from yahoo_fin.stock_info import get_data

    df = get_data(ticker)

    df = df.drop('ticker', axis=1)

    return df


if __name__ == "__main__":
    survey_market()
