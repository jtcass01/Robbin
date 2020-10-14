from robin_stocks import login, order_buy_market, order_sell_market
from json import load
from os import getcwd
from os.path import join


class RobinhoodInterface(object):
    def __init__(self, username: str, password: str, current_positions_json_path: str = ""):
        login(username=username, password=password)

        # load predescribed current positions.
        if current_positions_json_path == "":
            current_positions_json_path = join(getcwd(), "..", "History", "current_positions.json")
        with open(current_positions_json_path) as current_positions_file:
            self.current_positions = load(fp=current_positions_file)

    @staticmethod
    def buy_shares_at_market_price(symbol: str, quantity: int) -> None:
        order_buy_market(symbol=symbol, quantity=quantity)

    @staticmethod
    def sell_shares_at_market_price(symbol: str, quantity: int) -> None:
        order_sell_market(symbol=symbol, quantity=quantity)


if __name__ == "__main__":
    robinhood_account_info_json_path = join(getcwd(), "..", "AccountKeys", "robinhood.json")

    with open(robinhood_account_info_json_path) as robinhood_account_info_file:
        robinhood_account_info = load(fp=robinhood_account_info_file)

    test_interface = RobinhoodInterface(username=robinhood_account_info['username'],
                                        password=robinhood_account_info['password'])

    print(test_interface.current_positions)
