from zipline.api import order, record, symbol

def initialize(context):
    pass

def handle_data(context, data):
    order(symbol('SPY'), 10)
    record(SPY=data.current(symbol('SPY'), 'price'))