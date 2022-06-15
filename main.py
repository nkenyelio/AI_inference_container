# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from configs.config import CFG
from model.unet import MHet


def run():
    """Builds model, loads data, trains and evaluates"""
    model = MHet(CFG)
    model.load_data()
    model.build()
    #model.train()
    #model.evaluate()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

