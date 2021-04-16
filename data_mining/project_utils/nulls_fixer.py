import numpy as np
import pandas as pd
from data_mining import ColorizedLogger

logger = ColorizedLogger('NullsFixer', 'white')


class NullsFixer:

    def __init__(self):
        pass

    def fix_people_fully_vaccinated(self, df: pd.DataFrame):
        # people_fully_vaccinated = total_vaccines - people_vaccinated
        # If total_vaccines==0 -> people_fully_vaccinated = 0.0
        pass

    def fix_people_vaccinated(self, df: pd.DataFrame):
        # people_vaccinated = total_vaccines - people_fully_vaccinated
        # If total_vaccines==0 -> people_vaccinated = 0.0
        pass

    def fix_total_vaccines(self, df: pd.DataFrame):
        # total_vaccines = people_vaccinated + people_fully_vaccinated
        # total_vaccines = previous_total_vaccines + daily_vaccinations
        # total_vaccines = next_total_vaccines - next_daily_vaccinations
        pass

    def fix_daily_vaccinations(self, df: pd.DataFrame):
        # daily_vaccinations = total_vaccines - previous_total_vaccines
        pass
