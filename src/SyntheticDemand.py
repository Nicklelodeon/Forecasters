import math


class GenerateDemand:
    def __init__(self):
        self.demand = []

    def simulate(self, year):
        for z in range(year):
            for i in range(1, 13):
                # Estimate 70% demand of 2778 (1945) in months 1-3, 5-6, 10
                if i in [1, 2, 3, 5, 6, 10]:
                    demand.extend(
                        [round(x) for x in np.random.normal(1945, math.sqrt(1945), 30)]
                    )
                # Estimate 130% demand of 2778 (3611) in other months (peak months)
                else:
                    demand.extend(
                        [round(x) for x in np.random.normal(3611, math.sqrt(3611), 30)]
                    )
