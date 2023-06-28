
import numpy as np
import pandas as pd

from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
import matplotlib.pyplot as plt
import seaborn as sns 



#### Generate New Demand ####


df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()
state = State()
state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# Bayesian
# print('r1', state.run(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117))
bayesian = state.test_no_season(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
bayesian_poisson = state.test_poisson_no_season(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
bayesian_24 = state.test_no_season_24_period(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)


# GA
# print('r1', state.run(54, 63, 42, 47, 42, 49))
ga = state.test_no_season(54, 63, 42, 47, 42, 49)
ga_poisson = state.test_poisson_no_season(54, 63, 42, 47, 42, 49)
ga_24 = state.test_no_season_24_period(54, 63, 42, 47, 42, 49)

# OLS
# print('r1', state.run(36, 44, 41, 48, 34, 38))
ols = state.test_no_season(36, 44, 41, 48, 34, 38)
ols_poisson = state.test_poisson_no_season(36, 44, 41, 48, 34, 38)
ols_24 = state.test_no_season_24_period(36, 44, 41, 48, 34, 38)

ml = state.test_no_season(37, 41, 142, 149, 32, 35)
ml_poisson = state.test_poisson_no_season(37, 41, 142, 149, 32, 35)
ml_24 = state.test_no_season_24_period(37, 41, 142, 149, 32, 35)


# print(bayesian)
# print(ga)
# print(ols)
# 

rl = [103306.59999999995, 103306.59999999995, 106024.19999999992, 106024.19999999992, 106358.60000000002, 106358.60000000002, 97838.19999999997, 97838.19999999997, 105364.79999999997, 105364.79999999997, 105867.79999999996, 105867.79999999996, 105881.39999999998, 105881.39999999998, 107143.0, 107143.0, 107008.59999999999, 107008.59999999999, 109366.99999999999, 109366.99999999999, 103515.39999999997, 103515.39999999997, 101220.80000000003, 101220.80000000003, 106888.99999999997, 106888.99999999997, 102212.20000000001, 102212.20000000001, 94807.80000000002, 94807.80000000002, 107753.40000000007, 107753.40000000007, 105458.19999999998, 105458.19999999998, 101484.59999999995, 101484.59999999995, 106799.80000000006, 106799.80000000006, 104089.2, 104089.2, 107007.2, 107007.2, 101951.60000000005, 101951.60000000005, 104897.79999999994, 104897.79999999994, 105934.20000000001, 105934.20000000001, 104956.40000000001, 104956.40000000001, 99209.99999999999, 99209.99999999999, 108945.99999999996, 108945.99999999996, 107935.2, 107935.2, 103484.60000000003, 103484.60000000003, 103621.39999999995, 103621.39999999995, 102308.2, 102308.2, 97526.00000000001, 97526.00000000001, 108908.59999999996, 108908.59999999996, 104996.80000000002, 104996.80000000002, 108274.20000000003, 108274.20000000003, 107596.59999999998, 107596.59999999998, 104534.40000000002, 104534.40000000002, 99622.0, 99622.0, 104004.8, 104004.8, 108811.00000000003, 108811.00000000003, 98176.40000000001, 98176.40000000001, 105887.80000000003, 105887.80000000003, 96839.40000000004, 96839.40000000004, 101092.99999999994, 101092.99999999994, 102493.99999999999, 102493.99999999999, 95737.2, 95737.2, 106137.39999999998, 106137.39999999998, 104519.99999999997, 104519.99999999997, 106487.2, 106487.2, 92182.40000000002, 92182.40000000002, 104279.39999999998, 104279.39999999998, 107517.40000000002, 107517.40000000002, 103932.20000000006, 103932.20000000006, 101698.59999999999, 101698.59999999999, 99165.00000000001, 99165.00000000001, 106863.79999999999, 106863.79999999999, 104949.59999999998, 104949.59999999998, 108678.40000000005, 108678.40000000005, 105738.79999999994, 105738.79999999994, 94583.80000000002, 94583.80000000002, 91471.79999999999, 91471.79999999999, 71467.2, 71467.2, 103684.40000000004, 103684.40000000004, 103942.19999999997, 103942.19999999997, 110735.20000000003, 110735.20000000003, 100527.59999999999, 100527.59999999999, 98635.2, 98635.2, 105145.99999999999, 105145.99999999999, 108694.00000000003, 108694.00000000003, 105973.8, 105973.8, 109023.39999999997, 109023.39999999997, 104564.19999999998, 104564.19999999998, 98750.59999999999, 98750.59999999999, 102099.99999999997, 102099.99999999997, 100742.20000000001, 100742.20000000001, 104609.99999999997, 104609.99999999997, 99407.8, 99407.8, 103054.19999999994, 103054.19999999994, 101005.39999999997, 101005.39999999997, 99273.59999999999, 99273.59999999999, 107119.59999999996, 107119.59999999996, 108203.39999999997, 108203.39999999997, 104133.80000000002, 104133.80000000002, 103774.40000000002, 103774.40000000002, 107485.4, 107485.4, 93279.79999999994, 93279.79999999994, 105722.79999999996, 105722.79999999996, 107895.60000000003, 107895.60000000003, 107522.39999999997, 107522.39999999997, 105615.6, 105615.6, 104625.19999999995, 104625.19999999995, 106015.39999999997, 106015.39999999997, 105763.40000000001, 105763.40000000001, 102638.40000000002, 102638.40000000002, 106536.99999999997, 106536.99999999997, 103524.39999999998, 103524.39999999998, 100511.40000000001, 100511.40000000001, 91812.20000000003, 91812.20000000003, 107414.59999999996, 107414.59999999996, 105924.99999999997, 105924.99999999997, 106109.59999999999, 106109.59999999999, 93900.4, 93900.4, 102893.99999999999, 102893.99999999999, 105485.79999999999, 105485.79999999999, 107818.6, 107818.6, 106256.19999999995, 106256.19999999995, 102981.6, 102981.6, 102754.40000000004, 102754.40000000004, 106569.80000000003, 106569.80000000003, 106771.79999999999, 106771.79999999999, 92432.79999999997, 92432.79999999997, 107899.60000000005, 107899.60000000005, 108373.99999999997, 108373.99999999997, 102175.59999999999, 102175.59999999999, 90916.4, 90916.4, 106138.59999999996, 106138.59999999996, 102299.80000000002, 102299.80000000002, 99350.8, 99350.8, 108210.79999999997, 108210.79999999997, 106333.99999999996, 106333.99999999996, 97941.19999999995, 97941.19999999995, 105574.80000000002, 105574.80000000002, 98816.99999999997, 98816.99999999997, 107140.40000000001, 107140.40000000001, 102839.19999999997, 102839.19999999997, 98953.79999999996, 98953.79999999996, 108652.60000000002, 108652.60000000002, 105902.99999999994, 105902.99999999994, 99360.79999999997, 99360.79999999997, 107239.59999999996, 107239.59999999996, 103665.6, 103665.6, 102850.00000000003, 102850.00000000003, 97994.60000000003, 97994.60000000003, 106593.20000000008, 106593.20000000008, 105719.79999999999, 105719.79999999999, 100407.39999999995, 100407.39999999995, 105392.8, 105392.8, 85556.19999999998, 85556.19999999998, 106320.2, 106320.2, 103111.39999999998, 103111.39999999998, 105923.0, 105923.0, 107335.2, 107335.2, 103769.00000000001, 103769.00000000001, 106895.59999999999, 106895.59999999999, 89276.0, 89276.0, 107757.4, 107757.4, 109458.4, 109458.4, 98692.40000000001, 98692.40000000001, 108808.0, 108808.0, 104858.19999999997, 104858.19999999997, 107472.0, 107472.0, 106360.79999999994, 106360.79999999994, 98671.20000000003, 98671.20000000003, 96265.79999999997, 96265.79999999997, 104930.6, 104930.6, 104272.20000000003, 104272.20000000003, 105065.20000000001, 105065.20000000001, 96870.40000000001, 96870.40000000001, 104923.20000000001, 104923.20000000001, 103751.99999999999, 103751.99999999999, 102558.39999999997, 102558.39999999997, 90974.19999999997, 90974.19999999997, 110554.00000000004, 110554.00000000004, 99515.80000000002, 99515.80000000002, 105996.0, 105996.0, 107319.80000000006, 107319.80000000006, 105508.99999999999, 105508.99999999999, 102275.8, 102275.8, 99780.00000000001, 99780.00000000001, 102532.80000000005, 102532.80000000005, 105512.80000000005, 105512.80000000005, 103211.40000000001, 103211.40000000001, 101514.8, 101514.8, 97066.99999999997, 97066.99999999997, 104844.99999999997, 104844.99999999997, 106072.99999999999, 106072.99999999999, 104394.4, 104394.4, 103939.0, 103939.0, 106728.20000000001, 106728.20000000001, 107501.20000000001, 107501.20000000001, 105506.40000000002, 105506.40000000002, 104119.8, 104119.8, 103208.2, 103208.2, 104701.39999999997, 104701.39999999997, 96620.80000000002, 96620.80000000002, 102964.60000000003, 102964.60000000003, 102896.19999999998, 102896.19999999998, 105721.59999999998, 105721.59999999998, 105101.40000000002, 105101.40000000002, 106060.4, 106060.4, 105872.80000000002, 105872.80000000002, 106453.8, 106453.8, 102768.60000000003, 102768.60000000003, 103198.40000000002, 103198.40000000002, 95415.80000000003, 95415.80000000003, 100029.79999999999, 100029.79999999999, 102576.60000000002, 102576.60000000002, 106681.59999999996, 106681.59999999996, 106613.80000000006, 106613.80000000006, 107322.39999999998, 107322.39999999998, 98305.39999999998, 98305.39999999998, 96488.2, 96488.2, 103949.99999999994, 103949.99999999994, 103935.0, 103935.0, 102512.20000000004, 102512.20000000004, 106311.00000000001, 106311.00000000001, 101760.59999999996, 101760.59999999996, 105644.79999999999, 105644.79999999999, 86151.8, 86151.8, 109804.6, 109804.6, 104582.60000000005, 104582.60000000005, 101655.00000000003, 101655.00000000003, 100919.0, 100919.0, 106998.19999999992, 106998.19999999992, 109944.40000000004, 109944.40000000004, 105937.8, 105937.8, 104992.00000000001, 104992.00000000001, 107380.0, 107380.0, 105871.40000000002, 105871.40000000002, 108037.6, 108037.6, 109247.4, 109247.4, 104042.39999999997, 104042.39999999997, 104084.40000000001, 104084.40000000001, 102325.79999999999, 102325.79999999999, 108331.39999999997, 108331.39999999997, 101581.4, 101581.4, 107266.99999999999, 107266.99999999999, 101262.8, 101262.8, 106759.6, 106759.6, 99663.60000000002, 99663.60000000002, 102823.79999999993, 102823.79999999993, 105694.0, 105694.0, 102452.20000000001, 102452.20000000001, 102835.20000000003, 102835.20000000003, 104207.79999999996, 104207.79999999996, 105111.79999999999, 105111.79999999999, 104309.79999999999, 104309.79999999999, 97557.59999999998, 97557.59999999998, 101693.4, 101693.4, 104546.2, 104546.2, 106685.99999999999, 106685.99999999999, 99636.80000000003, 99636.80000000003, 107141.39999999995, 107141.39999999995, 106713.19999999998, 106713.19999999998, 103997.20000000001, 103997.20000000001, 106858.59999999998, 106858.59999999998, 105957.20000000004, 105957.20000000004, 102322.40000000001, 102322.40000000001, 92512.99999999997, 92512.99999999997, 97693.59999999998, 97693.59999999998, 105459.79999999996, 105459.79999999996, 103996.79999999996, 103996.79999999996, 93530.99999999997, 93530.99999999997, 99824.0, 99824.0, 102611.99999999996, 102611.99999999996, 108723.4, 108723.4, 105143.4, 105143.4, 103693.60000000002, 103693.60000000002, 105416.19999999997, 105416.19999999997, 110779.99999999999, 110779.99999999999, 103339.60000000002, 103339.60000000002, 108701.79999999994, 108701.79999999994, 105671.80000000002, 105671.80000000002, 103623.2, 103623.2, 101792.79999999997, 101792.79999999997, 107047.79999999999, 107047.79999999999, 103340.39999999998, 103340.39999999998, 104063.4, 104063.4, 97489.2, 97489.2, 104614.19999999998, 104614.19999999998, 108663.6, 108663.6, 109896.80000000002, 109896.80000000002, 99303.60000000003, 99303.60000000003, 104398.40000000001, 104398.40000000001, 105478.99999999997, 105478.99999999997, 104507.8, 104507.8, 102995.2, 102995.2, 104965.99999999997, 104965.99999999997, 103798.80000000006, 103798.80000000006, 96788.80000000002, 96788.80000000002, 107645.20000000001, 107645.20000000001, 109719.79999999994, 109719.79999999994, 102259.39999999997, 102259.39999999997, 104531.2, 104531.2, 93755.79999999999, 93755.79999999999, 106670.4, 106670.4, 103870.39999999998, 103870.39999999998, 105121.6, 105121.6, 106193.40000000004, 106193.40000000004, 107654.59999999993, 107654.59999999993, 109098.40000000002, 109098.40000000002, 104337.80000000002, 104337.80000000002, 105724.19999999998, 105724.19999999998, 106679.39999999998, 106679.39999999998, 108391.40000000001, 108391.40000000001, 110936.19999999995, 110936.19999999995, 107744.59999999995, 107744.59999999995, 105165.59999999999, 105165.59999999999, 105540.20000000001, 105540.20000000001, 106419.79999999996, 106419.79999999996, 102605.20000000006, 102605.20000000006, 110779.99999999999, 110779.99999999999, 106259.00000000003, 106259.00000000003, 105229.8, 105229.8, 105817.40000000002, 105817.40000000002, 106343.2, 106343.2, 104569.59999999999, 104569.59999999999, 107160.60000000002, 107160.60000000002, 114197.6, 114197.6, 106402.79999999996, 106402.79999999996, 104054.00000000003, 104054.00000000003, 110348.40000000004, 110348.40000000004, 105933.19999999997, 105933.19999999997, 99639.19999999998, 99639.19999999998, 96034.19999999998, 96034.19999999998, 106464.99999999996, 106464.99999999996, 104172.99999999994, 104172.99999999994, 106408.00000000001, 106408.00000000001, 104505.2, 104505.2, 102950.19999999998, 102950.19999999998, 102417.40000000002, 102417.40000000002, 99897.59999999998, 99897.59999999998, 104680.19999999998, 104680.19999999998, 107063.99999999997, 107063.99999999997, 103939.59999999996, 103939.59999999996, 104299.99999999999, 104299.99999999999, 104167.39999999998, 104167.39999999998, 104651.79999999999, 104651.79999999999, 88556.20000000004, 88556.20000000004, 110782.40000000002, 110782.40000000002, 101798.19999999997, 101798.19999999997, 106127.59999999996, 106127.59999999996, 97725.79999999999, 97725.79999999999, 103835.80000000003, 103835.80000000003, 106196.2, 106196.2, 108395.8, 108395.8, 107939.20000000004, 107939.20000000004, 108669.2, 108669.2, 104706.79999999999, 104706.79999999999, 106231.20000000001, 106231.20000000001, 107522.8, 107522.8, 107727.80000000002, 107727.80000000002, 101852.19999999998, 101852.19999999998, 105871.99999999996, 105871.99999999996, 105691.4, 105691.4, 103048.99999999997, 103048.99999999997, 106385.20000000003, 106385.20000000003, 109291.19999999995, 109291.19999999995, 107780.59999999996, 107780.59999999996, 110371.60000000003, 110371.60000000003, 105721.59999999998, 105721.59999999998, 109963.0, 109963.0, 106562.19999999995, 106562.19999999995, 100286.19999999997, 100286.19999999997, 99867.00000000003, 99867.00000000003, 107695.20000000007, 107695.20000000007, 97142.59999999999, 97142.59999999999, 103660.59999999996, 103660.59999999996, 103827.79999999997, 103827.79999999997, 105383.4, 105383.4, 99934.60000000002, 99934.60000000002, 102350.20000000001, 102350.20000000001, 105067.2, 105067.2, 100698.79999999997, 100698.79999999997, 100031.19999999998, 100031.19999999998, 103433.39999999997, 103433.39999999997, 104583.99999999997, 104583.99999999997, 108602.99999999999, 108602.99999999999, 102310.39999999997, 102310.39999999997, 103126.19999999995, 103126.19999999995, 100752.99999999997, 100752.99999999997, 108478.19999999997, 108478.19999999997, 104393.19999999997, 104393.19999999997, 107339.19999999998, 107339.19999999998, 106048.0, 106048.0, 105832.60000000003, 105832.60000000003, 101611.39999999997, 101611.39999999997, 104978.99999999999, 104978.99999999999, 91011.79999999999, 91011.79999999999, 106210.99999999999, 106210.99999999999, 107390.40000000005, 107390.40000000005, 105638.6, 105638.6, 107358.80000000005, 107358.80000000005, 101057.39999999997, 101057.39999999997, 103304.80000000003, 103304.80000000003, 97995.20000000001, 97995.20000000001, 100577.20000000003, 100577.20000000003, 110903.40000000002, 110903.40000000002, 107203.6, 107203.6, 103973.00000000003, 103973.00000000003, 105936.79999999994, 105936.79999999994, 105492.4, 105492.4, 105557.40000000001, 105557.40000000001, 100748.59999999996, 100748.59999999996, 104184.00000000001, 104184.00000000001, 105277.80000000002, 105277.80000000002, 101077.79999999999, 101077.79999999999, 106144.4, 106144.4, 106442.00000000003, 106442.00000000003, 100194.59999999996, 100194.59999999996, 103747.6, 103747.6, 105677.8, 105677.8, 90438.79999999997, 90438.79999999997, 101214.60000000006, 101214.60000000006, 106129.20000000006, 106129.20000000006, 105050.20000000003, 105050.20000000003, 98100.19999999998, 98100.19999999998, 101008.59999999995, 101008.59999999995, 106961.79999999997, 106961.79999999997, 105554.39999999997, 105554.39999999997, 104870.79999999997, 104870.79999999997, 106337.00000000003, 106337.00000000003, 99568.6, 99568.6, 99732.8, 99732.8, 106400.99999999997, 106400.99999999997, 104130.80000000003, 104130.80000000003, 106734.80000000003, 106734.80000000003, 104094.59999999996, 104094.59999999996, 102157.40000000001, 102157.40000000001, 104236.40000000004, 104236.40000000004, 102051.0, 102051.0, 91685.59999999999, 91685.59999999999, 103932.2, 103932.2, 106826.99999999999, 106826.99999999999, 105387.40000000004, 105387.40000000004, 96182.59999999999, 96182.59999999999, 105409.59999999998, 105409.59999999998, 103094.99999999997, 103094.99999999997, 108400.00000000004, 108400.00000000004, 102795.80000000002, 102795.80000000002, 102971.4, 102971.4, 104474.20000000001, 104474.20000000001, 106314.80000000003, 106314.80000000003, 108088.19999999998, 108088.19999999998, 103442.40000000002, 103442.40000000002, 108020.80000000002, 108020.80000000002, 111303.19999999997, 111303.19999999997, 98092.19999999998, 98092.19999999998, 103862.79999999997, 103862.79999999997, 103910.6, 103910.6, 103266.00000000003, 103266.00000000003, 107271.60000000003, 107271.60000000003, 97230.59999999999, 97230.59999999999, 104237.6, 104237.6, 97221.20000000003, 97221.20000000003, 96617.6, 96617.6, 99022.20000000001, 99022.20000000001, 106216.40000000002, 106216.40000000002, 108348.40000000007, 108348.40000000007, 100175.59999999995, 100175.59999999995, 103298.8, 103298.8, 101564.2, 101564.2, 110223.00000000003, 110223.00000000003, 102412.59999999996, 102412.59999999996, 86024.8, 86024.8, 100451.40000000001, 100451.40000000001, 103538.59999999998, 103538.59999999998, 99667.59999999998, 99667.59999999998, 100212.40000000001, 100212.40000000001, 103586.59999999995, 103586.59999999995, 106579.59999999999, 106579.59999999999, 108519.99999999993, 108519.99999999993, 104196.59999999998, 104196.59999999998, 107763.40000000002, 107763.40000000002, 111002.60000000002, 111002.60000000002, 108367.80000000002, 108367.80000000002, 105705.59999999996, 105705.59999999996, 103504.59999999996, 103504.59999999996, 100235.0, 100235.0, 96480.40000000001, 96480.40000000001, 104064.59999999999, 104064.59999999999, 108732.40000000001, 108732.40000000001, 111710.60000000005, 111710.60000000005, 95594.8, 95594.8, 109503.00000000001, 109503.00000000001, 99943.19999999995, 99943.19999999995, 102846.80000000002, 102846.80000000002, 105984.39999999994, 105984.39999999994, 104362.00000000001, 104362.00000000001, 96264.0, 96264.0, 101435.40000000001, 101435.40000000001, 106462.99999999996, 106462.99999999996, 104725.80000000002, 104725.80000000002, 106885.40000000001, 106885.40000000001, 105363.00000000009, 105363.00000000009, 92449.2, 92449.2, 103429.00000000003, 103429.00000000003, 108660.20000000001, 108660.20000000001, 107593.39999999998, 107593.39999999998, 109618.80000000008, 109618.80000000008, 104480.2, 104480.2, 98711.80000000002, 98711.80000000002, 104315.6, 104315.6, 105847.40000000004, 105847.40000000004, 97776.59999999998, 97776.59999999998, 107760.39999999998, 107760.39999999998, 102237.39999999992, 102237.39999999992, 105082.59999999999, 105082.59999999999, 99031.99999999997, 99031.99999999997, 106215.4, 106215.4]

# ax = sns.boxplot(data=[bayesian_poisson, ga_poisson, ols_poisson, rl, ml_poisson])
# ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML'])
# ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods, mean = {}', round(mean, 2)))
# plt.show()

print(np.mean(bayesian))
print(np.mean(ga))
print(np.mean(ols))
print(np.mean(rl))
print(np.mean(ml))

ax = sns.boxplot(data=[bayesian, ga, ols, rl, ml])
ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML'])
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods, mean = {}, std = {}', round(mean, 2), round(std, 2)))
plt.show()


# ax = sns.boxplot(data=[bayesian_24, ga_24, ols_24, rl, ml_24])
# ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML'])
# ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 24 periods, mean = {}, std = {}', round(mean, 2), round(std, 2)))
# plt.show()
