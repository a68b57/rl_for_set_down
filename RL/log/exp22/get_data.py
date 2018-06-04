import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


# sns.set(color_codes=True)
sns.set(style="darkgrid")


#### for overlapped histo#########
# sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
#
# dt_23_7_1 = pd.read_json('following_23.7+1000000_log.csv')
# dt_23_7_2 = pd.read_json('following_23.7+1000000+1000000_log.csv')
# dt_23_7_3 = pd.read_json('following_23.7+1000000+1000000+1000000_log.csv')
#
# d1 = np.array(dt_23_7_1['mean_q'])
# d2 = np.array(dt_23_7_2['mean_q'])
# d3 = np.array(dt_23_7_3['mean_q'])
#
# print(d1)
# print(d2)
# print(d3)

# test_all_data = np.concatenate((d1, d2, d3, d4, d5, d6, d7))
# test_group = np.concatenate((10*np.ones(len(d1)), 30*np.ones(len(d2)), 50*np.ones(len(d3)), 70*np.ones(len(d4)), 90*np.ones(len(d5)),110*np.ones(len(d6)), 130*np.ones(len(d7))))
# df = pd.DataFrame(dict(g=test_group.astype(np.int), impact_velocity=test_all_data))
#
# pal = sns.cubehelix_palette(10, rot=-.25, light=.4)
# g = sns.FacetGrid(df, row="g", hue="g", aspect=2.5, size=1, palette=pal)
#
# g.map(sns.kdeplot, "impact_velocity", clip_on=False, shade=True, alpha=0.5, lw=1.5, bw=.2)
# g.map(sns.kdeplot, "impact_velocity", clip_on=False, color="w", lw=2, bw=.2)
# g.map(plt.axhline, y=0, lw=2, clip_on=False)
#
#
# def label(x, color, label):
#     ax = plt.gca()
#     ax.text(-.15, 0.1, label+'k steps', fontweight="bold", color=color,
#             ha="left", va="center", transform=ax.transAxes)
#
# g.map(label, "impact_velocity")
# g.fig.subplots_adjust(hspace=-0.4)
#
# g.set_titles("")
# g.set(yticks=[])
# g.despine(bottom=True, left=True)
# plt.xlim(-0.1,1)
# plt.show()
####################

##################### overlapped histros in one figure


d1 = np.array([0.0034801487405911,
0.00556327138283575,
0.00693477226510364,
0.00721134744841034,
0.0073186284912552,
0.00778479330981075,
0.00984981228886461,
0.0101022394771766,
0.0104603464071884,
0.0105395377692785,
0.0106553699072354,
0.0107445270574091,
0.0131360652662771,
0.014163457001648,
0.0147885909956091,
0.0157049813628563,
0.0158680009362877,
0.0158695429697575,
0.0158821972140322,
0.0161894742210444,
0.0165138847984103,
0.0171041703300601,
0.0174579398997921,
0.0178342975466084,
0.018219714611849,
0.0184802445030741,
0.0186764232515024,
0.0186853682055199,
0.0199220111442155,
0.0204052293313817,
0.0205212167933988,
0.0206203971079688,
0.020778216667785,
0.0209403683090814,
0.0211047009806897,
0.0220974559743148,
0.0226052172143465,
0.0228146394986561,
0.0230905149499217,
0.0231008393606968,
0.0242687668809838,
0.024736416325486,
0.0248171604084346,
0.0250315141466473,
0.0259728793254199,
0.0264629824033813,
0.0268887082997704,
0.0280505748950644,
0.0296184227824625,
0.0296788263159531,
0.0297694467536358,
0.030327553693974,
0.0304133855503741,
0.0304999511425219,
0.0306221180644917,
0.0311967506745825,
0.0316433941139671,
0.032763708182757,
0.0328739455586602,
0.033292390334716,
0.033584419672894,
0.0339292013150549,
0.0340102792668828,
0.0340778590866853,
0.0345397920372736,
0.0348894062891514,
0.0356309491884987,
0.0359065233313993,
0.0362191191093686,
0.0371934471581836,
0.0378993011144946,
0.037980785920646,
0.0383698768665619,
0.0383736545040403,
0.0384048114640478,
0.0387905282962642,
0.0392559931412739,
0.0392704098434571,
0.0398953491865717,
0.0401731543646466,
0.0403041463697296,
0.0405010121942917,
0.0405785534977454,
0.0410143780546601,
0.041280624818385,
0.0416668375755069,
0.0420313664183736,
0.0421639667948837,
0.0426713320688688,
0.0428963129052029,
0.0429197062672149,
0.0430361435267202,
0.0432467653111557,
0.0434446660150201,
0.0436505450602764,
0.0438585860581631,
0.0439698285439816,
0.0450765087660443,
0.0458932779979682,
0.0460030199454176,
0.0467073289254705,
0.0471793057232883,
0.04735743363113,
0.0478260569343192,
0.0480103861101489,
0.0480497358057397,
0.0482935871137968,
0.0483190396467537,
0.049274259594001,
0.0497999108264624,
0.0498106193872516,
0.0500334089899956,
0.0503764064981604,
0.0503859204646728,
0.0511084091580027,
0.0515507702993734,
0.0521632753129175,
0.0521740631075973,
0.0531406868042716,
0.0542655863637842,
0.0543956691781222,
0.0547733818090723,
0.0555558765967579,
0.055725716460171,
0.0559883522204974,
0.0560715235866072,
0.0560912135892311,
0.0565414538490261,
0.0570305140502736,
0.0571595421801785,
0.0575839239033771,
0.0587930205923382,
0.0589595345435479,
0.059456935170843,
0.059650353792442,
0.0599040920800187,
0.0599890079793974,
0.0600308954953288,
0.0604547501197494,
0.0607213831545828,
0.0607415053040539,
0.0614042213374555,
0.061405845178002,
0.0617069188517894,
0.0619426379427557,
0.0623171595550165,
0.0625177691408974,
0.0635171627076536,
0.0636166507528069,
0.063737226365137,
0.0639879285103584,
0.0641512464818739,
0.0641674972103035,
0.0649573567721884,
0.0667480973319234,
0.0674793739168633,
0.0676391926135667,
0.068602942073559,
0.0687169287499057,
0.0687865488513628,
0.0688075829320578,
0.068884739636621,
0.0689375994292885,
0.069469629797867,
0.0694962038561187,
0.0696064854787615,
0.070204091272168,
0.0707639967600926,
0.0713102117347431,
0.0722295228064462,
0.0728416286990541,
0.0733033870304656,
0.0735870463226673,
0.0736472015306644,
0.0739579603323293,
0.0743288757211458,
0.0745957757599269,
0.0746623155865533,
0.0747475053821223,
0.0748785895562776,
0.0752013774395355,
0.0753461461894167,
0.0754346473190815,
0.07596392585143,
0.0761296888029017,
0.0763570061584096,
0.0764351216219295,
0.076629729979949,
0.0767328830061986,
0.076762017716332,
0.076884469852474,
0.0775787303415054,
0.0775898399787378,
0.0779547390825375,
0.0783174637860284,
0.0784753455867104,
0.0789209640078648,
0.0789457160106544,
0.0793414576064722,
0.0793828647324668,
0.0797416356835523,
0.080089919289299,
0.0803448356854686,
0.0809367840397757,
0.0823556140485593,
0.0825198000909566,
0.0828492311498685,
0.0841601891650923,
0.0843313573585647,
0.0845139622095026,
0.0850159704818809,
0.0868050576276858,
0.0868795003543355,
0.087337194126329,
0.0879310526512844,
0.088037934098617,
0.0882341505530215,
0.0883933315271079,
0.0889814570093206,
0.0905878685559225,
0.0922794309791186,
0.0925900151546966,
0.0926586323153078,
0.0933741344835415,
0.0937473222681451,
0.0944893813968983,
0.0949705419072844,
0.0956813427249914,
0.0957262728122732,
0.0958150992083651,
0.0963667284537717,
0.0964175740199291,
0.0964749326342851,
0.0964874430007079,
0.0965208385615135,
0.0975272439167618,
0.0977303413607134,
0.0978623673433576,
0.0984484627666582,
0.0986243035048551,
0.0986324177633957,
0.0991132550265084,
0.0991215759283693,
0.0995126372617517,
0.0998172283752563,
0.100290318408156,
0.101940134630487,
0.102171127735144,
0.102770568090924,
0.102903433287262,
0.102921260600861,
0.103316025315574,
0.103568406261259,
0.104007916148055,
0.104336224752188,
0.104338273497677,
0.10494086643599,
0.105253629313511,
0.105445239800721,
0.107546017652469,
0.108108288039155,
0.108709603175932,
0.109040485238032,
0.109572007562777,
0.109933862983125,
0.110075751140259,
0.110187816517282,
0.110533111712035,
0.110634082118688,
0.110998443159955,
0.111155084183321,
0.111432825912243,
0.111439626086236,
0.11154879731023,
0.113695798235032,
0.114113298003833,
0.114409452197952,
0.114425675784293,
0.114917999686797,
0.115474949699603,
0.116676443528374,
0.117639960649272,
0.118440183955957,
0.118826229036362,
0.119037715283445,
0.119239173449714,
0.119487045271103,
0.119701358473985,
0.119868555122884,
0.12294837156269,
0.123162753531041,
0.123492178928792,
0.123883176015971,
0.12407575845204,
0.124558030465991,
0.124961651262381,
0.127877428971326,
0.12797032377347,
0.129113737336004,
0.129978542634124,
0.130891117569929,
0.131322630288175,
0.131461163989566,
0.132202445569316,
0.13308019767031,
0.133116236457567,
0.133270552601812,
0.133313007169726,
0.134568911114119,
0.135052593467475,
0.135868189367487,
0.137160760475576,
0.137461495388567,
0.137464178653901,
0.138174251720535,
0.138845095375575,
0.139922721161043,
0.140442756075907,
0.140511483378289,
0.140646341580704,
0.140912844594818,
0.140961201757581,
0.141723290211253,
0.142456313516517,
0.142876037654354,
0.144042825952204,
0.145275908293079,
0.146332218465761,
0.147775083606008,
0.147884199864583,
0.148029528030156,
0.148839614051588,
0.149089577880766,
0.150283527246708,
0.150585594076058,
0.150987528612463,
0.151453366204586,
0.151998433324669,
0.155365086028647,
0.15550128881284,
0.156498855716181,
0.158398767710155,
0.161758417492863,
0.161989006536269,
0.162201971471627,
0.162723754768659,
0.164020959766216,
0.164311632861383,
0.164639671468829,
0.164815549693778,
0.165184732617183,
0.166070632325477,
0.166384226458476,
0.166410533659138,
0.167355883020575,
0.168113256714122,
0.16896476765174,
0.169074902354924,
0.16946395431757,
0.169631673020039,
0.17038395790224,
0.17047068851249,
0.171160055183637,
0.171318823090911,
0.171416195950043,
0.172511654383514,
0.175072631257032,
0.176795361731204,
0.180426475488602,
0.181840183871143,
0.18276186418813,
0.184531715578991,
0.185723523965811,
0.186250344645926,
0.187290592161791,
0.187431207064628,
0.187918023410041,
0.189412066218604,
0.19067788011538,
0.19121741769978,
0.192161383655956,
0.192819510256621,
0.193129445370199,
0.193860328534492,
0.196896104837792,
0.197664689621795,
0.197958195590253,
0.199166547943643,
0.20516250928591,
0.207776180113139,
0.208119885820319,
0.209239091245077,
0.210283032527339,
0.211332071898789,
0.215954597275321,
0.216384673088998,
0.217955926571509,
0.219954006969392,
0.220052927268983,
0.220361476645725,
0.222525705892447,
0.223236766825785,
0.223344183761394,
0.223820443729479,
0.225234680895463,
0.228391058648048,
0.2298853292571,
0.231413150455144,
0.232184525646275,
0.232422761875308,
0.235598795140031,
0.238499401989323,
0.239765258297187,
0.241156265931051,
0.242602461473216,
0.243909080952238,
0.244087774086221,
0.244353052706607,
0.245266992066036,
0.247288958255081,
0.247514357105256,
0.248222626594947,
0.248432924815569,
0.249065145121392,
0.249108333885673,
0.250873531214189,
0.251983470448458,
0.255403463884698,
0.256127898942737,
0.256143932052608,
0.261175558786935,
0.263514582546409,
0.263813777506998,
0.26419646326429,
0.264631071505055,
0.264976432234265,
0.269773256183967,
0.270031774088375,
0.273162149534074,
0.274188109098188,
0.27424489901243,
0.276216745472651,
0.276371408048344,
0.276804859654596,
0.277119287781522,
0.278488532350325,
0.281010796084309,
0.283484890688412,
0.283639603237207,
0.284361116106591,
0.284718945281557,
0.284991338356417,
0.287376293551893,
0.287586543350038,
0.287861483026246,
0.290073882563511,
0.29169216581002,
0.291765828347073,
0.292679929882422,
0.293309892043592,
0.293842887635587,
0.294089604131793,
0.2945993234191,
0.297557596864553,
0.299593261464866,
0.30194941272665,
0.302626899676564,
0.302795685251938,
0.302934140144555,
0.303341788590172,
0.303541912310728,
0.305595006669397,
0.308214979119925,
0.313458145127781,
0.315317576825329,
0.321668925794047,
0.337346589261678,
0.349108775034348,
0.352053480606922,
0.369594648398324,
0.372243359962563,
0.38543541437686,
0.411392167708371,
0.418268007044289,
0.452262558700847,
0.464797600612088,
0.470939658901592,
0.480385240575432,
0.50182268160043,
0.510077017025119,
0.5432754595379,
0.553484378765328,
])
d2 = np.array([0.00525609903008206,
0.0073186284912552,
0.00778479330981075,
0.00887034583779034,
0.0101022394771766,
0.0126635978771183,
0.0131360652662771,
0.0147885909956091,
0.0158680009362877,
0.0158695429697575,
0.0158821972140322,
0.0165138847984103,
0.0171041703300601,
0.0174346374166934,
0.0178342975466084,
0.018219714611849,
0.0184802445030741,
0.0186764232515024,
0.0199389847052878,
0.0204052293313817,
0.0205212167933988,
0.020778216667785,
0.0209403683090814,
0.0211047009806897,
0.021127290535774,
0.0211682562447946,
0.0220974559743148,
0.0226052172143465,
0.0227684125392269,
0.0231935421156093,
0.0242687668809838,
0.0248171604084346,
0.0259728793254199,
0.02892833183326,
0.0294271542201496,
0.0296788263159531,
0.030327553693974,
0.0304551926174579,
0.0304999511425219,
0.0312870419508204,
0.0316433941139671,
0.0321220718075344,
0.0324030015533072,
0.0328739455586602,
0.033584419672894,
0.0336917340881149,
0.0338088049782082,
0.0339292013150549,
0.0345397920372736,
0.0348490733355122,
0.0354486746061333,
0.0356309491884987,
0.0360969428083369,
0.0369808867279175,
0.0371934471581836,
0.0372700633139766,
0.0378993011144946,
0.037980785920646,
0.0382983872593323,
0.0383698768665619,
0.0384048114640478,
0.0387090106068388,
0.0391187584350394,
0.0392559931412739,
0.0392633436077894,
0.0400099486999928,
0.0401731543646466,
0.0403041463697296,
0.0405010121942917,
0.0410143780546601,
0.041280624818385,
0.0415620254286653,
0.0416668375755069,
0.0420313664183736,
0.0422535661732493,
0.0426713320688688,
0.0429197062672149,
0.0430361435267202,
0.0431426808062696,
0.0432467653111557,
0.0434446660150201,
0.0436505450602764,
0.0438585860581631,
0.0439698285439816,
0.0440403003957091,
0.0450765087660443,
0.0458932779979682,
0.0460030199454176,
0.0467073289254705,
0.0467486513008097,
0.04735743363113,
0.04776219585096,
0.0478260569343192,
0.0480103861101489,
0.0480497358057397,
0.0481554809844065,
0.0482901980376305,
0.0483190396467537,
0.0497999108264624,
0.0500334089899956,
0.0503859204646728,
0.0507390450349377,
0.0511084091580027,
0.0515507702993734,
0.0521632753129175,
0.0521740631075973,
0.0530806181032117,
0.0531406868042716,
0.0542656497222049,
0.0543956691781222,
0.055725716460171,
0.0560715235866072,
0.0560912135892311,
0.0562733969778417,
0.0564163516502658,
0.0565414538490261,
0.0571595421801785,
0.0576127349736977,
0.0578301358624822,
0.0579617774299912,
0.0589595345435479,
0.059456935170843,
0.0595116028639842,
0.059650353792442,
0.0599890079793974,
0.0604547501197494,
0.0615982014720018,
0.0623171595550165,
0.0625177691408974,
0.0635171627076536,
0.0636166507528069,
0.063737226365137,
0.0639879285103584,
0.0641512464818739,
0.0641674972103035,
0.0649573567721884,
0.0663811683066706,
0.0667480973319234,
0.0667594572229691,
0.0673127590686917,
0.0674793739168633,
0.067602381494023,
0.0676391926135667,
0.068171087818758,
0.068602942073559,
0.0687169287499057,
0.0687865488513628,
0.0688075829320578,
0.068884739636621,
0.0689375994292885,
0.0701157526797003,
0.070204091272168,
0.0707639967600926,
0.0713102117347431,
0.0716574162725303,
0.0735870463226673,
0.0740788131145198,
0.0743288757211458,
0.0746623155865533,
0.0747475053821223,
0.0748785895562776,
0.0752013774395355,
0.0753461461894167,
0.07596392585143,
0.0760992829647744,
0.0761296888029017,
0.0763570061584096,
0.0763843628658823,
0.0764351216219295,
0.0765314935694672,
0.076570915662586,
0.0767328830061986,
0.076884469852474,
0.0775787303415054,
0.0775898399787378,
0.0782523101905408,
0.0782992851126707,
0.0783174637860284,
0.0784407753993355,
0.0784753455867104,
0.0786738438294909,
0.0789209640078648,
0.0793083513431325,
0.0793828647324668,
0.0796551681500146,
0.080089919289299,
0.0803448356854686,
0.0807932778645082,
0.0809367840397757,
0.0825198000909566,
0.0830304427608208,
0.0842689437431865,
0.0843313573585647,
0.0850762123295645,
0.0864131514657895,
0.0868050576276858,
0.0868795003543355,
0.087337194126329,
0.0873781493698678,
0.0879310526512844,
0.088037934098617,
0.0883933315271079,
0.0885845383799033,
0.0889814570093206,
0.0890123366749984,
0.0905878685559225,
0.0922703606133535,
0.0922794309791186,
0.0925900151546966,
0.0926586323153078,
0.0927157525901112,
0.0933741344835415,
0.0936354243704729,
0.0937473222681451,
0.0944197659628232,
0.0946860556390572,
0.0951953887012369,
0.0956813427249914,
0.0957262728122732,
0.0961502758484167,
0.0963246556766029,
0.0963667284537717,
0.0964749326342851,
0.0964874430007079,
0.0965208385615135,
0.0966304482243574,
0.0976337191470789,
0.0984484627666582,
0.0986243035048551,
0.0986324177633957,
0.0991132550265084,
0.0993505317581534,
0.0995126372617517,
0.0996011567013122,
0.0998050077856139,
0.0998172283752563,
0.10053586456563,
0.10187091682905,
0.101940134630487,
0.102171127735144,
0.102770568090924,
0.102903433287262,
0.102921260600861,
0.103316025315574,
0.103553267375602,
0.103568406261259,
0.103912246085218,
0.104007916148055,
0.104336224752188,
0.104338273497677,
0.10494086643599,
0.106117067823632,
0.106241890759993,
0.107116060577659,
0.10799642114228,
0.108108288039155,
0.108707652047406,
0.108709603175932,
0.108736709115767,
0.109040485238032,
0.109572007562777,
0.110025641140408,
0.110075751140259,
0.110187816517282,
0.110533111712035,
0.111432825912243,
0.111439626086236,
0.11154879731023,
0.113190042744447,
0.114113298003833,
0.114425675784293,
0.114917999686797,
0.115474949699603,
0.11581429234548,
0.116676443528374,
0.11733675022453,
0.117639960649272,
0.117794846987644,
0.118412815442901,
0.118440183955957,
0.118826229036362,
0.119037715283445,
0.119239173449714,
0.119487045271103,
0.119504247177376,
0.119701358473985,
0.119868555122884,
0.119965545270007,
0.120758036375785,
0.121203397694298,
0.12294837156269,
0.123144706248519,
0.123492178928792,
0.123883176015971,
0.12407575845204,
0.124558030465991,
0.124961651262381,
0.1257609524093,
0.126020161276807,
0.127284874825122,
0.127617618229081,
0.12797032377347,
0.129113737336004,
0.129831373654428,
0.129978542634124,
0.130891117569929,
0.131497179370679,
0.131903928427903,
0.132202445569316,
0.132348955193478,
0.13308019767031,
0.133116236457567,
0.133270552601812,
0.133313007169726,
0.13342611853314,
0.134568911114119,
0.135868189367487,
0.136193041326713,
0.137160760475576,
0.137461495388567,
0.137999468533869,
0.140442756075907,
0.140511483378289,
0.141723290211253,
0.142456313516517,
0.142876037654354,
0.144042825952204,
0.145275908293079,
0.145413924738769,
0.146332218465761,
0.147775083606008,
0.14817375617687,
0.148839614051588,
0.149089577880766,
0.150283527246708,
0.150585594076058,
0.150987528612463,
0.151428922757124,
0.151453366204586,
0.151658005023085,
0.15281362214135,
0.153675889908662,
0.155035365387319,
0.15550128881284,
0.155582790223985,
0.156498855716181,
0.158172955912943,
0.160112807334016,
0.161103833324541,
0.161758417492863,
0.162201971471627,
0.162723754768659,
0.164020959766216,
0.165184732617183,
0.166070632325477,
0.166384226458476,
0.166410533659138,
0.167355883020575,
0.16896476765174,
0.169074902354924,
0.16946395431757,
0.169631673020039,
0.17038395790224,
0.171318823090911,
0.172511654383514,
0.175072631257032,
0.176795361731204,
0.178305295340593,
0.180426475488602,
0.180556746421883,
0.180757400998215,
0.181690366416785,
0.181840183871143,
0.18276186418813,
0.184352835413955,
0.184531715578991,
0.186250344645926,
0.187290592161791,
0.187431207064628,
0.187918023410041,
0.189412066218604,
0.189461232276069,
0.189616081849393,
0.19067788011538,
0.192161383655956,
0.193129445370199,
0.19360189180023,
0.194931403900609,
0.195533770038527,
0.196896104837792,
0.197958195590253,
0.199166547943643,
0.200878225665861,
0.20516250928591,
0.207041644063586,
0.207776180113139,
0.208119885820319,
0.210283032527339,
0.211332071898789,
0.215581693417919,
0.218658641644103,
0.219954006969392,
0.220052927268983,
0.223344183761394,
0.223820443729479,
0.228391058648048,
0.231413150455144,
0.232184525646275,
0.232757012086537,
0.233803352225208,
0.239765258297187,
0.242120780409976,
0.244087774086221,
0.244353052706607,
0.245493938254726,
0.247288958255081,
0.248222626594947,
0.249065145121392,
0.251952586843132,
0.251983470448458,
0.254912135422449,
0.255403463884698,
0.25586723474472,
0.259313044510976,
0.260144568448446,
0.263514582546409,
0.26370194026438,
0.264139058450583,
0.26419646326429,
0.264631071505055,
0.267638240615371,
0.269773256183967,
0.270031774088375,
0.271331110804098,
0.273157090298644,
0.274188109098188,
0.27424489901243,
0.274483466468816,
0.277119287781522,
0.277514909184529,
0.278488532350325,
0.280290171892719,
0.28057504355588,
0.281010796084309,
0.281503472080904,
0.282432184536789,
0.283639603237207,
0.284718945281557,
0.285911236837739,
0.287576012515367,
0.287586543350038,
0.28983577155345,
0.290073882563511,
0.291544587334349,
0.29169216581002,
0.291765828347073,
0.293309892043592,
0.294089604131793,
0.297557596864553,
0.299580994159854,
0.302626899676564,
0.302795685251938,
0.302934140144555,
0.303541912310728,
0.303581965925863,
0.3039313432873,
0.305166841316935,
0.305595006669397,
0.306486958670642,
0.313458145127781,
0.317887769743672,
0.318357190731033,
0.326607374136469,
0.329284148390796,
0.337346589261678,
0.338986076270484,
0.349108775034348,
0.352053480606922,
0.366259928131534,
0.369594648398324,
0.372243359962563,
0.38543541437686,
0.411392167708371,
0.449836339450695,
0.452262558700847,
0.471299668582743,
0.480385240575432,
0.495746400844483,
0.502507102131893,
0.510077017025119,
0.516853142310918,
0.559054225956661,
0.576882634377882,
])
# d3 = np.array([])

sns.kdeplot(d1, label='cheating', shade=True, color='g')
sns.kdeplot(d2, label='using prediction module', shade=True, color='r')
# sns.kdeplot(d3, label='hrl with "specific" set-down skill', shade=True)

# label = 'mean:{:.4f}, std:{:.4f}'.format(np.mean(d1), np.std(d1))
# sns.kdeplot(d1, label=label, shade=True)

plt.xlim(-0.1,0.8)
plt.xlabel('impact velocity (m/s)')
plt.title('Hs:1.5, Tp:15, set-down skill from 1m')
plt.show()
###################################################

######show tsplot###########
# window = 300
#
#
# def running_mean(x, N):
#     cumsum = np.cumsum(np.insert(x, 0, 0))
#     return (cumsum[N:] - cumsum[:-N]) / float(N)
#
#
# cur1 = pd.read_json('following_23.1.2.4_log.csv')['episode_reward']
# cur2 = pd.read_json('following_23.1.2.4_log.csv')['mean_q']
#
# cur1 = running_mean(np.array(cur1), window)
# std_cur1 = np.std(cur1)
#
# cur2 = running_mean(np.array(cur2), window)
# std_cur2 = np.std(cur2)
#
# l = min(len(cur1),len(cur2))
# C_params = np.linspace(1, l, l)
# cur1 = cur1[0:l]
# cur2 = cur2[0:l]
#
# cur1_min = 0
# cur1_max = 200
#
# cur2_min = 0
# cur2_max = 200
#
# sns.set_style("darkgrid")
#
#
# host = host_subplot(111, axes_class=AA.Axes)
# plt.subplots_adjust(right=0.75)
#
# par1 = host.twinx()
#
# new_fixed_axis_1 = par1.get_grid_helper().new_fixed_axis
# par1.axis["left"] = new_fixed_axis_1(loc="right",
#                                     axes=par1,
#                                     offset=(0, 0))
#
# par1.axis["left"].toggle(all=True)
#
#
# host.set_ylim(cur1_min, cur1_max)
# host.set_xlabel("episode")
# par1.set_ylabel("mean Q-value")
# host.set_ylabel("episode reward")
#
# p1, = host.plot(C_params, cur1,label="episode reward", color='red')
# p2, = par1.plot(C_params, cur2, label="mean Q", color='green')
# host.fill_between(C_params, cur1 - std_cur1, cur1 + std_cur1, alpha = 0.1, color="red")
# par1.fill_between(C_params, cur2 - std_cur2, cur2 + std_cur2, alpha = 0.1, color="green")
# par1.set_ylim(cur2_min, cur2_max)
# host.legend()
#
# host.axis["left"].label.set_color(p1.get_color())
# par1.axis["left"].label.set_color(p2.get_color())
#
# host.axis["left"].major_ticklabels.set_color(p1.get_color())
# par1.axis["left"].major_ticklabels.set_color(p2.get_color())
#
# plt.title('Initial distance: 7m')
# plt.draw()
# plt.show()






