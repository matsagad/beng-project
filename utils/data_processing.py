""" JSON Test """
import json
import numpy as np
from sklearn import linear_model


def getTS(tf):
    with open("data/fig3_stress_level_expts.json", "r") as read_file:
        data = json.load(read_file)

    data = data["gluc"][tf]["gluc_0_1"]

    return data


def getTSgluc(tf):
    with open("data/fig3_stress_level_expts.json", "r") as read_file:
        data = json.load(read_file)

    data = data["gluc"][tf]["gluc_0_1"]

    return data


def getTSnacl(tf):
    with open("data/fig3_stress_level_expts.json", "r") as read_file:
        data = json.load(read_file)

    data = data["nacl"][tf]["nacl_0_4M"]

    return data


def getTSoxid(tf):
    with open("data/fig3_stress_level_expts.json", "r") as read_file:
        data = json.load(read_file)

    data = data["oxid"][tf]["oxid_0_50mM"]

    return data


"""
def getTSst(tf, stresstype):
    with open("fig2_stress_type_expts.json", "r") as read_file:
        data = json.load(read_file)
    
    data = data[tf][stresstype]
    return data
"""


def scaleFactCalcs(tf):
    with open("data/figS1_nuclear_marker_expts.json", "r") as read_file:
        ncdata = json.load(read_file)

    max5 = np.array(ncdata[tf]["GFP"]["max5"]).astype("float")
    median = np.array(ncdata[tf]["GFP"]["median"]).astype("float")
    background = np.array(ncdata[tf]["GFP"]["imBackground"]).astype("float")

    scaledTS = np.divide(np.subtract(max5, background), np.subtract(median, background))

    scaledTS = scaledTS[~np.isnan(scaledTS).any(axis=1)]
    scaledTS = np.mean(scaledTS, axis=0)

    with open("data/figS1_nuclear_marker_expts.json", "r") as read_file:
        ncdata = json.load(read_file)

    ncMe = np.array(ncdata[tf]["GFP"]["nucInnerMedian"]).astype("float")
    ncIm = np.array(ncdata[tf]["GFP"]["imBackground"]).astype("float")
    nc = ncMe - ncIm

    nc = nc[~np.isnan(nc).any(axis=1)]
    nc = np.mean(nc, axis=0)
    scaledTS = np.reshape(scaledTS, (len(scaledTS), 1))
    nc = np.reshape(nc, (len(nc), 1))
    lm = linear_model.LinearRegression()
    lm.fit(scaledTS, nc)

    return lm.coef_, lm.intercept_, scaledTS


def scaleFactors(tf):
    if tf == "dot6" or tf == "mig1" or tf == "sfp1" or tf == "maf1":
        lc, li, ts = scaleFactCalcs(tf)
        scale = lc / (np.mean(ts) * li + lc)
        offset = li / (np.mean(ts) * li + lc)
    else:
        dot6lc, dot6li, dot6ts = scaleFactCalcs("dot6")
        mig1lc, mig1li, mig1ts = scaleFactCalcs("mig1")
        sfp1lc, sfp1li, sfp1ts = scaleFactCalcs("sfp1")

        dot6scale = dot6lc / (np.mean(dot6ts) * dot6li + dot6lc)
        sfp1scale = sfp1lc / (np.mean(sfp1ts) * sfp1li + sfp1lc)
        mig1scale = mig1lc / (np.mean(mig1ts) * mig1li + mig1lc)

        dot6offset = dot6li / (np.mean(dot6ts) * dot6li + dot6lc)
        sfp1offset = sfp1li / (np.mean(sfp1ts) * sfp1li + sfp1lc)
        mig1offset = mig1li / (np.mean(mig1ts) * mig1li + mig1lc)

        scale = np.mean([dot6scale, sfp1scale, mig1scale])
        offset = np.mean([dot6offset, sfp1offset, mig1offset])

    return scale, offset


def scaleTS(tf):
    data = getTS(tf)

    scale, offset, ts = scaleFactCalcs(tf)

    max5 = np.array(data["GFP"]["max5"]).astype("float")
    background = np.array(data["GFP"]["imBackground"]).astype("float")
    median = np.array(data["GFP"]["median"]).astype("float")
    times = np.array(data["general"]["times"]).astype("float")

    scaledTS = np.divide(np.subtract(max5, background), np.subtract(median, background))
    times = times[~np.isnan(scaledTS).any(axis=1)]
    scaledTS = scaledTS[~np.isnan(scaledTS).any(axis=1)]
    scaled = scale * scaledTS + offset

    scaled = scaled / np.mean(scaled)

    a, b = scaled.shape

    origin = data["general"]["origin"]

    return scaled, origin, times


def scaleTS2st(tf, st1, st2):
    st = [st1, st2]
    scale, offset = scaleFactors(tf)

    scaled = [0, 0]
    timeList = [0, 0]
    origin = [0, 0]

    data = [0, 0]

    for i in range(2):
        if st[i] == "gluc":
            data[i] = getTSgluc(tf)
        elif st[i] == "oxid":
            data[i] = getTSoxid(tf)
        else:
            data[i] = getTSnacl(tf)

        max5 = np.array(data[i]["GFP"]["max5"]).astype("float")
        background = np.array(data[i]["GFP"]["imBackground"]).astype("float")
        median = np.array(data[i]["GFP"]["median"]).astype("float")
        times = np.array(data[i]["general"]["times"]).astype("float")

        scaledTS = np.divide(
            np.subtract(max5, background), np.subtract(median, background)
        )
        times = times[~np.isnan(scaledTS).any(axis=1)]
        scaledTS = scaledTS[~np.isnan(scaledTS).any(axis=1)]

        scaled[i] = scale * scaledTS + offset
        timeList[i] = times
        origin[i] = data[i]["general"]["origin"]

    n = np.min([timeList[i].shape[0] for i in range(2)])

    timeList[0] = timeList[0][:n]
    timeList[1] = timeList[1][:n]
    scaled[0] = scaled[0][:n]
    scaled[0] = scaled[0] / scaled[0][:, origin[0] - 1].mean()
    scaled[1] = scaled[1][:n]
    scaled[1] = scaled[1] / scaled[1][:, origin[1] - 1].mean()

    return scaled, origin, timeList


def scaleTSall(tf):
    st = ["gluc", "oxid", "nacl"]
    scale, offset = scaleFactors(tf)

    scaled = [0, 0, 0]
    timeList = [0, 0, 0]
    origin = [0, 0, 0]

    data = [0, 0, 0]

    for i in range(3):
        if st[i] == "gluc":
            data[i] = getTSgluc(tf)
        elif st[i] == "oxid":
            data[i] = getTSoxid(tf)
        else:
            data[i] = getTSnacl(tf)

        max5 = np.array(data[i]["GFP"]["max5"]).astype("float")
        background = np.array(data[i]["GFP"]["imBackground"]).astype("float")
        median = np.array(data[i]["GFP"]["median"]).astype("float")
        times = np.array(data[i]["general"]["times"]).astype("float")

        scaledTS = np.divide(
            np.subtract(max5, background), np.subtract(median, background)
        )
        times = times[~np.isnan(scaledTS).any(axis=1)]
        scaledTS = scaledTS[~np.isnan(scaledTS).any(axis=1)]

        scaled[i] = scale * scaledTS + offset
        timeList[i] = times
        origin[i] = data[i]["general"]["origin"]

    n = np.min([timeList[i].shape[0] for i in range(3)])

    timeList[0] = timeList[0][:n]
    timeList[1] = timeList[1][:n]
    timeList[2] = timeList[2][:n]
    scaled[0] = scaled[0][:n]
    scaled[0] = scaled[0] / scaled[0][:, origin[0] - 1].mean()
    scaled[1] = scaled[1][:n]
    scaled[1] = scaled[1] / scaled[1][:, origin[1] - 1].mean()
    scaled[2] = scaled[2][:n]
    scaled[2] = scaled[2] / scaled[2][:, origin[2] - 1].mean()

    return scaled, origin, timeList


def scaleTS2(tf, stresstype):
    data = getTSst(tf, stresstype)

    scale, offset = scaleFactors(tf)

    origin = data["general"]["origin"]

    max5 = np.array(data["GFP"]["max5"]).astype("float")
    background = np.array(data["GFP"]["imBackground"]).astype("float")
    median = np.array(data["GFP"]["median"]).astype("float")
    times = np.array(data["general"]["times"]).astype("float")

    scaledTS = np.divide(np.subtract(max5, background), np.subtract(median, background))
    times = times[~np.isnan(scaledTS).any(axis=1)]
    scaledTS = scaledTS[~np.isnan(scaledTS).any(axis=1)]
    scaled = scale * scaledTS + offset

    return scaled, origin, times
