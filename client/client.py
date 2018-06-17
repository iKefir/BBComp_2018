from test import rastriginf, Problem, testAll

from ctypes import c_int, c_char_p, c_double, CDLL
from numpy.ctypeslib import ndpointer
import numpy as np

import diversipy
import cma
import sys
import platform
import time

# get library name
dllname = ""
if platform.system() == "Windows":
    dllname = "./bbcomp.dll"
elif platform.system() == "Linux":
    dllname = "./libbbcomp.so"
elif platform.system() == "Darwin":
    dllname = "./libbbcomp.dylib"
else:
    sys.exit("unknown platform")

TESTING = True


if TESTING:
    bbcomp = Problem()
else:
    # initialize dynamic library
    bbcomp = CDLL(dllname)
    bbcomp.configure.restype = c_int
    bbcomp.login.restype = c_int
    bbcomp.numberOfTracks.restype = c_int
    bbcomp.trackName.restype = c_char_p
    bbcomp.setTrack.restype = c_int
    bbcomp.numberOfProblems.restype = c_int
    bbcomp.setProblem.restype = c_int
    bbcomp.dimension.restype = c_int
    bbcomp.numberOfObjectives.restype = c_int
    bbcomp.budget.restype = c_int
    bbcomp.evaluations.restype = c_int
    bbcomp.evaluate.restype = c_int
    bbcomp.evaluate.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS")]
    bbcomp.history.restype = c_int
    bbcomp.history.argtypes = [c_int, ndpointer(c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS")]
    bbcomp.errorMessage.restype = c_char_p

# configuration
LOGFILEPATH = "logs/"
USERNAME = "demoaccount"
PASSWORD = "demopassword"
TRACKNAME = "trial"
LOGIN_DELAY_SECONDS = 10
LOCK_DELAY_SECONDS = 60


# network failure resilient functions
def safeLogin():
    while True:
        result = bbcomp.login(USERNAME.encode('ascii'), PASSWORD.encode('ascii'))
        if result != 0:
            return
        msg = bbcomp.errorMessage().decode('ascii')
        print("WARNING: login failed: ", msg)
        time.sleep(LOGIN_DELAY_SECONDS)
        if msg == "already logged in":
            return


def safeSetTrack():
    while True:
        result = bbcomp.setTrack(TRACKNAME.encode('ascii'))
        if result != 0:
            return
        print("WARNING: setTrack failed: ", bbcomp.errorMessage().decode('ascii'))
        safeLogin()


def safeGetNumberOfProblems():
    while True:
        result = bbcomp.numberOfProblems()
        if result != 0:
            return result
        print("WARNING: numberOfProblems failed: ", bbcomp.errorMessage().decode('ascii'))
        safeSetTrack()


def safeSetProblem(problemID):
    while True:
        # print(problemID)
        result = bbcomp.setProblem(problemID)
        if result != 0:
            return
        msg = bbcomp.errorMessage().decode('ascii')
        print("WARNING: setProblem failed: ", msg)
        if len(msg) >= 22 and msg[0:22] == "failed to acquire lock":
            time.sleep(LOCK_DELAY_SECONDS)
        else:
            safeSetTrack()


def safeEvaluate(problemID, point):
    while True:
        value = np.zeros(1)
        result = bbcomp.evaluate(point, value)
        if result != 0:
            return value[0]
            print("WARNING: evaluate failed: ", bbcomp.errorMessage().decode('ascii'))
            print(point)
            safeSetProblem(problemID)


def safeGetDimension(problemID):
    while True:
        result = bbcomp.dimension()
        if result != 0:
            return result
        print("WARNING: dimension failed: ", bbcomp.errorMessage().decode('ascii'))
        safeSetProblem(problemID)


def safeGetBudget(problemID):
    while True:
        result = bbcomp.budget()
        if result != 0:
            return result
        print("WARNING: budget failed: ", bbcomp.errorMessage().decode('ascii'))
        safeSetProblem(problemID)


def safeGetEvaluations(problemID):
    while True:
        result = bbcomp.evaluations()
        if result >= 0:
            return result
        print("WARNING: evaluations failed: " + bbcomp.errorMessage().decode('ascii'))
        safeSetProblem(problemID)


class Point:
    def __init__(self, point, value):
        self.point = point
        self.value = value

    def __repr__(self):
        return "Point (point:%s value:%s)" % (str(self.point), self.value)


# simple constraint handling by truncation
def truncate2bounds(vec):
    for i in range(len(vec)):
        if vec[i] < 0.0:
            vec[i] = 0.0
        elif vec[i] > 1.0:
            vec[i] = 1.0
    return vec


def solveProblem(problemID):
    # set the problem
    safeSetProblem(problemID)

    # obtain problem properties
    bud = safeGetBudget(problemID)
    dim = safeGetDimension(problemID)
    evals = safeGetEvaluations(problemID)

    # output status
    if evals == bud:
        print("problem ", problemID, ": already solved")
        return
    elif evals == 0:
        print("problem ", problemID, ": starting from scratch")
    else:
        print("problem ", problemID, ": starting from evaluation ", evals)

    # reserve some budget for final optimization
    reserved = 100 * dim
    storedResults = np.array([])

    if (dim <= 5):
        # use Nelder-Mead
        for i in range(dim):
            results = nelderMead(problemID, (bud - reserved) / dim, dim)
            storedResults = np.append(storedResults, results)
    else:
        # chose super-uniformly distributed starting points
        startingAmount = 2 * dim
        startingPoints = diversipy.hycusampling.maximin_reconstruction(startingAmount, dim)
        startingPoints = np.array([Point(x, safeEvaluate(problemID, x)) for x in startingPoints])
        startingPoints = np.array([x.point for x in sorted(startingPoints, key=lambda a: a.value)])
        reserved += startingAmount
        curPoint = 0
        # use CMA-ES on chosen points
        while (evals < bud - reserved):
            # print("started", evals)
            results = cmaES(problemID, bud - reserved, dim, evals, startingPoints[curPoint])
            storedResults = np.append(storedResults, results)
            evals = safeGetEvaluations(problemID)
            # print("finished", evals)
            curPoint += 1
            if len(startingPoints) == curPoint:
                startingPoints = diversipy.hycusampling.maximin_reconstruction(len(startingPoints) + 1, dim, existing_points=startingPoints)

    # use ASSLS
    storedResults = sorted(storedResults, key=lambda a: a.value)
    bestPoints = min(dim, len(storedResults))
    for i in range(bestPoints):
        ASSLS(problemID, reserved / bestPoints, dim, storedResults[i], storedResults[0].value)


def nelderMead(problemID, bud, dim):
    points = diversipy.hycusampling.maximin_reconstruction(dim + 1, dim)
    simplex = [Point(x, 1e100) for x in points]

    countEvals = 0

    for i in range(0, dim + 1):
        if (countEvals < bud):
            simplex[i].value = safeEvaluate(problemID, simplex[i].point)
            countEvals += 1
        else:
            return simplex

    while countEvals < bud:
        simplex = sorted(simplex, key=lambda a: a.value)

        centroid = np.zeros(dim)
        for i in range(dim):
            centroid += simplex[i].point
        centroid = truncate2bounds(centroid / dim)

        xrp = truncate2bounds(centroid + 1 * (centroid - simplex[dim].point))
        xr = Point(xrp, safeEvaluate(problemID, xrp))
        countEvals += 1
        if (xr.value < simplex[dim - 1].value and xr.value >= simplex[0].value):
            simplex[dim] = xr
        elif (xr.value < simplex[0].value):
            if (countEvals < bud):
                xep = truncate2bounds(centroid + 2 * (xr.point - centroid))
                xe = Point(xep, safeEvaluate(problemID, xep))
                countEvals += 1
                if (xe.value < xr.value):
                    simplex[dim] = xe
                else:
                    simplex[dim] = xr
            else:
                return simplex
        else:
            if (countEvals < bud):
                xcp = truncate2bounds(centroid + 0.5 * (simplex[dim].point - centroid))
                xc = Point(xcp, safeEvaluate(problemID, xcp))
                countEvals += 1
                if (xc.value < simplex[dim].value):
                    simplex[dim] = xc
                else:
                    for i in range(1, dim + 1):
                        simplex[i].point = truncate2bounds(simplex[0].point + 0.5 * (simplex[i].point - simplex[0].point))
                        if (countEvals < bud):
                            simplex[i].value = safeEvaluate(problemID, simplex[i].point)
                            countEvals += 1
                        else:
                            return simplex

    return simplex


cmaResults = np.array([])


# TODO change starting step size
def cmaES(problemID, bud, dim, evals, startPoint):
    global cmaResults
    stepSize = 0.15
    minBounds = [0.0] * dim
    maxBounds = [1.0] * dim
    bounds = [minBounds, maxBounds]
    tolfun = 0.0
    tolfunhist = 0.0
    optionsDict = {"maxfevals": bud - evals,
                   "maxiter": float("inf"),
                   "bounds": bounds,
                   "tolfun": tolfun,
                   "tolfunhist": tolfunhist,
                   "verb_time": False,
                   "verb_log": 0,
                   "verbose": -1,
                   "verb_disp": 0}
    cmaResults = np.array([])
    try:
        result = cma.fmin(lambda point: cmaESFunc(problemID, bud, point), startPoint, stepSize, options = optionsDict)
        return np.array(Point(result[0], result[1]))
    except (StopIteration, ValueError) as e:
        return cmaResults


def cmaESFunc(problemID, bud, point):
    if safeGetEvaluations(problemID) < bud:
        global cmaResults
        point = np.array(point)
        value = safeEvaluate(problemID, point)
        cmaResults = np.append(cmaResults, np.array(Point(point, value)))
        return value
    else:
        raise StopIteration()


def ASSLS(problemID, bud, dim, startPoint, valBest):
    x = startPoint.point
    val = startPoint.value
    stepSize = 0.01
    alpha = 0.2
    beta = 2
    succFailLimit = 10
    succStepsLimit = 100
    countEvals = 0
    succFail = 0
    succSteps = 0
    while (countEvals < bud):
        d = np.random.normal(x, 1, dim)
        r = 0
        for i in d:
            r += i * i
        r = r ** (.5)
        d /= r

        xn = truncate2bounds(x + stepSize * d)
        valn = safeEvaluate(problemID, xn)
        countEvals += 1
        if countEvals >= bud:
            break
        xna = truncate2bounds(x + stepSize * (1 + alpha) * d)
        valna = safeEvaluate(problemID, xna)
        countEvals += 1
        if valn < val or valna < val:
            if valn < valna:
                x = xn
                val = valn
            else:
                x = xna
                val = valna
                stepSize += stepSize * alpha
            if val < valBest:
                print("ASSLS improved: ", valBest, " -> ", val)
            succFail = 0
        else:
            succFail += 1

        if succFail >= succFailLimit:
            stepSize -= stepSize * alpha
            succFail = 0

        if countEvals >= bud:
            break

        if succSteps % succStepsLimit == 0:
            xnl = truncate2bounds(x + stepSize * (1 + beta) * d)
            valnl = safeEvaluate(problemID, xnl)

            if valnl < val:
                x = xnl
                val = valnl
                stepSize = stepSize * (1 + beta)

                if val < valBest:
                    print("ASSLS improved: ", valBest, " -> ", val, " on extended size")

        succSteps += 1


if (TESTING):
    testAll(bbcomp, solveProblem)
else:
    # setup
    # TODO: turn logging on if needed
    result = bbcomp.configure(0, LOGFILEPATH.encode('ascii'))
    if result == 0:
        sys.exit("configure() failed: " + bbcomp.errorMessage())

    safeLogin()
    safeSetTrack()
    n = safeGetNumberOfProblems()

    for i in range(595, 1000):
        solveProblem(i)
