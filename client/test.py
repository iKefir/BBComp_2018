from math import cos, pi
from functools import reduce
import numpy as np


amountPerDim = 2


class Problem:
    def __init__(self):
        self.dims = [2, 4, 5, 8, 10, 16, 20, 32, 40, 64]
        self.logging = True
        self.fn = spheref

    def dimension(self):
        return self.dim

    def budget(self):
        return 100 * self.dim ** 2

    def evaluate(self, point, value):
        if (self.evals >= self.budget()):
            return 0
        else:
            self.evals += 1
            value[0] = self.fn(point + self.dlt)
            self.avVal += value[0]
            if (self.bestVal > value[0]):
                self.bestVal = value[0]
                self.bvEval = self.evals
                if self.logging:
                    print("Eval: ", self.evals, "Point: ", point, "Value: ", value[0])

            return 1

    def evaluations(self):
        return self.evals

    def setProblem(self, pID):
        self.dim = self.dims[pID // amountPerDim]
        self.dlt = np.array([0] * self.dim)
        self.evals = 0
        self.bestVal = 1e100
        self.bvEval = -1
        self.avVal = 0
        return 1

    def setFn(self, fn):
        self.fn = fn

    def setLogging(self, enabled):
        self.logging = enabled


def rastriginf(point):
    point += -0.5
    point *= 10
    A = 10
    dim = len(point)
    res = A * dim
    for x in point:
        res += x*x - A * cos(2 * pi * x)

    return res


def spheref(point):
    point += -0.5
    point *= 10
    res = 0
    for x in point:
        res += x ** 2

    return res


def zakharovf(point):
    point += -(1 / 3)
    point *= 15
    res = 0
    acc = 0
    for i in range(len(point)):
        res += point[i] ** 2
        acc += 0.5 * (i + 1) * point[i]
    res += acc ** 2 + acc ** 4

    return res

def rosenbrockf(point):
    point += -(1 / 3)
    point *= 15
    res = 0
    for i in range(len(point) - 1):
        res += 100 * ((point[i + 1] - point[i] ** 2) ** 2) + (1 - point[i]) ** 2

    return res


def styblinskyTangPure(point):
    res = 0
    for x in point:
        res += (x ** 4 - 16 * x ** 2 + 5 * x) / 2

    return res


def styblinskyTangf(point):
    point += -0.5
    point *= 10
    return styblinskyTangPure(point)


def styblinskyTangFromZero(point):
    res = styblinskyTangf(point)
    min = styblinskyTangPure([-2.903534] * len(point)) - 1e-10
    return res - min


def testOne(bbcomp, solveProblem, pID, logging):
    bbcomp.setLogging(logging)
    solveProblem(pID)
    print("Best: ", bbcomp.bestVal, "Eval: ", bbcomp.bvEval)
    print("Average: ", bbcomp.avVal / bbcomp.evals)
    return (bbcomp.bestVal, bbcomp.bvEval)


problems = {
            "Rastrigin": rastriginf,
            "Sphere": spheref,
            "Zakharov": zakharovf,
            "Rosenbrock": rosenbrockf,
            "StyblinskyTang": styblinskyTangFromZero,
            }


def testAll(bbcomp, solveProblem):
    print("TESTING")
    file = open("testlog.txt", "w")
    file.write("Test results\n\n")
    file.close()
    for dim in range(10):
        file = open("testlog.txt", "a+")
        file.write("Dimensions: " + str(bbcomp.dims[dim]) + "\n")
        file.close()
        resForDim = []
        for f in problems.items():
            file = open("testlog.txt", "a+")
            file.write("\t" + f[0] + "\n")
            bbcomp.setFn(f[1])
            res = []
            for run in range(amountPerDim):
                res.append(testOne(bbcomp, solveProblem, dim * amountPerDim + run, False))
            best = res[0]
            avg = 0.
            for p in res:
                avg += p[0]
                if p[0] < best[0]:
                    best = p
            avg /= len(res)
            file.write("\tBest: " + str(best[0]) + "\n")
            file.write("\tAverage: " + str(avg) + "\n\n")
            file.close()
            resForDim.append(avg)
        sum = 0.
        for p in resForDim:
            sum += p
        file = open("testlog.txt", "a+")
        file.write("AvgForDim: " + str(sum / len(resForDim)) + "\n\n")
        file.close()
