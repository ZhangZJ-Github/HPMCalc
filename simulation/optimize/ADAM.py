import typing

import matplotlib
import numpy
from _logging import logger

matplotlib.use('tkagg')


class OptimizedResult:
    def __init__(self, status: bool, params, function_value):
        self.status, self.params, self.function_value = status, params, function_value


class AdamZhihu:
    """
    主要代码来自：
    简单认识Adam优化器 - Emerson的文章 - 知乎
    https://zhuanlan.zhihu.com/p/32698042
    """

    def __init__(self, loss_func, x0, how_to_get_grad: typing.Callable, lb: numpy.ndarray,
                 ub: numpy.ndarray, lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8, ):
        self.loss = loss_func
        self.theta = x0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.get_gradient = how_to_get_grad  # grad(loss)
        self.m = 0
        self.v = 0
        self.t = 0
        # 参数上下边界
        self.lb = lb
        self.ub = ub
        # self.opt_res = OptimizedResult(False,self.theta,None)

    def minimize_raw(self):
        self.t += 1
        g = self.get_gradient(self.theta)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.m_hat = self.m / (1 - self.beta1 ** self.t)
        self.v_hat = self.v / (1 - self.beta2 ** self.t)
        self.theta -= self.lr * self.m_hat / (self.v_hat ** 0.5 + self.epislon)

    def minimize(self):
        self.t += 1
        g = self.get_gradient(self.theta)
        lr = self.lr * (1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.theta -= lr * self.m / (self.v ** 0.5 + self.epislon)

        low_index = self.theta < self.lb
        self.theta[low_index] = self.theta[low_index]

        high_index = self.theta > self.ub
        self.theta[high_index] = self.theta[high_index]

        logger.info("参数越界情况：\n%s\n%s" % (low_index, high_index))

    def run(self, tols: numpy.ndarray, maxiter=100) -> OptimizedResult:
        old_theta = self.theta.copy()
        for i in range(maxiter):
            self.minimize_raw()
            func_res = self.loss(self.theta)
            if numpy.all(numpy.abs((self.theta - old_theta)) < tols):
                return OptimizedResult(True, self.theta, func_res)
        logger.warning("迭代到达最大步数，结果仍不收敛")
        return OptimizedResult(False, self.theta, func_res)


# Adam之实现

import numpy
from matplotlib import pyplot as plt


# 目标函数0阶信息
def func(X):
    funcVal = 5 * X[0, 0] ** 2 + 2 * X[1, 0] ** 2 + 3 * X[0, 0] - 10 * X[1, 0] + 4
    return funcVal


# 目标函数1阶信息
def grad(X):
    grad_x1 = 10 * X[0, 0] + 3
    grad_x2 = 4 * X[1, 0] - 10
    gradVec = numpy.array([[grad_x1], [grad_x2]])
    return gradVec


# 定义迭代起点
def seed(n=2):
    seedVec = numpy.random.uniform(-100, 100, (n, 1))
    return seedVec


class Adam(object):
    """
    https://www.cnblogs.com/xxhbdk/p/15063793.html
    """

    def __init__(self, func, x0, _get_partial_J, _partial_x_for_get_partial_diff, lb, ub, unit_steps: numpy.ndarray):
        '''
        _func: 待优化目标函数
        _grad: 待优化目标函数之梯度
        _seed: 迭代起始点
        '''
        self.__func = func
        # self.__grad = _grad
        self.__x0 = x0
        self._get_partial_J = _get_partial_J
        self._partial_x_for_get_partial_diff = _partial_x_for_get_partial_diff
        self.__xPath = list()
        self.__JPath = list()
        self._lb = lb
        self._ub = ub
        self._unit_step = unit_steps

    # def get_partial_J(self,x):
    #     return

    def get_solu(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1.e-8, zeta=1.e-6, maxIter=3000000):
        '''
        获取数值解,
        alpha: 步长参数
        beta1: 一阶矩指数衰减率
        beta2: 二阶矩指数衰减率
        epsilon: 足够小正数
        zeta: 收敛判据
        maxIter: 最大迭代次数
        '''
        self.__init_path()

        x = self.__init_x()
        JVal = self.__calc_JVal(x)
        self.__add_path(x, JVal)
        # grad = self.__calc_grad(x)
        grad = self.__calc_grad(x, JVal)

        m, v = numpy.zeros(x.shape), numpy.zeros(x.shape)
        for k in range(1, maxIter + 1):
            # print("k: {:3d},   JVal: {}".format(k, JVal))
            if self.__converged1(grad, zeta):
                self.__print_MSG(x, JVal, k)
                return x, JVal, True

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad * grad
            m_ = m / (1 - beta1 ** k)
            v_ = v / (1 - beta2 ** k)

            alpha_ = alpha / (numpy.sqrt(v_) + epsilon)
            d = -m_
            xNew = x + alpha_ * d

            # 修复越界参数
            low_index = xNew < self._lb
            xNew[low_index] = self._lb[low_index]
            high_index = xNew > self._ub
            xNew[high_index] = self._ub[high_index]
            xNew = x + ((xNew - x) // self._unit_step) * self._unit_step  # 步长为单位步长的整数倍

            JNew = self.__calc_JVal(xNew)
            self.__add_path(xNew, JNew)
            if self.__converged2(xNew - x, JNew - JVal, zeta ** 2):
                self.__print_MSG(xNew, JNew, k + 1)
                return xNew, JNew, True

            gNew = self.__calc_grad(xNew, JNew)
            x, JVal, grad = xNew, JNew, gNew
            # else:
            if self.__converged1(grad, zeta):
                self.__print_MSG(x, JVal, maxIter)
                return x, JVal, True

        print("Adam not converged after {} steps!".format(maxIter))
        return x, JVal, False

    def get_path(self):
        return self.__xPath, self.__JPath

    def __converged1(self, grad, epsilon):
        if numpy.linalg.norm(grad, ord=numpy.inf) < epsilon:
            return True
        return False

    def __converged2(self, xDelta, JDelta, epsilon):
        val1 = numpy.linalg.norm(xDelta, ord=numpy.inf)
        val2 = numpy.abs(JDelta)
        if val1 < epsilon or val2 < epsilon:
            return True
        return False

    def __print_MSG(self, x, JVal, iterCnt):
        print("Iteration steps: {}".format(iterCnt))
        print("Solution:\n{}".format(x.flatten()))
        print("JVal: {}".format(JVal))

    def __calc_JVal(self, x):
        return self.__func(x)

    def __calc_grad(self, x, function_value):
        # return self.__grad(x)
        return (self._get_partial_J(x,
                                    self._partial_x_for_get_partial_diff) - function_value) / self._partial_x_for_get_partial_diff

    def __init_x(self):
        return self.__x0

    def __init_path(self):
        self.__xPath.clear()
        self.__JPath.clear()

    def __add_path(self, x, JVal):
        self.__xPath.append(x)
        self.__JPath.append(JVal)


class AdamPlot(object):

    @staticmethod
    def plot_fig(adamObj, learn_rate=0.5):
        x, JVal, tab = adamObj.get_solu(learn_rate)
        xPath, JPath = adamObj.get_path()

        fig = plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        ax1.plot(numpy.arange(len(JPath)), JPath, "k.", markersize=1)
        ax1.plot(0, JPath[0], "go", label="starting point")
        ax1.plot(len(JPath) - 1, JPath[-1], "r*", label="solution")

        ax1.legend()
        ax1.set(xlabel="$iterCnt$", ylabel="$JVal$")

        x1 = numpy.linspace(-100, 100, 300)
        x2 = numpy.linspace(-100, 100, 300)
        x1, x2 = numpy.meshgrid(x1, x2)
        f = numpy.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                f[i, j] = func(numpy.array([[x1[i, j]], [x2[i, j]]]))
        ax2.contour(x1, x2, f, levels=36)
        x1Path = list(item[0] for item in xPath)
        x2Path = list(item[1] for item in xPath)
        ax2.plot(x1Path, x2Path, "k--", lw=2)
        ax2.plot(x1Path[0], x2Path[0], "go", label="starting point")
        ax2.plot(x1Path[-1], x2Path[-1], "r*", label="solution")
        ax2.set(xlabel="$x_1$", ylabel="$x_2$")
        ax2.legend()

        fig.tight_layout()
        # plt.show()
        fig.savefig("plot_fig.png")


def _test_func(X):
    x, y = X

    return x ** 2 + y ** 2


def _test_grad(X):
    x, y = X

    return numpy.array((2 * x, 2 * y))


if __name__ == '__main__':
    # adam = Adam(_test_func, numpy.array([1.,111.]),_test_grad,numpy.array([-numpy.inf,-numpy.inf]),numpy.array([+numpy.inf, +numpy.inf]),)
    # res = adam.run(numpy.array((1e-8,1e-8,)),1000)

    # adamObj = Adam(_test_func, _test_grad, numpy.array([500.0, -120]), numpy.array([-100, -100]),
    #                numpy.array([100, 100]),,
    # AdamPlot.plot_fig(adamObj, learn_rate=10)
    plt.show()
