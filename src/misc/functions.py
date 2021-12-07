import numpy as np


def create_linear_function(x, last_x, y, last_y):
    def lf(x_q):
        return last_y + (y - last_y) * (x_q - last_x) / (x - last_x)
    return lf


class PiecewiseContinuousLinearFunction:
    def __init__(self, list_xy_tuples):
        if len(list_xy_tuples) < 2:
            raise IOError(f"PiecewiseLinearFunction initialization requires at least two xy tuples.")
        last_x = None
        last_y = None
        self.list_bounds = []
        self.func_list = []
        sorted_list_xy_tuples = sorted(list_xy_tuples)
        for xy_tuple in sorted_list_xy_tuples:
            x, y = xy_tuple
            if last_x is not None and last_y is not None:
                self.list_bounds.append((last_x, x))
                self.func_list.append(create_linear_function(x, last_x, y, last_y))
            last_x = x
            last_y = y
        #self.func_list.append(-1)

    def get_y(self, x):
        """This function returns the function value of a piecewise continuous linear function defined by
        the xy-tuples of the instance init. Returns 0 in not defined intervals.

        :param x: x-value
        :type x: float
        :return: y-value
        :rtype: int/float
        """
        return np.piecewise(x, [(x >= xl) & (x <= xu) for xl, xu in self.list_bounds],
                            self.func_list)


class PolynomialFunction:
    def __init__(self, list_coefficients):
        self.f = np.polynomial.polynomial.Polynomial(list_coefficients)

    def get_y(self, x):
        return self.f(x)


class HardtDemandFunction:
    def __init__(self, a, b):
        """This function can be used to describe the price sensitivity to users according to
        Hardt, Cornelius; Bogenberger, Klaus (2021): Dynamic Pricing in Free-Floating Carsharing Systems - A Model
        Predictive Control Approach. In: TRB Annual Meeting

        :param a: scale_factor 1
        :param b: scale_factor 2
        """
        self.f = lambda x: (1 + np.exp(a - b)) / (1 + np.exp(a * x - b))

    def get_y(self, x):
        """This function returns the function value of the Hardt'sche demand functions.

        :param x: x-value
        :type x: float
        :return: y-value
        :rtype: int/float
        """
        return self.f(x)


def load_function(input_dict):
    """This function can be used to load functions based on scenario input.

    :param input_dict: dictionary containing the function type and parameters
    :type input_dict: dict
    :return: Function instance
    """
    func_key = input_dict.pop("func_key")
    if func_key == "pcw_lf":
        # x-y points have to be given in dictionary
        list_xy_tuples = input_dict.items()
        return PiecewiseContinuousLinearFunction(list_xy_tuples)
    elif func_key == "poly_f":
        # values saved as dictionary entries with key giving the order
        list_coefficients = []
        for k in sorted(input_dict.keys()):
            list_coefficients.append(input_dict[k])
        return PolynomialFunction(list_coefficients)
    elif func_key == "hardt_demand":
        # values saved as dictionary entries 'a' and 'b'
        a = input_dict["a"]
        b = input_dict["b"]
        return HardtDemandFunction(a, b)
    else:
        raise IOError(f"Unknown function key {func_key}!")


if __name__ == "__main__":
    # PLF
    sample_tuples = [(0,0), (1,1), (2,5)]
    plf = PiecewiseContinuousLinearFunction(sample_tuples)
    print(plf.func_list)
    print(plf.get_y(-2.0))
    print(plf.get_y(0))
    print(plf.get_y(0.5))
    print(plf.get_y(1))
    print(plf.get_y(1.5))
    print(plf.get_y(2))
    print(plf.get_y(3))

    # PF
    a = [1,1,1]
    pf = PolynomialFunction(a)
    print(pf)
    print(pf.get_y(0))
    print(pf.get_y(1))
    print(pf.get_y(2))