# %%
import math
import numpy as np
import matplotlib.pyplot as plt


class NormalDistribution:
    def __init__(self, data):
        self.data = data
        self.size = len(data)

        self.expectation = np.mean(data)

        self.var = np.var(data)

        self.var_corrected = np.var(data, ddof=1)

    def __str__(self):
        return f"""
            Математичне сподівання: {self.expectation}
            Виправлена дисперсія: {self.var_corrected}
        """

    def __len__(self):
        return self.size


class BaseDistribution:
    def __init__(self, lam, data):
        self.lam = lam
        self.data = data

        self.size = len(data)

        self.mean = np.mean(data)

    def log_likelihood(self, lam):
        pass

    def lambda_mle(self, n):
        pass

    def plot(self, lams, lam_true):
        self.plot_log_likelihood(lams)
        self.plot_lambda_mle(lam_true)

    def plot_log_likelihood(self, lams):
        mle = [self.log_likelihood(lam) for lam in lams]

        plt.figure(figsize=(10, 5))
        plt.plot(lams, mle, color="r", label="LnL($\lambda$)")
        plt.xlabel("$\lambda$")
        plt.ylabel("LnL($\lambda$)")
        plt.legend()
        plt.show()

    def plot_lambda_mle(self, lam_true):
        mle = [self.lambda_mle(n) for n in range(1, len(self.data) + 1)]

        plt.figure(figsize=(10, 5))
        plt.scatter(
            range(1, len(self.data) + 1), mle, label="$\lambda(n)$", marker="o", s=5
        )
        plt.xlabel("n")
        plt.ylabel("$\lambda(n)$")
        plt.axhline(y=lam_true, color="r", linestyle="--", label=f"{lam_true}")
        plt.legend()
        plt.show()

    def lambdas(self, n: list):
        return [self.lambda_mle(i) for i in n]


class PoissonDistribution(BaseDistribution):
    def __init__(self, lam, size):
        data = np.random.poisson(lam, size)
        super().__init__(lam, data)

    def log_likelihood(self, lam):
        return (
            -self.size
            + lam
            + np.sum(self.data) * np.log(lam)
            - np.sum(np.log(np.vectorize(math.factorial)(self.data)))
        )

    def lambda_mle(self, n):
        return np.sum(self.data[:n]) / n


class ExponentialDistribution(BaseDistribution):
    def __init__(self, lam, size):
        data = np.random.exponential(1 / lam, size)
        super().__init__(lam, data)

    def log_likelihood(self, lam):
        return self.size * np.log(lam) - lam * np.sum(self.data)

    def lambda_mle(self, n):
        return n / np.sum(self.data[:n])


def print_lambdas(dist: BaseDistribution, n: list):
    lambdas = dist.lambdas(n)
    print("Оцінки параметра λ: ")
    print("\n".join([f"λ({n[i]})={lam}" for i, lam in enumerate(lambdas)]))


def main():
    np.random.seed(0)

    data = np.random.normal(10, 9, 2002)

    normal_dist = NormalDistribution(data)

    print(normal_dist)

    lam = 0.6

    N = [i * 6 for i in range(10, 60, 10)]

    n = len(data)

    poisson_dist = PoissonDistribution(lam, n)

    lams = np.linspace(0.01, lam * 2, 1000)

    poisson_dist.plot(lams, poisson_dist.lam)

    print_lambdas(poisson_dist, N)

    exponential_dist = ExponentialDistribution(lam, n)

    exponential_dist.plot(lams, exponential_dist.lam)

    print_lambdas(exponential_dist, N)


if __name__ == "__main__":
    main()

# %%
