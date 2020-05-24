import numpy as np


class DailyData():
    def __init__(self, w_1, w_2, w_3, mu_1, mu_2, mu_3, sigma_1, sigma_2, sigma_3):
        super().__init__()
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.sigma_3 = sigma_3


class DailyGraph():
    def __init__(self, daily_data: DailyData, reso=500, time_shift=None):
        super().__init__()
        self.daily_data = daily_data
        self._x = np.linspace(0, 24, reso)
        self.time_shift = list(time_shift) if time_shift != None else list()
        if 0 in self.time_shift:
            self.time_shift.remove(0)

    def _daily_func(self, x):
        from scipy.stats import norm

        def _base(x, offset=0): return self.daily_data.w_1*norm.pdf(x, self.daily_data.mu_1+offset, self.daily_data.sigma_1) + self.daily_data.w_2 * norm.pdf(
            x, self.daily_data.mu_2+offset, self.daily_data.sigma_2) + self.daily_data.w_3 * norm.pdf(x, self.daily_data.mu_3 + offset, self.daily_data.sigma_3)
        ris = _base(x) + _base(x, -24) + _base(x, 24)
        for shift in self.time_shift:
            ris += _base(x, shift) + _base(x, shift-24) + _base(x, shift+24)
        return ris

    def run(self, file_name='daily_graph.png', title='Probabilty distribution during a normal day'):
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        # calculate y
        y = self._daily_func(self._x)
        # normalize y
        y = y / self._daily_func(np.linspace(0, 24, 25)).sum()

        fig, ax = plt.subplots(figsize=(9, 6))
        plt.plot(self._x, y, color='k', lw=2, path_effects=[
            pe.Stroke(linewidth=5, foreground='b'), pe.Normal()])

        hour_function = y[(np.linspace(0, 1, 25)*499).astype(np.int)]
        print(hour_function)

        plt.plot(np.linspace(0, 24, 25), hour_function, 'ro')
        ax.fill_between(self._x, y, 0, alpha=0.1, color='b')
        ax.set_xlim([-0, 25])
        ax.set_xticks(np.arange(0, 25, 1))
        ax.set_xlabel('Time of day')
        ax.set_title(title)

        plt.style.use('fivethirtyeight')
        plt.savefig(file_name, dpi=72, bbox_inches='tight')
        plt.grid()
        plt.show()


def normal_day(time_shift=None):
    normal_day_data = DailyData(w_1=1.2/6, w_2=2/6, w_3=2.2/6, mu_1=8, mu_2=12,
                                mu_3=19, sigma_1=1, sigma_2=1.5, sigma_3=1.8)
    normal_daily_graph = DailyGraph(normal_day_data, time_shift=time_shift)
    normal_daily_graph.run(file_name='normal_daily_graph.png', title="Probabilty distribution during a normal day." +
                           ((" GMT" + str(time_shift)) if time_shift != None else ""))


def weekend_day(time_shift=None):
    weekend_day_data = DailyData(w_1=1.2/6, w_2=2/6, w_3=3.1/6, mu_1=8, mu_2=12,
                                 mu_3=20, sigma_1=1, sigma_2=1.5, sigma_3=2.3)
    weekend_daily_graph = DailyGraph(weekend_day_data, time_shift=time_shift)
    weekend_daily_graph.run(file_name='week_end_daily_graph.png', title="Probabilty distribution during the weekend days." +
                            ((" GMT" + str(time_shift)) if time_shift != None else ""))


if __name__ == "__main__":
    # normal/weekend analysis
    ris = ""
    while(ris.upper() not in ["N", "W"]):
        ris = input("normal (N) / weekend (W): ")

    # timezones analysis
    time_ris = ""
    while(time_ris.upper() not in ["Y", "N"]):
        time_ris = input("time shift Y/N: ")
    time_shift = [0, 1, 2] if time_ris.upper() == "Y" else None

    # execution
    normal_day(time_shift) if ris.upper() == "N" else weekend_day(time_shift)
