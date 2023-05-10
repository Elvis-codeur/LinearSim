from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,\
    NavigationToolbar2QT as NavigationToolbar
import sys
import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")


class AxesCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(AxesCanvas, self).__init__(fig)


class StepWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(StepWindow, self).__init__(*args, **kwargs)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.canvas = AxesCanvas(self, width=5, height=4, dpi=100)

        self.defautltToolbar = NavigationToolbar(self.canvas, self)

        toolbar_layout = QtWidgets.QHBoxLayout()
        toolbar_layout.addWidget(self.defautltToolbar)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.defautltToolbar)
        main_layout.addWidget(self.canvas)

        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(main_layout)
        self.setCentralWidget(centralWidget)

        self.show()

    def plot_h_line(self, t, y):
        self.canvas.axes.plot([0, t], [y, y],
                              color="#21130d", linestyle='dashed',
                              linewidth=1)

    def plot_v_line(self, t, y):
        self.canvas.axes.plot([t, t], [0, y],
                              color="#21130d", linestyle='dashed',
                              linewidth=2)

    def scatter_point(self, t, y, s=100):
        self.canvas.axes.scatter([t, ], [y, ], color="b", s=s)

    def plot(self, time, sortie):
        M = np.max(sortie)
        M_t = time[sortie == M][0]

        steady = sortie[-1]
        steady_t = time[-1]

        self.plot_h_line(steady_t, steady)

        self.scatter_point(M_t, M)

        self.plot_h_line(M_t, M)

        self.plot_v_line(M_t, M)

        print((M_t, M), (steady_t, steady))

        (settling_bas_t, settling_bas), (settling_haut_t,
                                         settling_haut) = self.get_settling_informations(time, sortie, 0.02)

        self.plot_h_line(steady_t, settling_bas)

        self.plot_h_line(steady_t, settling_haut)

        if settling_haut_t > settling_bas_t:
            self.scatter_point(settling_haut_t, settling_haut, 50)
            self.plot_v_line(settling_haut_t, settling_haut)
        else:
            self.scatter_point(settling_bas_t, settling_bas, 50)
            self.plot_v_line(settling_bas_t, settling_bas)



        self.canvas.axes.plot(time, sortie, color="b")
        self.canvas.flush_events()

    def get_settling_informations(self, temps, sortie, pourcentage):
        steady = sortie[-1]
        steady_t = temps[-1]

        derivate_kernel = np.array([1, -1])

        limite_haute = steady*(1 + pourcentage)
        limite_basse = steady*(1 - pourcentage)

        limite_haute_cross = sortie < limite_haute
        limite_basse_cross = sortie > limite_basse

        d_limite_haute_cross = np.convolve(
            limite_haute_cross, derivate_kernel, "same")

        d_limite_basse_cross = np.convolve(
            limite_basse_cross, derivate_kernel, "same")

        limite_haute_temps = temps[d_limite_haute_cross > 0]
        limite_haute_temps = limite_haute_temps[-1]

        limite_basse_temps = temps[d_limite_basse_cross > 0]
        limite_basse_temps = limite_basse_temps[-1]

        print((limite_basse, limite_basse_temps),
              (limite_haute, limite_haute_temps))

        return (limite_basse_temps, limite_basse),\
            (limite_haute_temps, limite_haute)


def test():
    app = QtWidgets.QApplication(sys.argv)
    w = StepWindow()
    app.exec_()


if __name__ == "__main__":
    test()
