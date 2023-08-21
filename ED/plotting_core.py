from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,\
    NavigationToolbar2QT as NavigationToolbar
import sys
import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")


class AxesCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=200):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(AxesCanvas, self).__init__(self.fig)


class StepWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(StepWindow, self).__init__(*args, **kwargs)

        self.times_plotted = []
        self.sortie_plotted = []
        self.scatter_info_dict = {}

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
                              linewidth=1)

    def scatter_point(self, t, y, s=60, info={}):
        self.scatter_info_dict[(t, y)] = info

        self.times_plotted.append(t)
        self.sortie_plotted.append(y)
        self.colors = [(0, 0, 0) for i in self.times_plotted]
        self.scatter = self.canvas.axes.scatter(self.times_plotted, self.sortie_plotted,
                                                s=s,
                                                c=self.colors,
                                                cmap=self.plt_cmap,
                                                norm=self.plt_norm)
        print(self.scatter.get_offsets())

    def plot_depassement_info(self, Max_time, Max):

        self.plot_h_line(Max_time, Max)

        self.plot_v_line(Max_time, Max)

        self.scatter_point(Max_time, Max)

    def plot_steady_info(self, steady_t, steady):
        self.plot_h_line(steady_t, steady)
        self.scatter_point(steady_t, steady)

    def plot_settling_info(self, time, sortie, steady_t, pourcentage):
        (settling_bas_t, settling_bas), (settling_haut_t,
                                         settling_haut) = self.get_settling_informations(time, sortie, pourcentage)

        self.plot_h_line(steady_t, settling_bas)

        self.plot_h_line(steady_t, settling_haut)

        if settling_haut_t > settling_bas_t:
            self.scatter_point(settling_haut_t, settling_haut, 50)
            self.plot_v_line(settling_haut_t, settling_haut)
        else:
            self.scatter_point(settling_bas_t, settling_bas, 50)
            self.plot_v_line(settling_bas_t, settling_bas)

    def plot_rising_information(self, time, sortie, pourcentage1, pourcente2):
        (p1_temps, p1), (p2_temps, p2) = self.get_rise_informations(
            time, sortie, pourcentage1, pourcente2)

        self.plot_h_line(p1_temps, p1)
        self.plot_h_line(p2_temps, p2)

        self.plot_v_line(p1_temps, p1)
        self.plot_v_line(p2_temps, p2)

        info = {}
        info["name"] = "RiseTime"
        info["pourcentage"] = pourcentage1

        self.scatter_point(p2_temps, p2, info = info)
        self.scatter_point(p1_temps, p1)

    def plot(self, time, sortie):
        M = np.max(sortie)
        M_t = time[sortie == M][0]

        steady = sortie[-1]
        steady_t = time[-1]

        #print((M_t, M), (steady_t, steady))

        self.plt_norm = plt.Normalize(1, 4)
        self.plt_cmap = plt.cm.PiYG

        # Step 2. Create Annotation Object
        self.annotation = self.canvas.axes.annotate(
            text='',
            xy=(0, 0),
            xytext=(15, 15),  # distance from x, y
            textcoords='offset points',
            bbox={'boxstyle': 'round', 'fc': 'w'},
            arrowprops={'arrowstyle': '->'}
        )
        self.annotation.set_visible(False)

        self.plot_steady_info(steady_t,steady)

        self.canvas.axes.plot(time, sortie, color="b")
        self.canvas.flush_events()
        self.canvas.axes.set_xlabel("temps")
        self.canvas.axes.set_ylabel("Amplitude")
        self.canvas.axes.set_title("Réponse du système")
        self.canvas.axes.legend()
        self.canvas.fig.canvas.mpl_connect(
            'motion_notify_event', self.motion_hover)

    def motion_hover(self, event):
        annotation_visbility = self.annotation.get_visible()
        if event.inaxes == self.canvas.axes:
            is_contained, annotation_index = self.scatter.contains(event)
            if is_contained:
                data_point_location = self.scatter.get_offsets()[
                    annotation_index['ind'][0]]
                print(data_point_location)
                self.annotation.xy = data_point_location

                text_label = ""

                info = self.scatter_info_dict[tuple(data_point_location)]

                if(info["name"] == "RiseTime"):

                    text_label = 'Temps de montée à {:.2f} : {:.2f}'.format(
                        info["pourcentage"], data_point_location[1])

                self.annotation.set_text(text_label)

                """
                self.annotation.get_bbox_patch().set_facecolor(
                    self.plt_cmap(self.plt_norm(self.colors[annotation_index['ind'][0]])))  # ]
                """

                self.annotation.set_alpha(0.4)

                self.annotation.set_visible(True)
                self.canvas.fig.canvas.draw_idle()
            else:
                if annotation_visbility:
                    self.annotation.set_visible(False)
                    self.canvas.fig.canvas.draw_idle()

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

    def get_rise_informations(self, temps, sortie, pourcentage1, pourcentage2):
        steady = sortie[-1]

        pourcentage1_cross = sortie > steady*pourcentage1

        pourcentage1_temps = temps[pourcentage1_cross]
        pourcentage1_temps = pourcentage1_temps[0]

        pourcentage1_sortie = sortie[pourcentage1_cross]
        pourcentage1_sortie = pourcentage1_sortie[0]

        pourcentage2_cross = sortie > steady*pourcentage2

        pourcentage2_temps = temps[pourcentage2_cross]
        pourcentage2_temps = pourcentage2_temps[0]

        pourcentage2_sortie = sortie[pourcentage2_cross]
        pourcentage2_sortie = pourcentage2_sortie[0]

        return (pourcentage1_temps, pourcentage1_sortie),\
            (pourcentage2_temps, pourcentage2_sortie)


def test():
    app = QtWidgets.QApplication(sys.argv)
    w = StepWindow()
    app.exec_()


if __name__ == "__main__":
    test()
