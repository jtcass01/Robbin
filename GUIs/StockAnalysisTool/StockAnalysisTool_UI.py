from os import sep
from enum import IntEnum, Enum
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Union
import dateutil.relativedelta

from PyQt5.QtCore import Qt, QMetaObject, QCoreApplication
from PyQt5.QtGui import QFont, QColor, QFontMetrics
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QComboBox, QCheckBox, \
    QPushButton, QSizePolicy, QStatusBar, QApplication, QGridLayout, QTextEdit
from qdarkstyle import load_stylesheet
import pyqtgraph as pg

from utilities.FeatureEngineering import chande_momentum_oscillator, WilliamsR, CMO, MACD, ROC, WMA, HMA, TRIX, CCI, DPO, CMF, ADX, ForceIndex
from MarketInterfaces.MarketDataInterface import retrieve_yahoo_fin_stock_data

from ta.momentum import rsi, wr, roc
from ta.trend import macd_signal, macd, cci, dpo, adx
from ta.volume import chaikin_money_flow


class Ui_StockAnalysisTool(QMainWindow):
    def __init__(self, relative_path_correction: str = ""):
        super(Ui_StockAnalysisTool, self).__init__()

        # TODO
        self.valid_stock_tickers = ["AMZN"]

        # Initialize Object Attributes
        self.source = list(Ui_StockAnalysisTool.StockSources)[0]
        self.sample_rates = [sample_rate.value for sample_rate in Ui_StockAnalysisTool.SampleRates]
        self.time_deltas = [time_delta.value for time_delta in Ui_StockAnalysisTool.TimeDeltas]
        self.stock = self.valid_stock_tickers[0]
        self.df = retrieve_yahoo_fin_stock_data(ticker=self.stock)
        self.first_date = self.df.index[0]
        self.last_date = self.df.index[-1]
        self.data_lines = {}
        self.indicators = {}
        self.colors = {
            'blue': QColor(0, 0, 255, 255),
            'cyan': QColor(0, 255, 255, 255),
            'magenta': QColor(255, 0, 255, 255),
            'yellow': QColor(255, 255, 0, 255),
            'white': QColor(255, 255, 255, 255),
            'dark-gray': QColor(125, 125, 125, 255),
            'light-gray': QColor(200, 200, 200, 255),
            'gray': QColor(100, 100, 150, 255),
            'orange': QColor(255, 165, 0, 255),
            'salmon': QColor(250, 128, 144, 255),
            'violet': QColor(230, 130, 238, 255),
            'aqua-marine': QColor(127, 255, 212, 255)
        }
        self.color_iterator = 0
        self.protected_colors = {
            'green': QColor(0, 255, 0, 255),
            'red': QColor(255, 0, 0, 255)
        }
        self.rsi_n = 14
        self.cmo_n = 7
        self.macd_slow = 26
        self.macd_fast = 12
        self.macd_sign = 9
        self.roc_n = 12
        self.cci_n = 20
        self.dpo_n = 20
        self.cmf_n = 20
        self.adx_n = 14

        # Initialize UI Attributes. Then Initialize UI
        self.central_widget = None
        self.central_layout = None
        self.title_label = None
        self.title_h_divider = None
        self.display_layout = None
        self.graph_layout = None
        self.graph = None
        self.date_axis = None
        self.graph_legend = None
        self.data_selection_layout = None
        self.open_data_cb = None
        self.high_data_cb = None
        self.low_data_cb = None
        self.close_data_cb = None
        self.adjclose_data_cb = None
        self.volume_data_cb = None
        self.save_fig_btn = None
        self.graph_options_layout = None
        self.sample_rate_label = None
        self.sample_rate_combo = None
        self.time_delta_label = None
        self.time_delta_combo = None
        self.main_v_divider = None
        self.options_layout = None
        self.source_selection_layout = None
        self.source_label = None
        self.source_combo = None
        self.stock_selection_layout = None
        self.stock_label = None
        self.stock_combo = None
        self.top_h_divider = None
        self.momentum_indicator_label = None
        self.momentum_indicator_layout = None
        self.rsi_cb = None
        self.rsi_time_frame_label = None
        self.rsi_time_frame_text = None
        self.williams_r_cb = None
        self.cmo_cb = None
        self.macd_cb = None
        self.roc_cb = None
        self.middle_h_divider = None
        self.averages_label = None
        self.averages_layout = None
        self.wma_cb = None
        self.ema_cb = None
        self.sma_cb = None
        self.hma_cb = None
        self.trix_cb = None
        self.bottom_h_divider = None
        self.trend_indicators_label = None
        self.trend_indicators_layout = None
        self.cci_cb = None
        self.dpo_cb = None
        self.cmf_cb = None
        self.adx_cb = None
        self.force_index_cb = None
        self.statusbar = None
        self.setupUi()

    def setupUi(self) -> None:
        """

        :return:
        """
        # Generated Setup Code
        self.setObjectName("StockAnalysisTool_Ui")
        self.central_widget = QWidget(self)
        self.central_widget.setObjectName("central_widget")
        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.setObjectName("central_layout")
        self.title_label = QLabel(self.central_widget)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.title_label.sizePolicy().hasHeightForWidth())
        self.title_label.setSizePolicy(sizePolicy)
        font = QFont()
        font.setFamily("Garamond")
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.title_label.setFont(font)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName("title_label")
        self.central_layout.addWidget(self.title_label)
        self.title_h_divider = QFrame(self.central_widget)
        self.title_h_divider.setFrameShape(QFrame.HLine)
        self.title_h_divider.setFrameShadow(QFrame.Sunken)
        self.title_h_divider.setObjectName("title_h_divider")
        self.central_layout.addWidget(self.title_h_divider)
        self.display_layout = QHBoxLayout()
        self.display_layout.setObjectName("display_layout")
        self.graph_layout = QVBoxLayout()
        self.graph_layout.setObjectName("graph_layout")
        self.date_axis = pg.DateAxisItem(orientation='bottom')
        self.graph = pg.PlotWidget(axisItems={'bottom': self.date_axis})
        self.graph_legend = self.graph.addLegend()
        self.graph_layout.addWidget(self.graph)
        self.data_selection_layout = QGridLayout()
        self.open_data_cb = QCheckBox(self.central_widget)
        self.open_data_cb.setObjectName("open")
        self.data_selection_layout.addWidget(self.open_data_cb, 0, 0, 1, 1)
        self.high_data_cb = QCheckBox(self.central_widget)
        self.high_data_cb.setObjectName("high")
        self.data_selection_layout.addWidget(self.high_data_cb, 0, 1, 1, 1)
        self.low_data_cb = QCheckBox(self.central_widget)
        self.low_data_cb.setObjectName("low")
        self.data_selection_layout.addWidget(self.low_data_cb, 0, 2, 1, 1)
        self.close_data_cb = QCheckBox(self.central_widget)
        self.close_data_cb.setObjectName("close")
        self.data_selection_layout.addWidget(self.close_data_cb, 0, 3, 1, 1)
        self.adjclose_data_cb = QCheckBox(self.central_widget)
        self.adjclose_data_cb.setObjectName("adjclose")
        self.data_selection_layout.addWidget(self.adjclose_data_cb, 0, 4, 1, 1)
        self.volume_data_cb = QCheckBox(self.central_widget)
        self.volume_data_cb.setObjectName("volume")
        self.data_selection_layout.addWidget(self.volume_data_cb, 0, 5, 1, 1)
        self.graph_layout.addLayout(self.data_selection_layout)
        self.save_fig_btn = QPushButton(self.central_widget)
        self.save_fig_btn.setObjectName("save_fig_btn")
        self.graph_layout.addWidget(self.save_fig_btn)
        self.graph_options_layout = QGridLayout()
        self.sample_rate_label = QLabel(self.central_widget)
        self.sample_rate_label.setText("Sample Rate:")
        self.graph_options_layout.addWidget(self.sample_rate_label, 0, 0, 1, 1)
        self.sample_rate_combo = QComboBox(self.central_widget)
        self.sample_rate_combo.addItems(self.sample_rates)
        self.graph_options_layout.addWidget(self.sample_rate_combo, 0, 1, 1, 1)
        self.time_delta_label = QLabel(self.central_widget)
        self.time_delta_label.setText("Time delta:")
        self.graph_options_layout.addWidget(self.time_delta_label, 0, 2, 1, 1)
        self.time_delta_combo = QComboBox(self.central_widget)
        self.time_delta_combo.addItems(self.time_deltas)
        self.graph_options_layout.addWidget(self.time_delta_combo, 0, 3, 1, 1)
        self.graph_layout.addLayout(self.graph_options_layout)
        self.display_layout.addLayout(self.graph_layout)
        self.main_v_divider = QFrame(self.central_widget)
        self.main_v_divider.setFrameShape(QFrame.VLine)
        self.main_v_divider.setFrameShadow(QFrame.Sunken)
        self.main_v_divider.setObjectName("main_v_divider")
        self.display_layout.addWidget(self.main_v_divider)
        self.options_layout = QVBoxLayout()
        self.options_layout.setObjectName("options_layout")
        self.source_selection_layout = QHBoxLayout()
        self.source_selection_layout.setObjectName("source_selection_layout")
        self.source_label = QLabel(self.central_widget)
        self.source_label.setObjectName("source_label")
        self.source_selection_layout.addWidget(self.source_label)
        self.source_combo = QComboBox(self.central_widget)
        self.source_combo.setObjectName("source_combo")
        self.source_combo.addItem("")
        self.source_combo.addItem("")
        self.source_combo.addItem("")
        self.source_selection_layout.addWidget(self.source_combo)
        self.options_layout.addLayout(self.source_selection_layout)
        self.stock_selection_layout = QHBoxLayout()
        self.stock_selection_layout.setObjectName("stock_selection_layout")
        self.stock_label = QLabel(self.central_widget)
        self.stock_label.setObjectName("stock_label")
        self.stock_selection_layout.addWidget(self.stock_label)
        self.stock_combo = QComboBox(self.central_widget)
        self.stock_combo.setObjectName("stock_combo")
        self.stock_combo.addItems(self.valid_stock_tickers)
        self.stock_selection_layout.addWidget(self.stock_combo)
        self.options_layout.addLayout(self.stock_selection_layout)
        self.top_h_divider = QFrame(self.central_widget)
        self.top_h_divider.setFrameShape(QFrame.HLine)
        self.top_h_divider.setFrameShadow(QFrame.Sunken)
        self.top_h_divider.setObjectName("top_h_divider")
        self.options_layout.addWidget(self.top_h_divider)

        self.momentum_indicator_label = QLabel(self.central_widget)
        self.momentum_indicator_label.setObjectName("momentum_indicator_label")
        self.options_layout.addWidget(self.momentum_indicator_label)
        # Momentum Indicators
        self.momentum_indicator_layout = QGridLayout()
        self.rsi_cb = QCheckBox(self.central_widget)
        self.rsi_cb.setObjectName("rsi_cb")
        self.momentum_indicator_layout.addWidget(self.rsi_cb, 0, 0, 1, 1)
        self.rsi_time_frame_label = QLabel(self.central_widget)
        self.rsi_time_frame_label.setText("Time frame:")
        self.momentum_indicator_layout.addWidget(self.rsi_time_frame_label, 0, 1, 1, 1)
        self.rsi_time_frame_text = QTextEdit(self.central_widget)
        font_metric = QFontMetrics(self.rsi_time_frame_text.font())
        self.rsi_time_frame_text.setFixedHeight(font_metric.lineSpacing())
        self.rsi_time_frame_text.setFixedWidth(50)
        self.rsi_time_frame_text.setText(str(self.rsi_n))
        self.momentum_indicator_layout.addWidget(self.rsi_time_frame_text, 0, 2, 1, 1)
        self.williams_r_cb = QCheckBox(self.central_widget)
        self.williams_r_cb.setObjectName("williams_r_cb")
        self.momentum_indicator_layout.addWidget(self.williams_r_cb, 1, 0, 1, 1)
        self.cmo_cb = QCheckBox(self.central_widget)
        self.cmo_cb.setObjectName("cmo_cb")
        self.momentum_indicator_layout.addWidget(self.cmo_cb, 2, 0, 1, 1)
        self.macd_cb = QCheckBox(self.central_widget)
        self.macd_cb.setObjectName("macd_cb")
        self.momentum_indicator_layout.addWidget(self.macd_cb, 3, 0, 1, 1)
        self.roc_cb = QCheckBox(self.central_widget)
        self.roc_cb.setObjectName("roc_cb")
        self.momentum_indicator_layout.addWidget(self.roc_cb, 4, 0, 1, 1)
        self.middle_h_divider = QFrame(self.central_widget)
        self.middle_h_divider.setFrameShape(QFrame.HLine)
        self.middle_h_divider.setFrameShadow(QFrame.Sunken)
        self.middle_h_divider.setObjectName("middle_h_divider")
        self.options_layout.addLayout(self.momentum_indicator_layout)
        self.options_layout.addWidget(self.middle_h_divider)

        # Averages Indicators
        self.averages_label = QLabel(self.central_widget)
        self.averages_label.setObjectName("averages_label")
        self.options_layout.addWidget(self.averages_label)
        self.wma_cb = QCheckBox(self.central_widget)
        self.wma_cb.setObjectName("wma_cb")
        self.options_layout.addWidget(self.wma_cb)
        self.ema_cb = QCheckBox(self.central_widget)
        self.ema_cb.setObjectName("ema_cb")
        self.options_layout.addWidget(self.ema_cb)
        self.sma_cb = QCheckBox(self.central_widget)
        self.sma_cb.setObjectName("sma_cb")
        self.options_layout.addWidget(self.sma_cb)
        self.hma_cb = QCheckBox(self.central_widget)
        self.hma_cb.setObjectName("hma_cb")
        self.options_layout.addWidget(self.hma_cb)
        self.trix_cb = QCheckBox(self.central_widget)
        self.trix_cb.setObjectName("trix_cb")
        self.options_layout.addWidget(self.trix_cb)
        self.bottom_h_divider = QFrame(self.central_widget)
        self.bottom_h_divider.setFrameShape(QFrame.HLine)
        self.bottom_h_divider.setFrameShadow(QFrame.Sunken)
        self.bottom_h_divider.setObjectName("bottom_h_divider")
        self.options_layout.addWidget(self.bottom_h_divider)

        # Trend Indicators
        self.trend_indicators_label = QLabel(self.central_widget)
        self.trend_indicators_label.setObjectName("trend_indicators_label")
        self.options_layout.addWidget(self.trend_indicators_label)
        self.cci_cb = QCheckBox(self.central_widget)
        self.cci_cb.setObjectName("cci_cb")
        self.options_layout.addWidget(self.cci_cb)
        self.dpo_cb = QCheckBox(self.central_widget)
        self.dpo_cb.setObjectName("dpo_cb")
        self.options_layout.addWidget(self.dpo_cb)
        self.cmf_cb = QCheckBox(self.central_widget)
        self.cmf_cb.setObjectName("cmf_cb")
        self.options_layout.addWidget(self.cmf_cb)
        self.adx_cb = QCheckBox(self.central_widget)
        self.adx_cb.setObjectName("adx_cb")
        self.options_layout.addWidget(self.adx_cb)
        self.force_index_cb = QCheckBox(self.central_widget)
        self.force_index_cb.setObjectName("checkBox_14")
        self.options_layout.addWidget(self.force_index_cb)
        self.display_layout.addLayout(self.options_layout)
        self.central_layout.addLayout(self.display_layout)
        self.setCentralWidget(self.central_widget)
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        self.setCallbacks()
        QMetaObject.connectSlotsByName(self)

    def setCallbacks(self) -> None:
        """

        :return:
        """
        self.sample_rate_combo.currentIndexChanged.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.sample_rate_combo_changed(combo=self.sample_rate_combo)
        )
        self.time_delta_combo.currentIndexChanged.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.time_delta_combo_changed(stock_analysis_tool=self,
                                                                            combo=self.time_delta_combo)
        )
        self.source_combo.currentIndexChanged.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.source_combo_changed(combo=self.source_combo)
        )
        self.stock_combo.currentIndexChanged.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.stock_combo_changed(stock_analysis_tool=self,
                                                                       combo=self.stock_combo,
                                                                       source=self.source))
        self.open_data_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.open_data_cb_pressed(stock_analysis_tool=self,
                                                                        cb=self.open_data_cb)
        )
        self.high_data_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.high_data_cb_pressed(stock_analysis_tool=self,
                                                                        cb=self.high_data_cb)
        )
        self.low_data_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.low_data_cb_pressed(stock_analysis_tool=self,
                                                                       cb=self.low_data_cb)
        )
        self.close_data_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.close_data_cb_pressed(stock_analysis_tool=self,
                                                                         cb=self.close_data_cb)
        )
        self.adjclose_data_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.adjclose_data_cb_pressed(stock_analysis_tool=self,
                                                                            cb=self.adjclose_data_cb)
        )
        self.volume_data_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.volume_data_cb_pressed(stock_analysis_tool=self,
                                                                          cb=self.volume_data_cb)
        )
        self.rsi_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.rsi_cb_pressed(stock_analysis_tool=self,
                                                                  cb=self.rsi_cb))
        self.rsi_time_frame_text.textChanged.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.rsi_time_frame_text_changed(
                stock_analysis_tool=self,
                rsi_time_frame_text=self.rsi_time_frame_text
            )
        )
        self.williams_r_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.williams_r_cb_pressed(stock_analysis_tool=self,
                                                                         cb=self.williams_r_cb))
        self.cmo_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.cmo_cb_pressed(stock_analysis_tool=self,
                                                                  cb=self.cmo_cb))
        self.macd_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.macd_cb_pressed(stock_analysis_tool=self,
                                                                   cb=self.macd_cb))
        self.roc_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.roc_cb_pressed(stock_analysis_tool=self,
                                                                  cb=self.roc_cb))
        self.wma_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.wma_cb_pressed(cb=self.wma_cb))
        self.ema_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.ema_cb_pressed(cb=self.ema_cb))
        self.sma_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.sma_cb_pressed(cb=self.sma_cb))
        self.hma_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.hma_cb_pressed(cb=self.hma_cb))
        self.trix_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.trix_cb_pressed(cb=self.trix_cb))
        self.cci_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.cci_cb_pressed(stock_analysis_tool=self,
                                                                  cb=self.cci_cb))
        self.dpo_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.dpo_cb_pressed(stock_analysis_tool=self,
                                                                  cb=self.dpo_cb))
        self.cmf_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.cmf_cb_pressed(stock_analysis_tool=self,
                                                                  cb=self.cmf_cb))
        self.adx_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.adx_cb_pressed(stock_analysis_tool=self,
                                                                  cb=self.adx_cb))
        self.force_index_cb.clicked.connect(
            lambda: Ui_StockAnalysisTool.Callbacks.force_index_cb_pressed(cb=self.force_index_cb))

    def retranslateUi(self) -> None:
        """

        :return:
        """
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("StockAnalysisTool_Ui", "Stock Analysis Tool"))
        self.title_label.setText(_translate("StockAnalysisTool_Ui", "ASTRA Stock Analysis Tool"))
        self.save_fig_btn.setText(_translate("StockAnalysisTool_Ui", "Export Graph"))
        self.source_label.setText(_translate("StockAnalysisTool_Ui", "Source:"))
        self.source_combo.setItemText(0, _translate("StockAnalysisTool_Ui", "yahoo_fin"))
        self.source_combo.setItemText(1, _translate("StockAnalysisTool_Ui", "ASTRASS"))
        self.source_combo.setItemText(2, _translate("StockAnalysisTool_Ui", "yfinance"))
        self.stock_label.setText(_translate("StockAnalysisTool_Ui", "Stock:"))
        self.momentum_indicator_label.setText(_translate("StockAnalysisTool_Ui", "Momentum Indicators"))
        self.open_data_cb.setText(_translate("StockAnalysisTool_Ui", "open"))
        self.high_data_cb.setText(_translate("StockAnalysisTool_Ui", "high"))
        self.low_data_cb.setText(_translate("StockAnalysisTool_Ui", "low"))
        self.close_data_cb.setText(_translate("StockAnalysisTool_Ui", "close"))
        self.adjclose_data_cb.setText(_translate("StockAnalysisTool_Ui", "adjclose"))
        self.volume_data_cb.setText(_translate("StockAnalysisTool_Ui", "volume"))
        self.rsi_cb.setText(_translate("StockAnalysisTool_Ui", "RSI"))
        self.williams_r_cb.setText(_translate("StockAnalysisTool_Ui", "WilliamsR"))
        self.cmo_cb.setText(_translate("StockAnalysisTool_Ui", "CMO"))
        self.macd_cb.setText(_translate("StockAnalysisTool_Ui", "MACD"))
        self.roc_cb.setText(_translate("StockAnalysisTool_Ui", "ROC"))
        self.averages_label.setText(_translate("StockAnalysisTool_Ui", "Averages"))
        self.wma_cb.setText(_translate("StockAnalysisTool_Ui", "WMA"))
        self.ema_cb.setText(_translate("StockAnalysisTool_Ui", "EMA"))
        self.sma_cb.setText(_translate("StockAnalysisTool_Ui", "SMA"))
        self.hma_cb.setText(_translate("StockAnalysisTool_Ui", "HMA"))
        self.trix_cb.setText(_translate("StockAnalysisTool_Ui", "TRIX"))
        self.trend_indicators_label.setText(_translate("StockAnalysisTool_Ui", "Trend Indicators"))
        self.cci_cb.setText(_translate("StockAnalysisTool_Ui", "CCI"))
        self.dpo_cb.setText(_translate("StockAnalysisTool_Ui", "DPO"))
        self.cmf_cb.setText(_translate("StockAnalysisTool_Ui", "CMF"))
        self.adx_cb.setText(_translate("StockAnalysisTool_Ui", "ADX"))
        self.force_index_cb.setText(_translate("StockAnalysisTool_Ui", "Force Index"))

    def add_column_to_graph(self, column_name: str,
                            color: Union[QColor, None] = None) -> None:
        """

        :param column_name:
        :param color:
        :return:
        """
        relevant_df = self.get_relevant_data_frame()

        x = [datetime.timestamp(date_time) for date_time in relevant_df.index]
        if color is None:
            self.data_lines[column_name] = self.graph.plot(x=x,
                                                           y=relevant_df[column_name],
                                                           pen=pg.mkPen(color=self.colors[list(self.colors.keys())[self.color_iterator]]),
                                                           name=column_name)
            self.color_iterator += 1
            self.color_iterator %= len(self.colors.keys())
        else:
            self.data_lines[column_name] = self.graph.plot(x=x,
                                                           y=relevant_df[column_name],
                                                           pen=pg.mkPen(color=color),
                                                           name=column_name)

    def remove_column_from_graph(self, column_name: str) -> None:
        """

        :param column_name:
        :return:
        """
        self.graph.removeItem(self.data_lines[column_name])
        del self.data_lines[column_name]

    def update_data_on_graph(self, data_frame: DataFrame) -> None:
        """

        :param data_frame:
        :return:
        """
        x = [datetime.timestamp(date_time) for date_time in data_frame.index]

        for data_line_key, data_line in self.data_lines.items():
            data_line.setData(x=x, y=data_frame[data_line_key])

    def get_relevant_data_frame(self) -> DataFrame:
        """

        :return:
        """
        return self.df.truncate(before=self.first_date, after=self.last_date)

    class Callbacks:
        @staticmethod
        def generic_data_cb_pressed(stock_analysis_tool: QMainWindow,
                                    checked: bool,
                                    df_column_name: str) -> None:
            if checked:
                # Add open data to graph
                stock_analysis_tool.add_column_to_graph(column_name=df_column_name)
            else:
                # Remove open data from graph
                stock_analysis_tool.remove_column_from_graph(column_name=df_column_name)

        @staticmethod
        def source_combo_changed(combo: QComboBox) -> None:
            combo_text = combo.currentText()

        @staticmethod
        def stock_combo_changed(stock_analysis_tool: QMainWindow,
                                combo: QComboBox,
                                source: IntEnum) -> None:
            combo_text = combo.currentText()

            if source == Ui_StockAnalysisTool.StockSources.YAHOO_FIN:
                stock_analysis_tool.df = retrieve_yahoo_fin_stock_data(ticker=combo_text)
            elif source == Ui_StockAnalysisTool.StockSources.ASTRASS:
                # retrieve data from ASTRASS
                stock_analysis_tool.df = None
            elif source == Ui_StockAnalysisTool.StockSources.YFINANCE:
                # retrieve data from YFinance
                stock_analysis_tool.df = None

            stock_analysis_tool.update_data_on_graph(data_frame=stock_analysis_tool.df)

        @staticmethod
        def sample_rate_combo_changed(combo: QComboBox) -> None:
            combo_text = combo.currentText()

        @staticmethod
        def time_delta_combo_changed(stock_analysis_tool: QMainWindow,
                                     combo: QComboBox) -> None:
            """

            :param stock_analysis_tool:
            :param combo:
            :return:
            """
            new_time_delta = Ui_StockAnalysisTool.TimeDeltas(combo.currentText())
            min_date = stock_analysis_tool.df.index[0].to_pydatetime()
            max_date = stock_analysis_tool.df.index[-1].to_pydatetime()

            if new_time_delta == Ui_StockAnalysisTool.TimeDeltas.FIVE_YEARS:
                min_date = max_date - timedelta(days=365*5)
            elif new_time_delta == Ui_StockAnalysisTool.TimeDeltas.ONE_YEAR:
                min_date = max_date - timedelta(days=365*1)
            elif new_time_delta == Ui_StockAnalysisTool.TimeDeltas.YEAR_TO_DATE:
                min_date = min_date.replace(year=max_date.year, month=1, day=1)
            elif new_time_delta == Ui_StockAnalysisTool.TimeDeltas.SIX_MONTHS:
                min_date = max_date - dateutil.relativedelta.relativedelta(months=6)
            elif new_time_delta == Ui_StockAnalysisTool.TimeDeltas.ONE_MONTH:
                min_date = max_date - dateutil.relativedelta.relativedelta(months=1)
            elif new_time_delta == Ui_StockAnalysisTool.TimeDeltas.FIVE_DAYS:
                min_date = max_date - timedelta(days=5)

            stock_analysis_tool.first_date = min_date
            stock_analysis_tool.last_date = max_date
            stock_analysis_tool.update_data_on_graph(stock_analysis_tool.get_relevant_data_frame())

        @staticmethod
        def open_data_cb_pressed(stock_analysis_tool: QMainWindow,
                                 cb: QCheckBox) -> None:
            Ui_StockAnalysisTool.Callbacks.generic_data_cb_pressed(
                stock_analysis_tool=stock_analysis_tool,
                checked=cb.isChecked(),
                df_column_name='open'
            )

        @staticmethod
        def high_data_cb_pressed(stock_analysis_tool: QMainWindow,
                                 cb: QCheckBox) -> None:
            Ui_StockAnalysisTool.Callbacks.generic_data_cb_pressed(
                stock_analysis_tool=stock_analysis_tool,
                checked=cb.isChecked(),
                df_column_name='high'
            )

        @staticmethod
        def low_data_cb_pressed(stock_analysis_tool: QMainWindow,
                                cb: QCheckBox) -> None:
            Ui_StockAnalysisTool.Callbacks.generic_data_cb_pressed(
                stock_analysis_tool=stock_analysis_tool,
                checked=cb.isChecked(),
                df_column_name='low'
            )

        @staticmethod
        def close_data_cb_pressed(stock_analysis_tool: QMainWindow,
                                  cb: QCheckBox) -> None:
            Ui_StockAnalysisTool.Callbacks.generic_data_cb_pressed(
                stock_analysis_tool=stock_analysis_tool,
                checked=cb.isChecked(),
                df_column_name='close'
            )

        @staticmethod
        def adjclose_data_cb_pressed(stock_analysis_tool: QMainWindow,
                                     cb: QCheckBox) -> None:
            Ui_StockAnalysisTool.Callbacks.generic_data_cb_pressed(
                stock_analysis_tool=stock_analysis_tool,
                checked=cb.isChecked(),
                df_column_name='adjclose'
            )

        @staticmethod
        def volume_data_cb_pressed(stock_analysis_tool: QMainWindow,
                                   cb: QCheckBox) -> None:
            Ui_StockAnalysisTool.Callbacks.generic_data_cb_pressed(
                stock_analysis_tool=stock_analysis_tool,
                checked=cb.isChecked(),
                df_column_name='volume'
            )

        @staticmethod
        def rsi_cb_pressed(stock_analysis_tool: QMainWindow,
                           cb: QCheckBox) -> None:
            """

            :param stock_analysis_tool:
            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add RSI to Display Graph
                stock_analysis_tool.df['rsi'] = rsi(close=stock_analysis_tool.df['close'],
                                                    n=stock_analysis_tool.rsi_n)
                stock_analysis_tool.df['rsi overbought'] = 70
                stock_analysis_tool.df['rsi oversold'] = 30
                stock_analysis_tool.add_column_to_graph(column_name='rsi')
                stock_analysis_tool.add_column_to_graph(column_name='rsi overbought',
                                                        color=stock_analysis_tool.protected_colors['red'])
                stock_analysis_tool.add_column_to_graph(column_name='rsi oversold',
                                                        color=stock_analysis_tool.protected_colors['green'])
            else:
                # Remove RSI from Display Graph
                stock_analysis_tool.remove_column_from_graph(column_name='rsi')
                stock_analysis_tool.remove_column_from_graph(column_name='rsi overbought')
                stock_analysis_tool.remove_column_from_graph(column_name='rsi oversold')
                stock_analysis_tool.df = stock_analysis_tool.df.drop("rsi", axis=1)
                stock_analysis_tool.df = stock_analysis_tool.df.drop("rsi overbought", axis=1)
                stock_analysis_tool.df = stock_analysis_tool.df.drop("rsi oversold", axis=1)

        @staticmethod
        def rsi_time_frame_text_changed(stock_analysis_tool: QMainWindow,
                                        rsi_time_frame_text: QTextEdit):
            """

            :param stock_analysis_tool:
            :param rsi_time_frame_text:
            :return:
            """
            text = rsi_time_frame_text.toPlainText()

            if text != "":
                try:
                    stock_analysis_tool.rsi_n = int(text)
                    if 'rsi' in stock_analysis_tool.df.columns:
                        stock_analysis_tool.df = stock_analysis_tool.df.drop("rsi", axis=1)
                        stock_analysis_tool.df['rsi'] = rsi(close=stock_analysis_tool.df['close'],
                                                            n=stock_analysis_tool.rsi_n)
                        x = [datetime.timestamp(date_time) for date_time in stock_analysis_tool.df.index]
                        stock_analysis_tool.data_lines['rsi'].setData(x, stock_analysis_tool.df['rsi'])
                except ValueError:
                    print("Invalid RSI Input")

        @staticmethod
        def williams_r_cb_pressed(stock_analysis_tool: QMainWindow,
                                  cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add WilliamsR to Display Graph
                stock_analysis_tool.df['WilliamsR'] = wr(stock_analysis_tool.df['high'],
                                                         stock_analysis_tool.df['low'],
                                                         stock_analysis_tool.df['close'])
                stock_analysis_tool.df['WilliamsR overbought'] = -20
                stock_analysis_tool.df['WilliamsR oversold'] = -80
                stock_analysis_tool.add_column_to_graph(column_name='WilliamsR')
                stock_analysis_tool.add_column_to_graph(column_name='WilliamsR overbought',
                                                        color=stock_analysis_tool.protected_colors['red'])
                stock_analysis_tool.add_column_to_graph(column_name='WilliamsR oversold',
                                                        color=stock_analysis_tool.protected_colors['green'])
            else:
                # Remove WilliamsR from Display Graph
                stock_analysis_tool.remove_column_from_graph(column_name='WilliamsR')
                stock_analysis_tool.remove_column_from_graph(column_name='WilliamsR overbought')
                stock_analysis_tool.remove_column_from_graph(column_name='WilliamsR oversold')
                stock_analysis_tool.df = stock_analysis_tool.df.drop("WilliamsR", axis=1)
                stock_analysis_tool.df = stock_analysis_tool.df.drop("WilliamsR overbought", axis=1)
                stock_analysis_tool.df = stock_analysis_tool.df.drop("WilliamsR oversold", axis=1)

        @staticmethod
        def cmo_cb_pressed(stock_analysis_tool: QMainWindow,
                           cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                stock_analysis_tool.df['cmo'] = chande_momentum_oscillator(close_data=stock_analysis_tool.df['close'],
                                                                           period=stock_analysis_tool.cmo_n)
                stock_analysis_tool.df['cmo overbought'] = 50
                stock_analysis_tool.df['cmo oversold'] = -50
                # Add CMO to Display Graph
                stock_analysis_tool.add_column_to_graph(column_name='cmo')
                stock_analysis_tool.add_column_to_graph(column_name='cmo overbought',
                                                        color=stock_analysis_tool.protected_colors['red'])
                stock_analysis_tool.add_column_to_graph(column_name='cmo oversold',
                                                        color=stock_analysis_tool.protected_colors['green'])
            else:
                # Remove CMO from Display Graph
                stock_analysis_tool.remove_column_from_graph(column_name='cmo')
                stock_analysis_tool.remove_column_from_graph(column_name='cmo overbought')
                stock_analysis_tool.remove_column_from_graph(column_name='cmo oversold')
                stock_analysis_tool.df = stock_analysis_tool.df.drop("cmo", axis=1)
                stock_analysis_tool.df = stock_analysis_tool.df.drop("cmo overbought", axis=1)
                stock_analysis_tool.df = stock_analysis_tool.df.drop("cmo oversold", axis=1)

        @staticmethod
        def macd_cb_pressed(stock_analysis_tool: QMainWindow,
                            cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add MACD to Display Graph
                stock_analysis_tool.df['MACD'] = macd(close=stock_analysis_tool.df['close'],
                                                      n_slow=stock_analysis_tool.macd_slow,
                                                      n_fast=stock_analysis_tool.macd_fast) - \
                                                 macd_signal(close=stock_analysis_tool.df['close'],
                                                             n_slow=stock_analysis_tool.macd_slow,
                                                             n_fast=stock_analysis_tool.macd_fast,
                                                             n_sign=stock_analysis_tool.macd_sign)
                stock_analysis_tool.add_column_to_graph(column_name='MACD')
            else:
                # Remove MACD from Display Graph
                stock_analysis_tool.remove_column_from_graph(column_name='MACD')
                stock_analysis_tool.df = stock_analysis_tool.df.drop("MACD", axis=1)

        @staticmethod
        def roc_cb_pressed(stock_analysis_tool: QMainWindow,
                           cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add ROC to Display Graph
                stock_analysis_tool.df['roc'] = roc(close=stock_analysis_tool.df['close'],
                                                    n=stock_analysis_tool.roc_n)
                stock_analysis_tool.add_column_to_graph(column_name='roc')
            else:
                # Remove ROC from Display Graph
                stock_analysis_tool.remove_column_from_graph(column_name='roc')
                stock_analysis_tool.df = stock_analysis_tool.df.drop("roc", axis=1)

        @staticmethod
        def wma_cb_pressed(cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add RSI to Display Graph
                pass
            else:
                # Remove RSI from Display Graph
                pass

        @staticmethod
        def ema_cb_pressed(cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add RSI to Display Graph
                pass
            else:
                # Remove RSI from Display Graph
                pass

        @staticmethod
        def sma_cb_pressed(cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add RSI to Display Graph
                pass
            else:
                # Remove RSI from Display Graph
                pass

        @staticmethod
        def hma_cb_pressed(cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add RSI to Display Graph
                pass
            else:
                # Remove RSI from Display Graph
                pass

        @staticmethod
        def trix_cb_pressed(cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add RSI to Display Graph
                pass
            else:
                # Remove RSI from Display Graph
                pass

        @staticmethod
        def cci_cb_pressed(stock_analysis_tool: QMainWindow,
                           cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add CCI to Display Graph
                stock_analysis_tool.df['cci'] = cci(close=stock_analysis_tool.df['close'],
                                                    low=stock_analysis_tool.df['low'],
                                                    high=stock_analysis_tool.df['high'],
                                                    n=stock_analysis_tool.cci_n)
                stock_analysis_tool.add_column_to_graph(column_name='cci')
            else:
                # Remove CCI from Display Graph
                stock_analysis_tool.remove_column_from_graph(column_name='cci')
                stock_analysis_tool.df = stock_analysis_tool.df.drop("cci", axis=1)

        @staticmethod
        def dpo_cb_pressed(stock_analysis_tool: QMainWindow,
                           cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add DPO to Display Graph
                stock_analysis_tool.df['dpo'] = dpo(close=stock_analysis_tool.df['close'],
                                                    n=stock_analysis_tool.dpo_n)
                stock_analysis_tool.add_column_to_graph(column_name='dpo')
            else:
                # Remove DPO from Display Graph
                stock_analysis_tool.remove_column_from_graph(column_name='dpo')
                stock_analysis_tool.df = stock_analysis_tool.df.drop("dpo", axis=1)

        @staticmethod
        def cmf_cb_pressed(stock_analysis_tool: QMainWindow,
                           cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add CMF to Display Graph
                stock_analysis_tool.df['cmf'] = chaikin_money_flow(high=stock_analysis_tool.df['high'],
                                                                   low=stock_analysis_tool.df['low'],
                                                                   close=stock_analysis_tool.df['close'],
                                                                   volume=stock_analysis_tool.df['volume'],
                                                                   n=stock_analysis_tool.cmf_n)
                stock_analysis_tool.add_column_to_graph(column_name='cmf')
            else:
                # Remove CMF from Display Graph
                stock_analysis_tool.remove_column_from_graph(column_name='cmf')
                stock_analysis_tool.df = stock_analysis_tool.df.drop("cmf", axis=1)

        @staticmethod
        def adx_cb_pressed(stock_analysis_tool: QMainWindow,
                           cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add ADX to Display Graph
                stock_analysis_tool.df['adx'] = adx(high=stock_analysis_tool.df['high'],
                                                    low=stock_analysis_tool.df['low'],
                                                    close=stock_analysis_tool.df['close'],
                                                    n=stock_analysis_tool.adx_n)
                stock_analysis_tool.add_column_to_graph(column_name='adx')
            else:
                # Remove ADX from Display Graph
                stock_analysis_tool.remove_column_from_graph(column_name='adx')
                stock_analysis_tool.df = stock_analysis_tool.df.drop("adx", axis=1)

        @staticmethod
        def force_index_cb_pressed(cb: QCheckBox) -> None:
            """

            :param cb:
            :return:
            """
            if cb.isChecked():
                # Add RSI to Display Graph
                pass
            else:
                # Remove RSI from Display Graph
                pass

    class StockSources(IntEnum):
        YAHOO_FIN = 0
        ASTRASS = 1
        YFINANCE = 2

    class SampleRates(Enum):
        ONE_DAY = "1 Day"
        ONE_HOUR = "1 Hour"
        THIRTY_MINUTES = "30 Minutes"

    class TimeDeltas(Enum):
        MAX = "MAX"
        FIVE_YEARS = "5 years"
        ONE_YEAR = "1 year"
        YEAR_TO_DATE = "YTD"
        SIX_MONTHS = "6 months"
        ONE_MONTH = "1 month"
        FIVE_DAYS = "5 days"


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    sat = Ui_StockAnalysisTool(relative_path_correction=".." + sep + ".." + sep)
    sat.show()
    app.setStyleSheet(load_stylesheet(qt_api="pyqt5"))
    sys.exit(app.exec_())
