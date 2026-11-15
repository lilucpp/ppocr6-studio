#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QComboBox>
#include <QTextEdit>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QSplitter>
#include <QGroupBox>
#include <QFormLayout>
#include <QScrollArea>

#include "ocr_engine.h"
#include "params.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void on_browse();
    void on_run();

private:
    OcrEngine engine_;
    std::string models_dir_ = "models";

    // UI builders
    QWidget* build_param_panel();
    QWidget* build_result_panel();
    QDoubleSpinBox* add_spin(QFormLayout* form, const QString& label, double min, double max, double step, double def);
    QSpinBox* add_int_spin(QFormLayout* form, const QString& label, int min, int max, int step, int def);

    // Controls
    QComboBox*       cbo_model_;
    QComboBox*       cbo_limit_type_;
    QSpinBox*        spn_limit_side_len_;
    QSpinBox*        spn_max_side_limit_;
    QDoubleSpinBox*  spn_det_thresh_;
    QDoubleSpinBox*  spn_box_thresh_;
    QDoubleSpinBox*  spn_unclip_ratio_;
    QDoubleSpinBox*  spn_merge_overlap_;
    QDoubleSpinBox*  spn_merge_height_;
    QSpinBox*        spn_merge_gap_;
    QSpinBox*        spn_pad_lr_;
    QDoubleSpinBox*  spn_conf_thr_;

    // Results
    QTextEdit* txt_result_;
    QLabel*    lbl_timing_;
    QLineEdit* edt_image_;
    QLabel*    lbl_info_;

    OcrParams collect_params() const;
};

#endif
