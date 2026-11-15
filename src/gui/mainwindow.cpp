#include "mainwindow.h"

#include <QApplication>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QStatusBar>
#include <QVBoxLayout>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("PP-OCRv6 Tuner");
    resize(1200, 780);

    // Top bar
    auto* top = new QHBoxLayout;
    edt_image_ = new QLineEdit;
    edt_image_->setPlaceholderText("Path to image...");
    top->addWidget(edt_image_, 1);
    auto* btn_browse = new QPushButton("Browse...");
    auto* btn_run    = new QPushButton("▶ Run");
    btn_run->setStyleSheet("QPushButton { font-weight: bold; padding: 4px 16px; }");
    top->addWidget(btn_browse);
    top->addWidget(btn_run);
    connect(btn_browse, &QPushButton::clicked, this, &MainWindow::on_browse);
    connect(btn_run,    &QPushButton::clicked, this, &MainWindow::on_run);

    // Splitter
    auto* split = new QSplitter(Qt::Horizontal);
    split->addWidget(build_param_panel());
    split->addWidget(build_result_panel());
    split->setStretchFactor(0, 0);
    split->setStretchFactor(1, 1);
    split->setSizes({380, 820});

    auto* central = new QWidget;
    auto* lay = new QVBoxLayout(central);
    lay->setContentsMargins(8, 8, 8, 8);
    lay->addLayout(top);
    lay->addWidget(split, 1);
    setCentralWidget(central);

    statusBar()->showMessage("Ready. Select image and click Run.");
}

// ── Parameter Panel ──

QWidget* MainWindow::build_param_panel() {
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setMinimumWidth(320);
    auto* container = new QWidget;
    auto* lay = new QVBoxLayout(container);
    lay->setSpacing(12);

    // Model
    auto* gm = new QGroupBox("Model");
    auto* fm = new QFormLayout(gm);
    cbo_model_ = new QComboBox;
    cbo_model_->addItems({"tiny", "small", "medium"});
    fm->addRow("Size:", cbo_model_);
    lay->addWidget(gm);

    // Detection
    auto* gd = new QGroupBox("Detection");
    auto* fd = new QFormLayout(gd);
    cbo_limit_type_ = new QComboBox;
    cbo_limit_type_->addItems({"min", "max", "resize_long"});
    fd->addRow("Limit type:", cbo_limit_type_);
    spn_limit_side_len_ = add_int_spin(fd, "Limit side len:", 32, 2048, 32, 64);
    spn_max_side_limit_ = add_int_spin(fd, "Max side limit:", 256, 8192, 64, 4000);
    spn_det_thresh_  = add_spin(fd, "Prob threshold:",  0.05, 0.50, 0.05, 0.30);
    spn_box_thresh_  = add_spin(fd, "Box score thresh:", 0.20, 0.90, 0.05, 0.60);
    spn_unclip_ratio_= add_spin(fd, "Unclip ratio:",     1.0,  4.0,  0.1,  1.5);
    lay->addWidget(gd);

    // Merge
    auto* grm = new QGroupBox("Box Merge");
    auto* frm = new QFormLayout(grm);
    spn_merge_overlap_ = add_spin(frm,   "Overlap ratio:", 0.1, 0.9, 0.05, 0.50);
    spn_merge_height_  = add_spin(frm,   "Height ratio:",  1.5, 5.0, 0.1,  2.50);
    spn_merge_gap_     = add_int_spin(frm, "Gap (px):",     5,   100, 5,   30);
    lay->addWidget(grm);

    // Crop
    auto* gc = new QGroupBox("Crop Padding");
    auto* fc = new QFormLayout(gc);
    spn_pad_lr_ = add_int_spin(fc, "Padding:", 0, 20, 1, 3);
    lay->addWidget(gc);

    // Recognition
    auto* gf = new QGroupBox("Recognition");
    auto* ff = new QFormLayout(gf);
    spn_conf_thr_ = add_spin(ff, "Conf threshold:", 0.00, 0.99, 0.01, 0.00);
    lay->addWidget(gf);

    // Info label
    lbl_info_ = new QLabel;
    lbl_info_->setWordWrap(true);
    lbl_info_->setStyleSheet("QLabel { color: #666; padding: 4px; }");
    lay->addWidget(lbl_info_);
    lay->addStretch();

    scroll->setWidget(container);
    return scroll;
}

QWidget* MainWindow::build_result_panel() {
    auto* w = new QWidget;
    auto* lay = new QVBoxLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lbl_timing_ = new QLabel("Timing: —");
    lbl_timing_->setStyleSheet("QLabel { color: #555; padding: 2px 4px; }");
    txt_result_ = new QTextEdit;
    txt_result_->setReadOnly(true);
    txt_result_->setFont(QFont("Consolas", 10));
    lay->addWidget(lbl_timing_);
    lay->addWidget(txt_result_, 1);
    return w;
}

QDoubleSpinBox* MainWindow::add_spin(QFormLayout* form, const QString& label,
                                      double min, double max, double step, double def) {
    auto* sp = new QDoubleSpinBox;
    sp->setRange(min, max); sp->setSingleStep(step); sp->setValue(def);
    sp->setDecimals(2); sp->setFixedWidth(100);
    form->addRow(label, sp);
    return sp;
}

QSpinBox* MainWindow::add_int_spin(QFormLayout* form, const QString& label,
                                    int min, int max, int step, int def) {
    auto* sp = new QSpinBox;
    sp->setRange(min, max); sp->setSingleStep(step); sp->setValue(def);
    sp->setFixedWidth(100);
    form->addRow(label, sp);
    return sp;
}

// ── Slots ──

void MainWindow::on_browse() {
    auto path = QFileDialog::getOpenFileName(this, "Select Image",
        {}, "Images (*.jpg *.jpeg *.png *.bmp);;All (*)");
    if (!path.isEmpty()) edt_image_->setText(path);
}

void MainWindow::on_run() {
    QString img = edt_image_->text().trimmed();
    if (img.isEmpty()) {
        QMessageBox::warning(this, "No Image", "Please select an image first.");
        return;
    }
    OcrParams params = collect_params();
    std::string ms = cbo_model_->currentText().toStdString();

    if (!engine_.is_initialized()) {
        statusBar()->showMessage("Initializing engine...");
        QApplication::processEvents();
        auto err = engine_.init(models_dir_, ms);
        if (!err.empty()) {
            QMessageBox::critical(this, "Init Failed", QString::fromStdString(err));
            return;
        }
    }

    statusBar()->showMessage("Running OCR...");
    QApplication::processEvents();
    auto res = engine_.run(img.toStdString(), params);

    txt_result_->clear();
    if (!res.error.empty()) {
        txt_result_->setPlainText(QString::fromStdString(res.error));
        statusBar()->showMessage("Error.");
        return;
    }

    QString out;
    for (size_t i = 0; i < res.lines.size(); ++i)
        out += QString("[%1] %2\n").arg((int)i).arg(QString::fromStdString(res.lines[i].text));

    lbl_timing_->setText(
        QString("Det: %1 ms  |  Merge: %2→%3  |  Rec: %4 ms  |  Total: %5 ms")
            .arg(res.det_ms, 0, 'f', 1)
            .arg(res.raw_boxes).arg(res.merged_boxes)
            .arg(res.rec_ms, 0, 'f', 1)
            .arg(res.total_ms, 0, 'f', 1));
    txt_result_->setPlainText(out);

    lbl_info_->setText(QString("Params: %1").arg(QString::fromStdString(params.to_string())));
    statusBar()->showMessage(QString("Done — %1 lines, %2 ms").arg((int)res.lines.size()).arg(res.total_ms, 0, 'f', 0));
}

OcrParams MainWindow::collect_params() const {
    OcrParams p;
    p.model_size          = cbo_model_->currentText().toStdString();
    p.det_limit_type      = cbo_limit_type_->currentText().toStdString();
    p.limit_side_len      = spn_limit_side_len_->value();
    p.max_side_limit      = spn_max_side_limit_->value();
    p.det_thresh          = (float)spn_det_thresh_->value();
    p.det_box_thresh      = (float)spn_box_thresh_->value();
    p.unclip_ratio        = (float)spn_unclip_ratio_->value();
    p.merge_overlap_ratio = (float)spn_merge_overlap_->value();
    p.merge_height_ratio  = (float)spn_merge_height_->value();
    p.merge_gap_px        = (float)spn_merge_gap_->value();
    p.pad_lr              = spn_pad_lr_->value();
    p.pad_tb              = spn_pad_lr_->value();
    p.rec_conf_threshold  = (float)spn_conf_thr_->value();
    return p;
}
