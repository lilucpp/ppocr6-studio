#include <QApplication>
#include "mainwindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("ppocr6-tuner");
    MainWindow win;
    win.show();
    return app.exec();
}
