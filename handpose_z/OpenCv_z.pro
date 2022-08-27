#-------------------------------------------------
#
# Project created by QtCreator 2020-11-30T11:02:08
#
#-------------------------------------------------

QT       += core gui datavisualization

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = OpenCv_z
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


#pylon



SOURCES += \
        main.cpp \
        widget.cpp

HEADERS += \
        widget.h

FORMS += \
        widget.ui





#opencv



unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv4


unix:!macx: LIBS += -L$$PWD/../../../../../usr/local/cuda-10.2/lib64/ -lcudart

INCLUDEPATH += $$PWD/../../../../../usr/local/cuda-10.2/include
DEPENDPATH += $$PWD/../../../../../usr/local/cuda-10.2/include

unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/aarch64-linux-gnu/ -lnvinfer

INCLUDEPATH += $$PWD/../../../../../usr/include/aarch64-linux-gnu
DEPENDPATH += $$PWD/../../../../../usr/include/aarch64-linux-gnu
