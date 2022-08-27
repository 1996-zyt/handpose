#ifndef WIDGET_H
#define WIDGET_H



//#include<QWindow>
#include <opencv2/highgui/highgui_c.h>
#include <QWidget>
#include<opencv2/opencv.hpp>
#include<QFileDialog>
#include<QMessageBox>
#include<QDebug>
#include<vector>
#include<QTimer>
#include<QDateTime>
#include <QWidget>
#include <QtDataVisualization>
#include <QAbstract3DInputHandler>


#include<NvInfer.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cublas_v2.h>
#include<fstream>
#include<iostream>
#include<string>


typedef QVector<QVector3D> MY_BUF3D ;

using namespace std;
using namespace nvinfer1;
using namespace QtDataVisualization;


//#pragma execution_character_set("utf-8")
//#pragma comment  (lib, "User32.lib")

using namespace cv;
using namespace std;

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();


private slots:
    void on_pushButton_clicked();

    void on_pushButton_6_clicked();

    void on_pushButton_5_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

    void on_pushButton_7_clicked();





private:
    Ui::Widget *ui;

    QImage Mat2QImage(Mat& image);

    void outImage(Mat& image);

    void showImg(Mat& image);

    Mat img;
    Mat output;
    Mat input;
    Mat img_0;
    Mat outImg;

    VideoCapture cap;
    QTimer* capTimer;
    String str1;



    void* buffers[3];
    cudaStream_t stream;
    IExecutionContext* context;
    clock_t start,end;


    Q3DScatter *m_3Dgraph;
    QScatter3DSeries* m_3Dseries;
    QScatterDataArray *dataArray;
    float*a,*b,*c;



};

#endif // WIDGET_H
