#include "widget.h"
#include "ui_widget.h"

class Logger : public ILogger
{
    //void log(Severity severity, const char* msg) override
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;



Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    capTimer=new QTimer(this);
    capTimer->stop();

    //读取模型二进制文件
    IRuntime* runtime = createInferRuntime(gLogger);
    std::string cached_path = "/home/zyt/QT/palm/trt_palm/hand_3d.engine";
    std::ifstream trtModelFile(cached_path, std::ios_base::in | std::ios_base::binary);
    trtModelFile.seekg(0, ios::end);
    int size = trtModelFile.tellg();
    trtModelFile.seekg(0, ios::beg);
    char* buff = new char[size];
    trtModelFile.read(buff, size);
    trtModelFile.close();

    //准备engine和显存
    ICudaEngine* engine = runtime->deserializeCudaEngine((void*)buff, size, nullptr);
    delete buff;
    //
    context = engine->createExecutionContext();
    cudaMallocManaged(&buffers[0], 1*256*256*3*sizeof(float), cudaMemAttachHost);
    cudaMallocManaged(&buffers[1], 1*63*sizeof(float), cudaMemAttachHost);
    cudaMallocManaged(&buffers[2], 1*1*sizeof(float), cudaMemAttachHost);
    cudaStreamCreate(&stream);
    cudaStreamAttachMemAsync( stream, buffers[0], 1*256*256*3*sizeof(float), cudaMemAttachGlobal );
    cudaStreamAttachMemAsync( stream, buffers[1], 1*63*sizeof(float), cudaMemAttachGlobal );
    cudaStreamAttachMemAsync( stream, buffers[2], 1*1*sizeof(float), cudaMemAttachGlobal );
    //执行一次推理
    context->enqueue(1, buffers, stream, nullptr);
    cudaDeviceSynchronize();


    qDebug()<<buffers[0];
    qDebug()<<buffers[1];
    m_3Dgraph = new Q3DScatter();
    QWidget *a = QWidget::createWindowContainer(m_3Dgraph);
    a->show();


    QScatterDataProxy *proxy = new QScatterDataProxy(); //数据代理

    m_3Dseries = new QScatter3DSeries(proxy);//创建序列
    m_3Dseries->setMeshSmooth(true);
    m_3Dgraph->addSeries(m_3Dseries);
    m_3Dseries->setMesh(QAbstract3DSeries::MeshSphere);//数据点为圆球
    //m_3Dseries->setMesh(QAbstract3DSeries::M);//数据点为圆球
    m_3Dseries->setSingleHighlightColor(QColor(0,0,255));//设置点选中时的高亮颜色
    m_3Dseries->setBaseColor(QColor(100,50,200));//设置点的颜色
    m_3Dseries->setItemSize(0.3);//设置点的大小
    dataArray = new QScatterDataArray();
    dataArray->resize(46);




    //创建坐标轴
        //m_3Dgraph->axisX()->setTitle("axis X");
        //m_3Dgraph->axisX()->setTitleVisible(true);
    m_3Dgraph->axisX()->setRange(-50,300);
    //m_3Dgraph->axisY()->setTitle("axis Y");
    //m_3Dgraph->axisY()->setTitleVisible(true);
    m_3Dgraph->axisY()->setRange(0,255);
    //m_3Dgraph->axisZ()->setTitle("axis Z");
    //m_3Dgraph->axisZ()->setTitleVisible(true);
    m_3Dgraph->axisZ()->setRange(-100,100);
    m_3Dgraph->activeTheme()->setLabelBackgroundEnabled(false);
    m_3Dgraph->activeTheme()->setBackgroundColor(QColor(70,70,70));//设置背景色










    //output=Mat(1,21,CV_32FC3,(void*)buffers[1]);

}

Widget::~Widget()
{
    delete ui;
}

void Widget::on_pushButton_clicked()//打开图片并显示
{
    QString fileName = QFileDialog::getOpenFileName(this,tr("选择输入图像"),"E:/opencv","图片(*.jpg *.bmp *.png)");
    if(fileName.isEmpty())
        return;
    //qDebug()<<"filename : "<<fileName;
    img=imread(fileName.toStdString());
    //ui->label->setPixmap(QPixmap::fromImage(Mat2QImage(img)));

    start=std::time(nullptr);
    int s=256;
    if(img.cols>img.rows)
    {
        cv::resize(img,img,Size(s,img.rows*s/img.cols));
        int i=(256-img.rows)/2;
        copyMakeBorder(img,img,i,256-img.rows-i,0,0,BORDER_CONSTANT);
      }
    else
    {
        cv::resize(img,img,Size(img.cols*s/img.rows,s));
        int i=(256-img.cols)/2;
        copyMakeBorder(img,img,0,0,i,256-img.cols-i,BORDER_CONSTANT);
    }
    cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img,CV_32FC3,2.0/255,-1);
    cv::imshow("123",img);
    a=(float*)img.data;
    b=(float*)buffers[0];
    c=(float*)buffers[1];
    for(int i=0;i<256;i++)
        for(int j=0;j<256;j++)
            for(int k=0;k<3;k++)
                *(b++)=*(a++);
    context->enqueue(1, buffers, stream, nullptr);
    cudaDeviceSynchronize();

    if(*((float*)buffers[2])<0.5)
    {
        qDebug()<<*((float*)buffers[2]);
        return;
    }
    for(auto it=dataArray->begin();it<dataArray->end();it++)
    {
        it->setPosition(QVector3D(*c,255-*(c+1),-*(c+2)));
        //qDebug()<<*c<<*(c+1)<<*(c+2);
        c=c+3;
    }
    m_3Dseries->dataProxy()->resetArray(dataArray);
    end=std::time(nullptr);

    qDebug()<<end-start;

}

QImage Widget::Mat2QImage(Mat& image0)//Mat转换为QPixmap
{
    Mat image=image0.clone();
    QImage img;
//    int s=600;
//    if(image.cols>image.rows)
//        cv::resize(image,image,Size(s,image.rows*s/image.cols));
//    else cv::resize(image,image,Size(image.cols*s/image.rows,s));
    if (image.channels() == 3) {
        cvtColor(image, image, CV_BGR2RGB);
        img = QImage((const unsigned char*)(image.data), image.cols, image.rows,
                     image.cols * image.channels(), QImage::Format_RGB888);
    }
    else if (image.channels() == 1) {
        img = QImage((const unsigned char*)(image.data), image.cols, image.rows,
                     image.cols * image.channels(), QImage::Format_Indexed8);
    }
    else {
        img = QImage((const unsigned char*)(image.data), image.cols, image.rows,
                     image.cols * image.channels(), QImage::Format_RGB888);
    }
    return img;
}

void Widget::on_pushButton_6_clicked()//打开/关闭相机
{

    if(!cap.isOpened())
    {
        //cap.open("http://192.168.1.101:4747/video  ");
        cap.open(0);
        if(!cap.isOpened()) return;
        capTimer->start(10);
        cap.read(img);
        ui->pushButton_6->setText("关闭相机");
        ui->pushButton->setDisabled(1);
        ui->pushButton_2->setDisabled(1);
        ui->pushButton_3->setDisabled(1);
        ui->pushButton_4->setDisabled(1);
        ui->pushButton_5->setDisabled(1);
        ui->pushButton_7->setDisabled(0);
        ui->label_2->setText("结果");
        connect(capTimer,&QTimer::timeout,[=](){
            //cap.read(img);
            capTimer->stop();
            medianBlur(img,img,21);
//            flip(outImg,outImg,1);
//            img_0= outImg.clone();
//            ui->label->setPixmap(QPixmap::fromImage(Mat2QImage(img_0)));


            start=std::time(nullptr);
            int s=256;

            img.convertTo(img,CV_32FC3,2.0/255,-1);
            if(img.cols>img.rows)
            {
                cv::resize(img,img,Size(s,img.rows*s/img.cols));
                int i=(256-img.rows)/2;
                copyMakeBorder(img,img,i,256-img.rows-i,0,0,BORDER_CONSTANT);
              }
            else
            {
                cv::resize(img,img,Size(img.cols*s/img.rows,s));
                int i=(256-img.cols)/2;
                copyMakeBorder(img,img,0,0,i,256-img.cols-i,BORDER_CONSTANT);
            }
            cvtColor(img, img, cv::COLOR_BGR2RGB);
            //cv::imshow("123",img);
            a=(float*)img.data;
            b=(float*)buffers[0];
            c=(float*)buffers[1];
            for(int i=0;i<256;i++)
                for(int j=0;j<256;j++)
                    for(int k=0;k<3;k++)
                        *(b++)=*(a++);
            context->enqueue(1, buffers, stream, nullptr);
            cudaDeviceSynchronize();
//            if(*((float*)buffers[2])<0.5)
//            {
//                //qDebug()<<*((float*)buffers[2]);
//                return;
//            }
            auto it=dataArray->begin();
            for(int i=0;i<21;i++)
            {
                it->setPosition(QVector3D(*c,255-*(c+1),*(c+2)));
                it++;
                if(i%4!=0||i==0){
                     it->setPosition(QVector3D((*c+(*(c+3)))/2,255-(*(c+1)+(*(c+4)))/2,(*(c+2)+(*(c+5)))/2));
                     it++;
                }
                else{
                    it->setPosition(QVector3D((*(c+3)+(*(c-9)))/2,255-(*(c+4)+(*(c-8)))/2,(*(c+5)+(*(c-7)))/2));
                    it++;
                }

                if(i==5){
                    it->setPosition(QVector3D((*c)/4+(*(c-12))*3/4,255-(*(c+1)/4+(*(c-11))*3/4),(*(c+2))/4+(*(c-10))*3/4));
                    it++;
                    it->setPosition(QVector3D((*c)*3/4+(*(c-12))/4,255-(*(c+1)*3/4+(*(c-11))/4),(*(c+2))*3/4+(*(c-10))/4));
                    it++;
                }
                if(i==17){
                    it->setPosition(QVector3D((*c)/4+(*(c-51))*3/4,255-(*(c+1)/4+(*(c-50))*3/4),(*(c+2))/4+(*(c-49))*3/4));
                    it++;
                    it->setPosition(QVector3D((*c)*3/4+(*(c-51))/4,255-(*(c+1)*3/4+(*(c-50))/4),(*(c+2))*3/4+(*(c-49))/4));
                    it++;
                    it->setPosition(QVector3D((*c+(*(c-51)))/2,255-(*(c+1)+(*(c-50)))/2,(*(c+2)+(*(c-49)))/2));
                    it++;
                }



                //qDebug()<<*c<<*(c+1)<<*(c+2);
                c=c+3;
            }




            m_3Dseries->dataProxy()->resetArray(dataArray);
            waitKey(10);
            end=std::time(nullptr);

            qDebug()<<end-start;
            cap.read(img);
            cv::imshow("123",img);
            capTimer->start(20);





        });

    }
    else {
        capTimer->stop();
        cap.release();
        ui->label->setText("原图");
        ui->label_2->setText("结果");
        ui->pushButton_6->setText("打开相机");
        ui->pushButton->setDisabled(0);
        ui->pushButton_5->setDisabled(1);
        ui->pushButton_7->setDisabled(1);
        disconnect(capTimer);
    }


}

void Widget::on_pushButton_5_clicked()//保存图片
{
    QString fileName = QFileDialog::getSaveFileName(this,"保存输出图像",
                               "D:/opencv","图片(*jpg)");
    if(fileName.isEmpty())
        return;
    fileName+=".jpg";
    //qDebug()<<"filename : "<<fileName;
    bool save=imwrite(fileName.toStdString(),outImg);
    if(save)
        QMessageBox::information(this,"输出图像",
                                 "图片成功保存至："+fileName);
    else QMessageBox::warning(this,"输出图像","保存失败");

}

void Widget::on_pushButton_2_clicked()//孔径检测
{

}

void Widget::on_pushButton_3_clicked()//角点检测
{

    img_0=img.clone();
    outImg=img.clone();
    cvtColor(img_0, img_0, CV_BGR2GRAY);//转化为灰度图
    img_0=img_0>242;//二值化
    vector <vector <Point>> contours;
    vector<Vec4i> hierarchy;
    vector <Point> maxCont;
    findContours(img_0,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE, Point(0,0));//轮廓查找
    for (int i = 0; i < contours.size(); ++i) {
        if(contourArea(contours[i])>60000)//筛选面积较大的轮廓
        {
            //drawContours( outImg, contours, i, Scalar(0,0,255), 10, 8, hierarchy, 0, Point());
            maxCont.insert(maxCont.end(), contours[i].begin(), contours[i].end());
        }
    }
    int x_max=0,x_min=2591,y_max=0,y_min=1943;
    for(int i=0;i<maxCont.size();i++)//获得轮廓位置
    {
        if(x_max<maxCont[i].x)
            x_max=maxCont[i].x;
        if(x_min>maxCont[i].x)
            x_min=maxCont[i].x;
        if(y_max<maxCont[i].y)
            y_max=maxCont[i].y;
        if(y_min>maxCont[i].y)
            y_min=maxCont[i].y;
    }
    Point dot;
    int n=10;
    if(x_min<500&&y_min<500)//判断角点位置
        dot=Point(x_max-n,y_max);
    else if(x_min<500&&y_max>1400)
        dot=Point(x_max,y_min+n);
    else if(x_max>2000&&y_min<500)
        dot=Point(x_min,y_max-n);
    else if(x_max>2000&&y_max>1400)
        dot=Point(x_min+n,y_min);
    //qDebug()<<dec.x<<"00"<<dec.y;
    circle(outImg, dot, 20, Scalar(255,0,0), -1, 8, 0);
    putText(outImg, tr("(%1,%2)").arg(dot.x).arg(dot.y).toStdString(),dot+Point(50,-50),FONT_HERSHEY_SIMPLEX,3,Scalar(255, 0,255 ),8);
    //qDebug()<<"x1:"<<x_min<<"x2:"<<x_max<<"y1"<<y_min<<"y2"<<y_max;
    outImage(outImg);
}

void Widget::on_pushButton_4_clicked()//黄点检测
{

    img_0=img.clone();
    outImg=img.clone();
    GaussianBlur(img_0, img_0, Size(9, 9),0);//高斯滤波使黄点颜色均匀
    cvtColor(img_0, img_0, CV_BGR2HSV);//转化为hsv
    vector<Mat> hsv;
    split(img_0,hsv);
    img_0=hsv[1]>90;//取第二通道二值化
    Moments moment;
    int id=1;
    vector <vector <Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img_0,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE, Point(0,0));//查找轮廓
    for (int i = 0; i < contours.size(); ++i) {
        if(contourArea(contours[i])>1500)//轮廓筛选并在图上标注
        {
            moment=moments(Mat(contours[i]));
            Point center=Point(moment.m10/moment.m00,moment.m01/moment.m00);//使用轮廓矩计算重心
            circle(outImg, center, 10, Scalar(0,255,0), -1, 8, 0);
            drawContours( outImg, contours, i, Scalar(0,255,0), 6, 8, hierarchy, 0, Point());
            putText(outImg, tr("%1").arg(id).toStdString(),center+Point(-20,20),FONT_HERSHEY_SIMPLEX,2,Scalar(0, 0,255 ),6);
            QString str=tr("S%1:%2, L%1:(%3,%4)").arg(id).arg(contourArea(contours[i])).arg(center.x).arg(center.y);
            putText(outImg, str.toStdString(),Point(0,id*40),FONT_HERSHEY_SIMPLEX,1.3,Scalar(225, 0,255 ),3);
            id++;
            //qDebug()<<str.toUtf8();
        }
    }
    outImage(outImg);
}

void Widget::on_pushButton_7_clicked()//拍照
{
    cap>>outImg;
    flip(outImg,outImg,1);
    img_0=outImg.clone();
    ui->label_2->setPixmap(QPixmap::fromImage(Mat2QImage(img_0)));
    ui->pushButton_5->setDisabled(0);
}
void Widget::outImage(Mat& image)//显示输出结果
{
    ui->label_2->setPixmap(QPixmap::fromImage(Mat2QImage(image)));
    ui->pushButton_5->setDisabled(0);
}




