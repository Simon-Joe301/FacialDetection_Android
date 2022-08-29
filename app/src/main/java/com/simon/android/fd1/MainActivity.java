package com.simon.android.fd1;



import static org.opencv.core.CvType.CV_8UC3;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.nfc.Tag;
import android.os.Bundle;
import android.os.Environment;
import android.provider.ContactsContract;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;


public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {
    private static final String TAG = "FD2";
    private JavaCameraView mOpenCvCameraView;
    private Mat mIntermediateMat;
    private CascadeClassifier classifier;
    private MatOfRect faces;
    Rect[] facesArray;
    private Mat mGray;
    private Mat mRgba;
    private int fps=3;
    private int mAbsoluteFaceSize = 0;
    private int idx=1;
    private int[] cameraIdx={CameraBridgeViewBase.CAMERA_ID_BACK,CameraBridgeViewBase.CAMERA_ID_FRONT};
    private ImageButton button;
    private ImageButton shotButton;
    private ImageView imageView;
    private static double[] Color_list= {
            1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 33, 35, 37, 39,
            41, 43, 44, 46, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 66, 67, 69, 71, 73, 74,
            76, 78, 79, 81, 83, 84, 86, 87, 89, 91, 92, 94, 95, 97, 99, 100, 102, 103, 105,
            106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 128,
            130, 131, 133, 134, 135, 137, 138, 139, 141, 142, 143, 145, 146, 147, 149, 150,
            151, 153, 154, 155, 156, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170,
            171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            204, 205, 205, 206, 207, 208, 209, 210, 211, 211, 212, 213, 214, 215, 215, 216,
            217, 218, 219, 219, 220, 221, 222, 222, 223, 224, 224, 225, 226, 226, 227, 228,
            228, 229, 230, 230, 231, 232, 232, 233, 233, 234, 235, 235, 236, 236, 237, 237,
            238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 244, 245,
            245, 246, 246, 246, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 250,
            251, 251, 251, 251, 252, 252, 252, 252, 253, 253, 253, 253, 253, 254, 254, 254,
            254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255};


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    initClassifier();
                    mOpenCvCameraView.setCameraPermissionGranted();
                    mOpenCvCameraView.enableView();

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

     /**
     * 磨皮美颜方法
     * 返回处理后的Mat
     * 使用双边滤波进行磨皮，使用颜色查找表进行美白
     **/
    private Mat FaceBeauty(){
        Mat bPic=new Mat(mRgba.rows(),mRgba.cols(),CV_8UC3);
        Imgproc.cvtColor(mRgba,bPic,Imgproc.COLOR_BGRA2BGR);
        //人脸磨皮美白
        if(facesArray.length>0) {
            for (Rect faceRect : facesArray) {
                Point left, right;
                left = faceRect.tl();
                right = faceRect.br();
                //起始点结束点坐标,找到要进行磨皮美白的区域
                int startrow = (int) left.y;
                int endrow = (int) right.y;
                int startcol = (int) left.x;
                int endcol = (int) right.x;
                int height = endrow-startrow;
                int offset = height / 5;
                startrow -= offset;
                endrow += offset / 2;
                //Mat roi=new Mat(bPic,faceRect);
                Mat roi = new Mat(bPic, new Range(startrow, endrow), new Range(startcol, endcol));
                Log.d(TAG,roi.size().toString());
                Mat dst = new Mat(roi.rows(), roi.cols(), roi.type());

                //人脸磨皮
                Imgproc.bilateralFilter(roi, dst, 19, 75, 75);
                Log.d(TAG, "bilateralfilter");
                double[] asrc;//原像素值
                double[] adst = new double[3];//查表得到的目标像素值
                for (int x = 0; x < dst.rows(); x++) {
                    for (int y = 0; y < dst.cols(); y++) {
                        asrc = dst.get(x, y);
                        adst[0]=Color_list[(int) asrc[0]];
                        adst[1]=Color_list[(int) asrc[1]];
                        adst[2]=Color_list[(int) asrc[2]];

                        bPic.put(startrow + x, startcol + y, adst);
                    }
                }

                Log.d(TAG, "Beautfied!");
                roi.release();
                dst.release();
            }
        }
        return bPic;
    }
    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

//        if(this.getResources().getConfiguration().orientation== Configuration.ORIENTATION_PORTRAIT)
//            Log.i("FD2","portrait");
//        else
//            Log.i("FD2","landscape");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.image_manipulations_surface_view);

        //组件对象
        button =(ImageButton) findViewById(R.id.btn);
        shotButton =(ImageButton) findViewById(R.id.btn1);
        imageView=(ImageView)findViewById(R.id.imgv);
        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.jcv);

        //初始化javaCameraView对象
        initJCV();

        //转换摄像头按钮组件监听器设置
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.i(TAG,"clicked");
                idx=(idx+1)%2;
                mOpenCvCameraView.disableView();
                mOpenCvCameraView.setCameraIndex(cameraIdx[idx]);
                mOpenCvCameraView.enableView();
            }
        });
        //拍摄按钮组件监听器设置
        shotButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.d(TAG,"clicked");
                //生成美颜后的bmp图
                Bitmap bmp=null;
                try{
                    Mat temp=FaceBeauty();
                    bmp=Bitmap.createBitmap(temp.cols(),temp.rows(),Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(temp,bmp);
                    Log.d(TAG,"mat2bmp");
                }catch (CvException e){
                    Log.d(TAG,e.getMessage());
                }
                //预览结果图像
                imageView.setImageBitmap(bmp);
                //保存结果图像
                filestore(bmp);
            }
        });
    }
    /*
    保存结果图像方法
     */
    public void filestore(Bitmap bmp){
        FileOutputStream out=null;
        SimpleDateFormat sdf=new SimpleDateFormat("yyyy-mm-dd_hh-mm-ss");
        String fileName =sdf.format(new Date())+".png";
        File sd = new File(Environment.getExternalStorageDirectory()+"/Pictures/FD2");
        boolean success=true;
        if(!sd.exists()){
            success=sd.mkdir();
        }
        if(success){
            File dest =new File(sd,fileName);

            try{
                out =new FileOutputStream(dest);
                bmp.compress(Bitmap.CompressFormat.PNG,100,out);
                Toast.makeText(MainActivity.this,"Pictures have been saved in"+dest.toString(),Toast.LENGTH_SHORT).show();
            }catch (Exception e){
                e.printStackTrace();
                Log.i(TAG,e.getMessage());
            }finally {
                try{
                    if(out!=null){
                        out.close();
                        Log.i(TAG,"success saved!");
                    }
                }catch (IOException e){
                    Log.i(TAG,e.getMessage()+"Error");
                    e.printStackTrace();
                }
            }
        }
    }

    /*
     * 初始化JavaCameraView组件
     */
    public void initJCV(){
        mOpenCvCameraView.setOrientation(this.getResources().getConfiguration().orientation== Configuration.ORIENTATION_PORTRAIT);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    // 初始化人脸级联分类器，必须先初始化
    private void initClassifier() {
        try {
            InputStream is = getResources()
                    .openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "lbpcascade_frontalface_improved.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            classifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /*
    后面都是MainActivity类对CvCameraViewListener2接口方法的实现
     */
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mIntermediateMat != null)
            mIntermediateMat.release();
        mIntermediateMat = null;
        mGray.release();
        mRgba.release();
    }

    /*
    * 最重要的方法，实现人脸检测，并画框
    * 本方法每三帧检测一次，避免频繁检测带来较大的卡顿
    */
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
//        Mat rgba = inputFrame.rgba();
//        return rgba;
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        Mat rgba=new Mat(mRgba.rows(),mRgba.cols(),mRgba.type());
        mRgba.copyTo(rgba);
//        String s=String.valueOf(mPic.dims());
//        Log.i(TAG,s);
        if(fps==3) {
            float mRelativeFaceSize = 0.2f;
            if (mAbsoluteFaceSize == 0) {
                int height = mGray.rows();
                if (Math.round(height * mRelativeFaceSize) > 0) {
                    mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
                }
            }
            //mAbsoluteFaceSize=30;
            faces = new MatOfRect();
            if (classifier != null)
                classifier.detectMultiScale(mGray, faces, 1.05, 4, 2,
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
            facesArray = faces.toArray();
            fps = 0;
        }
        if(facesArray.length>0) {
            //Toast.makeText(MainActivity.this,"Detected!",Toast.LENGTH_SHORT).show();
            for (Rect faceRect : facesArray) {
                Imgproc.rectangle(rgba, faceRect.tl(), faceRect.br(), new Scalar(255, 255, 255, 255), 2);

            }
        }
        fps++;
        return rgba;
    }


}
