//
//  GraphCannySeg.h
//  GraphDepthImgSeg
//
//  Created by Giorgio on 24/10/15.
//  Copyright (c) 2015 Giorgio. All rights reserved.
//

#ifndef GraphDepthImgSeg_GraphCannySeg_h
#define GraphDepthImgSeg_GraphCannySeg_h

#include <cstring>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"

#include <opencv2/contrib/contrib.hpp>//for cv::applyColorMap
#include <opencv2/photo/photo.hpp>//for inpaintDepth

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <limits.h>
#include <climits>
#include <iostream>
#include <fstream>
#include <typeinfo>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifndef rad2deg
#define rad2deg 180.0f/M_PI
#endif

#ifndef deg2rad
#define deg2rad M_PI/180.0f
#endif

using namespace std;
//typedef unsigned char uchar;

//typedef signed char schar;

//typedef unsigned short ushort;

// threshold function
#define THRESHOLD(size, c) (c/size)

namespace GraphCanny {

struct edge{
    float w;
    int a, b;
    edge();
    edge(int a, int b, float w_);
};



struct rgb{
    
    uchar r, g, b;
    
    rgb();
    rgb(uchar r_,uchar g_,uchar b_);
    rgb& operator=(const rgb& other);
    
};

struct hsv{
    
    uchar h, s, v;
    
    hsv();
    hsv(uchar h_,uchar s_,uchar v_);
    hsv& operator=(const hsv& other);
    
};


struct CIELab{
    
    float L, a, b;
    
    CIELab();
    CIELab(float L_,float a_,float b_);
    CIELab& operator=(const CIELab& other);
    
};


inline bool operator==(const rgb &a, const rgb &b) {
    return ((a.r == b.r) && (a.g == b.g) && (a.b == b.b));
}
inline bool operator==(const hsv &a, const hsv &b) {
    return ((a.h == b.h) && (a.s == b.s) && (a.v == b.v));
}
//TODO: Floating Point == !!! Please use range > && <
inline bool operator==(const CIELab &a, const CIELab &b) {
    return ((a.L == b.L) && (a.a == b.a) && (a.b == b.b));
}

template <class T>
inline T abs(const T &x) { return (x > 0 ? x : -x); };

template <class T>
inline int sign(const T &x) { return (x >= 0 ? 1 : -1); };

template <class T>
inline T square(const T &x) { return x*x; };

template <class T>
inline T bound(const T &x, const T &min, const T &max) {
    return (x < min ? min : (x > max ? max : x));
}

template <class T>
inline bool check_bound(const T &x, const T&min, const T &max) {
    return ((x < min) || (x > max));
}

inline int vlib_round(float x) { return (int)(x + 0.5F); }

inline int vlib_round(double x) { return (int)(x + 0.5); }

inline double gaussian(double val, double sigma) {
    return exp(-square(val/sigma)/2)/(sqrt(2*M_PI)*sigma);
}



#define	RED_WEIGHT	0.299
#define GREEN_WEIGHT	0.587
#define BLUE_WEIGHT	0.114


// IMAGE Class
template <class T>
class image {
public:
    /* create an image */
    image(const int width, const int height, const bool init = true);
    
    /* delete an image */
    ~image();
    
    /* init an image */
    void init(const T &val);
    
    /* init an image from OpenCV */
    void init(const cv::Mat& m);
    
    /* copy an image */
    image<T> *copy() const;
    
    /* copy from OpenCV image */
    image<T> *copy(const cv::Mat& m) const;
    
    /* get the width of an image. */
    inline int width() const { return w; }
    
    /* get the height of an image. */
    inline int height() const { return h; }
    
    /* get the size of an image. */
    inline const int WxH() const { return wXh; }
    
    /* image data. */
    T *data;
    
    /* row pointers. */
    T **access;
    
private:
    int w, h, wXh;
};

/* use imRef to access image data. */
#define imRef(im, x, y) (im->access[y][x])

/* use imPtr to get pointer to image data. */
#define imPtr(im, x, y) &(im->access[y][x])



/* compute minimum and maximum value in an image */
template <class T>
inline void min_max(image<T> *im, T *ret_min, T *ret_max) {
    int width = im->width();
    int height = im->height();
    T min = imRef(im, 0, 0);
    T max = imRef(im, 0, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            T val = imRef(im, x, y);
            if (min > val)
                min = val;
            if (max < val)
                max = val;
        }
    }
    
    *ret_min = min;
    *ret_max = max;
}

/* compute minimum and maximum value in an image
 Overload for rgb datatype NO IF-else in compile time*/
template <>
inline void min_max(image<rgb> *im, rgb *ret_min, rgb *ret_max) {
    int width = im->width();
    int height = im->height();
    rgb min = imRef(im, 0, 0);
    rgb max = imRef(im, 0, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            rgb val = imRef(im, x, y);
            if (min.r > val.r)
                min.r = val.r;
            if (max.r < val.r)
                max.r = val.r;
            
            if (min.g > val.g)
                min.g = val.g;
            if (max.g < val.g)
                max.g = val.g;
            
            if (min.b > val.b)
                min.b = val.b;
            if (max.b < val.b)
                max.b = val.b;
            
            
        }
    }
    
    *ret_min = min;
    *ret_max = max;
}

/* compute minimum and maximum value in an image
 Overload for rgb datatype NO IF-else in compile time*/
template <>
inline void min_max(image<CIELab> *im, CIELab *ret_min, CIELab *ret_max) {
    int width = im->width();
    int height = im->height();
    CIELab min = imRef(im, 0, 0);
    CIELab max = imRef(im, 0, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            CIELab val = imRef(im, x, y);
            if (min.L > val.L)
                min.L = val.L;
            if (max.L < val.L)
                max.L = val.L;
            
            if (min.a > val.a)
                min.a = val.a;
            if (max.a < val.a)
                max.a = val.a;
            
            if (min.b > val.b)
                min.b = val.b;
            if (max.b < val.b)
                max.b = val.b;
            
            
        }
    }
    
    *ret_min = min;
    *ret_max = max;
}


static inline image<uchar> *imageRGBtoGRAY(image<rgb> *input) {
    int width = input->width();
    int height = input->height();
    image<uchar> *output = new image<uchar>(width, height, false);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(output, x, y) = (uchar)
            (imRef(input, x, y).r * RED_WEIGHT +
             imRef(input, x, y).g * GREEN_WEIGHT +
             imRef(input, x, y).b * BLUE_WEIGHT);
        }
    }
    return output;
}

static inline image<rgb> *imageGRAYtoRGB(image<uchar> *input) {
    int width = input->width();
    int height = input->height();
    image<rgb> *output = new image<rgb>(width, height, false);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(output, x, y).r = imRef(input, x, y);
            imRef(output, x, y).g = imRef(input, x, y);
            imRef(output, x, y).b = imRef(input, x, y);
        }
    }
    return output;
}

static inline image<float> *imageUCHARtoFLOAT(image<uchar> *input) {
    int width = input->width();
    int height = input->height();
    image<float> *output = new image<float>(width, height, false);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(output, x, y) = imRef(input, x, y);
        }
    }
    return output;
}

static inline image<float> *imageINTtoFLOAT(image<int> *input) {
    int width = input->width();
    int height = input->height();
    image<float> *output = new image<float>(width, height, false);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(output, x, y) = imRef(input, x, y);
        }
    }
    return output;
}

static inline image<uchar> *imageFLOATtoUCHAR(image<float> *input,
                                       float min, float max) {
    int width = input->width();
    int height = input->height();
    image<uchar> *output = new image<uchar>(width, height, false);
    
    if (max == min)
        return output;
    
    float scale = UCHAR_MAX / (max - min);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uchar val = (uchar)((imRef(input, x, y) - min) * scale);
            imRef(output, x, y) = bound(val, (uchar)0, (uchar)UCHAR_MAX);
        }
    }
    return output;
}

static inline image<uchar> *imageFLOATtoUCHAR(image<float> *input) {
    float min, max;
    min_max(input, &min, &max);
    return imageFLOATtoUCHAR(input, min, max);
}

static inline image<long> *imageUCHARtoLONG(image<uchar> *input) {
    int width = input->width();
    int height = input->height();
    image<long> *output = new image<long>(width, height, false);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(output, x, y) = imRef(input, x, y);
        }
    }
    return output;
}

static inline image<uchar> *imageLONGtoUCHAR(image<long> *input, long min, long max) {
    int width = input->width();
    int height = input->height();
    image<uchar> *output = new image<uchar>(width, height, false);
    
    if (max == min)
        return output;
    
    float scale = UCHAR_MAX / (float)(max - min);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uchar val = (uchar)((imRef(input, x, y) - min) * scale);
            imRef(output, x, y) = bound(val, (uchar)0, (uchar)UCHAR_MAX);
        }
    }
    return output;
}

static inline image<uchar> *imageLONGtoUCHAR(image<long> *input) {
    long min, max;
    min_max(input, &min, &max);
    return imageLONGtoUCHAR(input, min, max);
}

static inline image<uchar> *imageSHORTtoUCHAR(image<short> *input,
                                       short min, short max) {
    int width = input->width();
    int height = input->height();
    image<uchar> *output = new image<uchar>(width, height, false);
    
    if (max == min)
        return output;
    
    float scale = UCHAR_MAX / (float)(max - min);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uchar val = (uchar)((imRef(input, x, y) - min) * scale);
            imRef(output, x, y) = bound(val, (uchar)0, (uchar)UCHAR_MAX);
        }
    }
    return output;
}

static inline image<uchar> *imageSHORTtoUCHAR(image<short> *input) {
    short min, max;
    min_max(input, &min, &max);
    return imageSHORTtoUCHAR(input, min, max);
}

/* convolve src with mask.  dst is flipped! */
static inline void convolve_even(image<float> *src, image<float> *dst,
                          std::vector<float> &mask) {
    int width = src->width();
    int height = src->height();
    int len = mask.size();
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = mask[0] * imRef(src, x, y);
            for (int i = 1; i < len; i++) {
                sum += mask[i] *
                (imRef(src, std::max(x-i,0), y) +
                 imRef(src, std::min(x+i, width-1), y));
            }
            imRef(dst, y, x) = sum;
        }
    }
}

/* convolve src with mask.  dst is flipped! */
static inline void convolve_odd(image<float> *src, image<float> *dst,
                         std::vector<float> &mask) {
    int width = src->width();
    int height = src->height();
    int len = mask.size();
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = mask[0] * imRef(src, x, y);
            for (int i = 1; i < len; i++) {
                sum += mask[i] *
                (imRef(src, std::max(x-i,0), y) -
                 imRef(src, std::min(x+i, width-1), y));
            }
            imRef(dst, y, x) = sum;
        }
    }
}

#define WIDTH 4.0

/* normalize mask so it integrates to one */
static inline void normalize(std::vector<float> &mask) {
    int len = mask.size();
    float sum = 0;
    for (int i = 1; i < len; i++) {
        sum += fabs(mask[i]);
    }
    sum = 2*sum + fabs(mask[0]);
    for (int i = 0; i < len; i++) {
        mask[i] /= sum;
    }
}

/* make filters */
#define MAKE_FILTER(name, fun)                                \
static std::vector<float> make_ ## name (float sigma) {       \
sigma = std::max(sigma, 0.01F);			      \
int len = (int)ceil(sigma * WIDTH) + 1;                     \
std::vector<float> mask(len);                               \
for (int i = 0; i < len; i++) {                             \
mask[i] = fun;                                            \
}                                                           \
return mask;                                                \
}

MAKE_FILTER(fgauss, exp(-0.5*square(i/sigma)));

/* convolve image with gaussian filter */
static inline image<float> *smooth(image<float> *src, float sigma) {
    std::vector<float> mask = make_fgauss(sigma);
    normalize(mask);
    
    image<float> *tmp = new image<float>(src->height(), src->width(), false);
    image<float> *dst = new image<float>(src->width(), src->height(), false);
    convolve_even(src, tmp, mask);
    convolve_even(tmp, dst, mask);
    
    delete tmp;
    return dst;
}

/* convolve image with gaussian filter */
inline image<float> *smooth(image<uchar> *src, float sigma) {
    image<float> *tmp = imageUCHARtoFLOAT(src);
    image<float> *dst = smooth(tmp, sigma);
    delete tmp;
    return dst;
}

/* compute laplacian */
inline static image<float> *laplacian(image<float> *src) {
    int width = src->width();
    int height = src->height();
    image<float> *dst = new image<float>(width, height);
    
    for (int y = 1; y < height-1; y++) {
        for (int x = 1; x < width-1; x++) {
            float d2x = imRef(src, x-1, y) + imRef(src, x+1, y) -
            2*imRef(src, x, y);
            float d2y = imRef(src, x, y-1) + imRef(src, x, y+1) -
            2*imRef(src, x, y);
            imRef(dst, x, y) = d2x + d2y;
        }
    }
    return dst;
}


// disjoint-set forests using union-by-rank and path compression (sort of).

typedef struct {
    int rank;
    int p;
    int size;
    int link; //for circular list //All node in a set are all connected through a circular list
    bool printed; //Set already printed??
} uni_elt;

class universe {
public:
    universe(int elements);
    ~universe();
    int find(int x);
    void join(int x, int y);
    bool printSet(int x, cv::Mat_<uchar>& gray);
    static inline void getPixelCoordFromIdx(int idx, int w,int& x, int& y)
    {
        x = (idx % w);
        y = (idx / w); //integer division
        return;
    }
    int size(int x) const { return elts[x].size; }
    int num_sets() const { return num; }
    void collectSets(std::vector<std::vector<cv::Point3i> >& xyi, int H, int W);
    
    
private:
    uni_elt *elts;
    int num;
};




template <class T>
inline T mapminmax(const T& x, const T& xmin, const T& xmax,
                   const T& ymin, const T& ymax)
{
    return( (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin );
}


/* threshold image */
template <class T>
inline image<uchar> *threshold(image<T> *src, int t) {
    int width = src->width();
    int height = src->height();
    image<uchar> *dst = new image<uchar>(width, height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(dst, x, y) = (imRef(src, x, y) >= t);
        }
    }
    
    return dst;
}

enum eClassType: uint8_t{ RGB=0, HSV, CIELAB};

struct SegResults {
    cv::Point3f centroid3D;
    cv::Point2i centroid2D;
    cv::Point3f centroid3DFake;
    float eigenVal1;
    float eigenVal2;
    float angle;
    size_t num_points;
    std::vector<cv::Point3i> pxs;//x=x,y=y,z=idx
    cv::Mat_<cv::Vec3b> clusterRGB;
    cv::Mat_<uint16_t> clusterDepth;
    cv::Rect rect_aabb_;
    
    SegResults(const cv::Point3f& centroid3D_,
               const cv::Point3f& centroid3DFake_,
               const cv::Point2i& centroid2D_,
               float eigenVal1_,
               float eigenVal2_,
               float angle,
               size_t num_points_,
               const std::vector<cv::Point3i>& pxs_,
               const cv::Mat_<cv::Vec3b>& clusterRGB_,
               const cv::Mat_<uint16_t>& clusterDepth_,
               const cv::Rect& rect_aabb);
};

template <class T>
class GraphCannySeg {
public:
    float mSigma;
    float mK;//range da 0.003 a 0.001 x Shampoo //0.0031
    int mMin_size;
    
    uint mNumClustersFounds;
    
    //fcn
    float mKx,mKy,mKs;
    //hsv
    float mKdv, mKdc;
    
    image<T>* mInput_img;
    image<rgb>* mInputRGB_img;
    image<uint16_t> * mInput_depth;
    image<uint16_t> * mSmoothedDepth;
    image<uint16_t> * mInpaintedDepth;//After it has been Smoothed
    
    cv::Mat_<uchar> mcvGray_img;
    cv::Mat_<float> K_inv;
    cv::Mat_<float> K_invt;
    cv::Mat_<uchar> mImageMask;
    
    //CANNY
    cv::Mat_<uchar> mMagOut;
    cv::Mat_<uchar> mOriOut;
    cv::Mat mSaliencyMyCanny;
    float*  mSaliencyMyCannyPtr;
    
    //Canny Depth Ths
    uint16_t mDTH ; //[mm]
    uint16_t mplusD ; //for depth boundary
    uint16_t mpoint3D ; //for contact boundary
    float mg_angle ;//2.f/3.f*M_PI;
    float ml_angle ; //M_PI/3.f;
    float mLcannyTH;
    float mHcannyTH;
    
    
    eClassType mClass_type;
    
    uint mWidth, mHeight, mRxC;
    
    //we leave the white color only for clusters < min_size
    //i.e. the one we throw away.
    const rgb rgbWHITE;//(255,255,255);
    const float c_mapVeS = 1.f/255.f;
    
    //for debug
    std::vector<float> DrgbV;
    std::vector<float> DdepthV;
    
    //PCA filtering
    float mMax_eccentricity;
    float mMax_L1;//lamba1
    float mMax_L2;//lambda2
    float mFarObjZ;//object centroid > is rejected [mm]
    
    //To Save the results
    std::vector<SegResults> vecSegResults;

    // added by STE
//    cv::Mat rgbimg;
//    cv::Mat depthimg;
    
public:
    
    GraphCannySeg();
    
    GraphCannySeg(const cv::Mat& rgb_img, const cv::Mat_<uint16_t>& depth_img, float sigma_, float k_, float min_size, float kx_,float ky_, float ks_, float k_vec[9],float Lcannyth_=0.05f,float Hcannyth_=0.075f, float kdv_=4.5f, float kdc_=0.1f,float max_ecc=0.97f, float max_l1=2000.f, float max_l2=970.0f,uint16_t DTH = 30, uint16_t plusD = 5,
                  uint16_t point3D = 10, float g_angle = 120.f*deg2rad, float l_angle = 60.f*deg2rad, float FarObjZ=1100);
    template <class Q>
    inline image<Q>* convertMat2Image(const cv::Mat& m)
    {        
        int  width = m.cols;
        int  height = m.rows;
        
        if(m.channels()==3 && typeid(Q)==typeid(rgb))//RGB
        {
            //load the RGB image
            printf("Opencv RGB Image\n");
            //        cv::imwrite("/Users/giorgio/Documents/Polito/PhD/Slides/PresentazionePhDfineAnno/rgbAPK.jpg", m);
            cv::Mat m_rgb;
            cv::cvtColor(m, m_rgb, CV_BGR2RGB);
            uchar* m_rgbPtr = m_rgb.ptr();
            image<Q> *im = new image<Q>(width, height);
            memcpy(im->data, m_rgbPtr, width * height * sizeof(Q));
            return im;
        }
        else if(m.channels()==3 && typeid(Q)==typeid(hsv))
        {
            //Load the HSV image
            printf("Opencv HSV Image\n");
            //Gaussina Bluer
            //Size(5,5) is obtained from sigma 0.8=> ceil(0.8*4)+1;
            //cv::GaussianBlur(m, m, cv::Size(5,5), sigma_);
            cv::Mat m_hsv;
            cv::cvtColor(m, m_hsv, CV_BGR2HSV);
            uchar* m_hsvPtr = m_hsv.ptr();
            image<Q> *im = new image<Q>(width, height);
            memcpy(im->data, m_hsvPtr, width * height * sizeof(Q));
            return im;
            
        }
        else if(m.channels()==3 && typeid(Q)==typeid(CIELab))
        {
            //Load the CIELab image
            printf("Opencv CIELab Image\n");
            //Gaussina Bluer
            //Size(5,5) is obtained from sigma 0.8=> ceil(0.8*4)+1;
            //cv::GaussianBlur(m, m, cv::Size(5,5), sigma_);
            cv::Mat_<cv::Vec3f> m_cielab;
            cv::Mat_<cv::Vec3f> mfloat;
            m.convertTo( mfloat, CV_32FC3, 1.0f/255.0f);
            cv::cvtColor(mfloat, m_cielab, CV_BGR2Lab);
            float* m_cielabPtr = m_cielab.ptr<float>(0);
            image<Q> *im = new image<Q>(width, height);
            memcpy(im->data, m_cielabPtr, width * height * sizeof(Q));
            CIELab min_,max_;
            min_max<CIELab>((image<CIELab> *)im,&min_,&max_);
            printf("min: L %f; a %f; b %f\n",min_.L,min_.a,min_.b);
            printf("max: L %f; a %f; b %f\n",max_.L,max_.a,max_.b);
            
            return im;
            
        }
        else if (m.channels()==1)
        {
//            printf("m.depth %d\n",m.depth());
//            if(m.depth() == CV_32F)
//            {
//                printf("32F \n");
//            }
            printf("Opencv 1 CH Image\n");
            const Q* m_Ptr = m.ptr<Q>(0);
            image<Q> *im = new image<Q>(width, height);
            memcpy(im->data, m_Ptr, width * height * sizeof(Q));
            
            //Debug...
            /*
             cv::Mat debmat = cv::Mat(m.size(),CV_16U,im->data);
             //convert in Gray scale
             double minVal, maxVal;
             cv::Mat J1_gray_img;
             cv::minMaxLoc(debmat,&minVal,&maxVal);
             debmat.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
             cv::imshow("debug gray", J1_gray_img);
             cv::waitKey(0);
             */
            
            return im;
        }
        else
        {
            printf("ERROR unknown File Format\n");
            exit(EXIT_FAILURE);
        }

    }
    template <class Q>
    static inline void visualizeImage(image<Q> *im, std::string win_name="Segmentation Result", int wait_key_=0, bool bInpaintDepth=false)
    {
        if(typeid(Q)==typeid(rgb) )
        {
            cv::Mat seg = cv::Mat(im->height(),im->width(),CV_8UC3,im->data);
            cv::Mat seg_bgr;
            cv::cvtColor(seg, seg_bgr, CV_RGB2BGR);
            cv::imshow(win_name, seg_bgr);
            cv::waitKey(wait_key_);
        }
        else if(typeid(Q)==typeid(hsv))
        {
            cv::Mat seg = cv::Mat(im->height(),im->width(),CV_8UC3,im->data);
            
            cv::imshow(win_name, seg);
            cv::waitKey(wait_key_);
        }
        else //uint16_t
        {
            
            cv::Mat m(im->height(),im->width(),CV_16U,im->data);
            
            if (bInpaintDepth)
            {
                //inside visualization the depth_mm image inpaint modification has no meaninig...just for visualization purposes
                image<uint16_t>* inpimg = inpaintDepth(m,true,wait_key_,win_name);
            }
            else
            {
                //convert in Gray scale
                double minVal, maxVal;
                cv::Mat J1_gray_img;
                cv::minMaxLoc(m,&minVal,&maxVal);
                m.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
                
                cv::Mat colorMatJ1;
                cv::applyColorMap(J1_gray_img, colorMatJ1, cv::COLORMAP_JET);
                cv::imshow(win_name, colorMatJ1);
                cv::waitKey(wait_key_);
            }
            
        }
        
        return;
    }
    static inline void dynamicDepthSmoothing(image<uint16_t> *im, image<uint16_t> * smoothedDepth, const float beta = 1500.0f, const float gamma = 18000.0f)
    {
        
        image<uint16_t> *ModDepth_mm = im->copy();
        
        /** Depth Dependent Smoothing Area Map : B(r,c) **/
        
        //since here the depth is in mm...
        const float alfa = 0.0028f/1e6f;
        //const float beta = 1500;
        //% Fdc (eq 4)
        std::vector<float> Fdc(im->WxH(),0.f);
        std::vector<float> B(im->WxH(),0.f);
        
        for (int ii=0; ii<im->WxH(); ++ii) {
            
            Fdc[ii] = im->data[ii]*im->data[ii]*alfa;
            B[ii] = beta*Fdc[ii];
        }
        std::cout << "min Fdc: " << *std::min_element(Fdc.begin(),Fdc.end())<<"\n";
        std::cout << "max Fdc: " << *std::max_element(Fdc.begin(),Fdc.end())<<"\n";
        cv::Mat cvB = cv::Mat(im->height(),im->width(),CV_32F,&B[0]);
        //visualizeColorMap(cvB,"B",5);
        
        /** Depth Change Indicator Map : C(r,c) **/
        //const float gamma = 18000.f;
        std::vector<uchar> C(im->WxH(),0);
        
        for(int r=0; r<im->height(); ++r) //y
        {
            for(int c=0; c<im->width(); ++c) //x
            {
                
                int r_1 = r+1;
                int r__1 = r-1;
                int c_1 = c+1;
                int c__1 = c-1;
                if(c_1>=im->width() || r_1>=im->height() || r__1<0 || c__1<0)
                    continue;
                
                int pivot = r*im->width() + c;
                //pixel IDX +1
                int r_p1  = (r_1)*im->width() + c;
                int c_p1  = r*im->width() + (c_1);
                //pixel IDX -1
                int r_m1  = (r__1)*im->width() + c;
                int c_m1  = r*im->width() + (c__1);
                
                
                int Ddx = static_cast<int>(im->data[c_p1] - im->data[pivot]);
                int Ddy = static_cast<int>(im->data[r_p1] - im->data[pivot]);
                
                float Tdc = gamma*Fdc[pivot];
                //printf(" %f, ",Tdc);
                //printf(" %d, ",Ddx);
                if(static_cast<float>(abs(Ddx))>=Tdc)
                {
                    C[pivot] = 255; //1
                    C[c_p1] = 255; //1
                    C[c_m1] = 255; //1
                    ModDepth_mm->data[pivot] = 0;
                    ModDepth_mm->data[c_p1] = 0;
                    ModDepth_mm->data[c_m1] = 0;
                }
                if (static_cast<float>(abs(Ddy))>=Tdc)
                {
                    C[pivot] = 255;
                    C[r_p1] = 255; //1
                    C[r_m1] = 255; //1
                    ModDepth_mm->data[pivot] = 0;
                    ModDepth_mm->data[r_p1] = 0;
                    ModDepth_mm->data[r_m1] = 0;
                }
                
            }
        }
        cv::Mat cvC = cv::Mat(im->height(),im->width(),CV_8U,&C[0]);
        //    cv::imshow("C",cvC);
        //    cv::waitKey(0);
        //visualizeColorMap(cvC,"C",5,true);
        
        /** Final Smoothing Area Map : R(r,c) **/
        
        cv::Mat_<uchar> Cmat = cv::Mat_<uchar>(im->height(),im->width(), C.data()); //or simplier &C[0];
        cv::Mat Cmatnot;
        cv::bitwise_not(Cmat, Cmatnot);
        cv::Mat Tmat_;
        cv::distanceTransform(Cmatnot, Tmat_, CV_DIST_L2, CV_DIST_MASK_PRECISE);
        Tmat_ = Tmat_*M_SQRT1_2; //Tmat_./sqrt(2);
        //visualizeColorMap(Tmat_,"Tsqrt",5,true);
        
        float* Tptr = Tmat_.ptr<float>(0);
        
        std::vector<float> R(im->WxH(),0.f);
        for (int ii=0; ii<im->WxH(); ++ii) {
            
            R[ii] = std::min(B[ii], Tptr[ii]);
        }
        
        cv::Mat cvR = cv::Mat(im->height(),im->width(),CV_32F,&R[0]);
        visualizeColorMap(cvR,"R",5,false);
        
        /** Smoothing **/
        
        //integral image
        cv::Mat cvModDepth_mm(im->height(),im->width(),CV_16U,ModDepth_mm->data);
        cv::Mat cvModDepth_mmFloat;
        cvModDepth_mm.convertTo(cvModDepth_mmFloat, CV_32F);
        cv::Mat cvModDepth_mmIntegral;
        cv::integral(cvModDepth_mmFloat, cvModDepth_mmIntegral, CV_32F);
        /* I added this by myself even though not present in the original paper....it is not visible directly on the depth map, but if the depth map is converted in PC, a lot of noise is added....with this trick boarders are free from noise...the demostration of this is saved in my own matlab script i've made during the test of the original code !!
         % Since the average filter filters also the 0 (NaN) that
         % finds in the kernel thus modifying the value of the border
         % we need to adjust the average value computing the zeros found in that
         % kernel hence modifying the average value.
         */
        //Convert the Depth image in logical image setting all zero (NaN) values to 1
        cv::Mat logicalDepthmm = cv::Mat::zeros(im->height(),im->width(),CV_8U);
        uchar* logicalDepthmmPtr = logicalDepthmm.ptr<uchar>(0);
        for (int ii=0; ii<im->WxH(); ++ii) {
            
            if(ModDepth_mm->data[ii] == 0)
            {
                logicalDepthmmPtr[ii] = 1;//use 1 since i need it for integral image
            }
        }
        
        //    cv::imshow("logicalDepthmmPtr",logicalDepthmm*255);
        //    cv::waitKey(0);
        
        //The integral image counts the NaN (1) elements inside a given Kernel
        cv::Mat IntegralDepth_mmLogical;
        cv::integral(logicalDepthmm,IntegralDepth_mmLogical,CV_8U);
        
        cv::Mat SDepth_mm = cv::Mat::zeros(im->height(),im->width(),CV_16U);
        uint16_t* SDepth_mmPtr = SDepth_mm.ptr<uint16_t>(0);
        
        //TODO: Smoothing X Y ranges...
        for(int r=15; r<im->height()-15; ++r) //y
        {
            for(int c=15; c<im->width()-15; ++c) //x
            {
                int pivot = r*im->width() + c;
                int radi = int(R[pivot]+0.5f);
                
                if(radi == 0)
                {
                    //boarder or NaN: No smoothing at all
                    SDepth_mmPtr[pivot] = im->data[pivot];
                    continue;
                }
                
                int denumSqrt = (2*radi+1);
                int denum_ = denumSqrt*denumSqrt;
                
                //cv::Rect_<int> Ri(c-radi,r-radi,denumSqrt,denumSqrt);
                
                cv::Point2i P1, P2;
                
                P1.x = c-radi;
                P1.y = r-radi;
                
                P2.x = c-radi + denumSqrt;
                P2.y = r-radi + denumSqrt;
                
                
                //TODO:optimize through pointers
                int NumNaN = static_cast<int>(
                                              (IntegralDepth_mmLogical.at<int>(P2.y,P2.x)) + //D
                                              (IntegralDepth_mmLogical.at<int>(P1.y,P1.x)) - //A
                                              (IntegralDepth_mmLogical.at<int>(P2.y,P1.x)) - //C
                                              (IntegralDepth_mmLogical.at<int>(P1.y,P2.x)));//B
                
                
                if(denum_==NumNaN)
                {
                    //all pixels are NaN
                    SDepth_mmPtr[pivot] = im->data[pivot];
                    continue;
                }
                if(denum_<NumNaN)
                {
                    //impossibile!!!!!!
                    printf("ERROR denum_<NumNaN impossibile!!!!\n");
                    printf("row: %d; col: %d\n",r,c);
                    printf("denum_: %d; NumNaN: %d\n",denum_,NumNaN);
                    exit(EXIT_FAILURE);
                }
                
                SDepth_mmPtr[pivot] = static_cast<uint16_t>
                (
                 (1.0f/(static_cast<float>(denum_-NumNaN))*
                  (
                   (cvModDepth_mmIntegral.at<float>(P2.y, P2.x)) +//D
                   (cvModDepth_mmIntegral.at<float>(P1.y, P1.x)) -//A
                   (cvModDepth_mmIntegral.at<float>(P2.y,P1.x)) -//C
                   (cvModDepth_mmIntegral.at<float>(P1.y,P2.x)) //B
                   ))+0.5f
                 );
                
                
            }
        }
        
        smoothedDepth->init(SDepth_mm);
        
        
        
    }
    //use the member variable
    void dynamicDepthSmoothing(const float beta = 1500.0f, const float gamma = 18000.0f);
    
    static inline image<uint16_t>* inpaintDepth(image<uint16_t>* imd, bool viz=false,int waitKey_=5,std::string winname="depth-inpainted map")
    {
        
        //Convert depth in mm to gray since inpaint is performed on the gray scale image
        
        cv::Mat depth_mm(imd->height(),imd->width(),CV_16U,imd->data);
        
        double minVal, maxVal;
        cv::Mat J1_gray_img;
        cv::minMaxLoc(depth_mm,&minVal,&maxVal);
        depth_mm.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
        
        //Prova Inpaint Depth Map
        const unsigned char noDepth = 0; // change to 255, if values no depth uses max value
        cv::Mat temp, temp2;
        
        // 1 step - downsize for performance, use a smaller version of depth image
        cv::Mat small_depthf;
        cv::resize(J1_gray_img, small_depthf, cv::Size(), 0.2, 0.2);
        
        // 2 step - inpaint only the masked "unknown" pixels
        cv::inpaint(small_depthf, (small_depthf == noDepth), temp, 20.0, cv::INPAINT_TELEA);
        
        // 3 step - upscale to original size and replace inpainted regions in original depth image
        cv::resize(temp, temp2, J1_gray_img.size());
        temp2.copyTo(J1_gray_img, (J1_gray_img == noDepth)); // add to the original signal
        
        uint16_t depth_gray_scale = static_cast<uint16_t>((maxVal-minVal)/255.0);
        
        int mRxC = depth_mm.rows*depth_mm.cols;
        uint16_t* m_cvDepth_mmPtr = depth_mm.ptr<uint16_t>(0);
        uchar* m_cvDepth_grayPtr = J1_gray_img.ptr<uchar>(0);
        for(int ii=0;ii<mRxC;++ii)
        {
            if(m_cvDepth_mmPtr[ii]==0)//NaN
            {
                //fill the [mm] map with the impainted value
                m_cvDepth_mmPtr[ii] =
                static_cast<uint16_t>(m_cvDepth_grayPtr[ii])*depth_gray_scale + static_cast<uint16_t>(minVal);
                
            }
        }
        //update new min & max of m_cvDepth_mm after inpainting
        //TODO: Maybe unnecessary because these values are no more used outside this Fcn.
        cv::minMaxLoc(depth_mm,&minVal,&maxVal);
        /*        //for debug
         double minVal, maxVal;
         cv::Mat m_cvDepth_grayDebug;
         cv::minMaxLoc(m_cvDepth_mm,&minVal,&maxVal);
         m_cvDepth_mm.convertTo(m_cvDepth_grayDebug,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
         cv::imshow("m_cvDepth_grayDebug",m_cvDepth_grayDebug);
         cv::imshow("m_cvDepth_gray org inpainted",m_cvDepth_gray);
         */
        if(viz)
            visualizeColorMap(J1_gray_img,winname,waitKey_);
        
        image<uint16_t> *im = new image<uint16_t>(depth_mm.cols, depth_mm.rows);
        memcpy(im->data, m_cvDepth_mmPtr , mRxC*sizeof(uint16_t));
        
        return im;
        
        
        
        
    }
    
    static inline image<uint16_t>* inpaintDepth(cv::Mat& depth_mm, bool viz=false,int waitKey_=5,std::string winname="depth-inpainted map")
    {
        
        //Convert depth in mm to gray since inpaint is performed on the gray scale image
        double minVal, maxVal;
        cv::Mat J1_gray_img;
        cv::minMaxLoc(depth_mm,&minVal,&maxVal);
        depth_mm.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
        
        //Prova Inpaint Depth Map
        const unsigned char noDepth = 0; // change to 255, if values no depth uses max value
        cv::Mat temp, temp2;
        
        // 1 step - downsize for performance, use a smaller version of depth image
        cv::Mat small_depthf;
        cv::resize(J1_gray_img, small_depthf, cv::Size(), 0.2, 0.2);
        
        // 2 step - inpaint only the masked "unknown" pixels
        cv::inpaint(small_depthf, (small_depthf == noDepth), temp, 5.0, cv::INPAINT_TELEA);
        
        // 3 step - upscale to original size and replace inpainted regions in original depth image
        cv::resize(temp, temp2, J1_gray_img.size());
        temp2.copyTo(J1_gray_img, (J1_gray_img == noDepth)); // add to the original signal
        
        uint16_t depth_gray_scale = static_cast<uint16_t>((maxVal-minVal)/255.0);
        
        int mRxC = depth_mm.rows*depth_mm.cols;
        uint16_t* m_cvDepth_mmPtr = depth_mm.ptr<uint16_t>(0);
        uchar* m_cvDepth_grayPtr = J1_gray_img.ptr<uchar>(0);
        for(int ii=0;ii<mRxC;++ii)
        {
            if(m_cvDepth_mmPtr[ii]==0)//NaN
            {
                //fill the [mm] map with the impainted value
                m_cvDepth_mmPtr[ii] =
                static_cast<uint16_t>(m_cvDepth_grayPtr[ii])*depth_gray_scale + static_cast<uint16_t>(minVal);
                
            }
        }
        //update new min & max of m_cvDepth_mm after inpainting
        //TODO: Maybe unnecessary because these values are no more used outside this Fcn.
        cv::minMaxLoc(depth_mm,&minVal,&maxVal);
        /*        //for debug
         double minVal, maxVal;
         cv::Mat m_cvDepth_grayDebug;
         cv::minMaxLoc(m_cvDepth_mm,&minVal,&maxVal);
         m_cvDepth_mm.convertTo(m_cvDepth_grayDebug,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
         cv::imshow("m_cvDepth_grayDebug",m_cvDepth_grayDebug);
         cv::imshow("m_cvDepth_gray org inpainted",m_cvDepth_gray);
         */
        if(viz)
            visualizeColorMap(J1_gray_img,winname,waitKey_);
        
        image<uint16_t> *im = new image<uint16_t>(depth_mm.cols, depth_mm.rows);
        memcpy(im->data, m_cvDepth_mmPtr , mRxC*sizeof(uint16_t));
        
        return im;
        
        
        
        
    }

    
    //use the member variable
    inline void inpaintDepth(bool viz=false,int waitKey_=5,std::string winname="depth-inpainted map")
    {
        
        //Convert depth in mm to gray since inpaint is performed on the gray scale image
        //TODO: It takes the mSmoothedDepth image...so be sure DynamicDepthSmoothing is performed first !!!
        
        cv::Mat depth_mm(mSmoothedDepth->height(),mSmoothedDepth->width(),CV_16U,mSmoothedDepth->data);
        
        double minVal, maxVal;
        cv::Mat J1_gray_img;
        cv::minMaxLoc(depth_mm,&minVal,&maxVal);
        depth_mm.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
        
        //Prova Inpaint Depth Map
        const unsigned char noDepth = 0; // change to 255, if values no depth uses max value
        cv::Mat temp, temp2;
        
        // 1 step - downsize for performance, use a smaller version of depth image
        cv::Mat small_depthf;
        cv::resize(J1_gray_img, small_depthf, cv::Size(), 0.2, 0.2);
        
        // 2 step - inpaint only the masked "unknown" pixels
        cv::inpaint(small_depthf, (small_depthf == noDepth), temp, 20.0, cv::INPAINT_TELEA);
        
        // 3 step - upscale to original size and replace inpainted regions in original depth image
        cv::resize(temp, temp2, J1_gray_img.size());
        temp2.copyTo(J1_gray_img, (J1_gray_img == noDepth)); // add to the original signal
        
        uint16_t depth_gray_scale = static_cast<uint16_t>((maxVal-minVal)/255.0);
        
        int mRxC = depth_mm.rows*depth_mm.cols;
        uint16_t* m_cvDepth_mmPtr = depth_mm.ptr<uint16_t>(0);
        uchar* m_cvDepth_grayPtr = J1_gray_img.ptr<uchar>(0);
        for(int ii=0;ii<mRxC;++ii)
        {
            if(m_cvDepth_mmPtr[ii]==0)//NaN
            {
                //fill the [mm] map with the impainted value
                m_cvDepth_mmPtr[ii] =
                static_cast<uint16_t>(m_cvDepth_grayPtr[ii])*depth_gray_scale + static_cast<uint16_t>(minVal);
                
            }
        }
        //update new min & max of m_cvDepth_mm after inpainting
        //TODO: Maybe unnecessary because these values are no more used outside this Fcn.
        cv::minMaxLoc(depth_mm,&minVal,&maxVal);
        /*        //for debug
         double minVal, maxVal;
         cv::Mat m_cvDepth_grayDebug;
         cv::minMaxLoc(m_cvDepth_mm,&minVal,&maxVal);
         m_cvDepth_mm.convertTo(m_cvDepth_grayDebug,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
         cv::imshow("m_cvDepth_grayDebug",m_cvDepth_grayDebug);
         cv::imshow("m_cvDepth_gray org inpainted",m_cvDepth_gray);
         */
        if(viz)
            visualizeColorMap(J1_gray_img,winname,waitKey_);
        
//        image<uint16_t> *im = new image<uint16_t>(depth_mm.cols, depth_mm.rows);
//        memcpy(im->data, m_cvDepth_mmPtr , mRxC*sizeof(uint16_t));
//        
//        return im;
        mInpaintedDepth->init(depth_mm);
        
    }
    
    static inline void myCannyWithOri2(const cv::Mat& grayImg,
                                cv::Mat_<uchar>& MagOut,
                                cv::Mat_<uchar>& OriOut,
                                float T_Low, float T_High,
                                const cv::Mat_<float>& Kinv,
                                const cv::Mat& mask,
                                image<uint16_t> *depth,
                                float* saliency,
                                bool blurTheResult = true)
    {
        
        const int nx = grayImg.cols;
        const int ny = grayImg.rows;
        const int RxC = nx*ny;
        
        //std::cout<< cv::getGaussianKernel(5,0)<<"\n";
        
        cv::GaussianBlur( grayImg, grayImg, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );
        
        cv::Mat Gx;
        //cv::Sobel(grayImg, Gx, CV_32F, 1, 0, 3);
        cv::Scharr( grayImg, Gx, CV_32F, 1, 0, 1, 0, cv::BORDER_DEFAULT );
        
        cv::Mat Gy;
        //cv::Sobel(grayImg, Gy, CV_32F, 0, 1, 3, -1);
        cv::Scharr( grayImg, Gy, CV_32F, 0, 1, -1, 0, cv::BORDER_DEFAULT );
        
        cv::Mat_<float> Ori = cv::Mat_<float>::zeros(grayImg.size());
        cv::Mat_<uchar> Ori2 = cv::Mat_<uchar>::zeros(grayImg.size());
        cv::Mat_<float> G = cv::Mat_<float>::zeros(grayImg.size());
        cv::magnitude(Gx, Gy, G);
        //cv::phase(Gx, Gy, Ori, true);//true = [deg]
        
        float* GxPtr = Gx.ptr<float>(0);
        float* GyPtr = Gy.ptr<float>(0);
        float* GPtr = G.ptr<float>(0);
        float* OriPtr = Ori.ptr<float>(0);
        uchar* OriPtr2 = Ori2.ptr<uchar>(0);
        //Adjustment for negative directions, making all directions positive [0-360]
        for (int i = 0; i < RxC; ++i)
        {
            //        if (OriPtr[i]<0)
            //            OriPtr[i]=360.0f+OriPtr[i];
            OriPtr[i] = std::atan2(-GyPtr[i],-GxPtr[i])/M_PI*180.f + 180.0f;
            
        }
        
        //Adjusting directions to nearest 0, 45, 90, or 135 degree
        cv::Mat_<uint16_t> oppositeAngle = cv::Mat_<uint16_t>::zeros(grayImg.size());
        uint16_t* oppositeAnglePtr = oppositeAngle.ptr<uint16_t>(0);
        for (int i = 0; i < RxC; ++i)
        {
            if ( ((OriPtr[i] >= 0 ) && (OriPtr[i] < 22.5)) || ((OriPtr[i] >= 157.5) && (OriPtr[i] < 202.5)) || ((OriPtr[i] >= 337.5) && (OriPtr[i] <= 360)))
            {
                OriPtr2[i] = 0;
                if((OriPtr[i] >= 157.5) && (OriPtr[i] < 202.5))
                {
                    oppositeAnglePtr[i] = 180;
                }
            }
            else if ( ((OriPtr[i] >= 22.5) && (OriPtr[i] < 67.5)) ||   ((OriPtr[i] >= 202.5) && (OriPtr[i] < 247.5)) )
            {
                OriPtr2[i] = 45;
                if((OriPtr[i] >= 202.5) && (OriPtr[i] < 247.5))
                {
                    oppositeAnglePtr[i] = 225;
                }
            }
            else if ( ((OriPtr[i] >= 67.5 && OriPtr[i] < 112.5)) || ((OriPtr[i] >= 247.5 && OriPtr[i] < 292.5)) )
            {
                OriPtr2[i] = 90;
                if((OriPtr[i] >= 247.5 && OriPtr[i] < 292.5))
                {
                    oppositeAnglePtr[i] = 270;
                }
            }
            else if ( ((OriPtr[i] >= 112.5 && OriPtr[i] < 157.5)) || ((OriPtr[i] >= 292.5 && OriPtr[i] < 337.5)) )
            {
                OriPtr2[i] = 135;
                if((OriPtr[i] >= 292.5 && OriPtr[i] < 337.5))
                {
                    oppositeAnglePtr[i] = 315;
                }
            }
            
        }
        
        cv::Mat_<float> BW = cv::Mat_<float>::zeros(grayImg.size());
        OriOut = cv::Mat_<uchar>::zeros(grayImg.size());
        uchar* OriOutPtr = OriOut.ptr<uchar>(0);
        
        cv::Mat_<uint16_t> oppositeAngle2 = cv::Mat_<uint16_t>::zeros(grayImg.size());
        // Non-Maximum Supression
        for (int i=1; i < ny-1; ++i)
        {
            for(int j=1; j < nx-1; ++j )
            {
                if (Ori2(i,j)==0)
                {
                    float G3[] = { G(i,j), G(i,j+1), G(i,j-1) };
                    if(G(i,j) == *std::max_element(G3,G3+3))
                    {
                        BW(i,j) = G(i,j);
                        oppositeAngle2(i,j) = oppositeAngle(i,j);
                    }
                }
                else if (Ori2(i,j)==45)
                {
                    float G3[] = { G(i,j), G(i+1,j-1), G(i-1,j+1) };
                    if(G(i,j) == *std::max_element(G3,G3+3))
                    {
                        BW(i,j) = G(i,j);
                        oppositeAngle2(i,j) = oppositeAngle(i,j);
                    }
                }
                else if (Ori2(i,j)==90)
                {
                    float G3[] = { G(i,j), G(i+1,j), G(i-1,j) };
                    if(G(i,j) == *std::max_element(G3,G3+3))
                    {
                        BW(i,j) = G(i,j);
                        oppositeAngle2(i,j) = oppositeAngle(i,j);
                    }
                }
                else if (Ori2(i,j)==135)
                {
                    float G3[] = { G(i,j), G(i+1,j+1), G(i-1,j-1) };
                    if(G(i,j) == *std::max_element(G3,G3+3))
                    {
                        BW(i,j) = G(i,j);
                        oppositeAngle2(i,j) = oppositeAngle(i,j);
                    }
                }
            }
        }
        
        // Hysteresis Thresholding
        double maxBW;
        cv::minMaxLoc(BW, 0, &maxBW);
        T_Low = T_Low * maxBW;
        T_High = T_High * maxBW;
        
        cv::Mat_<uint16_t> oppositeAngleOut = cv::Mat_<uint16_t>::zeros(grayImg.size());
        uint16_t* oppositeAngleOutPtr = oppositeAngleOut.ptr<uint16_t>(0);
        
        MagOut = cv::Mat_<uchar>::zeros(grayImg.size());
        uchar* MagOutPtr = MagOut.ptr<uchar>(0);
        for (int i=0; i < ny; ++i)
        {
            for(int j=0; j < nx; ++j )
            {
                
                if (BW(i, j) < T_Low)
                    MagOut(i, j) = 0;
                else if (BW(i, j) > T_High)
                {
                    MagOut(i, j) = 255;
                    //TODO: Added 100 only for visualization...
                    OriOut(i, j) = Ori2(i,j)+100;
                    oppositeAngleOut(i,j) = oppositeAngle2(i,j);
                }
                //Using 8-connected components
                else if ( BW(i+1,j)>T_High || BW(i-1,j)>T_High || BW(i,j+1)>T_High || BW(i,j-1)>T_High || BW(i-1, j-1)>T_High || BW(i-1, j+1)>T_High || BW(i+1, j+1)>T_High || BW(i+1, j-1)>T_High)
                {
                    MagOut(i, j) = 255;
                    OriOut(i, j) = Ori2(i,j)+100;
                    oppositeAngleOut(i,j) = oppositeAngle2(i,j);
                }
            }
        }
        
        if(!mask.empty())
        {
            //filter by mask
            const uchar* maskPtr = mask.ptr<uchar>(0);
            for(int i=0;i<RxC;++i)
            {
                if(maskPtr[i]==0)
                {
                    MagOutPtr[i] = 0;
                    OriOutPtr[i] = 0;
                    oppositeAngleOutPtr[i] = 0;
                }
            }
        }
        if(depth==0)
            return;
        //filter by depth try to remove some textures leaving real borders unchanged
        cv::Mat_<cv::Vec3b> BOWOut = cv::Mat_<cv::Vec3b>::zeros(grayImg.size());
        for (int i = 0; i < ny ; ++i)//riga y
        {
            for (int j = 0; j < nx ; ++j)//colonna x
            {
                const int c = j + nx * i;
                if(i<11 || i>ny-11 || j<11 || j>nx-11)
                {
                    MagOutPtr[c]=0;
                    oppositeAngleOutPtr[c] = 0;
                    continue;
                }
                
                if(MagOutPtr[c]==0)
                    continue;
                //            if(imRef(depth, j, i)==0)
                //            {
                //                MagOutPtr[c]=0;
                //                continue;
                //            }
                //TODO:use these one
                const int p90 = c - nx;
                const int m90 = c + nx;
                const int p0 = c + 1;
                const int m0 = c - 1;
                const int p45 = p90 + 1;
                const int p135 = p90 - 1;
                const int m135 = m90 + 1;
                const int m45 = m90 - 1;
                
                
                //else border or texture: MagOutPtr[c]==255
                //TODO: Handle NaN in Depth
                uint16_t DTH = 30; //[mm]
                uint16_t plusD = 5; //for depth boundary
                uint16_t point3D = 10; //for contact boundary
                float g_angle = 120.f*M_PI/180.f;//2.f/3.f*M_PI;
                float l_angle = 60.f*M_PI/180.f; //M_PI/3.f;
                if ((OriOut(i,j)-100)==0)
                {
                    //float G3[] = { G(i,j), G(i,j+1), G(i,j-1) };
                    //                uint16_t dp = std::abs(imRef(depth, j, i) - imRef(depth, j+1, i));
                    //                uint16_t dm = std::abs(imRef(depth, j, i) - imRef(depth, j-1, i));
                    
                    uint16_t dp2 = std::abs(imRef(depth, j, i) - imRef(depth, j+plusD, i));
                    uint16_t dm2 = std::abs(imRef(depth, j, i) - imRef(depth, j-plusD, i));
                    
                    cv::Mat_<float> p0,p1,p2;
                    projectPixel2CameraRF(Kinv, j, i, imRef(depth, j, i), p0);
                    projectPixel2CameraRF(Kinv, j+point3D, i, imRef(depth, j+point3D, i), p1);
                    projectPixel2CameraRF(Kinv, j-point3D, i, imRef(depth, j-point3D, i), p2);
                    float theta;//in rad
                    angle3Points(p1, p0, p2, theta);
                    
                    if( (/*(dp<DTH && dm<DTH) ||*/ (dp2<DTH && dm2<DTH))
                       &&  (theta>g_angle || theta<l_angle))
                    {
                        MagOutPtr[c]=0;
                        oppositeAngleOutPtr[c]=0;
                        //                    MagOutPtr[c+plusD]=255;
                        //                    MagOutPtr[c-plusD]=255;
                        //                    MagOut(i,j+plusD) = 255;
                        //                    MagOut(i,j-plusD) = 255;
                    }
                    //                else
                    //                {
                    //                    //Border Ownership
                    //                    if(imRef(depth, j+plusD, i) < imRef(depth, j-plusD, i))
                    //                    {
                    //                        BOWOut(i,j)[2] = 255;//red
                    //                    }
                    //                    else
                    //                    {
                    //                         if(oppositeAngleOut(i,j)==180 || (Ori(i,j)< 15 &&  Ori(i,j)>=0))
                    //                            BOWOut(i,j)[2] = 255;//red
                    //                         else
                    //                            BOWOut(i,j)[1] = 255;//green
                    //                    }
                    //
                    //                }
                    
                }
                else if ((OriOut(i,j)-100)==45)
                {
                    //float G3[] = { G(i,j), G(i+1,j-1), G(i-1,j+1) };
                    //                uint16_t dp = std::abs(imRef(depth, j, i) - imRef(depth, j+1, i-1));
                    //                uint16_t dm = std::abs(imRef(depth, j, i) - imRef(depth, j-1, i+1));
                    
                    uint16_t dp2 = std::abs(imRef(depth, j, i) - imRef(depth, j+plusD, i-plusD));
                    uint16_t dm2 = std::abs(imRef(depth, j, i) - imRef(depth, j-plusD, i+plusD));
                    
                    cv::Mat_<float> p0,p1,p2;
                    projectPixel2CameraRF(Kinv, j, i, imRef(depth, j, i), p0);
                    projectPixel2CameraRF(Kinv, j+point3D, i-point3D, imRef(depth, j+point3D, i-point3D), p1);
                    projectPixel2CameraRF(Kinv, j-point3D, i+point3D, imRef(depth, j-point3D, i+point3D), p2);
                    float theta;//in rad
                    angle3Points(p1, p0, p2, theta);
                    
                    if( (/*(dp<DTH && dm<DTH) ||*/ (dp2<DTH && dm2<DTH))
                       &&  (theta>g_angle || theta<l_angle))
                    {
                        MagOutPtr[c]=0;
                    }
                }
                else if ((OriOut(i,j)-100)==90)
                {
                    //float G3[] = { G(i,j), G(i+1,j), G(i-1,j) };
                    //                uint16_t dp = std::abs(imRef(depth, j, i) - imRef(depth, j, i-1));
                    //                uint16_t dm = std::abs(imRef(depth, j, i) - imRef(depth, j, i+1));
                    
                    uint16_t dp2 = std::abs(imRef(depth, j, i) - imRef(depth, j, i-plusD));
                    uint16_t dm2 = std::abs(imRef(depth, j, i) - imRef(depth, j, i+plusD));
                    
                    cv::Mat_<float> p0,p1,p2;
                    projectPixel2CameraRF(Kinv, j, i, imRef(depth, j, i), p0);
                    projectPixel2CameraRF(Kinv, j, i-point3D, imRef(depth, j, i-point3D), p1);
                    projectPixel2CameraRF(Kinv, j, i+point3D, imRef(depth, j, i+point3D), p2);
                    float theta;//in rad
                    angle3Points(p1, p0, p2, theta);
                    
                    if(j==216 && i==259)
                    {
                        printf("j: %d; i: %d; dp2: %d; dm2: %d; theta: %f\n",j,i,dp2,dm2,theta*180.f/M_PI);
                    }
                    
                    if( ((dp2<DTH && dm2<DTH))
                       &&  (theta>g_angle || theta<l_angle))
                    {
                        MagOutPtr[c]=0;
                    }
                    
                    /*
                     if(j==215 && i==259)
                     {
                     printf("dp2: %d; dm2: %d; theta: %f\n",dp2,dm2,theta*180.f/M_PI);
                     }
                     if(j==189 && i==71)
                     {
                     printf("j==189 && i==71:\n dp2: %d; dm2: %d; theta: %f\n",dp2,dm2,theta*180.f/M_PI);
                     }
                     if(j==232 && i==182)
                     {
                     printf("j==232 && i==182:\n dp2: %d; dm2: %d; theta: %f\n",dp2,dm2,theta*180.f/M_PI);
                     }
                     
                     if( ( (dp2<DTH && dm2<DTH) || ( (theta<-20*M_PI/180.f && theta > -80*M_PI/180.f) && (dm2<5 && dp2>90)  ) )
                     && (theta>g_angle || theta<l_angle))
                     {
                     MagOutPtr[c]=0;
                     }
                     */
                }
                else if ((OriOut(i,j)-100)==135)
                {
                    //float G3[] = { G(i,j), G(i+1,j+1), G(i-1,j-1) };
                    //                uint16_t dp = std::abs(imRef(depth, j, i) - imRef(depth, j-1, i-1));
                    //                uint16_t dm = std::abs(imRef(depth, j, i) - imRef(depth, j+1, i+1));
                    //                
                    uint16_t dp2 = std::abs(imRef(depth, j, i) - imRef(depth, j-plusD, i-plusD));
                    uint16_t dm2 = std::abs(imRef(depth, j, i) - imRef(depth, j+plusD, i+plusD));
                    
                    cv::Mat_<float> p0,p1,p2;
                    projectPixel2CameraRF(Kinv, j, i, imRef(depth, j, i), p0);
                    projectPixel2CameraRF(Kinv, j-point3D, i-point3D, imRef(depth, j-point3D, i-point3D), p1);
                    projectPixel2CameraRF(Kinv, j+point3D, i+point3D, imRef(depth, j+point3D, i+point3D), p2);
                    float theta;//in rad
                    angle3Points(p1, p0, p2, theta);
                    
                    if( (/*(dp<DTH && dm<DTH) ||*/ (dp2<DTH && dm2<DTH))
                       &&  (theta>g_angle || theta<l_angle))
                    {
                        MagOutPtr[c]=0;
                    }
                }
                
            }
        }
        
        if(blurTheResult)
            cv::GaussianBlur(MagOut, MagOut, cv::Size(7,7), 0);
        
        
        
        cv::Mat SaliencyMyCanny;
        double minVal, maxVal;
        cv::minMaxIdx(MagOut, &minVal, &maxVal);
        MagOut.convertTo(SaliencyMyCanny,CV_32F,1.0/(maxVal - minVal), -minVal*1.0/(maxVal-minVal));
        
        cv::minMaxIdx(SaliencyMyCanny, &minVal, &maxVal);
        printf("SaliencyMyCanny Float min: %f; max: %f\n",minVal,maxVal);
        float* SaliencyMyCannyPtr = SaliencyMyCanny.ptr<float>(0);
        //copy Mat
        for (int i=0; i<RxC; ++i) {
            saliency[i] = SaliencyMyCannyPtr[i];
        }

        
        
        //    double minVal, maxVal;
        //    cv::Mat J1_gray_img;
        //    //oppositeAngle = oppositeAngle - 180;
        //    cv::minMaxLoc(oppositeAngleOut,&minVal,&maxVal);
        //    oppositeAngleOut.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
        //    printf("%s: min: %f, max: %f\n","Opposite Angle",minVal, maxVal);
        //    cv::imshow("BOW 0 deg", BOWOut);
        //    cv::imshow("Opposite Angle", J1_gray_img);
        //    cv::waitKey(0);
        
        
        
    }
    //use the member variable
    void myCannyWithOri2(float T_Low, float T_High,
                         const cv::Mat& mask,
                         bool blurTheResult = true);
    
    /* Specialization for hsv & depth image in [mm] & saliency if any*/
    //template <class T>
    image<rgb> *segment_image();
    inline void PostSegmFilter(universe* u)
    {
        //find only the possible objs
        cv::Mat_<uchar> gray = cv::Mat_<uchar>::zeros(mHeight, mWidth);
        
        std::vector<std::vector<cv::Point3i> > xyi;
        u->collectSets(xyi,mHeight,mWidth);
        
        printf("xyi->size() %lu\n",xyi.size());
        printf("disjoint sets %d\n",u->num_sets());
        
        /* Show Cluster by Cluster */
 /*
         cv::Mat rgb_img = cv::Mat::zeros(mHeight,mWidth,CV_8UC3);
         for(int idx_set=0; idx_set<xyi.size(); ++idx_set)
         {
         
         std::vector<cv::Point2d> eigen_vecs;
         std::vector<double> eigen_val;
         cv::Point cntr;
         double angle;
         double eccentricity;
         
         computePCA(xyi[idx_set], rgb_img, eigen_vecs, eigen_val, cntr, angle, eccentricity,true);
         
         cv::Mat cvHSV_ = cv::Mat(mInput_img->height(),mInput_img->width(),CV_8UC3,mInput_img->data);
         //compute the mask
         cv::Mat_<uchar> mask_ = cv::Mat_<uchar>::zeros(mInput_img->height(),mInput_img->width());
         uchar* mask_Ptr = mask_.ptr<uchar>(0);
         for(int ij=0;ij<xyi[idx_set].size();++ij)
         {
         mask_Ptr[ xyi[idx_set][ij].z ] = 255;
         }
         //debug
         imshow("maskDebug", mask_);
         
         cv::Mat_<float> Sat_HistNorm;
         cv::Mat histImage;
         computeSatHist(cvHSV_, true ,mask_, Sat_HistNorm,histImage,32,true );
         imshow("SathistImage", histImage);
         
         cv::Mat_<float> Hue_HistNorm;
         cv::Mat huehistImage;
         computeHueHist(cvHSV_, true, mask_, Hue_HistNorm, huehistImage,180,true);
         imshow("HuehistImage", huehistImage);
         
         cv::Mat_<float> Val_HistNorm;
         cv::Mat valhistImage;
         computeValueHist(cvHSV_, true, mask_, Val_HistNorm, valhistImage,xyi[idx_set].size(),32,true);
         imshow("ValuehistImage", valhistImage);
         
                 printf("V1: x: %.7f y: %.7f | L: %7f \n",eigen_vecs[0].x,eigen_vecs[0].y, eigen_val[0]);
                 printf("V2: x: %.7f y: %.7f | L: %7f \n",eigen_vecs[1].x,eigen_vecs[1].y, eigen_val[1]);
         
                 printf("Centroid x: %d , y: %d\n",cntr.x,cntr.y);
                 printf("angle: %f\n",angle);
                 printf("eccentricity: %f\n",eccentricity);
         
         imshow("PCA", rgb_img);
         cv::waitKey(0);
         }
   */
        
        /*FIlter Clusters through PCA*/

        cv::Mat rgb_img = cv::Mat::zeros(mHeight,mWidth,CV_8UC3);
        cv::Mat RealSegMat(mInput_img->height(),mInput_img->width(),CV_8UC3,cv::Scalar(0));
        //For each found cluster
        for(int idx_set=0; idx_set<xyi.size(); ++idx_set)
        {
            
            std::vector<cv::Point2d> eigen_vecs;
            std::vector<double> eigen_val;
            cv::Point cntr;
            double angle;
            double eccentricity;
            
            cv::Point3f Centroid3D = compute3DCentroid(xyi[idx_set]);
            
            //if Object is too far to be reached by the robot skip it (Here in camera frame in mm)
            float L2Centroid = std::sqrt(Centroid3D.z*Centroid3D.z+Centroid3D.y*Centroid3D.y+Centroid3D.x*Centroid3D.x);
            if(L2Centroid>mFarObjZ)
                continue;
            
            computePCA(xyi[idx_set], rgb_img, eigen_vecs, eigen_val, cntr, angle, eccentricity,false);
            
            //filter
            if(eccentricity>mMax_eccentricity || eigen_val[0]>mMax_L1
               || eigen_val[1]>mMax_L2)
                continue;
            
            //filter by Num NaN
            uint cluster_size = xyi[idx_set].size();
            //count the zeros[NaN] in the cluster
            uint NumNan = 0;
            for(int idxnan=0;idxnan<cluster_size;++idxnan)
            {
                if(mInpaintedDepth->data[ xyi[idx_set][idxnan].z ]==0)
                {
                    ++NumNan;
                }
            }
            //if >30% of points is NaN...drop the obj
            float NaNratio_ = ((float)NumNan)/((float)cluster_size);
            printf("NaNratio_: %f\n",NaNratio_);
            if(NaNratio_ >= 0.3f)
                continue;
            
            //Filter by Value (HSV)
            cv::Mat cvHSV_ = cv::Mat(mInput_img->height(),mInput_img->width(),CV_8UC3,mInput_img->data);
            //compute the mask
            cv::Mat_<uchar> mask_ = cv::Mat_<uchar>::zeros(mInput_img->height(),mInput_img->width());
            uchar* mask_Ptr = mask_.ptr<uchar>(0);
            for(int ij=0;ij<xyi[idx_set].size();++ij)
            {
                mask_Ptr[ xyi[idx_set][ij].z ] = 255;
            }
            cv::Mat_<float> Val_HistNorm;
            cv::Mat valhistImage;
            computeValueHist(cvHSV_, true, mask_, Val_HistNorm, valhistImage,xyi[idx_set].size(),32,true);
            
            float* Vptr = Val_HistNorm.ptr<float>(0);
            float sum3Bins = Vptr[0] + Vptr[1] + Vptr[2];
            printf("sum3Bins: %f\n",sum3Bins);
            //if 30% of cluster pixels fall within the first 3 bins (0-3*8) == (0-24) Value Intensity, then drop the obj
            if(sum3Bins>=0.3f)
                continue;
  
            printf("cluster_size: %u\n",cluster_size);
            printf("NumNan: %u\n",NumNan);
            
            //Draw the Objects Clusters
            uchar* imgPtr = rgb_img.ptr<uchar>(0);
            //Draw real RGB of the Segmented Objs
            
            uchar* RealSegMatPtr = RealSegMat.ptr<uchar>(0);
            
            //To Save the single cluster RGB
            cv::Mat_<cv::Vec3b> clusterRGBmat = cv::Mat_<cv::Vec3b>::zeros(RealSegMat.size());
            uchar* clusterRGBmatPtr = clusterRGBmat.ptr<uchar>(0);
            //To //Save the single cluster DEPTH
            cv::Mat_<uint16_t> clusterDepthmat = cv::Mat_<uint16_t>::zeros(RealSegMat.size());
            uint16_t* clusterDepthmatPtr = clusterDepthmat.ptr<uint16_t>(0);
            
            rgb rgb_ = random_rgb();
            for (int i = 0; i < xyi[idx_set].size(); ++i)
            {
                int idx = xyi[idx_set][i].z*3; //pixel_idx*3
                
                //PCA image result
                imgPtr[idx] = rgb_.b;
                imgPtr[idx+1] = rgb_.g;
                imgPtr[idx+2] = rgb_.r;
                
                //Real COlors
                RealSegMatPtr[idx] =
                imRef(mInputRGB_img,xyi[idx_set][i].x,xyi[idx_set][i].y).b; //b
                RealSegMatPtr[idx+1] =
                imRef(mInputRGB_img,xyi[idx_set][i].x,xyi[idx_set][i].y).g; //g
                RealSegMatPtr[idx+2] =
                imRef(mInputRGB_img,xyi[idx_set][i].x,xyi[idx_set][i].y).r; //r
                
                //Save the single cluster RGB
                clusterRGBmatPtr[idx] = RealSegMatPtr[idx];
                clusterRGBmatPtr[idx+1] = RealSegMatPtr[idx+1];
                clusterRGBmatPtr[idx+2] = RealSegMatPtr[idx+2];
                //Save the single cluster Depth
//                clusterDepthmatPtr[xyi[idx_set][i].z] =
//                imRef(mInpaintedDepth,xyi[idx_set][i].x,xyi[idx_set][i].y); //r
                clusterDepthmatPtr[xyi[idx_set][i].z] =
                mInpaintedDepth->data[xyi[idx_set][i].z];
                
            }
            
            cv::circle(rgb_img, cntr, 3, cv::Scalar(255, 0, 255), 2);
            cv::Point p1 = cntr + 0.02 * cv::Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
            cv::Point p2 = cntr - 0.02 * cv::Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
            drawAxis(rgb_img, cntr, p1, cv::Scalar(0, 255, 0), 1);
            drawAxis(rgb_img, cntr, p2, cv::Scalar(255, 255, 0), 3);
            
            
            cv::Point3f Centroid3DFake = computeFake3DCentroid(cntr);
            
            //copy the x-y pixel only for AABB
            std::vector<cv::Point2f> bbpp;
            for (int pixIdx=0; pixIdx<xyi[idx_set].size(); ++pixIdx) {
                
                cv::Point2f p2(xyi[idx_set][pixIdx].x,
                               xyi[idx_set][pixIdx].y);
                
                bbpp.push_back(p2);
                
            }
            cv::Rect rect_aabb_ = cv::boundingRect(bbpp);
            
            //Save the results
            SegResults segres_(Centroid3D, Centroid3DFake ,cntr, eigen_val[0], eigen_val[1], angle, xyi[idx_set].size(), xyi[idx_set], clusterRGBmat,clusterDepthmat, rect_aabb_);
            vecSegResults.push_back(segres_);
            
            //Print Info PCA
            printf("V1: x: %.7f y: %.7f | L: %7f \n",eigen_vecs[0].x,eigen_vecs[0].y, eigen_val[0]);
            printf("V2: x: %.7f y: %.7f | L: %7f \n",eigen_vecs[1].x,eigen_vecs[1].y, eigen_val[1]);
            
            printf("Centroid x: %d , y: %d\n",cntr.x,cntr.y);
            printf("Centroid3D x: %f , y: %f, z: %f\n",Centroid3D.x,Centroid3D.y,Centroid3D.z);
            
            
            printf("angle: %f\n",angle);
            printf("eccentricity: %f\n",eccentricity);
            
            //imshow("PCA", rgb_img);
            //cv::waitKey(0);
            
        }
        cv::imshow("PCA", rgb_img);
        cv::imshow("RealColorSeg", RealSegMat);
        //cv::waitKey(0);
        
            //cv::imwrite("/Users/giorgio/Documents/PCA.png", rgb_img);
        //cv::imwrite("/Users/giorgio/Documents/ColorCoffee.png", RealSegMat);
        //    cv::imwrite("/Users/giorgio/Documents/Polito/PhD/Slides/PresentazionePhDfineAnno/RealColorSeg.jpg", RealSegMat);
        
        
        
        //DEBUG
        /*
         for(int idx=0; idx<xyi.size(); ++idx)
         {
         for(int dx=0; dx<xyi[idx].size(); ++dx)
         {
         gray(xyi[idx][dx].y,xyi[idx][dx].x) = (uchar)idx*8;
         }
         }
         */
    }
    
    //template <class T>
    inline float diff(image<float> *r, image<float> *g, image<float> *b,
                             int x1, int y1, int x2, int y2, image<float> *depth, float* saliency) {

        
        
        //We return the edges weights between 0 and 1 !!
        
        //only RGB
        if(typeid(T)==typeid(rgb) && depth==0)
        {
            //printf("diff rgb\n");
            //TODO: RGB Alone is not normalized leaving the original code unchanged !!
            return sqrtf(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
                         square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
                         square(imRef(b, x1, y1)-imRef(b, x2, y2)));
        }
        //only CIELab
        if(typeid(T)==typeid(CIELab) && depth==0)
        {
            //printf("diff rgb\n");
            //TODO: CIELab Alone is not normalized leaving the original code unchanged !!
            return sqrtf(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
                         square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
                         square(imRef(b, x1, y1)-imRef(b, x2, y2)));
        }
        //RGB + DEPTH
        if(typeid(T)==typeid(rgb) && depth!=0 && saliency == 0)
        {
                    float Drgb =
                         sqrtf(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
                         square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
                         square(imRef(b, x1, y1)-imRef(b, x2, y2)));
            
//            float Wc = 1.f;
//            float Wd = 0.0f;
//            float Drgb =
//            sqrtf(square(Wc*(imRef(r, x1, y1)-imRef(r, x2, y2)) ) +
//                  square(Wc*(imRef(g, x1, y1)-imRef(g, x2, y2))) +
//                  square(Wc*(imRef(b, x1, y1)-imRef(b, x2, y2))) +
//                  square(Wd*(imRef(depth, x1, y1)-imRef(depth, x2, y2))));
//            Drgb /= std::sqrt(3*Wc*Wc+Wd*Wd); //sqrtf(3);
//            return Drgb;
            //R,G,B is given already normalized...
            // so need to normalize the error dividing by sqrt(3)
            Drgb /= 1.732050807568877f; //sqrtf(3);
            Drgb = std::pow(Drgb, 1.f/2.f);
            
            DrgbV.push_back(Drgb);
            
            //Get the delta Depth dDepth
            //TODO: How to Handle NaN (==0 mm) ??
            float Zpivot = imRef(depth, x1, y1);
            float Zcurrent = imRef(depth, x2, y2);
            
            //Depth is given normalized between [0,1] so DeltaDepth
            //is max == to 1
            float DeltaDepth = fabs(Zpivot - Zcurrent);
            if((Zpivot>1e-3f && Zcurrent<1e-3f) ||
               (Zpivot<1e-3f && Zcurrent>1e-3f))
            {
                DeltaDepth=0.1f;
            }
            else if(Zpivot<1e-3f && Zcurrent<1e-3f)
            {
                DeltaDepth=0.0f;
            }
            else
            {
                DeltaDepth = std::pow(DeltaDepth, 1.f/2.f);
            }
            DdepthV.push_back(DeltaDepth);
            //printf("%f, ",Drgb);
            
            //float Wscale=2.f;
            //        return Wscale*(dist_hsv/(static_cast<float>(sqrt(mKdv*mKdv + mKdc*mKdc))));
            
            //TODO: implement one of the crazy Fcn....
            //        float k_ = 1.0f;
            //        float weight = (sqrtf(DeltaDepth)*powf(dist_hsv, 1.0f+DeltaDepth) + k_*DeltaDepth*DeltaDepth)/(1.0f+k_);
            //        return Wscale*weight;
            
            /*        float weight = (mKy*std::log1p(Drgb) + DeltaDepth*std::pow(Drgb, (1.0f+DeltaDepth)) + mKx*DeltaDepth*DeltaDepth)/(1+mKy*M_LN2+mKx);
             */
            float weight = (mKy*Drgb*Drgb + DeltaDepth*std::pow(Drgb, (1.0f+DeltaDepth)) + mKx*DeltaDepth*DeltaDepth)/(1+mKy+mKx);
            return weight;
            
        }
        //RGB + DEPTH + SALIENCY
        if(typeid(T)==typeid(rgb) && depth!=0 && saliency!=0)
        {
            float Drgb =
            sqrtf(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
                  square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
                  square(imRef(b, x1, y1)-imRef(b, x2, y2)));
            //R,G,B is given already normalized...
            // so need to normalize the error dividing by sqrt(3)
            Drgb /= 1.732050807568877f; //sqrtf(3);
            //Drgb = std::pow(Drgb, 1.f/2.f);
            
            //printf("CIAO\n");
            
            //Get the delta Depth dDepth
            //TODO: How to Handle NaN (==0 mm) ??
            float Zpivot = imRef(depth, x1, y1);
            float Zcurrent = imRef(depth, x2, y2);
            
            //Depth is given normalized between [0,1] so DeltaDepth
            //is max == to 1
            float DeltaDepth = fabs(Zpivot - Zcurrent);
            //printf("%f, ",DeltaDepth);
            if( (Zpivot <= 1e-3f && Zcurrent > 1e-3f) ||
               (Zcurrent<= 1e-3f && Zpivot> 1e-3f) ) //float 0
            {
                //printf("\f, ",DeltaDepth);
                DeltaDepth=0.0f;
            }
            //DeltaDepth=0.0f;
            
            if(Zpivot>1e-3f && Zcurrent>1e-3f && DeltaDepth<=30.0f/1030.0f)//10mm/maxDepth[mm]
            {
                //printf("%f, ",DeltaDepth);
                //Drgb = 0.1f;
                float k_ = 1.0f;
                float weight = (/*mKy*std::log2(1.0f+Drgb) +
                                 mKs*std::log2(1.0f+ds) + */
                                /*DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) + */
                                DeltaDepth*std::pow(Drgb, (1.0f+DeltaDepth)) +
                                k_*std::log2(1.0f+DeltaDepth))/(1.0f+k_);
                
                DdepthV.push_back(DeltaDepth);
                DrgbV.push_back(Drgb);
                
                return weight;
            }
            
            
            //        else if(Zpivot<1e-3f && Zcurrent<1e-3f)
            //        {
            //            DeltaDepth=0.0f;
            //        }
            //        else
            //        {
            //            DeltaDepth = std::pow(DeltaDepth, 1.f/2.f);
            //        }
            DdepthV.push_back(DeltaDepth);
            DrgbV.push_back(Drgb);
            //printf("%f, ",Drgb);
            
            //float Wscale=2.f;
            //        return Wscale*(dist_hsv/(static_cast<float>(sqrt(mKdv*mKdv + mKdc*mKdc))));
            
            //TODO: implement one of the crazy Fcn....
            //        float k_ = 1.0f;
            //        float weight = (sqrtf(DeltaDepth)*powf(dist_hsv, 1.0f+DeltaDepth) + k_*DeltaDepth*DeltaDepth)/(1.0f+k_);
            //        return Wscale*weight;
            
            /*        float weight = (mKy*std::log1p(Drgb) + DeltaDepth*std::pow(Drgb, (1.0f+DeltaDepth)) + mKx*DeltaDepth*DeltaDepth)/(1+mKy*M_LN2+mKx);
             */
            
            float ps = saliency[y1*r->width()+ x1];
            float as = saliency[y2*r->width()+ x2];
            float ds = ps;//fabs(ps-as);
            
            
            //LOG2(1+x)
            float weight = (mKy*std::log2(1.0f+Drgb) +
                            /*mKs*std::log2(1.0f+ds) + */
                            /*DeltaDepth*std::pow(ds, (1.0f+DeltaDepth))*/ +
                            DeltaDepth*std::pow(Drgb, (1.0f+DeltaDepth)) +
                            mKx*std::log2(1.0f+DeltaDepth))/(1.0f/*2.0f*/+(/*mKs+*/mKy+mKx));
            
            return weight;
            
        }
        //HSV + DEPTH + SALIENCY
        else if(typeid(T)==typeid(hsv) && depth!=0 && saliency!=0)
        {
            //printf("diff hsv + depth + saliency\n");
            //exit(0);
            //r == h; g == s; b == v
            //x1,y1 = is the pivot pixel -> px,py
            //x2,y2 = is the neighboor pixel -> cx,cy
            //Arrays are 1 channel
            float h1 = imRef(r, x2, y2) * 2.0f; //to get Hue [0:360];
            float s1 = imRef(g, x2, y2) * c_mapVeS; //to get Sat [0:1];
            float v1 = imRef(b, x2, y2) * c_mapVeS; //to get Intensity [0:1];
            
            float h2 = imRef(r, x1, y1) * 2.0f; //to get Hue [0:360];
            float s2 = imRef(g, x1, y1) * c_mapVeS; //to get Sat [0:1];
            float v2 = imRef(b, x1, y1) * c_mapVeS; //to get Intensity [0:1];
            //Get the delta Depth dDepth
            //TODO: How to Handle NaN (==0 mm) ??
            float Zpivot = imRef(depth, x1, y1);
            float Zcurrent = imRef(depth, x2, y2);
            
            
            //        else if (Zpivot<=1e-3f && Zcurrent<=1e-3f)
            //        {
            //            //printf("both zeros: %f; %f\n",Zpivot,Zcurrent);
            //            DeltaDepth = 0.5f;
            //        }
            //        else
            //        {
            //            DeltaDepth = std::pow(DeltaDepth,5.0f); //con 1.8f stesso risultato d 1.3f
            //        }
            
            // compute difference between two colors in HSV space (new metric, intensity is separated)
            
            float delta_v = mKdv*fabs(v1 - v2);//4.5f
            float delta_h = fabs(h1 - h2);
            float theta = 0.0f;
            
            if(delta_h < 180)
                theta = delta_h;
            else
                theta = 360 - delta_h;
            
            float delta_c = mKdc*sqrt( s1*s1 + s2*s2 -
                                     2*s1*s2*cos(theta*deg2rad) );
            float dist_hsv = sqrt(delta_v*delta_v + delta_c*delta_c);//
            
            if( isnan(dist_hsv) || isinf(dist_hsv) )
                dist_hsv = 0.0f;
            
            if(v1 < 0.03f)
                dist_hsv = 0.01f;
            //Normalize dist_hsv between [0, 1]
            dist_hsv /= (sqrtf(mKdv*mKdv + mKdc*mKdc));
            
            //Normalize dist_hsv between [0, 255]
            //        dist_hsv = mapminmax(dist_hsv, 0.f, sqrtf(mKdv*mKdv + mKdc*mKdc), 0.0f, 255.f);
            
            
            //with saliency
            
            float ps = saliency[y1*r->width()+ x1];
            float as = saliency[y2*r->width()+ x2];
            float ds = ps;//fabs(ps-as);
            //normalize between [0-255];
            //ds *= 255.0f;
            //printf(" , %f",ps);
            
            //float Wscale=450.f;
            //        return Wscale*(dist_hsv/(static_cast<float>(sqrt(mKdv*mKdv + mKdc*mKdc))));
            
            
            //        return Wscale*weight;
            
            //without saliency
            /*
             float weight = (mKy*std::log1p(dist_hsv) + DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) + mKx*DeltaDepth*DeltaDepth)/(1.0f+mKy*M_LN2+mKx);
             return weight;
             */
            //        for(int ii=0;ii<640*480;++ii)
            //        {
            //            printf(", %f",saliency[ii]);
            //        }
            //        exit(0);
            
            
            
            
            //Depth is given normalized between [0,1] so DeltaDepth
            //is max == to 1
            float DeltaDepth = fabs(Zpivot - Zcurrent);
            
            //        if( (Zpivot <= 1e-3f && Zcurrent > 1e-3f) || (Zcurrent<=1e-3f && Zpivot>1e-3f) ) //float 0
            //        {
            //Only One of two pixels is NaN
            
            //            if(y2>=150 && y2<=400 && x2<560)
            //                DeltaDepth =5.0f/1030.0f; //*= DeltaDepth; //
            //            else
            //printf("%f,\n",DeltaDepth);
            //DeltaDepth = 0.0f;
            
            //                //LOG2(1+x)
            //                float weight = (mKy*std::log2(1.0f+dist_hsv) +
            //                                mKs*std::log2(1.0f+ds) +
            //                                DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
            //                                DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
            //                                mKx*std::log2(1.0f+DeltaDepth))/(2.0f+(mKs+mKy+mKx));
            //
            //                DrgbV.push_back(dist_hsv);
            //                DdepthV.push_back(DeltaDepth);
            //
            //                return weight;
            
            //dist_hsv = 0.0f;
            //ds=0.0f;
            //        }
            DrgbV.push_back(dist_hsv);
            DdepthV.push_back(DeltaDepth);
            //        else if(Zpivot>1e-3f && Zcurrent>1e-3f && DeltaDepth<10.0f/295.0f)//10mm/maxDepth[mm]
            //        {
            //            /* Case 1)   Z1 ~= Z2 != NaN ==> DeltaDepth~=0 */
            //
            //            float weight = (
            //            std::pow(DeltaDepth,1.f/1.f)*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
            //                    kc1*std::log2(1.0f+DeltaDepth))/(1.0f+kc1);
            //
            //            DrgbV.push_back(dist_hsv);
            //            DdepthV.push_back(DeltaDepth);
            //
            //            return weight;
            //        }
            //        else if(Zpivot<=1e-3f && Zcurrent<=1e-3f)//10mm/maxDepth[mm]
            //        {
            //            /* Case 2)  Z1 == Z2 == NaN ==> DeltaDepth == 0 */
            //            //LOG2(1+x)
            //            float weight = (mKy*std::log2(1.0f+dist_hsv) +
            //                            mKs*std::log2(1.0f+ds) +
            //                            DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
            //                            DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
            //                            mKx*std::log2(1.0f+DeltaDepth))/(2.0f+(mKs+mKy+mKx));
            //
            //            DrgbV.push_back(dist_hsv);
            //            DdepthV.push_back(DeltaDepth);
            //
            //            return weight;
            //        }
            //        else
            //        {
            //            //LOG2(1+x)
            //            float weight = (mKy*std::log2(1.0f+dist_hsv) +
            //                            mKs*std::log2(1.0f+ds) +
            //                            DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
            //                            DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
            //                            mKx*std::log2(1.0f+DeltaDepth))/(2.0f+(mKs+mKy+mKx));
            //
            //            DrgbV.push_back(dist_hsv);
            //            DdepthV.push_back(DeltaDepth);
            //
            //            return weight;
            //        }
            
            
            //mKy*LN(1+dhsv)
            /*        float weight = (mKy*std::log1p(dist_hsv) +
             mKs*std::log1p(ds) +
             DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
             DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
             mKx*DeltaDepth*DeltaDepth)/(2.0f+(mKs+mKy)*M_LN2 +mKx);
             */
            //LOG2();
            /*       float weight = (mKy*std::log2(1.0f+dist_hsv) +
             mKy*std::log2(-dist_hsv+2.0f) - mKy +
             mKs*std::log2(1.0f+ds) +
             DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
             DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
             mKx*DeltaDepth*DeltaDepth)/(2.0f+(mKs+mKy)+mKx - mKy);
             */
            /*  float weight =  std::log1p(DeltaDepth)*3.466649227809563E-1+std::log1p(dist_hsv)*3.466649227809563+std::log(-dist_hsv+2.56E2)*3.466649227809563+DeltaDepth*dist_hsv*3.466649227809563E-4+sqrtf(DeltaDepth)*3.466649227809563E-1-1.922318510597208E1;
             */
            //        float x = DeltaDepth;
            //        float y = dist_hsv;
            //
            //        float weight = 30.0f*(std::log1p(x) + mKx*sqrtf(x) + kxy*x*y + mKy*(std::log1p(y)+ std::log(255.0f-(y-1)) - std::log(256.0f)))/ (std::log(256.0f) + mKx*sqrtf(255.0f) + kxy*255.0f*255.0f);
            //
            
            //LOG(1+x)
            /*       float weight = (mKy*std::log1p(dist_hsv) +
             mKs*std::log1p(ds) +
             DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
             DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
             mKx*std::log1p(DeltaDepth))/(2.0f+(mKs+mKy+mKx)*M_LN2);
             */
            
            /*        //LOG2(1+x)
             float weight = (mKy*std::log2(1.0f+dist_hsv) +
             mKs*std::log2(1.0f+ds) +
             DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
             DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
             mKx*std::log2(1.0f+DeltaDepth))/(2.0f+(mKs+mKy+mKx));
             */
            //TODO: implement one of the crazy Fcn....
            
            //        float weight = (sqrtf(DeltaDepth)*powf(dist_hsv, 1.0f+DeltaDepth) + mKx*DeltaDepth*DeltaDepth + mKs*std::log2(1.0f+ds))/(1.0f+mKx+mKs);
            float weight = (/*mKy*std::log2(1.0f+dist_hsv)*/ + mKx*DeltaDepth*std::log2(1.0f+dist_hsv) + /*mKs*std::log2(1.0f+ds))*/mKs*ds)/(/*mKy+*/mKx+mKs);
            
            //added by STE
//             weight = (mKy*std::log1p(dist_hsv) +
//                            mKs*std::log1p(ds) +
//                            DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
//                            DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
//                            mKx*std::log1p(DeltaDepth))/(2.0f+(mKs+mKy+mKx)*M_LN2);
            
//            float weight = (mKy*dist_hsv*dist_hsv*dist_hsv*dist_hsv + mKx*DeltaDepth*std::log2(1.0f+dist_hsv) + /*mKs*std::log2(1.0f+ds))*/mKs*ds)/(mKy+mKx+mKs);
            
            
            /*        //LOG2(1+x) & Sigmoid
             float kt = 0.6f;
             float k = 10.f;
             float weight = (mKy*std::log2(1.0f+dist_hsv) +
             mKs*std::log2(1.0f+ds) +
             DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
             DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
             mKx*std::log2(1.0f+DeltaDepth))/(2.0f+(mKs+mKy+mKx))
             - 1.f/(1.f+std::exp(-(DeltaDepth-kt)*k)) + 1.f/(1.f+std::exp(-(0.f-kt)*k));
             
             if(weight<0.0f)
             weight = 0.0f;
             */
            //mKy*dHSV^2
            /*       float weight = (mKy*dist_hsv*dist_hsv +
             mKs*std::log1p(ds) +
             DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
             DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) +
             mKx*DeltaDepth*DeltaDepth)/(2.0f+mKs*M_LN2 +mKx+mKy);
             */
            //NO HSV
            /*     float weight = (
             mKs*std::log1p(ds) +
             DeltaDepth*std::pow(ds, (1.0f+DeltaDepth)) +
             mKx*DeltaDepth*DeltaDepth)/(1.0f+mKs*M_LN2 +mKx);
             */
            return weight;
            
        }
        //HSV + DEPTH
        else if(typeid(T)==typeid(hsv) && depth!=0 && saliency==0)
        {
            //        printf("diff hsv + depth\n");
            //        exit(0);
            //r == h; g == s; b == v
            //x1,y1 = is the pivot pixel -> px,py
            //x2,y2 = is the neighboor pixel -> cx,cy
            //Arrays are 1 channel
            float h1 = imRef(r, x2, y2) * 2.0f; //to get Hue [0:360];
            float s1 = imRef(g, x2, y2) * c_mapVeS; //to get Sat [0:1];
            float v1 = imRef(b, x2, y2) * c_mapVeS; //to get Intensity [0:1];
            
            float h2 = imRef(r, x1, y1) * 2.0f; //to get Hue [0:360];
            float s2 = imRef(g, x1, y1) * c_mapVeS; //to get Sat [0:1];
            float v2 = imRef(b, x1, y1) * c_mapVeS; //to get Intensity [0:1];
            //Get the delta Depth dDepth
            //TODO: How to Handle NaN (==0 mm) ??
            float Zpivot = imRef(depth, x1, y1);
            float Zcurrent = imRef(depth, x2, y2);
            
            //Depth is given normalized between [0,1] so DeltaDepth
            //is max == to 1
            float DeltaDepth = fabs(Zpivot - Zcurrent);
            
            // compute difference between two colors in HSV space (new metric, intensity is separated)
            
            float delta_v = mKdv*fabs(v1 - v2);//4.5f
            float delta_h = fabs(h1 - h2);
            float theta = 0.0f;
            
            if(delta_h < 180)
                theta = delta_h;
            else
                theta = 360 - delta_h;
            
            float delta_c = mKdc*sqrt( s1*s1 + s2*s2 -
                                     2*s1*s2*cos(theta*deg2rad) );
            float dist_hsv = sqrt(delta_v*delta_v + delta_c*delta_c);//
            
            if( isnan(dist_hsv) || isinf(dist_hsv) )
                dist_hsv = 0.0f;
            
            if(v1 < 0.03f)
                dist_hsv = 0.01f;
            //Normalize dist_hsv between [0, 1]
            dist_hsv /= (sqrtf(mKdv*mKdv + mKdc*mKdc));
            
            
            //float Wscale=450.f;
            //        return Wscale*(dist_hsv/(static_cast<float>(sqrt(mKdv*mKdv + mKdc*mKdc))));
            
            //TODO: implement one of the crazy Fcn....
            //        float k_ = 1.0f;
            //        float weight = (sqrtf(DeltaDepth)*powf(dist_hsv, 1.0f+DeltaDepth) + k_*DeltaDepth*DeltaDepth)/(1.0f+k_);
            //        return Wscale*weight;
            
            //with out saliency
            float weight = (mKy*std::log1p(dist_hsv) + DeltaDepth*std::pow(dist_hsv, (1.0f+DeltaDepth)) + mKx*DeltaDepth*DeltaDepth)/(1.0f+mKy*M_LN2+mKx);
            return weight;
            
        }
        //Only HSV
        else if(typeid(T)==typeid(hsv) && depth==0)
        {
            //printf("diff hsv\n");
            //r == h; g == s; b == v
            //x1,y1 = is the pivot pixel -> px,py
            //x2,y2 = is the neighboor pixel -> cx,cy
            //Arrays are 1 channel
            float h1 = imRef(r, x2, y2) * 2.0f; //to get Hue [0:360];
            float s1 = imRef(g, x2, y2) * c_mapVeS; //to get Sat [0:1];
            float v1 = imRef(b, x2, y2) * c_mapVeS; //to get Intensity [0:1];
            
            float h2 = imRef(r, x1, y1) * 2.0f; //to get Hue [0:360];
            float s2 = imRef(g, x1, y1) * c_mapVeS; //to get Sat [0:1];
            float v2 = imRef(b, x1, y1) * c_mapVeS; //to get Intensity [0:1];
            
            // compute difference between two colors in HSV space (new metric, intensity is separated)
            float delta_v = mKdv*fabs(v1 - v2);//4.5f
            float delta_h = fabs(h1 - h2);
            float theta = 0.0f;
            
            if(delta_h < 180)
                theta = delta_h;
            else
                theta = 360 - delta_h;
            
            float delta_c = mKdc*sqrtf( s1*s1 + s2*s2 -
                                      2*s1*s2*cos(theta*deg2rad) );
            float dist_hsv = sqrtf(delta_v*delta_v + delta_c*delta_c);//
            
            if( isnan(dist_hsv) || isinf(dist_hsv) )
                dist_hsv = 0.0f;
            
            if(v1 < 0.03f)
                dist_hsv = 0.01f;
            //normalize between [0, 1] and return
            //float Wscale=450.f;
            return (dist_hsv/(static_cast<float>(sqrtf(mKdv*mKdv + mKdc*mKdc))));
            
        }
        else
        {
            printf("ERROR Unknown data type...\n");
            return 0.f;
        }
    }
    
    
    static inline void visualizeColorMap(const cv::Mat& oneCHimg, std::string name="W1", int wait_key_=0, bool getgrayscale=false)
    {
        //if already 8 bit and we want to show only the gray image
        if(oneCHimg.depth()==CV_8U && getgrayscale)
        {
            double minVal, maxVal;
            cv::minMaxLoc(oneCHimg,&minVal,&maxVal);
            cv::imshow(name, oneCHimg);
            printf("%s: min: %f, max: %f\n",name.c_str(),minVal, maxVal);
            cv::waitKey(wait_key_);
            
        }
        //if already 8 bit and we want to show gray image in ColorMap
        else if(oneCHimg.depth()==CV_8U && !getgrayscale)
        {
            double minVal, maxVal;
            cv::minMaxLoc(oneCHimg,&minVal,&maxVal);
            
            cv::Mat colorMatJ1;
            cv::applyColorMap(oneCHimg, colorMatJ1, cv::COLORMAP_JET);
            
            cv::imshow(name, colorMatJ1);
            printf("%s: min: %f, max: %f\n",name.c_str(),minVal, maxVal);
            cv::waitKey(wait_key_);
        }
        //if other formats and we want to show gray image in ColorMap
        else if(oneCHimg.depth()!=CV_8U && !getgrayscale)
        {
            
            double minVal, maxVal;
            cv::Mat J1_gray_img;
            cv::minMaxLoc(oneCHimg,&minVal,&maxVal);
            oneCHimg.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
            cv::Mat colorMatJ1;
            cv::applyColorMap(J1_gray_img, colorMatJ1, cv::COLORMAP_JET);
            //std::string name(winname+std::string(" J1"));
            cv::imshow(name, colorMatJ1);
            printf("%s: min: %f, max: %f\n",name.c_str(),minVal, maxVal);
            cv::waitKey(wait_key_);
        }
        //if other formats and we want to show only gray image
        else if(oneCHimg.depth()!=CV_8U && getgrayscale)
        {
            
            double minVal, maxVal;
            cv::Mat J1_gray_img;
            cv::minMaxLoc(oneCHimg,&minVal,&maxVal);
            oneCHimg.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
            cv::imshow(name, J1_gray_img);
            printf("%s: min: %f, max: %f\n",name.c_str(),minVal, maxVal);
            cv::waitKey(wait_key_);
        }
        
        
    }
    
    static inline void convertDepth2ColorMap(const cv::Mat& oneCHimg, cv::Mat& dpCM)
    {
        //if already 8 bit
        if(oneCHimg.depth()==CV_8U)
        {
            double minVal, maxVal;
            cv::minMaxLoc(oneCHimg,&minVal,&maxVal);
            cv::applyColorMap(oneCHimg, dpCM, cv::COLORMAP_JET);
//            printf("%s: min: %f, max: %f\n",name.c_str(),minVal, maxVal);
        }
        else// if(oneCHimg.depth()!=CV_8U)
        {
            
            double minVal, maxVal;
            cv::Mat J1_gray_img;
            cv::minMaxLoc(oneCHimg,&minVal,&maxVal);
            oneCHimg.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
            cv::applyColorMap(J1_gray_img, dpCM, cv::COLORMAP_JET);
//            printf("%s: min: %f, max: %f\n",name.c_str(),minVal, maxVal);
        }


    }
    
    universe *segment_graph(int num_vertices, int num_edges, edge *edges);
    
    static inline void angle3Points(const cv::Mat_<float>& p1, const cv::Mat_<float>& p0, const cv::Mat_<float>& p2, float& theta)
    {
        //these vectors are wrt Camera RF (Z pointing out, X to the right, Y down)
        cv::Mat_<float> p01 = p1-p0;//p0 - p1;
        cv::Mat_<float> p02 = p2-p0;//p0 - p2;
        
        float absp01 = (float)cv::norm(p01,cv::NORM_L2);
        float absp02 = (float)cv::norm(p02,cv::NORM_L2);
        
        float costheta = (float)p01.dot(p02)/(absp01*absp02);
        
        cv::Mat_<float> cross_ = p01.cross(p02);
        //printf("%f, ",cross_(0));
        
        float abscross_ = (float)cv::norm(cross_,cv::NORM_L2);
        
        float sintheta = abscross_/(absp01*absp02);
        
        theta = std::atan2(sintheta,costheta);//in [rad]
        
        //check the Z unit vector of the cross product vector (wrt Camera frame hence we check the sign of X component)
        if(cross_(0)<0)//X component (Z unit vector)
            theta *= -1.0f;
        ////
        ////    theta = M_PI;
        //    printf("%f, ",theta*180.f/M_PI);
        
    }
    
    static inline void projectPixel2CameraRF(const cv::Mat& Kinv, int u , int v, uint16_t z, cv::Mat_<float>& XYZ)
    {
        cv::Mat_<float> Pixel(3,1);
        Pixel(0) = static_cast<float>(u*z);
        Pixel(1) = static_cast<float>(v*z);
        Pixel(2) = static_cast<float>(z);
        XYZ = Kinv*Pixel;
        
    }
    static inline void projectPixel2CameraRF(const cv::Mat& Kinv, int u , int v, float z, cv::Mat_<float>& XYZ)
    {
        cv::Mat_<float> Pixel(3,1);
        Pixel(0) = static_cast<float>(u)*z;
        Pixel(1) = static_cast<float>(v)*z;
        Pixel(2) = z;
        XYZ = Kinv*Pixel;
        
    }
    inline void projectPixel2CameraRF(int u , int v, uint16_t z, cv::Mat_<float>& XYZ)
    {
        cv::Mat_<float> Pixel(3,1);
        Pixel(0) = static_cast<float>(u*z);
        Pixel(1) = static_cast<float>(v*z);
        Pixel(2) = static_cast<float>(z);
        XYZ = K_inv*Pixel;
        //        //I used the traspose to get a point cloud of the type Nx3 where N is th enumber of points
        //        XYZ = Pixel*K_invt;
    }
    
    inline void getPointCloud(const cv::Mat_<uint16_t>& depth_img, cv::Mat_<float>& PC)
    {
        const uint16_t* depth_imgPtr = depth_img.ptr<uint16_t>(0);
        cv::Mat_<float> Pixel(1,3);
        float* PixelPtr = Pixel.ptr<float>(0);
        
        if(PC.empty())
        {
            PC = cv::Mat_<float>(0,3); //Nx3
        }
        
        for (int u=0; u<depth_img.cols; ++u) {
            for (int v=0; v<depth_img.rows; ++v) {
                
                uint16_t zz = *(depth_imgPtr + v*depth_img.cols + u );
                
                if(zz==0)
                    continue;
                
                PixelPtr[0] = static_cast<float>(u*zz);
                PixelPtr[1] = static_cast<float>(v*zz);
                PixelPtr[2] = static_cast<float>(zz);
                cv::Mat_<float> XYZ = Pixel*K_invt;
                PC.push_back(XYZ);
            }
        }
    }


    
    inline cv::Point3f compute3DCentroid(std::vector<cv::Point3i> pxs)
    {
        cv::Point3f centroid3D(0.f,0.f,0.f);
        size_t num_points=0;
        for(int i=0;i<pxs.size();++i)
        {
            //NAN
            if(imRef(mInpaintedDepth, pxs[i].x, pxs[i].y)==0)
                continue;
            
            cv::Mat_<float> XYZ;
            projectPixel2CameraRF(K_inv,
                                  pxs[i].x,
                                  pxs[i].y,
                                  imRef(mInpaintedDepth, pxs[i].x, pxs[i].y),
                                  XYZ);
            centroid3D.x += XYZ(0);
            centroid3D.y += XYZ(1);
            centroid3D.z += XYZ(2);
            
            ++num_points;
            
        }
        centroid3D.x /= (float)num_points;
        centroid3D.y /= (float)num_points;
        centroid3D.z /= (float)num_points;
        
        return centroid3D;
        
    }
    
    inline cv::Point3f computeFake3DCentroid(const cv::Point& cntr)
    {
        /*Fake 3D Centroid because it takes the 2d cluster centroid and generate te corresponding 3d point taking as depth value the one under the 2d centroid coordinates */
        cv::Mat_<float> XYZ;
        projectPixel2CameraRF(K_inv,
                              cntr.x,
                              cntr.y,
                              imRef(mInpaintedDepth, cntr.x, cntr.y),
                              XYZ);
        
        cv::Point3f c3d(XYZ(0),XYZ(1),XYZ(2));
        
        return c3d;
       
    }
 
    void computePCA(const std::vector<cv::Point3i>& pts,
                    cv::Mat &img,//BGR
                    std::vector<cv::Point2d>& eigen_vecs,
                    std::vector<double>& eigen_val,
                    cv::Point& cntr, double& angle,
                    double& eccentricity,bool viz_=true);
    
    void inline drawAxis(cv::Mat& img, cv::Point p, cv::Point q, cv::Scalar colour, const float scale = 0.2)
    {
        double angle;
        double hypotenuse;
        angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
        hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
        //    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
        //    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
        // Here we lengthen the arrow by a factor of scale
        q.x = (int) (p.x - scale * hypotenuse * cos(angle));
        q.y = (int) (p.y - scale * hypotenuse * sin(angle));
        cv::line(img, p, q, colour, 1, CV_AA);
        // create the arrow hooks
        p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
        p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
        cv::line(img, p, q, colour, 1, CV_AA);
        p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
        p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
        cv::line(img, p, q, colour, 1, CV_AA);
    }
    
    void inline computeValueHist(const cv::Mat& image, bool isHSV, const cv::Mat& mask, cv::Mat_<float>& Val_HistNorm, cv::Mat& histImage,int clusterArea,int Hbins=256, bool viz_=true )
    {
        /*****COMPUTE Value HISTOGRAM*******/
        cv::Mat hsv_mat;
        if(!isHSV)
        {
            //Conversione RBG -> HSV
            cv::cvtColor(image, hsv_mat, CV_BGR2HSV);
        }
        else
        {
            image.copyTo(hsv_mat);
        }
        std::vector<cv::Mat> hsv_planes;
        cv::split( hsv_mat, hsv_planes );
        
        //discretizzai valori di Sat a 30 livelli (Bins)
        int histSize = Hbins; //per Histogramma 1D
        // Hue varia da 0 a 179 compreso
        float sranges[] = { 0, 256 };
        const float* histRange = { sranges };
        cv::Mat_<float> Value_Hist;
        //          	cv::imshow("mask",mask);
        //          	cv::waitKey();
        //Compute Hist only on the mask
        cv::calcHist( &hsv_planes[2], 1, 0, mask, Value_Hist, 1, &histSize, &histRange, true, false );
        //normalizza da 0 1
        
        //cv::normalize(Value_Hist, Val_HistNorm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F, cv::Mat() );
        
        
        //Normalize such as each bins represent the probability of occurence of Value level (rk) in an image.
        
        Val_HistNorm = Value_Hist/(static_cast<float>(clusterArea));
        
        
        double maxVal=0;
        cv::minMaxLoc(Val_HistNorm, 0, &maxVal, 0, 0);
        
        
        //debug Val_HistNorm sums up to one!!
        /*
         float sum=0;
         for(int r=0;r<Val_HistNorm.rows;++r)
         {
         for(int c=0;c<Val_HistNorm.cols;++c)
         
         sum+=Val_HistNorm(r,c);
         }
         
         std::cout<<Val_HistNorm<<"\n";
         printf("SumUp: %f\n",sum);
         */
        
        /* Visualize the Hist : Line Hist */
        
        if(viz_)
        {
            int hist_w = 256; int hist_h = 400;
            int bin_w = cvRound( (double) hist_w/histSize );
            
            //cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
            histImage = cv::Mat::zeros(hist_h, hist_w, CV_8UC3);
            for( int i = 1; i < histSize; i++ )
            {
                cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(Val_HistNorm.at<float>(i-1)*700.f) ) ,
                         cv::Point( bin_w*(i), hist_h - cvRound(Val_HistNorm.at<float>(i)*700.f) ),
                         cv::Scalar( 255, 0, 0), 2, 8, 0  );
            }
        }
        
        
        /* Visualize the Hist : Bins Hist */
        /*
         if(viz_)
         {
         int hist_w = 256; int hist_h = 256;
         int bin_w = cvRound( (double) hist_w/histSize );
         
         //cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
         
         histImage = cv::Mat::zeros(hist_h, hist_w, CV_8UC3);
         for( int s = 1; s < histSize; ++s)
         {
         float binVal = Val_HistNorm.at<float>(s, 0);
         int intensity = cvRound(binVal*(hist_h-1)/float(maxVal));
         
         rectangle( histImage, cv::Point(s*bin_w, histImage.rows),
         cv::Point( (s+1)*bin_w - 1, histImage.rows - intensity),
         cv::Scalar::all(255),
         CV_FILLED );
         }
         }
         */
        
        
        
        /*****END COMPUTE VALUE HISTOGRAM*******/
    }
    
    void inline computeSatHist(const cv::Mat& image, bool isHSV, const cv::Mat& mask, cv::Mat_<float>& Sat_HistNorm, cv::Mat& histImage,int Hbins=256, bool viz_=true )
    {
        /*****COMPUTE SAT HISTOGRAM*******/
        cv::Mat hsv_mat;
        if(!isHSV)
        {
            //Conversione RBG -> HS
            cv::cvtColor(image, hsv_mat, CV_BGR2HSV);
        }
        else
        {
            image.copyTo(hsv_mat);
        }
        std::vector<cv::Mat> hsv_planes;
        cv::split( hsv_mat, hsv_planes );
        
        //discretizzai valori di Sat a 30 livelli (Bins)
        int histSize = Hbins; //per Histogramma 1D
        // Hue varia da 0 a 179 compreso
        float sranges[] = { 0, 256 };
        const float* histRange = { sranges };
        cv::Mat Sat_Hist;
        //          	cv::imshow("mask",mask);
        //          	cv::waitKey();
        //Compute Hist only on the mask
        cv::calcHist( &hsv_planes[1], 1, 0, mask, Sat_Hist, 1, &histSize, &histRange, true, false );
        //normalizza da 0 1
        //cv::Mat Sat_HistNorm;
        cv::normalize(Sat_Hist, Sat_HistNorm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F, cv::Mat() );
        
        
        /* Visualize the Hist */
        if(viz_)
        {
            int hist_w = 256; int hist_h = 400;
            int bin_w = cvRound( (double) hist_w/histSize );
            
            //cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
            histImage = cv::Mat::zeros(hist_h, hist_w, CV_8UC3);
            for( int i = 1; i < histSize; i++ )
            {
                cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(Sat_HistNorm.at<float>(i-1)*(float)hist_h) ) ,
                         cv::Point( bin_w*(i), hist_h - cvRound(Sat_HistNorm.at<float>(i)*(float)hist_h) ),
                         cv::Scalar( 255, 0, 0), 2, 8, 0  );
            }
        }
        
        
        /*****END COMPUTE SAT HISTOGRAM*******/
    }
    
    void inline computeHueHist(const cv::Mat& image, bool isHSV ,const cv::Mat& mask, cv::Mat_<float>& Hue_HistNorm, cv::Mat& histImage,int Hbins=180, bool viz_=true )
    {
        /*****COMPUTE HUE HISTOGRAM*******/
        //Conversione RBG -> HSV
        cv::Mat hsv_mat;
        if(!isHSV)
        {
            cv::cvtColor(image, hsv_mat, CV_BGR2HSV);
        }
        else
        {
            image.copyTo(hsv_mat);
        }
        
        std::vector<cv::Mat> hsv_planes;
        cv::split( hsv_mat, hsv_planes );
        
        //discretizzai valori di Hue a 30 livelli (Bins)
        //int Hbins = 180;
        int histSize = Hbins; //per Histogramma 1D
        // Hue varia da 0 a 179 compreso
        float hranges[] = { 0, 180 };
        const float* histRange = { hranges };
        cv::Mat Hue_Hist;
        //          	cv::imshow("mask",mask);
        //          	cv::waitKey();
        //Compute Hist only on the mask
        cv::calcHist( &hsv_planes[0], 1, 0, mask, Hue_Hist, 1, &histSize, &histRange, true, false );
        //normalizza da 0 1
        //cv::Mat Hue_HistNorm;
        cv::normalize(Hue_Hist, Hue_HistNorm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F, cv::Mat() );
        
        
        /* Visualize the Hist */
        if(viz_)
        {
            int hist_w = Hbins*2; int hist_h = 400;
            int bin_w = cvRound( (double) hist_w/histSize );
            
            //cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
            histImage = cv::Mat::zeros(hist_h, hist_w, CV_8UC3);
            for( int i = 1; i < histSize; i++ )
            {
                cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(Hue_HistNorm.at<float>(i-1)*(float)hist_h) ) ,
                         cv::Point( bin_w*(i), hist_h - cvRound(Hue_HistNorm.at<float>(i)*(float)hist_h) ),
                         cv::Scalar( 255, 0, 0), 2, 8, 0  );
            }
        }
        
        
        /*****END COMPUTE HUE HISTOGRAM*******/
    }

    // random color
    inline rgb random_rgb(){
        rgb c;
        //double r;
        
        //check if random color == white
        //we leave the white color only for clusters < min_size
        //i.e. the one we throw away.
        do
        {
            c.r = (uchar)random();
            c.g = (uchar)random();
            c.b = (uchar)random();
            
        }while(c==rgbWHITE);
        
        return c;
    }
    
    template <class R>
    static inline void save1chMat2MatlabCSV(const cv::Mat_<R>& mat, std::string path="/Users/giorgio/Documents/MATLAB/mypc.csv")
    {
        //save Depth mm for matlab in csv format
        std::ofstream ofs (path, std::ofstream::out);
        for (int r=0; r<mat.rows; ++r) {
            for (int c=0; c<mat.cols; ++c)
            {
                ofs<<mat(r,c);
                if (c < mat.cols-1) {
                    ofs<<", ";
                }
            }
            ofs<<"\n";
        }
        ofs.close();
    }
    
    template <class R>
    static inline void save1chMat2HeaderVector(
        const cv::Mat_<R>& mat,
        const cv::Point3f& centroid3D_,
        std::string path="/Users/giorgio/Documents/MATLAB/mypc.h",
        bool row_major = true)
    {
        std::ofstream ofs (path, std::ofstream::out);
        int rXc = mat.rows*mat.cols;
        
        ofs<<"#ifndef MATHEADER_H \n"<<"#define MATHEADER_H \n\n\n";
        ofs<<"//in meters \n";
        ofs<<"const float segCentroidX = "<<(centroid3D_.x*0.001f)<<"; \n";
        ofs<<"const float segCentroidY = "<<(centroid3D_.y*0.001f)<<"; \n";
        ofs<<"const float segCentroidZ = "<<(centroid3D_.z*0.001f)<<"; \n\n";
        if(row_major)
        {
            //save Depth mm in .h header as 1D array in row major order
            
            const R* matPrt = mat.template ptr<R>(0);
            
            ofs<<"//in millimeters \n";
            ofs<<"uint16_t savedmatMM[ "<< rXc << "] = { ";
            
            for (int r=0; r<rXc; ++r) {
                
               ofs<< matPrt[r];
                if(r==rXc-1)
                    break;
               ofs<<" , ";
                
                
            }
            
            ofs<<" }; \n\n";
            
            
            //the float version in meters
            ofs<<"float savedmatMeters[ "<< rXc << "] = { ";
            for (int r=0; r<rXc; ++r) {
                
                ofs << (float)matPrt[r]*0.001f;
                if(r==rXc-1)
                    break;
                ofs<<" , ";
                
                
            }
            
            ofs<<" }; \n\n";
            
            ofs<<"#endif \n";
        }//end if row major
        
        ofs.close();
    }//end fcn

    
    void run();
    


};//end GraphCannySeg Class


}//end namespace








#endif
