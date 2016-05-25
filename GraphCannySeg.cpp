//
//  GraphCannySeg.cpp
//  PSOQuatGraphSeg
//
//  Created by Giorgio on 13/11/15.
//  Copyright (c) 2015 Giorgio. All rights reserved.
//

//#include <stdio.h>
#include "GraphCannySeg.h"

namespace GraphCanny {
   
    
    edge::edge()
    {
        this->a = 0;
        this->b = 0;
        this->w = 0.f;
    }
    edge::edge(int a_, int b_, float w_)
    {
        this->a = a_;
        this->b = b_;
        this->w = w_;
    }
    bool operator<(const edge &a, const edge &b)
    {
        return a.w < b.w;
    }
    
    rgb::rgb()
    {
        r=0;
        g=0;
        b=0;
    }
    rgb::rgb(uchar r_,uchar g_,uchar b_)
    {
        r=r_;
        g=g_;
        b=b_;
    }
    rgb& rgb::operator=(const rgb& other)
    {
        if (this != &other) {
            // Deep copy code in here
            this->r = other.r;
            this->g = other.g;
            this->b = other.b;
        }
        return *this;
    }
    
    hsv::hsv()
    {
        h=0;
        s=0;
        v=0;
    }
    hsv::hsv(uchar h_,uchar s_,uchar v_)
    {
        h=h_;
        s=s_;
        v=v_;
    }
    hsv& hsv::operator=(const hsv& other)
    {
        if (this != &other) {
            // Deep copy code in here
            this->h = other.h;
            this->s = other.s;
            this->v = other.v;
        }
        return *this;
    }
    
    CIELab::CIELab()
    {
        L=0;
        a=0;
        b=0;
    }
    CIELab::CIELab(float L_,float a_,float b_)
    {
        L=L_;
        a=a_;
        b=b_;
    }
    CIELab& CIELab::operator=(const CIELab& other)
    {
        if (this != &other) {
            // Deep copy code in here
            this->L = other.L;
            this->a = other.a;
            this->b = other.b;
        }
        return *this;
    }
    
    //IMAGE
    template <class T>
    image<T>::image(const int width, const int height, const bool init) {
        w = width;
        h = height;
        wXh = w * h;
        data = new T[w * h];  // allocate space for image data
        access = new T*[h];   // allocate space for row pointers
        
        // initialize row pointers
        for (int i = 0; i < h; i++)
            access[i] = data + (i * w);
        
        if (init)
            memset(data, 0, w * h * sizeof(T));
    }
    
    template <class T>
    image<T>::~image() {
        delete [] data;
        delete [] access;
    }
    
    template <class T>
    void image<T>::init(const T &val) {
        T *ptr = imPtr(this, 0, 0);
        T *end = imPtr(this, w-1, h-1);
        while (ptr <= end)
            *ptr++ = val;
    }
    
    template <class T>
    image<T> *image<T>::copy() const {
        image<T> *im = new image<T>(w, h, false);
        memcpy(im->data, data, w * h * sizeof(T));
        return im;
    }
    
    template <class T>
    image<T> *image<T>::copy(const cv::Mat& m) const
    {
        int  width = m.cols;
        int  height = m.rows;
        
        if(m.channels()==3 && typeid(T)==typeid(rgb))//RGB
        {
            //load the RGB image
            //printf("Opencv RGB Image\n");
            cv::Mat m_rgb;
            cv::cvtColor(m, m_rgb, CV_BGR2RGB);
            uchar* m_rgbPtr = m_rgb.ptr();
            image<T> *im = new image<T>(width, height);
            memcpy(im->data, m_rgbPtr, width * height * sizeof(T));
            return im;
        }
        else if(m.channels()==3 && typeid(T)==typeid(hsv))
        {
            //Load the HSV image
            //printf("Opencv HSV Image\n");
            cv::Mat m_hsv;
            cv::cvtColor(m, m_hsv, CV_BGR2HSV);
            uchar* m_hsvPtr = m_hsv.ptr();
            image<T> *im = new image<T>(width, height);
            memcpy(im->data, m_hsvPtr, width * height * sizeof(T));
            return im;
            
        }
        else if (m.channels()==1)
        {
            //printf("Opencv 1 CH Image\n");
            T* m_Ptr = m.ptr<T>(0);
            image<T> *im = new image<T>(width, height);
            memcpy(im->data, m_Ptr, width * height * sizeof(T));
            
            return im;
        }
        else
        {
            printf("ERROR unknown File Format\n");
            exit(EXIT_FAILURE);
        }
    }
    
    template <class T>
    void image<T>::init(const cv::Mat& m)
    {
        int  width = m.cols;
        int  height = m.rows;
        
        if(m.channels()==3 && typeid(T)==typeid(rgb))//RGB
        {
            //load the RGB image
            //printf("Opencv RGB Image\n");
            cv::Mat m_rgb;
            cv::cvtColor(m, m_rgb, CV_BGR2RGB);
            uchar* m_rgbPtr = m_rgb.ptr();
            //image<T> *im = new image<T>(width, height);
            memcpy(this->data, m_rgbPtr, width * height * sizeof(T));
            //return im;
        }
        else if(m.channels()==3 && typeid(T)==typeid(hsv))
        {
            //Load the HSV image
            //printf("Opencv HSV Image\n");
            cv::Mat m_hsv;
            cv::cvtColor(m, m_hsv, CV_BGR2HSV);
            uchar* m_hsvPtr = m_hsv.ptr();
            //image<T> *im = new image<T>(width, height);
            memcpy(this->data, m_hsvPtr, width * height * sizeof(T));
            //return im;
            
        }
        else if (m.channels()==1)
        {
            //printf("Opencv 1 CH Image\n");
            //pointer to const
            const T* m_Ptr = m.ptr<const T>(0);
            //image<T> *im = new image<T>(width, height);
            memcpy(this->data, m_Ptr, width * height * sizeof(T));
            
            //return im;
        }
        else
        {
            printf("ERROR unknown File Format\n");
            exit(EXIT_FAILURE);
        }
    }
    
    //UNIVERSE
    universe::universe(int elements) {
        elts = new uni_elt[elements];
        num = elements;
        for (int i = 0; i < elements; i++) {
            elts[i].rank = 0;
            elts[i].size = 1;
            elts[i].p = i;
            elts[i].link = i;
            elts[i].printed = false;
        }
    }
    
    universe::~universe() {
        delete [] elts;
    }
    
    int universe::find(int x) {
        int y = x;
        while (y != elts[y].p)
            y = elts[y].p;
        elts[x].p = y;
        return y;
    }
    
    void universe::join(int x, int y) {
        
        //update the circular list first
        int temp = elts[y].link;     //Concatenation
        elts[y].link = elts[x].link; // of the two
        elts[x].link = temp;        // circular lists
        
        if (elts[x].rank > elts[y].rank) {
            elts[y].p = x;
            elts[x].size += elts[y].size;
        } else {
            elts[x].p = y;
            elts[y].size += elts[x].size;
            if (elts[x].rank == elts[y].rank)
                elts[y].rank++;
        }
        num--;
    }
    
    bool universe::printSet(int x, cv::Mat_<uchar>& gray) {
        
        
        if(elts[x].printed)
        {
            //Already printed
            //printf("X= %d Already Printed\n",x);
            return false;
        }
        uchar* pgray = gray.ptr<uchar>(0);
        
        //PRINT(x)
        *(pgray+x) = 255;
        elts[x].printed = true;
        int real_x = x;
        
        int root = elts[x].p;
        //PRINT(root)
        *(pgray+root) = 255;
        elts[root].printed = true;
        
        while( elts[x].link != real_x )
        {
            x = elts[x].link;
            elts[x].printed = true;
            //PRINT(x)
            *(pgray+x) = 255;
        }
        return true;
    }
    
    void universe::collectSets(std::vector<std::vector<cv::Point3i> >& xyi, int H, int W)
    {
        //collect all the disjoint set in a vector where each element is a vector(set ) of Point3i [x=x-coord; y=y-coord; z=pixelIdx==y*W+x]
        
        int im_WxH = H*W;
        
        
        for(int idx=0; idx<im_WxH; ++idx)
        {
            std::vector<cv::Point3i> pc3;
            int x = idx;
            if(elts[x].printed)
            {
                //Already printed
                //printf("X= %d Already Printed\n",x);
                continue;
            }
            //uchar* pgray = gray.ptr<uchar>(0);
            
            //PRINT(x)
            int xx,yy;
            getPixelCoordFromIdx(idx, W, xx, yy);
            pc3.push_back(cv::Point3i(xx,yy,idx) );//*(pgray+x) = 255;
            elts[x].printed = true;
            int real_x = x;
            
            int root = elts[x].p;
            //PRINT(root)
            getPixelCoordFromIdx(root, W, xx, yy);//*(pgray+root) = 255;
            pc3.push_back(cv::Point3i(xx,yy,root) );
            elts[root].printed = true;
            
            while( elts[x].link != real_x )
            {
                x = elts[x].link;
                elts[x].printed = true;
                //PRINT(x)
                //*(pgray+x) = 255;
                getPixelCoordFromIdx(x, W, xx, yy);
                pc3.push_back(cv::Point3i(xx,yy,x) );
            }
            
            //fill the sets containers
            xyi.push_back(pc3);
            
        }
        
    }
    
    //SEG RESULTS
    SegResults::SegResults(const cv::Point3f& centroid3D_,
               const cv::Point3f& centroid3DFake_,
               const cv::Point2i& centroid2D_,
               float eigenVal1_,
               float eigenVal2_,
               float angle,
               size_t num_points_,
               const std::vector<cv::Point3i>& pxs_,
               const cv::Mat_<cv::Vec3b>& clusterRGB_,
               const cv::Mat_<uint16_t>& clusterDepth_,
               const cv::Rect& rect_aabb)
    {
        centroid3D = centroid3D_;
        centroid3DFake = centroid3DFake_;
        centroid2D = centroid2D_;
        eigenVal1 = eigenVal1_;
        eigenVal2 = eigenVal2_;
        num_points = num_points_;
        pxs = pxs_;
        clusterRGB_.copyTo(clusterRGB);
        clusterDepth_.copyTo(clusterDepth);
        rect_aabb_ = rect_aabb;
    }
    
    //GRAPH SEG
    template <class T>
    GraphCannySeg<T>::GraphCannySeg(){}
    
    template <class T>
    GraphCannySeg<T>::GraphCannySeg(const cv::Mat& rgb_img, const cv::Mat_<uint16_t>& depth_img, float sigma_, float k_, float min_size, float kx_,float ky_, float ks_, float k_vec[9],float Lcannyth_,float Hcannyth_, float kdv_, float kdc_,float max_ecc, float max_l1, float max_l2,uint16_t DTH, uint16_t plusD,
                  uint16_t point3D, float g_angle, float l_angle, float FarObjZ):
    mSigma(sigma_),mK(k_),mMin_size(min_size),mKx(kx_),mKy(ky_),mKs(ks_),mKdv(kdv_),mKdc(kdc_),rgbWHITE(255,255,255),
    mMax_eccentricity(max_ecc),mMax_L1(max_l1),mMax_L2(max_l2),
    mDTH(DTH),mplusD(plusD),mpoint3D(point3D),mg_angle(g_angle),ml_angle(l_angle),mFarObjZ(FarObjZ),mLcannyTH(Lcannyth_),
    mHcannyTH(Hcannyth_)
    {
        
        //added by STE
//        rgbimg=rgb_img.clone();
//        depthimg=depth_img.clone();
        
        
        mInput_img = convertMat2Image<T>(rgb_img);
        mInput_depth = convertMat2Image<uint16_t>(depth_img);
        //backup of the RGB image
        mInputRGB_img = convertMat2Image<rgb>(rgb_img);
        
        mWidth = mInput_depth->width();
        mHeight = mInput_depth->height();
        mRxC = mInput_depth->WxH();
        
        //filled by DynamicDepthSmoothing and Inpaint Fcns
        mSmoothedDepth = new image<uint16_t>(mInput_depth->width(),mInput_depth->height(),false);
        mInpaintedDepth = new image<uint16_t>(mInput_depth->width(),mInput_depth->height(),false);
        //Convert to Gray img
        cv::cvtColor(rgb_img, mcvGray_img, CV_BGR2GRAY);
        
        //        float k_vec[9] = {571.9737, 0, 319.5000, 0, 571.0073, 239.5000, 0,0,1};
        cv::Mat_<float> K_ = cv::Mat_<float>(3,3,&k_vec[0]);
        K_inv = K_.inv();
        K_invt = K_inv.t();
        
        if(typeid(T) == typeid(rgb))
            mClass_type = RGB;
        else if(typeid(T) == typeid(hsv))
            mClass_type = HSV;
        else if(typeid(T) == typeid(CIELab))
            mClass_type = CIELAB;
        else
            exit(EXIT_FAILURE);
        //TODO: handle the mask, now empty
        mImageMask = cv::Mat_<uchar>();
        
        vecSegResults.empty();
        
    }
    template <class T>
    void GraphCannySeg<T>::dynamicDepthSmoothing(const float beta, const float gamma)
    {
        
        image<uint16_t> *ModDepth_mm = mInput_depth->copy();
        
        /** Depth Dependent Smoothing Area Map : B(r,c) **/
        
        //since here the depth is in mm...
        const float alfa = 0.0028f/1e6f;
        //const float beta = 1500;
        //% Fdc (eq 4)
        std::vector<float> Fdc(mInput_depth->WxH(),0.f);
        std::vector<float> B(mInput_depth->WxH(),0.f);
        
        for (int ii=0; ii<mInput_depth->WxH(); ++ii) {
            
            Fdc[ii] = mInput_depth->data[ii]*mInput_depth->data[ii]*alfa;
            B[ii] = beta*Fdc[ii];
        }
        std::cout << "min Fdc: " << *std::min_element(Fdc.begin(),Fdc.end())<<"\n";
        std::cout << "max Fdc: " << *std::max_element(Fdc.begin(),Fdc.end())<<"\n";
        cv::Mat cvB = cv::Mat(mInput_depth->height(),mInput_depth->width(),CV_32F,&B[0]);
        //visualizeColorMap(cvB,"B",5);
        
        /** Depth Change Indicator Map : C(r,c) **/
        //const float gamma = 18000.f;
        std::vector<uchar> C(mInput_depth->WxH(),0);
        
        for(int r=0; r<mInput_depth->height(); ++r) //y
        {
            for(int c=0; c<mInput_depth->width(); ++c) //x
            {
                
                int r_1 = r+1;
                int r__1 = r-1;
                int c_1 = c+1;
                int c__1 = c-1;
                if(c_1>=mInput_depth->width() || r_1>=mInput_depth->height() || r__1<0 || c__1<0)
                    continue;
                
                int pivot = r*mInput_depth->width() + c;
                //pixel IDX +1
                int r_p1  = (r_1)*mInput_depth->width() + c;
                int c_p1  = r*mInput_depth->width() + (c_1);
                //pixel IDX -1
                int r_m1  = (r__1)*mInput_depth->width() + c;
                int c_m1  = r*mInput_depth->width() + (c__1);
                
                
                int Ddx = static_cast<int>(mInput_depth->data[c_p1] - mInput_depth->data[pivot]);
                int Ddy = static_cast<int>(mInput_depth->data[r_p1] - mInput_depth->data[pivot]);
                
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
        cv::Mat cvC = cv::Mat(mInput_depth->height(),mInput_depth->width(),CV_8U,&C[0]);
        //    cv::imshow("C",cvC);
        //    cv::waitKey(0);
        //visualizeColorMap(cvC,"C",5,true);
        
        /** Final Smoothing Area Map : R(r,c) **/
        
        cv::Mat_<uchar> Cmat = cv::Mat_<uchar>(mInput_depth->height(),mInput_depth->width(), C.data()); //or simplier &C[0];
        cv::Mat Cmatnot;
        cv::bitwise_not(Cmat, Cmatnot);
        cv::Mat Tmat_;
        cv::distanceTransform(Cmatnot, Tmat_, CV_DIST_L2, CV_DIST_MASK_PRECISE);
        Tmat_ = Tmat_*M_SQRT1_2; //Tmat_./sqrt(2);
        //visualizeColorMap(Tmat_,"Tsqrt",5,true);
        
        float* Tptr = Tmat_.ptr<float>(0);
        
        std::vector<float> R(mInput_depth->WxH(),0.f);
        for (int ii=0; ii<mInput_depth->WxH(); ++ii) {
            
            R[ii] = std::min(B[ii], Tptr[ii]);
        }
        
        cv::Mat cvR = cv::Mat(mInput_depth->height(),mInput_depth->width(),CV_32F,&R[0]);
        visualizeColorMap(cvR,"R",5,false);
        
        /** Smoothing **/
        
        //integral image
        cv::Mat cvModDepth_mm(mInput_depth->height(),mInput_depth->width(),CV_16U,ModDepth_mm->data);
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
        cv::Mat logicalDepthmm = cv::Mat::zeros(mInput_depth->height(),mInput_depth->width(),CV_8U);
        uchar* logicalDepthmmPtr = logicalDepthmm.ptr<uchar>(0);
        for (int ii=0; ii<mInput_depth->WxH(); ++ii) {
            
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
        
        cv::Mat SDepth_mm = cv::Mat::zeros(mInput_depth->height(),mInput_depth->width(),CV_16U);
        uint16_t* SDepth_mmPtr = SDepth_mm.ptr<uint16_t>(0);
        
        //TODO: Smoothing X Y ranges...
        for(int r=15; r<mInput_depth->height()-15; ++r) //y
        {
            for(int c=15; c<mInput_depth->width()-15; ++c) //x
            {
                int pivot = r*mInput_depth->width() + c;
                int radi = int(R[pivot]+0.5f);
                
                if(radi == 0)
                {
                    //boarder or NaN: No smoothing at all
                    SDepth_mmPtr[pivot] = mInput_depth->data[pivot];
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
                    SDepth_mmPtr[pivot] = mInput_depth->data[pivot];
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
        
        mSmoothedDepth->init(SDepth_mm);
        
        
        
    }
    
    //use the member variable
    template <class T>
    void GraphCannySeg<T>::myCannyWithOri2(float T_Low, float T_High,
                         const cv::Mat& mask,
                         bool blurTheResult)
    {
        
        const int nx = mcvGray_img.cols;
        const int ny = mcvGray_img.rows;
        const int RxC = nx*ny;
        
        //std::cout<< cv::getGaussianKernel(5,0)<<"\n";
        
        //TODO: mcvGray_img member Gray image is Modified by Gaussian Blur
        cv::GaussianBlur( mcvGray_img, mcvGray_img, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );
        
        cv::Mat Gx;
        //cv::Sobel(mcvGray_img, Gx, CV_32F, 1, 0, 3);
        cv::Scharr( mcvGray_img, Gx, CV_32F, 1, 0, 1, 0, cv::BORDER_DEFAULT );
        
        cv::Mat Gy;
        //cv::Sobel(mcvGray_img, Gy, CV_32F, 0, 1, 3, -1);
        cv::Scharr( mcvGray_img, Gy, CV_32F, 0, 1, -1, 0, cv::BORDER_DEFAULT );
        
        cv::Mat_<float> Ori = cv::Mat_<float>::zeros(mcvGray_img.size());
        cv::Mat_<uchar> Ori2 = cv::Mat_<uchar>::zeros(mcvGray_img.size());
        cv::Mat_<float> G = cv::Mat_<float>::zeros(mcvGray_img.size());
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
        cv::Mat_<uint16_t> oppositeAngle = cv::Mat_<uint16_t>::zeros(mcvGray_img.size());
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
        
        cv::Mat_<float> BW = cv::Mat_<float>::zeros(mcvGray_img.size());
        mOriOut = cv::Mat_<uchar>::zeros(mcvGray_img.size());
        uchar* OriOutPtr = mOriOut.ptr<uchar>(0);
        
        cv::Mat_<uint16_t> oppositeAngle2 = cv::Mat_<uint16_t>::zeros(mcvGray_img.size());
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
        
        cv::Mat_<uint16_t> oppositeAngleOut = cv::Mat_<uint16_t>::zeros(mcvGray_img.size());
        uint16_t* oppositeAngleOutPtr = oppositeAngleOut.ptr<uint16_t>(0);
        
        mMagOut = cv::Mat_<uchar>::zeros(mcvGray_img.size());
        uchar* MagOutPtr = mMagOut.ptr<uchar>(0);
        for (int i=0; i < ny; ++i)
        {
            for(int j=0; j < nx; ++j )
            {
                
                if (BW(i, j) < T_Low)
                    mMagOut(i, j) = 0;
                else if (BW(i, j) > T_High)
                {
                    mMagOut(i, j) = 255;
                    //TODO: Added 100 only for visualization...
                    mOriOut(i, j) = Ori2(i,j)+100;
                    oppositeAngleOut(i,j) = oppositeAngle2(i,j);
                }
                //Using 8-connected components
                else if ( BW(i+1,j)>T_High || BW(i-1,j)>T_High || BW(i,j+1)>T_High || BW(i,j-1)>T_High || BW(i-1, j-1)>T_High || BW(i-1, j+1)>T_High || BW(i+1, j+1)>T_High || BW(i+1, j-1)>T_High)
                {
                    mMagOut(i, j) = 255;
                    mOriOut(i, j) = Ori2(i,j)+100;
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
        if(mInpaintedDepth==0)
            return;
        //filter by mInpaintedDepth try to remove some textures leaving real borders unchanged
        cv::Mat_<cv::Vec3b> BOWOut = cv::Mat_<cv::Vec3b>::zeros(mcvGray_img.size());
        
        const int start_px = std::max(mplusD, mpoint3D)+1;
        for (int i = 0; i < ny ; ++i)//riga y
        {
            for (int j = 0; j < nx ; ++j)//colonna x
            {
                const int c = j + nx * i;
                if(i<start_px || i>ny-start_px || j<start_px || j>nx-start_px)
                {
                    MagOutPtr[c]=0;
                    oppositeAngleOutPtr[c] = 0;
                    continue;
                }
                
                if(MagOutPtr[c]==0)
                    continue;
                //            if(imRef(mInpaintedDepth, j, i)==0)
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
                //                uint16_t DTH = 30; //[mm]
                //                uint16_t plusD = 5; //for depth boundary
                //                uint16_t point3D = 10; //for contact boundary
                //                float g_angle = 120.f*M_PI/180.f;//2.f/3.f*M_PI;
                //                float l_angle = 60.f*M_PI/180.f; //M_PI/3.f;
                if ((mOriOut(i,j)-100)==0)
                {
                    //float G3[] = { G(i,j), G(i,j+1), G(i,j-1) };
                    //                uint16_t dp = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j+1, i));
                    //                uint16_t dm = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j-1, i));
                    
                    uint16_t dp2 = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j+mplusD, i));
                    uint16_t dm2 = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j-mplusD, i));
                    
                    cv::Mat_<float> p0,p1,p2;
                    projectPixel2CameraRF(K_inv, j, i, imRef(mInpaintedDepth, j, i), p0);
                    projectPixel2CameraRF(K_inv, j+mpoint3D, i, imRef(mInpaintedDepth, j+mpoint3D, i), p1);
                    projectPixel2CameraRF(K_inv, j-mpoint3D, i, imRef(mInpaintedDepth, j-mpoint3D, i), p2);
                    float theta;//in rad
                    angle3Points(p1, p0, p2, theta);
                    
                    if( (/*(dp<mDTH && dm<mDTH) ||*/ (dp2<mDTH && dm2<mDTH))
                       &&  (theta>mg_angle || theta<ml_angle))
                    {
                        MagOutPtr[c]=0;
                        oppositeAngleOutPtr[c]=0;
                        //                    MagOutPtr[c+mplusD]=255;
                        //                    MagOutPtr[c-mplusD]=255;
                        //                    mMagOut(i,j+mplusD) = 255;
                        //                    mMagOut(i,j-mplusD) = 255;
                    }
                    //                else
                    //                {
                    //                    //Border Ownership
                    //                    if(imRef(mInpaintedDepth, j+mplusD, i) < imRef(mInpaintedDepth, j-mplusD, i))
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
                else if ((mOriOut(i,j)-100)==45)
                {
                    //float G3[] = { G(i,j), G(i+1,j-1), G(i-1,j+1) };
                    //                uint16_t dp = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j+1, i-1));
                    //                uint16_t dm = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j-1, i+1));
                    
                    uint16_t dp2 = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j+mplusD, i-mplusD));
                    uint16_t dm2 = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j-mplusD, i+mplusD));
                    
                    cv::Mat_<float> p0,p1,p2;
                    projectPixel2CameraRF(K_inv, j, i, imRef(mInpaintedDepth, j, i), p0);
                    projectPixel2CameraRF(K_inv, j+mpoint3D, i-mpoint3D, imRef(mInpaintedDepth, j+mpoint3D, i-mpoint3D), p1);
                    projectPixel2CameraRF(K_inv, j-mpoint3D, i+mpoint3D, imRef(mInpaintedDepth, j-mpoint3D, i+mpoint3D), p2);
                    float theta;//in rad
                    angle3Points(p1, p0, p2, theta);
                    
                    if( (/*(dp<mDTH && dm<mDTH) ||*/ (dp2<mDTH && dm2<mDTH))
                       &&  (theta>mg_angle || theta<ml_angle))
                    {
                        MagOutPtr[c]=0;
                    }
                }
                else if ((mOriOut(i,j)-100)==90)
                {
                    //float G3[] = { G(i,j), G(i+1,j), G(i-1,j) };
                    //                uint16_t dp = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j, i-1));
                    //                uint16_t dm = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j, i+1));
                    
                    uint16_t dp2 = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j, i-mplusD));
                    uint16_t dm2 = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j, i+mplusD));
                    
                    cv::Mat_<float> p0,p1,p2;
                    projectPixel2CameraRF(K_inv, j, i, imRef(mInpaintedDepth, j, i), p0);
                    projectPixel2CameraRF(K_inv, j, i-mpoint3D, imRef(mInpaintedDepth, j, i-mpoint3D), p1);
                    projectPixel2CameraRF(K_inv, j, i+mpoint3D, imRef(mInpaintedDepth, j, i+mpoint3D), p2);
                    float theta;//in rad
                    angle3Points(p1, p0, p2, theta);
                    
                    if(j==216 && i==259)
                    {
                        printf("j: %d; i: %d; dp2: %d; dm2: %d; theta: %f\n",j,i,dp2,dm2,theta*180.f/M_PI);
                    }
                    
                    if( ((dp2<mDTH && dm2<mDTH))
                       &&  (theta>mg_angle || theta<ml_angle))
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
                     
                     if( ( (dp2<mDTH && dm2<mDTH) || ( (theta<-20*M_PI/180.f && theta > -80*M_PI/180.f) && (dm2<5 && dp2>90)  ) )
                     && (theta>mg_angle || theta<ml_angle))
                     {
                     MagOutPtr[c]=0;
                     }
                     */
                }
                else if ((mOriOut(i,j)-100)==135)
                {
                    //float G3[] = { G(i,j), G(i+1,j+1), G(i-1,j-1) };
                    //                uint16_t dp = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j-1, i-1));
                    //                uint16_t dm = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j+1, i+1));
                    //
                    uint16_t dp2 = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j-mplusD, i-mplusD));
                    uint16_t dm2 = std::abs(imRef(mInpaintedDepth, j, i) - imRef(mInpaintedDepth, j+mplusD, i+mplusD));
                    
                    cv::Mat_<float> p0,p1,p2;
                    projectPixel2CameraRF(K_inv, j, i, imRef(mInpaintedDepth, j, i), p0);
                    projectPixel2CameraRF(K_inv, j-mpoint3D, i-mpoint3D, imRef(mInpaintedDepth, j-mpoint3D, i-mpoint3D), p1);
                    projectPixel2CameraRF(K_inv, j+mpoint3D, i+mpoint3D, imRef(mInpaintedDepth, j+mpoint3D, i+mpoint3D), p2);
                    float theta;//in rad
                    angle3Points(p1, p0, p2, theta);
                    
                    if( (/*(dp<mDTH && dm<mDTH) ||*/ (dp2<mDTH && dm2<mDTH))
                       &&  (theta>mg_angle || theta<ml_angle))
                    {
                        MagOutPtr[c]=0;
                    }
                }
                
            }
        }
        
        //TODO this filter Canny very well but the params must be optimized....now it works worse!!!
        /*  FindCountors Delete small segment  */
        //        std::vector<std::vector<cv::Point> > contours;
        //        std::vector<cv::Vec4i> hierarchy;
        //
        //        /// Find contours
        //        cv::findContours( mMagOut, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cv::Point(0, 0) );
        //
        //        for(int ic=0;ic<contours.size();++ic)
        //        {
        //            if(contours[ic].size()<30)
        //            {
        //                for(int ip=0;ip<contours[ic].size();++ip)
        //                {
        //                    //delete the pixels
        //                    mMagOut(contours[ic][ip].y,contours[ic][ip].x) = 0;
        //                }
        //            }
        //        }
        
        
        
        if(blurTheResult)
            cv::GaussianBlur(mMagOut, mMagOut, cv::Size(7,7), 0);
        
        
        
        //Normalize Canny and get the float pointer
        //cv::Mat SaliencyMyCanny;
        double minVal, maxVal;
        cv::minMaxIdx(mMagOut, &minVal, &maxVal);
        mMagOut.convertTo(mSaliencyMyCanny,CV_32F,1.0/(maxVal - minVal), -minVal*1.0/(maxVal-minVal));
        
        cv::minMaxIdx(mSaliencyMyCanny, &minVal, &maxVal);
        printf("SaliencyMyCanny Float min: %f; max: %f\n",minVal,maxVal);
        mSaliencyMyCannyPtr = mSaliencyMyCanny.ptr<float>(0);
        
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
    
    template <class T>
    image<rgb>* GraphCannySeg<T>::segment_image() {
        
        //test if HSV class
        //TODO: Implement segmentation for RGB & CIELab
        if(mClass_type==HSV)
        {
            
            if(mSaliencyMyCannyPtr)
                printf("***Segmenting HSV + DEPTH + SALIENCY***\n");
            else
                printf("***Segmenting HSV + DEPTH***\n");
            
            //            int width = im->width();
            //            int height = im->height();
            
            //    for(int ii=0;ii<640*480;++ii)
            //    {
            //        printf(", %f",saliency[ii]);
            //    }
            //    exit(0);
            
            //in case T==hsv -> r == h; g == s; b == v
            //TODO: Here we are smothing HSV directly...maybe it's ok...
            //      otherwise we need to smooth rgb first and than convert
            //      to HSV
            
            image<float> *h = new image<float>(mWidth, mHeight);
            image<float> *s = new image<float>(mWidth, mHeight);
            image<float> *v = new image<float>(mWidth, mHeight);
            
            //    TODO: DONE we have already smoothed the RGB image in the loading stage : But this one works worse !!
            //    image<float> *smooth_h = new image<float>(mWidth, mHeight);
            //    image<float> *smooth_s = new image<float>(mWidth, mHeight);
            //    image<float> *smooth_v = new image<float>(mWidth, mHeight);
            
            // smooth each color channel//Now just deep copy
            for (int y = 0; y < mHeight; y++) {
                for (int x = 0; x < mWidth; x++) {
                    imRef(h, x, y) = static_cast<float>(imRef(mInput_img, x, y).h);
                    imRef(s, x, y) = static_cast<float>(imRef(mInput_img, x, y).s);
                    imRef(v, x, y) = static_cast<float>(imRef(mInput_img, x, y).v);
                }
            }
            /*for HSV image, smooth is done in the loading stage of the RGB image when commented them*/
            image<float> *smooth_h = smooth(h, mSigma);
            image<float> *smooth_s = smooth(s, mSigma);
            image<float> *smooth_v = smooth(v, mSigma);
            delete h;
            delete s;
            delete v;
            
            //normalize depth between [0,1]
            uint16_t max_depth_val =
            *std::max_element(imPtr(mInpaintedDepth, 0, 0),
                              imPtr(mInpaintedDepth, mWidth-1, mHeight-1));
            uint16_t min_depth_val =
            *std::min_element(imPtr(mInpaintedDepth, 0, 0),
                              imPtr(mInpaintedDepth, mWidth-1, mHeight-1));
            
            printf("Max Depth Value: %d [mm]\n",max_depth_val);
            //min value is 0 for sure so: Not So sure now... :)
            printf("Min Depth Value: %d [mm]\n",min_depth_val);
            
            
            image<float> *depth_mm_norm = new image<float>(mWidth, mHeight);
            for(size_t idx=0;idx<depth_mm_norm->WxH();++idx)
            {
                /*normalize [0-1]*/
                //        depth_mm_norm->data[idx] =
                //        static_cast<float>(mInpaintedDepth->data[idx])/static_cast<float>(max_depth_val);
                
                float act_d = static_cast<float>(mInpaintedDepth->data[idx]);
                
                depth_mm_norm->data[idx] = mapminmax(act_d, (float)min_depth_val, (float)max_depth_val, 0.f, 1.f);
                
                
                /*normalize [0-255]*/
                //         depth_mm_norm->data[idx] =
                //        mapminmax(static_cast<float>(mInpaintedDepth->data[idx]), 0.f, static_cast<float>(max_depth_val), 0.f, 255.0f);
            }
            //    float ret_min, ret_max;
            //    min_max(depth_mm_norm, &ret_min, &ret_max);
            //    printf("minDepthNorm: %f  \n",ret_min);
            //    printf("maxDepthNorm: %f  \n",ret_max);
            
            //DEBUG
            /*
             cv::Mat debmat = cv::Mat(mInpaintedDepth->height(),mInpaintedDepth->mWidth(),CV_32F,depth_mm_norm->data);
             //convert in Gray scale
             double minVal, maxVal;
             cv::Mat J1_gray_img;
             cv::minMaxLoc(debmat,&minVal,&maxVal);
             debmat.convertTo(J1_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
             cv::imshow("debug gray", J1_gray_img);
             cv::waitKey(0);
             */
            
            // build graph
            edge *edges = new edge[mWidth*mHeight*4];
            cv::Mat W1 = cv::Mat::zeros(mHeight, mWidth, CV_32F);
            float* pW1 = W1.ptr<float>(0);
            cv::Mat W2 = cv::Mat::zeros(mHeight, mWidth, CV_32F);
            float* pW2 = W2.ptr<float>(0);
            cv::Mat W3 = cv::Mat::zeros(mHeight, mWidth, CV_32F);
            float* pW3 = W3.ptr<float>(0);
            cv::Mat W4 = cv::Mat::zeros(mHeight, mWidth, CV_32F);
            float* pW4 = W4.ptr<float>(0);
            
            // 8-connectivity
            /*       0
             /
             /w4
             /
             x-w1-0
             | \
             w2 \w3
             |   \
             0    0
             
             
             */
            int num = 0;
            for (int y = 0; y < mHeight; y++) {
                for (int x = 0; x < mWidth; x++) {
                    if (x < mWidth-1) {
                        edges[num].a = y * mWidth + x;
                        edges[num].b = y * mWidth + (x+1);
                        edges[num].w = diff(smooth_h, smooth_s, smooth_v, x, y, x+1, y, depth_mm_norm, mSaliencyMyCannyPtr);
                        
                        *(pW1+(y * mWidth + x)) = edges[num].w;
                        num++;
                    }
                    
                    if (y < mHeight-1) {
                        edges[num].a = y * mWidth + x;
                        edges[num].b = (y+1) * mWidth + x;
                        edges[num].w = diff(smooth_h, smooth_s, smooth_v, x, y, x, y+1, depth_mm_norm, mSaliencyMyCannyPtr);
                        
                        *(pW2+(y * mWidth + x)) = edges[num].w;
                        num++;
                    }
                    
                    if ((x < mWidth-1) && (y < mHeight-1)) {
                        edges[num].a = y * mWidth + x;
                        edges[num].b = (y+1) * mWidth + (x+1);
                        edges[num].w = diff(smooth_h, smooth_s, smooth_v, x, y, x+1, y+1, depth_mm_norm, mSaliencyMyCannyPtr);
                        *(pW3+(y * mWidth + x)) = edges[num].w;
                        num++;
                    }
                    
                    if ((x < mWidth-1) && (y > 0)) {
                        edges[num].a = y * mWidth + x;
                        edges[num].b = (y-1) * mWidth + (x+1);
                        edges[num].w = diff(smooth_h, smooth_s, smooth_v, x, y, x+1, y-1, depth_mm_norm, mSaliencyMyCannyPtr);
                        *(pW4+(y * mWidth + x)) = edges[num].w;
                        num++;
                    }
                }
            }
            delete smooth_h;
            delete smooth_s;
            delete smooth_v;
            //printf("num edges: %d\n",num);
            
            //PLOT Ws
            visualizeColorMap(W1,"W1",5);
            visualizeColorMap(W2,"W2",5);
            visualizeColorMap(W3,"W3",5);
            visualizeColorMap(W4,"W4",5);
            cv::Mat Wtotal = 0.25*W1+0.25*W2+0.25*W3+0.25*W4;
            visualizeColorMap(Wtotal,"Wtot",5);
            
            double minVal, maxVal;
            cv::minMaxLoc(Wtotal,&minVal,&maxVal);
            
            
            float minWdRGB = *std::min_element(DrgbV.begin(), DrgbV.end());
            float maxWdRGB = *std::max_element(DrgbV.begin(), DrgbV.end());
            printf("minWdHSV: %f; maxWdHSV: %f\n",minWdRGB,maxWdRGB);
            
            float minWdDepth = *std::min_element(DdepthV.begin(), DdepthV.end());
            float maxWdDepth = *std::max_element(DdepthV.begin(), DdepthV.end());
            printf("minWdDepth: %f; maxWdDepth: %f\n",minWdDepth,maxWdDepth);
            
            
            // segment
            universe *u = segment_graph(mRxC, num, edges);//c);
            
            // post process small components
            //TODO: Good because fill small gap in the depth with the true object. Bad because can add objects to a cluster regardless the weight informations provided!!
            for (int i = 0; i < num; i++) {
                int a = u->find(edges[i].a);
                int b = u->find(edges[i].b);
                if ((a != b) && ((u->size(a) < mMin_size) || (u->size(b) < mMin_size)))
                    u->join(a, b);
            }
            delete [] edges;
            
            mNumClustersFounds = u->num_sets();
            
            
            
            //Preliminary Segmentation Results
            image<rgb> *output = new image<rgb>(mWidth, mHeight);
            
            //pick random colors for each component
            rgb *colors = new rgb[mRxC];
            for (int i = 0; i < mRxC; i++)
                colors[i] = random_rgb();
            
            for (int y = 0; y < mHeight; y++) {
                for (int x = 0; x < mWidth; x++) {
                    int comp = u->find(y * mWidth + x);
                    
                    imRef(output, x, y) = colors[comp];
                    
                }
            }
            
            printf("got %d components\n", mNumClustersFounds);
            //visualizeImage(output,"Seg Res",0);
            
            
            //Prova Print Nodes for a given set
            /*
             cv::Mat_<uchar> gray = cv::Mat_<uchar>::zeros(mHeight, mWidth);
             u->printSet(234 * mWidth + 215, gray);
             cv::imshow("printSet", gray);
             cv::waitKey(5);
             u->printSet(234 * mWidth + 218, gray);
             u->printSet(434 * mWidth + 218, gray);
             cv::waitKey(5);
             cv::imshow("printSet2", gray);
             */
            PostSegmFilter(u);
            
            
            
            
            delete [] colors;
            delete depth_mm_norm;
            delete u;
            return output;
        }//end if test HSV class
        std::cout<<"CIELAB and RGB not Handled yet...only HSV\n";
        exit(EXIT_FAILURE);//return 0;
    }
    
    template <class T>
    universe* GraphCannySeg<T>::segment_graph(int num_vertices, int num_edges, edge *edges) {
        // sort edges by weight
        std::sort(edges, edges + num_edges);
        
        // make a disjoint-set forest
        universe *u = new universe(num_vertices);
        
        // init thresholds
        float *threshold = new float[num_vertices];
        for (int i = 0; i < num_vertices; ++i)
            threshold[i] = THRESHOLD(1,mK);
        
        // for each edge, in non-decreasing weight order...
        for (int i = 0; i < num_edges; ++i) {
            edge *pedge = &edges[i];
            
            // components conected by this edge
            //find ritorna la root a cui appartine quel pixel che ha a come starting edge ed b come ending edge
            int a = u->find(pedge->a);
            int b = u->find(pedge->b);
            //Se non appartengono gia' allo stesso set (component)
            //testa se e' possibile unire i due sets tramite i Th.
            if (a != b) {
                if ((pedge->w <= threshold[a]) &&
                    (pedge->w <= threshold[b])) {
                    u->join(a, b);
                    //ottieni la nuova root del set unito!!
                    a = u->find(a);
                    //Update the th since the C dimension is changed
                    threshold[a] = pedge->w + THRESHOLD(u->size(a), mK);
                    //          cv::Mat_<float> thMat = cv::Mat_<float>(480,640,&threshold[0]);
                    //          visualizeColorMap(thMat, "THs", 0);
                }
            }
        }
        
        //    for (int i = 0; i < num_edges; i++) {
        //        int a = u->find(edges[i].a);
        //        int b = u->find(edges[i].b);
        //        if ((a != b) && ((u->size(a) < 500) || (u->size(b) < 500)))
        //            u->join(a, b);
        //    }
        //
        //    float ths[640*480] = {};
        //    printf("u->size() %d\n",u->num_sets());
        //    std::vector<std::vector<cv::Point3i> > xyi;
        //    u->collectSets(xyi, 480, 640);
        //    for (int i=0; i<xyi.size(); ++i) {
        //        for (int j=0; j<xyi[i].size(); ++j)
        //        {
        //            ths[xyi[i][j].z] = threshold[xyi[i][j].z];
        //
        //        }
        //    }
        ///*    for(int i=0;i<num_vertices;++i)
        //    {
        //        int xx = u->find(i);
        //        ths[xx] = threshold[xx];
        //    }
        //    for(int i=0;i<num_vertices;++i)
        //    {
        //        if(ths[i]>0.f)
        //            printf("%f\n",ths[i]);
        //    }
        //*/
        //    cv::Mat_<float> thMat = cv::Mat_<float>(480,640,&ths[0]);
        //    visualizeColorMap(thMat, "THs", 0);
        
        // free up
        delete threshold;
        return u;
    }
    
    template <class T>
    void GraphCannySeg<T>::computePCA(const std::vector<cv::Point3i>& pts,
                    cv::Mat &img,//BGR
                    std::vector<cv::Point2d>& eigen_vecs,
                    std::vector<double>& eigen_val,
                    cv::Point& cntr, double& angle,
                    double& eccentricity,bool viz_)
    {
        //Construct a buffer used by the pca analysis
        int sz = static_cast<int>(pts.size());
        cv::Mat data_pts = cv::Mat(sz, 2, CV_64FC1);
        for (int i = 0; i < data_pts.rows; ++i)
        {
            data_pts.at<double>(i, 0) = (double)pts[i].x;
            data_pts.at<double>(i, 1) = (double)pts[i].y;
        }
        //Perform PCA analysis
        cv::PCA pca_analysis(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);
        //Store the center of the object
        cntr = cv::Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                         static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
        //Store the eigenvalues and eigenvectors
        //std::vector<cv::Point2d> eigen_vecs(2);
        eigen_vecs.resize(2);
        eigen_val.resize(2);
        //std::vector<double> eigen_val(2);
        for (int i = 0; i < 2; ++i)
        {
            eigen_vecs[i] = cv::Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                        pca_analysis.eigenvectors.at<double>(i, 1));
            eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
        }
        //Compute Eccentricity
        eccentricity = sqrt(1-eigen_val[1]/eigen_val[0]);
        
        if(viz_)
        {
            //cv::cvtColor(mask, img, CV_GRAY2BGR);
            // Draw the principal components
            //gray(xyi[idx][dx].y,xyi[idx][dx].x)
            uchar* imgPtr = img.ptr<uchar>(0);
            rgb rgb_ = random_rgb();
            for (int i = 0; i < pts.size(); ++i)
            {
                int idx = pts[i].z*3; //pixel_idx*3
                
                imgPtr[idx] = rgb_.b;
                imgPtr[idx+1] = rgb_.g;
                imgPtr[idx+2] = rgb_.r;
            }
            
            cv::circle(img, cntr, 3, cv::Scalar(255, 0, 255), 2);
            cv::Point p1 = cntr + 0.02 * cv::Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
            cv::Point p2 = cntr - 0.02 * cv::Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
            drawAxis(img, cntr, p1, cv::Scalar(0, 255, 0), 1);
            drawAxis(img, cntr, p2, cv::Scalar(255, 255, 0), 3);
        }
        angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
        return;
        
    }
    
    template <class T>
    void GraphCannySeg<T>::run()
    {
        //smooth the depth
       // dynamicDepthSmoothing(2000.f);
        mSmoothedDepth=mInput_depth->copy();

        visualizeImage(mSmoothedDepth,"SmoothedDepth",5);
        
        //inpaint & visualize if true the depth
        inpaintDepth(true,5);
        
        //My Canny and Blur the result if true, generate the float* pointer SaliencyMyCannyPtr used by segment_image
        //0.05f, 0.075f
        myCannyWithOri2(mLcannyTH,mHcannyTH, mImageMask,true);
        cv::imshow("MyCanny Gaussin Blur", mMagOut);
        visualizeColorMap(mOriOut,"PHASE",5,false);
        
        //Segment & PCA filter
        /* HSV + DEPTH + SALIENCY */
        image<rgb> *seg = segment_image();
        
        visualizeImage(seg,"Seg Res",5);
        
    }
    
};//end namespace

// No need to call this TemporaryFunction() function,
// it's just to avoid link error.
void TemporaryFunction()
{
    
    float kfloat ;
    float kxfloat;
    float kyfloat ;
    float ksfloat ;
    float gafloat ;
    float lafloat;
    float lcannyf ;
    float hcannyf ;
    
    int k=67;//30; //0.003 /10000
    int kx=2000;//2.0f;
    int ky=30;//0.03f;
    int ks=50;//0.05f;
    float kdv=4.5f;
    float kdc=0.1f;
    float min_size=500.0f;
    float sigma=0.8f;
    float max_ecc = 0.978f;
    float max_L1 = 3800.0f;
    float max_L2 = 950.0f;
    
    int DTH = 30; //[mm]
    int plusD = 7; //for depth boundary
    int point3D = 5; //10//for contact boundary
    int g_angle = 148;//2.f/3.f*M_PI;
    int l_angle = 56; //M_PI/3.f;
    int Lcanny = 50;
    int Hcanny = 75;
    int FarObjZ = 875;//1800; //[mm]
    
    cv::Mat kinect_rgb_img;
    cv::Mat kinect_depth_img_mm;
    
    //CHALLENGE 1 DATASET
    double fx = 571.9737;
    double fy = 571.0073;
    double cx = 319.5000;
    double cy = 239.5000;
    float k_vec[9] = {static_cast<float>(fx), 0, static_cast<float>(cx), 0, static_cast<float>(fy), static_cast<float>(cy), 0.f,0.f,1.f};

    GraphCanny::GraphCannySeg<GraphCanny::hsv> gcs(kinect_rgb_img, kinect_depth_img_mm, sigma, kfloat, min_size, kxfloat, kyfloat, ksfloat,k_vec,lcannyf,hcannyf,kdv, kdc,max_ecc,max_L1,max_L2,(uint16_t)DTH,(uint16_t)plusD,(uint16_t)point3D,gafloat,lafloat,(float)FarObjZ);
    
    gcs.run();
}
