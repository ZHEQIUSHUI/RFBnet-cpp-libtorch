#include <iostream>
#include <torch/script.h>
//#include <torch/torch.h>
#include <opencv2/opencv.hpp>
using namespace std;
class Obj
{
public:
    int cls_id;
    float score;
    cv::Rect2d bbox;
    void set_bbox(vector<double>rect)
    {
        bbox=cv::Rect2d(cv::Point2d(rect[0],rect[1]),cv::Point2d(rect[2],rect[3]));
    }

    double operator [](int i)
    {
        switch (i) {
        case 0:
            return bbox.tl().x;
        case 1:
            return bbox.tl().y;
        case 2:
            return bbox.br().x;
        case 3:
            return bbox.br().y;
        default:
            return -1;
        }
    }

    bool operator >(Obj&obj)
    {
        return score>obj.score;
    }

    bool operator <(Obj&obj)
    {
        return score<obj.score;
    }
};
bool cmp(Obj&a,Obj&b)
{
    return a>b;
}

//v={100,300,200} false->return {0,2,1}  true->return {1,2,0}
template <typename T>
inline vector<size_t> sort_indexes_e(vector<T> &v, bool reverse = false)
{
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    if (reverse)
    {
        std::reverse(idx.begin(), idx.end());
    }
    return idx;
}

//select element by indexs in vector
template <typename T>
inline vector<T> selectByindex(vector<T> &vec, vector<size_t>&idxs)
{
    vector<T> result(idxs.size());
    for (size_t i = 0; i < idxs.size(); i++)
    {
        result[i] = vec[idxs[i]];
    }
    return result;
}


void nms_(vector<Obj>&dets,float thresh=0.45)
{
    vector<int> keep;
    vector<double> x1s(dets.size());
    vector<double> y1s(dets.size());
    vector<double> x2s(dets.size());
    vector<double> y2s(dets.size());
    vector<double> areas(dets.size());
    vector<size_t> order = sort_indexes_e(dets,true);
    for (int i = 0; i < dets.size(); ++i) {
        Obj&obj=dets[i];
        auto x1 = obj[0];
        auto y1 = obj[1];
        auto x2 = obj[2];
        auto y2 = obj[3];
        x1s[i]=x1;
        y1s[i]=y1;
        x2s[i]=x2;
        y2s[i]=y2;
        auto area = (x2 - x1 + 1) * (y2 - y1 + 1);
        areas[i]=area;
    }
    while (order.size()>0) {
        size_t i = order[0];
        keep.push_back(i);
        auto t =vector<size_t>(order.begin()+1,order.end());
        auto x1 = selectByindex(x1s,t);
        x1.push_back(x1s[i]);

        auto y1 = selectByindex(y1s,t);
        y1.push_back(y1s[i]);

        auto x2 = selectByindex(x2s,t);
        x2.push_back(x2s[i]);

        auto y2 = selectByindex(y2s,t);
        y2.push_back(y2s[i]);
    }
}

vector<int> nms(vector<Obj>&dets,float thresh=0.45)
{
    vector<int>keep;
    vector<int> erase;
    for (int i = 0; i < dets.size(); ++i) {
        Obj& obj = dets[i];
        for (int j = 0; j < dets.size(); ++j) {
            if(i==j)
                continue;
            if(obj.cls_id==dets[j].cls_id)
            {
                if((obj.bbox&dets[j].bbox).area()/(obj.bbox|dets[j].bbox).area()>thresh)
                {
                    if(obj.score>dets[j].score)
                    {
                        erase.push_back(j);
                    }
                    else {
                        erase.push_back(i);
                    }
                }
            }
        }
    }
//    std::cout<<"ok"<<std::endl;
    sort(erase.begin(),erase.end(),greater<int>());
    auto end_iter = unique(erase.begin(),erase.end());
    erase.erase(end_iter,erase.end());
    for (int var = 0; var < erase.size(); ++var) {
        dets.erase(dets.begin()+erase[var]);
         //std::cout<<erase[var]<<std::endl;
    }
    return keep;
}

int main()
{
    vector<double> variance = {0.1, 0.2};
    vector<int> feature_maps= {64, 32, 16, 8, 4, 2, 1};
    int min_dim= 512;
    vector<int> steps= {8, 16, 32, 64, 128, 256, 512};
    vector<double> min_sizes={20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8};
    vector<double> max_sizes = {51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72};
    vector<vector<int>> aspect_ratios={vector<int>{2, 3}, vector<int>{2, 3}, vector<int>{2, 3}, vector<int>{2, 3}, vector<int>{2, 3}, vector<int>{2}, vector<int>{2}};
    auto clip = true;
    //        }
    vector<vector<double>> mean;
    for (int k = 0; k < feature_maps.size(); k++) {
         auto f = feature_maps[k];
         vector<vector<int>> product(f*f);
         int i=0;
         int j=0;
         for (int l = 0; l < f*f; ++l) {
             if(j==f)
             {
                 i++;
                 j=0;
             }
             product[l]=(vector<int>{i,j});
             j++;
         }
         for (int t = 0; t < product.size(); ++t) {
             int i =product[t][0];
             int j =product[t][1];
             auto f_k = min_dim / steps[k];
             auto cx = (j + 0.5) / f_k;
             auto cy = (i + 0.5) / f_k;
             auto s_k = min_sizes[k] / min_dim;
             mean.push_back(vector<double>{cx, cy, s_k, s_k});
             auto s_k_prime = sqrt(s_k * (max_sizes[k] / min_dim));
             mean.push_back(vector<double>{cx, cy, s_k_prime, s_k_prime});
             for (int b = 0; b < aspect_ratios[k].size(); ++b) {
                 auto ar = aspect_ratios[k][b];
                 mean.push_back(vector<double>{cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)});
                 mean.push_back(vector<double>{cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)});

             }
         }
    }

//    cout<<mean.size()<<endl;

    torch::jit::script::Module module = torch::jit::load("RFB512_E_34_4.pt");
    module.to(at::kCUDA);
    cv::VideoCapture cap(0);
    cv::Mat src;
    cv::Mat input;
    while (1) {
        if(!cap.read(src))
            break;
//        cv::cvtColor(src, input, cv::COLOR_BGR2RGB);
        cv::resize(src,input,cv::Size(512,512));
        input.convertTo(input,CV_32FC3);
        input-=cv::Scalar_<float>(103.94f, 116.78f, 123.68f);
//        std::cout<<"ok"<<std::endl;
        torch::Tensor tensor_image = torch::from_blob(input.data, {1,input.rows, input.cols,3}, torch::kF32);
        tensor_image = tensor_image.permute({0,3,1,2});
//        tensor_image = tensor_image.toType(torch::kFloat);
        tensor_image = tensor_image.to(torch::kCUDA);
        std::vector<torch::IValue> batch;
        batch.push_back(tensor_image);
        auto result = module.forward(batch);
//        cout<<*result.type()<<endl;
        auto tuple=result.toTuple();
        auto bbox_cls=tuple->elements();
//        cout<<bbox_cls.size()<<" 0:"<<*bbox_cls[0].type()<<" 1:"<<*bbox_cls[1].type()<<endl;
        auto bbox_tuple = bbox_cls[0].toTensor().to(torch::kCPU);
        auto softmax = bbox_cls[1].toTensor().to(torch::kCPU);
//        auto bbox_shape = bbox_tuple.sizes();
//        auto softmax_shape = softmax.sizes();
        auto label = softmax.argmax({1});
        vector<int64> order;
        vector<Obj> detect_result;
        for (int64 i=0;i<label.size(0);i++) {
            if(*label[i].data_ptr<long>()!=0&&*softmax[i][label[i]].data_ptr<float>()>0.5)
            {
                Obj obj;
                obj.cls_id=*label[i].data_ptr<long>();
                obj.score=*softmax[i][label[i]].data_ptr<float>();
//                cout<<*label[i].data_ptr<long>()<<" "<<*softmax[i][label[i]].data_ptr<float>()<<endl;
                order.push_back(i);
                detect_result.push_back(obj);
            }
        }
        bbox_tuple = bbox_tuple.index_select(1,torch::from_blob(order.data(),{int64(order.size())},torch::kInt64));
//        cout<<bbox_tuple.sizes()<<endl;
//        softmax = softmax.index_select(0,torch::from_blob(order.data(),{int64(order.size())},torch::kInt64));
//        cout<<softmax.sizes()<<endl;
//        vector<float> shape={float(src.cols),float(src.rows),float(src.cols),float(src.rows)};
//        torch::Tensor image_shape=torch::from_blob(shape.data(),{int(shape.size())},torch::kFloat32);
        vector<vector<double>> priors;
        for (int i = 0; i < order.size(); ++i) {
            priors.push_back(mean[order[i]]);
        }
        vector<vector<double>> boxes(order.size(),vector<double>(4,-1));
        for (int i = 0; i < priors.size(); ++i) {
            boxes[i][0] = priors[i][0]+(*bbox_tuple[0][i][0].data_ptr<float>())*variance[0]*priors[i][3];
            boxes[i][1] = priors[i][1]+(*bbox_tuple[0][i][1].data_ptr<float>())*variance[0]*priors[i][4];
            boxes[i][2] = priors[i][2]*exp((*bbox_tuple[0][i][2].data_ptr<float>())*variance[1]);
            boxes[i][3] = priors[i][3]*exp((*bbox_tuple[0][i][3].data_ptr<float>())*variance[1]);
            boxes[i][0] -= (boxes[i][2]/2);
            boxes[i][1] -= (boxes[i][3]/2);
            boxes[i][2] += boxes[i][0];
            boxes[i][3] += boxes[i][1];
            boxes[i][0] *= src.cols;
            boxes[i][1] *= src.rows;
            boxes[i][2] *= src.cols;
            boxes[i][3] *= src.rows;
            detect_result[i].set_bbox(boxes[i]);
//            cout<<boxes[i][0]<<" "<<boxes[i][1]<<" "<<boxes[i][2]<<" "<<boxes[i][3]<<endl;
        }
        //nms(detect_result);
        for (int var = 0; var < detect_result.size(); ++var) {
            cv::rectangle(src,detect_result[var].bbox,cv::Scalar(0,0,255),2);
        }

//        cout<<bbox_tuple<<endl;
//        auto bbox_size=1;
//        for (int i =0;i<bbox_shape.size();i++) {
//            bbox_size*=bbox_shape[i];
//        }
//        vector<float> bbox_result(bbox_tuple.data_ptr<float>(),bbox_tuple.data_ptr<float>()+bbox_size);
//        auto softmax_size=1;
//        for (int i =0;i<softmax_shape.size();i++) {
//            softmax_size*=softmax_shape[i];
//        }
//        vector<float> softmax_result(softmax.data_ptr<float>(),softmax.data_ptr<float>()+softmax_size);
        cv::imshow("",src);
        if(cv::waitKey(1)==27)
            break;
    }
    cout << "Hello World!" << endl;
    return 0;
}
