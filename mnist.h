#pragma once
#include "CppMNIST/include/mnist/mnist_reader.hpp"
#include "image.h"


auto inline
getData()
    {
    auto home = getenv("HOME");
    return mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(0,0,
            itensor::format("%s/software/mpsml/CppMNIST",home).c_str());
    }

//DataSet == MNIST_dataset
using DataSet = stdx::decay_t<decltype(getData())>;
//ImagesT == vector<vector<uint8_t>>
using ImagesT = stdx::decay_t<decltype(getData().training_images)>;
//LabelsT == vector<uint8_t>
using LabelsT = stdx::decay_t<decltype(getData().training_labels)>;
//Img == vector<uint8_t> ?
using Img = stdx::decay_t<decltype(getData().training_images[0])>;



class MNISTRef
    {
    ImagesT const* I_ = nullptr;
    LabelsT const* L_ = nullptr;
    long num_ = -1;
    ImageType type_ = None;
    public:

    using value_type = const Pix;

    MNISTRef() { }
    
    MNISTRef(DataSet const& D,
             ImageType type,
             long num)
      : I_(&((type==Train) ? D.training_images : D.test_images)), 
        L_(&((type==Train) ? D.training_labels : D.test_labels)),
        num_(num),
        type_(type)
        { 
        assert(num_ >= 0);
        }

    explicit operator bool() const { return I_!=nullptr && L_!=nullptr; }

    size_t
    size() const { return I_->at(num_).size(); }

    long
    label() const { return L_->at(num_); }

    Pix
    operator[](size_t n) const { return I_->at(num_).at(n); }

    long
    num() const { return num_; }

    ImageType
    type() const { return type_; }

    size_t
    width() const { return 28ul; }

    size_t
    height() const { return 28ul; }

    auto
    begin() -> decltype(I_->at(num_).begin()) const { return I_->at(num_).begin(); }

    auto
    end() -> decltype(I_->at(num_).end()) const { return I_->at(num_).end(); }

    Pix const*
    data() const { return I_->at(num_).data(); }
    };

using MNImage = ImageT<MNISTRef>;
using MNImageSet = std::vector<MNImage>;
using PImage = ImageT<Pixels>;
using PImgSet = std::vector<PImage>;
//PImgColl: collection of PImgSets, one for each label 
using PImgColl = std::array<PImgSet,10>;


class Images
    {
    DataSet const* D_ = nullptr;
    ImageType type_ = None;
    long Nimgs_ = 0;

    class ImgIter
        {
        DataSet const* D_ = nullptr;
        ImageType type_ = None;
        long num_ = 0;
        public:
        ImgIter(DataSet const* D,
                ImageType type,
                long num = 0)
            : D_(D),type_(type),num_(num)
            { }
        MNImage
        operator*() const { return MNImage(*D_,type_,num_); }
        void
        operator++() { ++num_; }
        bool
        operator==(ImgIter const& O) const { return O.num_==num_; }
        bool
        operator!=(ImgIter const& O) const { return O.num_!=num_; }
        };
    public:
    using iterator = ImgIter;
    using const_iterator = ImgIter;

    Images(DataSet const& D,
           ImageType type)
      : D_(&D),
        type_(type)
        { 
            {
            if(type_ == Train)
                {
                Nimgs_ = D.training_images.size();
                }
            else
            if(type_ == Test)
                {
                Nimgs_ = D.test_images.size();
                }
            }
        }

    long
    size() const { return Nimgs_; }

    const_iterator
    begin() const { return ImgIter(D_,type_,0); }
    const_iterator
    end() const { return ImgIter(D_,type_,Nimgs_); }
    };

MNImage inline
getImg(DataSet const& data,
       Args const& args = Global::args())
    {
    auto w = args.getInt("Num",-1);
    auto L = args.getInt("Label",-1);
    auto type = imageType(args.getInt("Type",Train));
    if(L < -1 || L > 9) throw std::runtime_error(itensor::format("label %d out of range",L).c_str());
    ImgLabel ll = -2;

    auto& iset = (type==Train) ? data.training_images : data.test_images;
    auto& lset = (type==Train) ? data.training_labels : data.test_labels;

    if(w == -1)
        {
        long nimg = iset.size();
        do
            {
            //Grab a random image
            w = (long)nimg*itensor::Global::random();
            if(w < 0) w = 0;
            if(w >= nimg) w = nimg-1;
            ll = lset.at(w);
            }
        while(ll != L && L != -1);
        }

    return MNImage(data,type,w);
    }

long inline
binarize(long pix, long thresh=100)
    { 
    return (pix < thresh) ? 0l : 1l;
    }

template<typename S>
void
show(ImageT<S> const& img,
     Args const& args = Global::args())
    {
    using itensor::print;
    using itensor::println;
    using itensor::printfln;
    auto thresh = args.getInt("Thresh",150);
    long rowc = 1;
    long colc = 1;
    for(auto& el : img)
        {
        if(colc == 1 || colc == 28) print("| ");
        else if(rowc == 1 || rowc == 28) print("--");
        else 
            {
            if(binarize(el,thresh)) print("X ");
            else                    print("  ");
            }
        ++colc;
        if(colc > 28) 
            {
            println();
            colc = 1;
            ++rowc;
            }
        }
    println();
    printfln("label = %d, num = %d",img.label(),img.num());
    }



PImgSet inline
getAllMNIST(DataSet const& D,
       Args const& args = Args::global())
    {
    auto type = imageType(args.getInt("Type",Train));
    auto imglen = args.getInt("imglen",28);

    //auto orig_images = std::array<ImageSet,10>{};
    //for(auto I : Images(D,type))
    //    {
    //    orig_images.at(I.label()).push_back(I);
    //    }

    auto images = Images(D,type);
    auto scal_images = PImgSet(images.size());
    auto n = 0;
    for(auto I : Images(D,type))
        {
        scal_images.at(n) = reduce<Pixels>(I,imglen);
        ++n;
        }
    return scal_images;
    }

