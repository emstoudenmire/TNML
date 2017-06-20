#pragma once
#include "png.hpp"
#include "itensor/util/stdx.h"
#include "itensor/global.h"
#include "itensor/util/args.h"
#include "itensor/indextype.h"

using std::array;

using itensor::range;
using itensor::Args;
using itensor::Global;
using itensor::printfln;
using itensor::Real;
using itensor::Cplx;
using itensor::Pi;
using itensor::sqr;

using Pix = uint8_t;
using ImgLabel = long;

class RGBPixel
    {
    array<Real,3> rgb;
    public:

    RGBPixel() : rgb{} { }

    RGBPixel(Real r, Real g, Real b) : rgb{{r,g,b}} { }

    Real
    operator[](size_t n) const { return rgb[n]; }

    Real &
    operator[](size_t n) { return rgb[n]; }

    size_t
    size() const { return 3ul; }

    RGBPixel&
    operator/=(Real x)
        {
        rgb[0] /= x;
        rgb[1] /= x;
        rgb[2] /= x;
        return *this;
        }

    RGBPixel&
    operator*=(Real x)
        {
        rgb[0] *= x;
        rgb[1] *= x;
        rgb[2] *= x;
        return *this;
        }

    RGBPixel&
    operator+=(RGBPixel p)
        {
        rgb[0] += p.rgb[0];
        rgb[1] += p.rgb[1];
        rgb[2] += p.rgb[2];
        return *this;
        }
    };

RGBPixel inline
operator+(RGBPixel p1, RGBPixel const& p2) { return (p1+=p2); }


enum ImageType { None, Train, Test};

ImageType inline
imageType(long t) { return (t==(long)Train) ? Train : Test; }

template<typename PixT>
class ImgStore
    {
    std::vector<PixT> pix_;
    size_t nrows_ = 0;
    size_t ncols_ = 0;
    long num_ = -1;
    ImageType type_ = None;
    long label_ = -1;
    public:

    using value_type = PixT;

    ImgStore() { }

    ImgStore(size_t nrows,
             size_t ncols,
             ImageType type = None,
             long num = -1,
             long label = -1)
      : pix_(nrows*ncols),
        nrows_(nrows),
        ncols_(ncols),
        num_(num),
        type_(type),
        label_(label)
        { }

    ImgStore(size_t len,
           ImageType type = None,
           long num = -1,
           long label = -1)
      : pix_(len*len),
        nrows_(len),
        ncols_(len),
        num_(num),
        type_(type),
        label_(label)
        { }

    explicit operator bool() const { return pix_.empty(); }

    size_t
    size() const { return pix_.size(); }

    long
    label() const { return label_; }

    value_type const&
    operator[](size_t n) const { return pix_.at(n); }

    value_type&
    operator[](size_t n) { return pix_.at(n); }

    long
    num() const { return num_; }

    ImageType
    type() const { return type_; }

    size_t
    width() const { return nrows_; }

    size_t
    height() const { return ncols_; }

    auto
    begin() -> decltype(pix_.begin()) const { return pix_.begin(); }

    auto
    end() -> decltype(pix_.end()) const { return pix_.end(); }

    auto
    data() -> decltype(pix_.data()) const { return pix_.data(); }
    };

using Pixels = ImgStore<Pix>;
using RGBStore = ImgStore<RGBPixel>;


template<typename Storage>
class ImageT
    {
    private:
    Storage store_;
    public:
    using value_type = stdx::remove_const_t<typename Storage::value_type>;

    ImageT() { }

    template<typename... SArgs>
    ImageT(SArgs&&... sargs)
      : store_(std::forward<SArgs>(sargs)...)
        { 
        assert(type()==Train || type()==Test);
        }

    explicit operator bool() const { return store_; }

    size_t
    size() const { return store_.size(); }

    size_t
    width() const { return store_.width(); }

    size_t
    height() const { return store_.height(); }

    //0-indexed
    value_type
    operator[](size_t n) const { return store_[n]; }

    //1-indexed
    value_type
    operator()(long n) const { return store_[n-1]; }

    //0-indexed
    value_type
    pixel(size_t x, size_t y) const
        {
        return store_[x+width()*y];
        }

    //template<typename V>
    //auto
    //set(size_t x, size_t y, V const& v)
    //    -> stdx::if_compiles_return<void,decltype(store_[0]=value_type(v))>
    //    {
    //    store_[x+width()*y] = v;
    //    }
      
    void
    set(size_t x, size_t y, value_type v)
        {
        store_[x+width()*y] = v;
        }

    ImgLabel
    label() const { return store_.label(); }

    long
    num() const { return store_.num(); }

    ImageType
    type() const { return store_.type(); }

    auto
    begin() -> decltype(store_.begin()) const { return store_.begin(); }

    auto
    end() -> decltype(store_.end()) const { return store_.end(); }

    auto
    data() -> decltype(store_.data()) const { return store_.data(); }
    };

using RGBImage = ImageT<RGBStore>;

template<typename N, typename S>
ImageT<N>
resize(ImageT<S> const& oimg,
       long newlen,
       bool debug = false)
    { 
    using Sizet = decltype(oimg.width());
    using PixT = decltype(oimg.pixel(0,0));
    assert(oimg.width()==oimg.height());
    long oldlen = oimg.width();

    auto nimg = ImageT<N>(newlen,oimg.type(),oimg.num(),oimg.label());

    if(oldlen < newlen)
        {
        Sizet pad = (newlen-oldlen)/2;
        for(auto nx : range(nimg.width()))
        for(auto ny : range(nimg.height()))
            {
            if(nx < pad || ny < pad ||
               nx >= pad+oldlen || ny >= pad+oldlen)
                {
                nimg.set(nx,ny,PixT{});
                }
            else
                {
                auto ox = nx-pad;
                auto oy = ny-pad;
                nimg.set(nx,ny,oimg.pixel(ox,oy));
                }
            }
        }
    else if(oldlen > newlen)
        {
        //Credit Mark Ransom for posting algorithm
        //below on Stack Overflow
		Real scale = (1.*newlen)/oldlen;
		PixT threshold = 0.5/(scale*scale);
		Real yend = 0.0;
		for(int f = 0; f < newlen; ++f) // y on output
			{
			Real ystart = yend;
			yend = (f+1)/scale;
			if(yend >= oldlen) yend = oldlen - 0.000001;
			Real xend = 0.0;
			for(int g = 0; g < newlen; ++g) // x on output
				{
				Real xstart = xend;
				xend = (g+1)/scale;
				if(xend >= oldlen) xend = oldlen - 0.000001;
				auto sum = PixT{};
				for(int y = (int)ystart; y <= (int)yend; ++y)
					{
					Real yportion = 1.0;
					if(y == (int)ystart) yportion -= ystart - y;
					if(y == (int)yend) yportion -= y+1 - yend;
					for(int x = (int)xstart; x <= (int)xend; ++x)
						{
						Real xportion = 1.0;
						if(x == (int)xstart) xportion -= xstart - x;
						if(x == (int)xend) xportion -= x+1 - xend;
						sum += oimg.pixel(x,y) * yportion * xportion;
						}
                    }
				//auto val = (sum > threshold) ? 255 : 0;
				nimg.set(g,f,sum);
                }
            }
        }
    else //oldlen == newlen
        {
        assert(oldlen == newlen);
        for(auto x : range(oimg.width()))
        for(auto y : range(oimg.height()))
            {
            nimg.set(x,y,oimg.pixel(x,y));
            }
        }
    return nimg;
    }

template<typename N, typename S>
ImageT<N>
reduce(ImageT<S> const& oimg,
       long newlen) 
    { 
    using PixT = decltype(oimg.pixel(0,0));
    assert(oimg.width()==oimg.height());
    auto bsize = oimg.width()/newlen;
    auto rem = oimg.width()%bsize;
    auto nimg = ImageT<N>(newlen,oimg.type(),oimg.num(),oimg.label());
    
    for(auto nx : range(nimg.width()))
    for(auto ny : range(nimg.height()))
        {
        auto avg = PixT{};
        long cnt = 0;
        auto oxs = rem+bsize*nx;
        auto oys = rem+bsize*ny;
        //printfln("nx,ny = %d,%d",nx,ny);
        for(auto ox : range(oxs,oxs+bsize))
        for(auto oy : range(oys,oys+bsize))
            {
            //printfln("  ox,oy = %d,%d",ox,oy);
            avg += oimg.pixel(ox,oy);
            ++cnt;
            }
        avg /= cnt;
        nimg.set(nx,ny,avg);
        }
    return nimg;
    }

template<class S>
void
writeGray(ImageT<S> const& image,
      std::string fname)
    {
    auto I = png::image<png::gray_pixel>(image.width(),image.height());
    for(auto y : range(I.get_height()))
    for(auto x : range(I.get_width()))
        {
        I[y][x] = png::gray_pixel((255-image.pixel(x,y)));
        }
    I.write(fname.c_str());
    }

void inline
writeColor(ImageT<RGBStore> const& image,
           std::string fname)
    {
    auto I = png::image<png::rgb_pixel>(image.width(),image.height());
    for(auto y : range(I.get_height()))
    for(auto x : range(I.get_width()))
        {
        auto p = image.pixel(x,y);
        I[y][x] = png::rgb_pixel(p[0],p[1],p[2]);
        }
    I.write(fname.c_str());
    }
