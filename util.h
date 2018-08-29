#pragma once
#include "itensor/util/print_macro.h"
#include "itensor/mps/mps.h"
#include "itensor/mps/sites/spinhalf.h"
#include "mllib/mnist.h"

using namespace itensor;
using std::min;
using std::max;
using std::vector;
using std::array;
using std::move;
using std::min;

using MPSArr = array<vector<MPS>,10>;

const auto Label = itensor::IndexType("Label");

ITensor inline
toverlap(MPS const& psi,
         MPS const& img,
         long c = 1)
    {
    auto N = psi.N();
    auto W = img.A(N)*psi.A(N);
    for(auto j = N-1; j >= c; --j)
        {
        W *= (img.A(j)*psi.A(j));
        }
    if(c > 1)
        {
        auto WL = img.A(1)*psi.A(1);
        for(auto j = 2; j < c; ++j)
            {
            WL *= (img.A(j)*psi.A(j));
            }
        W *= WL;
        }
    return W;
    }

template<typename C>
long 
argmax(C const& c)
    {
    auto mel = c.front();
    auto mn = 0;
    for(auto n : range(c.size()))
        {
        if(c[n] > mel)
            {
            mel = c[n];
            mn = n;
            }
        }
    return mn;
    }
template<typename C>
long 
argmin(C const& c)
    {
    auto mel = c.front();
    auto mn = 0;
    for(auto n : range(c.size()))
        {
        if(c[n] < mel)
            {
            mel = c[n];
            mn = n;
            }
        }
    return mn;
    }


template<typename ImgType, typename Func>
MPS 
makeMPS(SiteSet const& sites,
        ImgType const& img,
        Func const& phi)
    {
    auto N = sites.N();
    if(N != (decltype(N)) img.size()) throw std::runtime_error("Mismatched sizes");
    auto d = sites(1).m();
    auto psi = MPS(sites);
    auto links = vector<Index>(N+4);
    for(auto b : range(N+1)) links.at(b) = Index("l",1);
    for(auto j : range1(N))
        {
        auto& ll = links.at(j-1);
        auto& rl = links.at(j);
        auto A = ITensor(ll,sites(j),rl);
        for(auto n : range1(d))
            {
            A.set(ll(1),sites(j)(n),rl(1),phi(img(j),n));
            }
        psi.setA(j,A);
        }
    psi.Anc(N) *= setElt(links.at(N)(1));
    psi.Anc(1) *= setElt(links.at(0)(1));
    return psi;
    }

template<typename ImgType>
ImgType const&
randImg(std::vector<ImgType> const& imgs,
        long label)
    {
    auto max_tries = 1000;
    for(auto t = 0; t < max_tries; ++t)
        {
        auto w = (long)imgs.size()*itensor::Global::random();
        if(w < 0) w = 0;
        if(w >= imgs.size()) w = imgs.size()-1;

        auto& img = imgs.at(w);
        if(img.label == label) return img;
        }
    Error(format("Did not find image with requested label after %d tries",max_tries));
    return imgs.front();
    }

template<size_t nlabel>
void
fullTest(MPS const& psi,
         MPSArr const& set_test,
         array<long,nlabel> labels)
    {
    Index L;
    long cent = 0;
    for(auto j : range1(psi.N()))
        {
        L = findtype(psi.A(j),Label);
        if(L) 
            {
            cent = j;
            break;
            }
        }
    if(!L || cent==0) Error("expected Label index at some site of psi MPS");
    auto counts = array<long,10>{};
    auto ninc = array<long,10>{};
    auto tninc = 0;
    auto tncor = 0;
    auto nte = 0;

    while(true)
        {
        bool got_state = false;
        for(auto l : labels)
            {
            auto c = counts[l];
            auto& testL = set_test.at(l);
            if(c >= (long)testL.size()) continue;
            counts[l]++;
            ++nte;
            got_state = true;
            auto& testimg = testL.at(c);
            auto W = toverlap(psi,testimg,cent);
            auto weights = array<Real,10>{};
                {
                //printf("L=%d weights",L);
                for(auto k : range(10))
                    {
                    //printf(" %.0E",std::fabs(W.real(wi(1+k))));
                    weights[k] = std::fabs(W.real(L(1+k)));
                    }
                //println();
                auto pl = argmax(weights);
                if(pl == l) 
                    {
                    //println("Correct");
                    ++tncor;
                    }
                else        
                    {
                    //println("Incorrect");
                    ++tninc;
                    ++ninc[l];
                    }
                //PAUSE
                }
            }
        if(!got_state) break;
        }
    printfln("%d/%d correct (%.2f%%), %d/%d incorrect (%.2f%%)",
             tncor,nte,tncor*100./nte,tninc,nte,tninc*100./nte);
    auto tot_counts = 0l;
    for(auto l : range(10))
        {
        auto nt = counts[l];
        tot_counts += nt;
        if(nt == 0) continue;
        auto ni = ninc[l];
        auto nc = nt-ni;
        printfln("  Digit %d %d/%d correct (%.2f%%), %d/%d incorrect (%.2f%%)",
                 l,nc,nt,nc*100./nt,ni,nt,ni*100./nt);
        }
    printfln("Total # test images = %d",tot_counts);
    }

void inline
movePos(MPS & psi,
        long np,
        Args const& args = Args::global())
    {
    auto N = psi.N();
    if(np < 1 || np > N) Error("New position out of range");
    Index L;
    long p = 0;
    for(auto j : range1(N))
        {
        L = findtype(psi.A(j),Label);
        if(L) 
            {
            p = j;
            break;
            }
        if(j == N) Error("Expected psi to have Label-type Index");
        }
    //printfln("Found o.c. at position %d",p);
    auto moveBy = [&](long dp)
        {
        auto AA = psi.A(p)*psi.A(p+dp);
        ITensor U,S,V;
        if(1 < p && p < N) U = ITensor(findtype(psi.A(p),Site),uniqueIndex(psi.A(p),psi.A(p+dp),Link));
        else               U = ITensor(findtype(psi.A(p),Site));
        svd(AA,U,S,V,args);
        psi.setA(p,U);
        psi.setA(p+dp,S*V);
        psi.Aref(p+dp) *= norm(AA)/norm(psi.A(p+dp));
        if(!findtype(psi.A(p+dp),Label)) Error("No Label index on new o.c.");
        };
    while(np > p)
        {
        moveBy(+1);
        ++p;
        }
    while(np < p)
        {
        moveBy(-1);
        --p;
        }
    }
