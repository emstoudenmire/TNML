#pragma once
#include <future>
#include "util.h"


//Represents a range of integers, enumerating 
//the training data handled by a given thread
struct Bound
    {
    size_t n = 0;
    size_t begin = 0;
    size_t end = 0;
    Bound() { }
    Bound(size_t nn, size_t b, size_t e) : n(nn),begin(b), end(e) { } 
    };

//Function object for doing parallel tasks
class ParallelDo
    {
    vector<Bound> bounds;
    public:

    ParallelDo(vector<Bound> const& bs)
      : bounds(bs)
        { }

    size_t
    Nthread() const { return bounds.size(); }

    template<typename Task>
    void
    operator()(Task&& T) const
        {
        const auto Nf = 16ul;
        if(bounds.size() > Nf) Error("Need to increase size of futs");
        auto futs = std::array<std::future<void>,Nf>{};
        auto task = move(T);
        for(auto& b : bounds)
            {
            futs.at(b.n) = std::async(std::launch::async,task,b);
            }
        for(auto& b : bounds)
            {
            futs.at(b.n).wait();
            }
        }
    };

void inline
setOneThread()
    {
    setenv("VECLIB_NUM_THREADS","1",1);
    auto vnt = getenv("VECLIB_NUM_THREADS");
    if(vnt != NULL)
        println("pdmrg: VECLIB_NUM_THREADS = ",vnt);
    else
        println("pdmrg: OMP_NUM_THREADS not defined");

    setenv("OMP_NUM_THREADS","1",1);
    auto ont = getenv("OMP_NUM_THREADS");
    if(ont != NULL)
        println("pdmrg: OMP_NUM_THREADS = ",ont);
    else
        println("pdmrg: OMP_NUM_THREADS not defined");

    setenv("MKL_NUM_THREADS","1",1);
    auto mnt = getenv("MKL_NUM_THREADS");
    if(mnt != NULL)
        println("pdmrg: MKL_NUM_THREADS = ",mnt);
    else
        println("pdmrg: MKL_NUM_THREADS not defined");

    setenv("GOTO_NUM_THREADS","1",1);
    auto gnt = getenv("GOTO_NUM_THREADS");
    if(gnt != NULL)
        println("pdmrg: GOTO_NUM_THREADS = ",gnt);
    else
        println("pdmrg: GOTO_NUM_THREADS not defined");
    }
