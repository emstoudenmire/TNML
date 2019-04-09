#include "single.h"
#include "itensor/util/input.h"

const size_t NL = 10;

int 
main(int argc, char* argv[])
    {
    setOneThread();

    if(argc != 2) 
       { 
       printfln("Usage: %s inputfile",argv[0]); 
       return 0; 
       }
    auto input = InputGroup(argv[1],"input");

    auto datadir = input.getString("datadir","/Users/mstoudenmire/software/tnml/mllib/MNIST");
    const auto L = input.getInt("label",0);
    auto Ntrain = input.getInt("Ntrain",60000);
    auto Nsweep = input.getInt("Nsweep",50);
    auto cutoff = input.getReal("cutoff",1E-8);
    auto maxm = input.getInt("maxm",5000);
    auto minm = input.getInt("minm",max(10,maxm/2));
    auto noise = input.getReal("noise",0.);
    auto ninitial = input.getInt("ninitial",100);
    auto Nthread = input.getInt("nthread",4);
    auto pause_steps = input.getYesNo("pause_steps",false);
    auto feature = input.getString("feature","normal");

    enum Feature { Normal, Series };
    auto ftype = Normal;
    if(feature == "normal") { ftype = Normal; }
    else if(feature == "series") { ftype = Series; }
    else
        {
        Error(format("feature=%s not recognized",feature));
        }

    //Cost function settings
    auto lambda = input.getReal("lambda",0.);

    //Gradient settings
    auto method = input.getString("method","conj");
    auto alpha = input.getReal("alpha",1.0);
    auto clip = input.getReal("clip",1.0);
    auto Npass = input.getInt("Npass",4);
    auto cconv = input.getReal("cconv",1E-10);
    auto Ntarget = input.getInt("Ntarget",10);
    auto pcut = input.getReal("pcut",1E-8);
    auto precalc = input.getYesNo("precalc",true);

    auto Wname = format("W%d",L);
    auto labels = array<long,NL>{{0,1,2,3,4,5,6,7,8,9}};

    auto train = readMNIST(datadir,mllib::Train,{"NT=",Ntrain});

    auto N = train.front().size();
    printfln("%d sites",N);
    SpinHalf sites;
    if(fileExists("sites") )
        {
        sites = readFromFile<SpinHalf>("sites");
        }
    else
        {
        sites = SpinHalf(N);
        writeToFile("sites",sites);
        }

    auto phi = [ftype](Real g, int n) -> Cplx
        {
        if(g < 0 || g > 255.) Error(format("Expected g=%f to be in [0,255]",g));
        auto x = g/255.;
        if(ftype == Normal)
            {
            return n==1 ? cos(Pi/2.*x) : sin(Pi/2.*x);
            }
        else if(ftype == Series)
            {
            return n==1 ? 1. : x/4.;
            }
        return 0.;
        };

    println("Converting training set to MPS");
    auto totL = 0l;
    auto trainmps = MPSArr{};
    auto counts = array<int,10>{};
    for(auto& img : train)
        {
        auto l = img.label;
        if(counts[l] >= Ntrain) continue;
        trainmps.at(l).push_back(makeMPS(sites,img,phi));
        ++counts[l];
        if(l == L) ++totL;
        }
    auto totNtrain = stdx::accumulate(counts,0);

    printfln("Total of %d training images",totNtrain);
    printfln("%d training images with selected label L=%d",totL,L);


    MPS W;
    if(fileExists(Wname))
        {
        printfln("Reading %s from file",Wname);
        W = readFromFile<MPS>(Wname,sites);
        }
    else
        {
        //println("Making random initial state");
        W = MPS(sites);
        auto psis = vector<MPS>(ninitial);
        for(auto m : range(ninitial)) 
            {
            psis.at(m) = makeMPS(sites,randImg(train,labels.at(L)),phi);
            }
        printfln("Summing %d random label %d states",ninitial,labels.at(L));
        W = sum(psis,{"Cutoff",1E-10,"Maxm",10});
        W.orthogonalize();
        W.Aref(1) /= norm(W.A(1));
        }
    W.position(1,{"Cutoff",cutoff,"Maxm",maxm});
    println("Done making initial W");

    //
    // Setup parallel worker
    //
    auto bounds = vector<Bound>(Nthread);
    auto th_size = totNtrain/Nthread;
    auto bcount = 0;
    for(auto n : range(Nthread))
        {
        bounds.at(n) = Bound(n,bcount,bcount+th_size);
        bcount += th_size;
        }
    bounds.back().end = totNtrain;
    for(auto& b : bounds)
        {
        printfln("Thread %d %d -> %d (%d)",b.n,b.begin,b.end,b.end-b.begin);
        }
    auto parallel_do = ParallelDo{bounds};


    //
    // Setup ts
    //
    auto ts = vector<TState>(totNtrain);
    auto nextLabel = [&labels](long start_at = -1)
        {
        static long nl = 0;
        auto l = labels.at(nl);
        if(start_at >= 0) { nl = start_at; return l; }
        nl = (nl+1==(long)labels.size()) ? 0 : nl+1;
        return l;
        };
    auto nextTrainN = [&trainmps](long l)
        {
        static auto ncount = array<size_t,10>{};
        auto& set = trainmps.at(l);
        auto& cl = ncount.at(l);
        if(cl >= set.size()) return -1ul;
        return cl++;
        };

    for(auto& t : ts)
        {
        long count = 0;
        while(t.n == -1)
            {
            t.l = nextLabel();
            t.n = nextTrainN(t.l);
            if(++count > 20) Error("Infinite loop while setting up ts");
            }
        }

    print("Projecting training states...");
    parallel_do(
        [&](Bound b)
            {
            for(auto i = b.begin; i < b.end; ++i)
                {
                auto& t = ts.at(i);
                auto& tmps = *getTrainState(trainmps,t);
                auto& E = t.E;
                E.resize(N+2);
                E.at(N) = tmps.A(N)*W.A(N);
                for(auto j = N-1; j >= 3; --j)
                    {
                    E.at(j) = (tmps.A(j)*W.A(j))*E.at(j+1);
                    E.at(j).scaleTo(1.);
                    }
                }
            }
        );
    println("done");


    if(precalc)
        {
        parallel_do(
            [&](Bound b)
                {
                for(auto i = b.begin; i < b.end; ++i)
                    {
                    auto& t = ts.at(i);
                    auto& tmps = *getTrainState(trainmps,t);
                    t.v = tmps.A(1)*tmps.A(2);
                    t.v *= t.E.at(3);
                    }
                }
            );
        }
    auto C = quadcost(W.A(1)*W.A(2),ts,trainmps,L,parallel_do,{"lambda",lambda,"Precalc",precalc,"LC",0});
    printfln("Before DMRG, Cost = %.10f",C/Ntrain);

    auto sweeps = Sweeps(Nsweep);
    sweeps.maxm() = maxm;
    sweeps.cutoff() = cutoff;
    sweeps.minm() = minm;
    sweeps.noise() = noise;

    auto args = Args{"Label",L,
                     "lambda",lambda,
                     "Wname",Wname,
                     "Method",method,
                     "Npass",Npass,
                     "Ntarget",Ntarget,
                     "PCut",pcut,
                     "alpha",alpha,
                     "clip",clip,
                     "cconv",cconv,
                     "PauseSteps",pause_steps,
                     "Precalc",precalc
                    };

    mldmrg(W,trainmps,ts,sweeps,parallel_do,args);

    printfln("Writing %s to disk",Wname);
    writeToFile(Wname,W);

    return 0;
    }
