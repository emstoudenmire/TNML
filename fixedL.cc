#include <future>
#include "util.h"
#include "paralleldo.h"
#include "itensor/util/input.h"
#include "itensor/mps/sweeps.h"
#include "itensor/util/print_macro.h"

using namespace itensor;
using std::vector;
using std::array;
using std::move;
using std::min;

const size_t NL = 10;

//Struct holding info about training "states"
struct TState
    {
    long n = -1;
    int l = NL;
    ITensor v;
    vector<ITensor> E;
    };

MPS*
getTrainState(MPSArr & trainmps,
              TState const& t) 
    { 
    return &(trainmps.at(t.l).at(t.n)); 
    }

//
// Compute squared distance of the actual output
// of the model from the ideal output
//
Real
quadcost(ITensor B,
         vector<TState> const& ts,
         ParallelDo const& parallel_do,
         Args const& args = Args::global())
    {
    auto NT = ts.size();
    auto lambda = args.getReal("lambda",0.);
    auto showlabels = args.getBool("ShowLabels",false);

    auto L = findtype(B,Label);
    if(!L) L = findtype(ts.front().v,Label);
    if(!L) Error("Couldn't find Label index in quadcost");

    if(args.getBool("Normalize",false))
        {
        B /= norm(B);
        }

    //
    //Set up containers for multithreaded calculations
    auto deltas = array<ITensor,10>{};
    for(auto l : range(10)) deltas[l] = setElt(L(1+l));
    auto reals = array<vector<Real>,10ul>{};
    for(auto l : range(10))
        {
        reals[l] = vector<Real>(parallel_do.Nthread(),0.);
        }
    auto ints = vector<int>(parallel_do.Nthread(),0);
    //

    parallel_do(
        [&](Bound b)
            {
            auto weights = array<Real,10>{};
            for(auto i = b.begin; i < b.end; ++i)
                {
                auto& t = ts.at(i);
                // P is the model output
                auto P = B*t.v;
                // deltas[t.l] is the ideal output vector
                // (one-hot encoding) for the training state t
                auto dP = deltas[t.l] - P;
                // cost is square of dP
                reals[t.l].at(b.n) += sqr(norm(dP));
                // save all outputs to check if t correctly classified
                for(auto k : range(10))
                    {
                    weights[k] = std::abs(P.real(L(1+k)));
                    }
                if(t.l == argmax(weights)) ints[b.n] += 1;
                }
            }
        );
    auto CR = lambda*sqr(norm(B));
    auto C = 0.;
    for(auto l : range(10))
        {
        auto CL = stdx::accumulate(reals[l],0.);
        if(showlabels) printfln("  Label l=%d C%d = %.10f",l,l,CL/NT);
        C += CL;
        }
    if(showlabels) printfln("  Reg. cost CR = %.10f",CR/NT);
    C += CR;

    auto ncor = stdx::accumulate(ints,0);
    auto ninc = (ts.size()-ncor);
    printfln("Percent correct = %.4f%%, # incorrect = %d/%d",ncor*100./ts.size(),ninc,ncor+ninc);

    return C;
    }

//
// Conjugate gradient
//
void
cgrad(ITensor & B,
      vector<TState> & ts,
      ParallelDo const& parallel_do, 
      Args const& args)
    {
    auto NT = ts.size();
    auto Npass = args.getInt("Npass");
    auto lambda = args.getReal("lambda",0.);
    auto cconv = args.getReal("cconv",1E-10);

    auto L = findtype(B,Label);
    if(!L) L = findtype(ts.front().v,Label);
    if(!L) Error("Couldn't find Label index in cgrad");

    auto deltas = array<ITensor,10>{};
    for(auto l : range(10)) deltas[l] = setElt(L(1+l));

    //Workspace for parallel ops
    auto Nthread = parallel_do.Nthread();
    auto tensors = vector<ITensor>(Nthread);
    auto reals = vector<Real>(Nthread);
    auto ints = vector<int>(Nthread);

    // Compute initial gradient
    for(auto& T : tensors) T = ITensor{};
    parallel_do(
        [&](Bound b)
            {
            for(auto i = b.begin; i < b.end; ++i)
                {
                auto& t = ts.at(i);
                auto P = B*t.v;
                auto dP = deltas[t.l] - P;
                tensors.at(b.n) += dP*dag(t.v);
                }
            }
        );
    auto r = stdx::accumulate(tensors,ITensor{});
    if(lambda != 0.) r = r - lambda*B;

    auto p = r;
    for(auto pass : range1(Npass))
        {
        println("  Conj grad pass ",pass);
        // Compute p*A*p
        for(auto& r : reals) r = 0.;
        parallel_do(
            [&](Bound b)
                {
                for(auto i = b.begin; i < b.end; ++i)
                    {
                    auto& t = ts.at(i);
                    //
                    // The matrix A is like outer
                    // product of dag(v) and v, so
                    // dag(p)*A*p is |p*v|^2
                    // 
                    auto pv = p*t.v;
                    reals.at(b.n) += sqr(norm(pv));
                    }
                }
            );
        auto pAp = stdx::accumulate(reals,0.);
        pAp += lambda*sqr(norm(p));

        auto a = sqr(norm(r))/pAp;
        B = B + a*p;
        B.scaleTo(1.);

        if(pass == Npass) break;

        // Compute new gradient and cost function
        for(auto& T : tensors) T = ITensor();
        for(auto& r : reals) r = 0.;
        parallel_do(
            [&](Bound b)
                {
                for(auto i = b.begin; i < b.end; ++i)
                    {
                    auto& t = ts.at(i);
                    auto P = B*t.v;
                    auto dP = deltas[t.l] - P;
                    tensors.at(b.n) += dP*dag(t.v);
                    reals.at(b.n) += sqr(norm(dP));
                    }
                }
            );
        auto nr = stdx::accumulate(tensors,ITensor{});
        if(lambda != 0.) nr = nr - lambda*B;
        auto beta = sqr(norm(nr)/norm(r));
        r = nr;
        r.scaleTo(1.);

        auto C = stdx::accumulate(reals,0.);
        C += lambda*sqr(norm(B));
        printfln("  Cost = %.10f",C/NT);

        // Quit if gradient gets too small
        if(norm(r) < cconv) 
            {
            printfln("  |r| = %.1E < %.1E, breaking",norm(r),cconv);
            break;
            }
        else
            {
            printfln("  |r| = %.1E",norm(r));
            }

        p = r + beta*p;
        p.scaleTo(1.);
        }
    }


//         
// M.L. DMRG
//         
void
mldmrg(MPS & W,
       MPSArr & trainmps,
       vector<TState> & ts,
       Sweeps const& sweeps,
       ParallelDo const& parallel_do,
       Args const& args)
    {
    auto N = W.N();
    auto NT = ts.size();

    auto method = args.getString("Method");
    auto replace = args.getBool("Replace",false);

    auto Nthread = parallel_do.Nthread();
    auto reals = vector<Real>(Nthread);

    auto cargs = Args{args,"Normalize",false};

    // For loop over sweeps of the MPS
    for(auto sw : range1(sweeps))
    {
    printfln("\nSweep %d maxm=%d minm=%d",sw,sweeps.maxm(sw),sweeps.minm(sw));
    auto svd_args = Args{"Cutoff",sweeps.cutoff(sw),
                         "Maxm",sweeps.maxm(sw),
                         "Minm",sweeps.minm(sw),
                         "Sweep",sw};
    // Loop over individual bonds of the MPS
    for(int b = 1, ha = 1; ha <= 2; sweepnext(b,ha,N))
        {
        // c and c+dc are j,j+1 if sweeping right
        // if sweeping left they are j,j-1
        auto c = (ha==1) ? b : b+1;
        auto dc = (ha==1) ? +1 : -1;

        auto lc = min(c,c+dc)-1;
        auto rc = max(c,c+dc)+1;

        printfln("Sweep %d Half %d Bond %d",sw,ha,c);

        // Save old bond tensor
        auto origm = commonIndex(W.A(c),W.A(c+dc)).m();
        auto oB = W.A(c)*W.A(c+dc);

        // B is the bond tensor we will optimize
        auto B = oB;
        B.scaleTo(1.);

        //
        // Make effective image (4 site) tensors
        // Store in t.v of each elem t of ts
        //
        parallel_do(
            [&](Bound b)
                {
                for(auto i = b.begin; i < b.end; ++i)
                    {
                    auto& t = ts.at(i);
                    auto& tmps = *getTrainState(trainmps,t);
                    t.v = tmps.A(c)*tmps.A(c+dc);
                    if(lc > 0)   t.v *= t.E.at(lc);
                    if(rc < N+1) t.v *= t.E.at(rc);
                    }
                }
            );

        //
        // Optimize bond tensor B
        //
        if(method == "conj") cgrad(B,ts,parallel_do,args);
        else Error(format("method type \"%s\" not recognized",method));

        //
        // Report cost after optimization
        //
        printfln("Sweep %d Half %d Bond %d",sw,ha,c);

        auto oC = quadcost(oB,ts,parallel_do,cargs);
        auto C = quadcost(B,ts,parallel_do,cargs);
        printfln("Cost = %.10f -> %.10f",oC/NT,C/NT);

        //
        // SVD B back apart into MPS tensors
        //
        ITensor S;
        auto spec = svd(B,W.Aref(c),S,W.Aref(c+dc),svd_args);
        W.Aref(c+dc) *= S;
        auto newm = commonIndex(W.A(c),W.A(c+dc)).m();
        printfln("SVD trunc err = %.2E",spec.truncerr());

        printfln("Original m=%d, New m=%d",origm,newm);

        auto newB = W.A(c)*W.A(c+dc);
        Print(norm(newB));
        printfln("rank(newB) = %d",rank(newB));

        auto newC = quadcost(newB,ts,parallel_do,{cargs,"ShowLabels",true});
        printfln("--> After SVD, Cost = %.10f",newC/NT);

        if(replace)
            {
            if(newC > oC)
                {
                //println(" == New C is higher, using old B == ");
                //PAUSE
                auto spec = svd(oB,W.Aref(c),S,W.Aref(c+dc),svd_args);
                W.Aref(c+dc) *= S;
                newB = W.A(c)*W.A(c+dc);
                newC = quadcost(newB,ts,parallel_do,{cargs,"ShowLabels",true});
                printfln("--> After replacement, Cost = %.10f",newC/NT);
                }
            }

        //
        // Update E's (MPS environment tensors)
        // i.e. projection of training images into current "wings"
        // of the MPS W
        //
        parallel_do(
            [&](Bound b)
            {
            for(auto i = b.begin; i < b.end; ++i)
                {
                auto& t = ts.at(i);
                auto& tmps = *getTrainState(trainmps,t);
                auto& E = t.E;
                if(c == 1 || c == N) 
                    {
                    E.at(c) = tmps.A(c)*W.A(c);
                    }
                else
                    {
                    E.at(c) = E.at(c-dc)*(tmps.A(c)*W.A(c));
                    }
                E.at(c).scaleTo(1.);
                }
            });

        if(fileExists("WRITE_WF"))
            {
            println("File WRITE_WF found");
            system("rm -f WRITE_WF");
            println("Writing W to disk");
            writeToFile("W",W);
            }

        } //loop over c,dc

    println("Writing W to disk");
    writeToFile("W",W);

    } //loop over sweeps

    } //mldmrg


int 
main(int argc, const char* argv[])
    {
    // Set environment variables to use 1 thread
    setOneThread();

    if(argc != 2) 
       { 
       printfln("Usage: %s inputfile",argv[0]); 
       return 0; 
       }
    auto input = InputGroup(argv[1],"input");

    auto d = input.getInt("d",2);
    auto Ntrain = input.getInt("Ntrain",60000);
    auto Nsweep = input.getInt("Nsweep",50);
    auto imglen = input.getInt("imglen",14);
    auto cutoff = input.getReal("cutoff",1E-10);
    auto maxm = input.getInt("maxm",5000);
    auto minm = input.getInt("minm",max(10,maxm/2));
    auto ninitial = input.getInt("ninitial",100);
    auto Nthread = input.getInt("nthread",1);
    auto replace = input.getYesNo("replace",false);
    auto feature = input.getString("feature","normal");

    //Cost function settings
    auto lambda = input.getReal("lambda",0.);

    //Gradient settings
    auto method = input.getString("method","conj");
    auto alpha = input.getReal("alpha",0.01);
    auto clip = input.getReal("clip",1.0);
    auto Npass = input.getInt("Npass",4);
    auto cconv = input.getReal("cconv",1E-10);

    enum Feature { Normal, Series };
    auto ftype = Normal;
    if(feature == "normal") { ftype = Normal; }
    else if(feature == "series") { ftype = Series; }
    else
        {
        Error(format("feature=%s not recognized",feature));
        }

    auto labels = array<long,NL>{{0,1,2,3,4,5,6,7,8,9}};

    auto data = getData();
    auto train = getAllMNIST(data,{"Type",Train,"imglen",imglen});

    auto N = train.front().size();
    auto c = N/2;
    printfln("%d sites of dimension %d",N,d);
    SiteSet sites;
    if(fileExists("sites") )
        {
        sites = readFromFile<SiteSet>("sites");
        if(sites(1).m() != (long)d)
            {
            printfln("Error: d=%d but dimension of first site is %d",d,sites(1).m());
            EXIT
            }
        }
    else
        {
        sites = SiteSet(N,d);
        writeToFile("sites",sites);
        }

    //
    // Local feature map (a lambda function)
    //
    auto phi = [ftype,d](Real g, int n) -> Cplx
        {
        if(g < 0 || g > 255.) Error(format("Expected g=%f to be in [0,255]",g));
        auto x = g/255.;
        if(ftype == Normal)
            {
            if(d != 2)
                {
                println("d must be 2 for normal feature map");
                EXIT
                }
            return n==1 ? cos(Pi/2.*x) : sin(Pi/2.*x);
            }
        else if(ftype == Series)
            {
            return pow(x/4.,n-1);
            }
        return 0.;
        };

    println("Converting training set to MPS");
    auto trainmps = MPSArr{};
    auto counts = array<int,10>{};
    for(auto& img : train)
        {
        auto l = img.label();
        if(counts[l] >= Ntrain) continue;
        trainmps.at(l).push_back(makeMPS(sites,img,phi));
        ++counts[l];
        }
    auto totNtrain = stdx::accumulate(counts,0);
    printfln("Total of %d training images",totNtrain);

    Index L;
    MPS W;
    if(fileExists("W"))
        {
        println("Reading W from disk");
        W = readFromFile<MPS>("W",sites);
        L = findtype(W.A(c),Label);
        if(!L) 
            {
            printfln("Expected W to have Label type Index at site %d",c);
            EXIT
            }
        }
    else
        {
        //
        // If W not read from disk,
        // make initial W by summing training
        // states together
        //
        L = Index("L",10,Label);
        auto Lval = [&L](long n){ return L(1+n); };
        auto ipsis = vector<MPS>(labels.size());
        for(auto n : range(labels))
            {
            auto psis = vector<MPS>(ninitial);
            for(auto m : range(ninitial)) 
                {
                psis.at(m) = makeMPS(sites,randImg(train,labels[n]),phi);
                }
            printfln("Summing %d random label %d states",ninitial,labels[n]);
            ipsis.at(n) = sum(psis,{"Cutoff",1E-10,"Maxm",10});
            ipsis.at(n).Aref(c) *= 0.1*setElt(Lval(labels[n]));
            }
        printfln("Summing all %d label states together",ipsis.size());
        W = sum(ipsis,{"Cutoff",1E-8,"Maxm",10});
        W.Aref(c) /= norm(W.A(c));
        println("Done making initial W");
        }
    Print(overlap(W,W));
    println("Done making initial W");
    writeToFile("W",W);

    if(!findtype(W.A(c),Label)) Error(format("Label Index not on site %d",c));

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
    // Setup ts (training state structs)
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
    auto nextTrainN = [&labels,&trainmps](long l)
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

    //
    // Project training states (product states)
    // into environment of W MPS
    //
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

    // Set up initial t.v tensors
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

    auto C = quadcost(W.A(1)*W.A(2),ts,parallel_do,{"lambda",lambda});
    printfln("Before starting DMRG Cost = %.10f",C/totNtrain);

    train.clear(); //to save memory

    auto sweeps = Sweeps(Nsweep);
    sweeps.maxm() = maxm;
    sweeps.cutoff() = cutoff;
    sweeps.minm() = minm;

    auto args = Args{"lambda",lambda,
                     "Method",method,
                     "Npass",Npass,
                     "alpha",alpha,
                     "clip",clip,
                     "cconv",cconv,
                     "Replace",replace
                    };

    mldmrg(W,trainmps,ts,sweeps,parallel_do,args);

    println("Writing W to disk");
    writeToFile("W",W);

    return 0;
    }
