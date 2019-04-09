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
using std::string;

const size_t NL = 10;

//Struct holding info about training "states"
struct TState
    {
    SiteSet const& sites_;
    bool active = true;
    long n = -1;
    int l = -1;
    int d = 0;
    ITensor v;
    vector<Real> data;

    template<typename Func, typename ImgType>
    TState(int n_, int l_, 
           SiteSet const& sites, 
           ImgType const& img, 
           Func const& phi)
      : sites_(sites),
        n(n_),
        l(l_) 
        {
        auto N = sites.N();
        d = sites(1).m();
        data.resize(N*d);
        auto i = 0;
        for(auto j : range1(img.size()))
        for(auto n : range1(d))
            {
            data.at(i) = phi(img(j),n);
            ++i;
            }
        }
    Real
    operator()(int i, int n) const //1-indexed
        {
        //TODO: change .at() to []
        return data.at(d*i+n-d-1);
        }
    ITensor
    A(int i) const
        {
        auto store = DenseReal(d);
        for(auto n : range(d)) store[n] = operator()(i,1+n);
        return ITensor(IndexSet{sites_(i)},std::move(store));
        }

    };

class TrainStates
    {
    public:
    vector<TState> ts_;
    int N = 0;
    int currb_ = -1; //left env built to here
    bool dirmade_ = false;
    int Nbatch_ = 1;
    int batchSize_ = 0;
    int Nthread_ = 1;
    ParallelDo pd_;

    TrainStates(vector<TState> && ts,
                int N_, int Nthread, int Nbatch = 1)
      : ts_(move(ts)),
        N(N_), 
        Nbatch_(Nbatch),
        Nthread_(Nthread)
        { 
        int totNtrain = ts_.size();
        if(totNtrain%Nbatch != 0)
            {
            printfln("totNtrain=%d, Nbatch=%d, totNtrain%Nbatch=%d",
                     totNtrain, Nbatch, totNtrain%Nbatch);
            Error("totNtrain not commensurate with Nbatch");
            }
        batchSize_ = totNtrain/Nbatch;
        pd_ = ParallelDo(Nthread,batchSize_);
        for(auto& b : pd_.bounds())
            {
            printfln("Thread %d %d -> %d (%d)",b.n,b.begin,b.end,b.size());
            }
        }

    int
    size() const { return ts_.size(); }

    int
    Nthread() const { return Nthread_; }

    TState const& 
    front() const { return ts_.front(); }

    TState const&
    operator()(int i) const { return ts_.at(i); }
    TState &
    operator()(int i) { return ts_.at(i); }

    TState const&
    getState(int i) const { return ts_.at(i); }

    static string&
    writeDir() 
        {
        static string wd = "proj_images";
        return wd;
        }

    void
    init(MPS const& W)
        {
        if(not dirmade_)
            {
            auto cmd = "mkdir -p "+writeDir();
            std::system(cmd.c_str());
            dirmade_ = true;
            }
        auto nextE = vector<ITensor>(batchSize_);
        auto currE = vector<ITensor>(batchSize_);
        for(auto bn : range(Nbatch_))
            {
            auto batchStart = bn*batchSize_;
            for(auto n = N; n >= 3; --n)
                {
                pd_([&](Bound b){
                for(auto i = b.begin; i < b.end; ++i)
                    {
                    auto& t = ts_.at(batchStart+i);
                    if(n == N)
                        {
                        nextE.at(i) = (t.A(n)*W.A(n));
                        }
                    else
                        {
                        nextE.at(i) = (t.A(n)*W.A(n))*currE.at(i);
                        }
                    nextE[i].scaleTo(1.);
                    }});
                currE.swap(nextE);
                writeToFile(fname(bn,n),currE);
                }
            }
        setBond(1);
        }

    void
    setBond(int b)
        {
        if(currb_ == b) return;
        currb_ = b;
        auto lc = b-1;
        auto rc = b+2;
        auto useL = (lc > 0);
        auto useR = (rc < N+1);
        //TODO: don't realloc on every setBond call
        vector<ITensor> LE,RE;
        if(useL) LE = vector<ITensor>(batchSize_);
        if(useR) RE = vector<ITensor>(batchSize_);
        // Make effective image (4 site) tensors
        // Store in t.v of each elem t of ts
        for(auto bn : range(Nbatch_))
            {
            auto batchStart = bn*batchSize_;
            if(useL) readFromFile(fname(bn,lc),LE);
            if(useR) readFromFile(fname(bn,rc),RE);
            pd_([&](Bound b){
            for(auto i = b.begin; i < b.end; ++i)
                {
                auto& t = ts_.at(batchStart+i);
                t.v = t.A(lc+1)*t.A(rc-1);
                if(useL) t.v *= LE.at(i);
                if(useR) t.v *= RE.at(i);
                }});
            }
        LE.clear();
        RE.clear();
        }

    void
    shiftE(MPS const& W,
           int b, Direction dir)
        {
        auto c = (dir==Fromleft) ? b : b+1;
        auto dc = (dir==Fromleft) ? +1 : -1;

        auto prevc = (dir==Fromleft) ? b-1 : b+2;
        auto hasPrev = (prevc >= 1 && prevc <= N);

        if(hasPrev)
            {
            printfln("## Advancing E from %d to %d",prevc,c);
            }
        else
            {
            printfln("## Making new E at %d",c);
            }
        vector<ITensor> prevE;
        if(hasPrev) prevE = vector<ITensor>(batchSize_);
        auto nextE = vector<ITensor>(batchSize_);
        for(auto bn : range(Nbatch_))
            {
            auto batchStart = bn*batchSize_;
            if(hasPrev) readFromFile(fname(bn,prevc),prevE);
            pd_([&](Bound b){
            for(auto i = b.begin; i < b.end; ++i)
                {
                auto& t = ts_.at(batchStart+i);
                if(not hasPrev)
                    {
                    nextE.at(i) = t.A(c)*W.A(c);
                    }
                else
                    {
                    nextE.at(i) = prevE.at(i)*(t.A(c)*W.A(c));
                    }
                nextE.at(i).scaleTo(1.);
                }});
            writeToFile(fname(bn,c),nextE);
            }
        }


    template<typename Func>
    void
    execute(Func && f) const
        {
        for(auto bn : range(Nbatch_))
            {
            auto batchStart = bn*batchSize_;
            pd_([&f,batchStart,this](Bound b)
                {
                //printfln("B %d %d->%d",b.n,b.begin,b.end);
                for(auto i = batchStart+b.begin; i < batchStart+b.end; ++i)
                    {
                    auto& t = getState(i);
                    f(b.n,t);
                    }
                });
            }
        }

    private:

    string
    fname(int nb, int j)
        {
        return format("%s/B%03dE%05d",writeDir(),nb,j);
        }

    //ITensor&
    //E(int x, int nt)
    //    {
    //    return E_.at(x).at(nt);
    //    }
    //ITensor const&
    //E(int x, int nt) const
    //    {
    //    return E_.at(x).at(nt);
    //    }

    };

//
// Compute squared distance of the actual output
// of the model from the ideal output
//
Real
quadcost(ITensor B,
         TrainStates const& ts,
         Args const& args = Args::global())
    {
    auto NT = ts.size();
    auto lambda = args.getReal("lambda",0.);
    auto showlabels = args.getBool("ShowLabels",false);

    auto L = findtype(B,Label);
    if(!L) L = findtype(ts.front().v,Label);
    if(!L) 
        {
        Print(B);
        Print(ts.front().v);
        Error("Couldn't find Label index in quadcost");
        }

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
        reals[l] = vector<Real>(ts.Nthread(),0.);
        }
    auto ints = vector<int>(ts.Nthread(),0);
    //

    ts.execute([&](int nt, TState const& t)
        {
        auto weights = array<Real,10>{};
        auto P = B*t.v;
        auto dP = deltas[t.l] - P;
        reals[t.l].at(nt) += sqr(norm(dP));
        for(auto l : range(10))
            {
            weights[l] = std::abs(P.real(L(1+l)));
            }
        //print(t.n,": "); for(auto w : weights) print(" ",w); println();
        if(t.l == argmax(weights)) ints.at(nt) += 1;
        });

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
    auto ninc = (NT-ncor);
    printfln("Percent correct = %.4f%%, # incorrect = %d/%d",
             ncor*100./NT,ninc,ncor+ninc);
    return C;
    }

//
// Conjugate gradient
//
void
cgrad(ITensor & B,
      TrainStates & ts,
      Args const& args)
    {
    auto NT = ts.size();
    auto Npass = args.getInt("Npass");
    auto lambda = args.getReal("lambda",0.);
    auto cconv = args.getReal("cconv",1E-10);
    printfln("In cgrad, lambda = %.3E",lambda);

    auto L = findtype(B,Label);
    if(!L) L = findtype(ts.front().v,Label);
    if(!L) Error("Couldn't find Label index in cgrad");

    auto deltas = array<ITensor,10>{};
    for(auto l : range(10)) deltas[l] = setElt(L(1+l));

    //Workspace for parallel ops
    auto Nthread = ts.Nthread();
    auto tensors = vector<ITensor>(Nthread);
    auto reals = vector<Real>(Nthread);
    auto ints = vector<int>(Nthread);

    // Compute initial gradient
    for(auto& T : tensors) T = ITensor{};
    ts.execute([&](int nt, TState const& t)
            {
            auto P = B*t.v;
            auto dP = deltas[t.l] - P;
            tensors.at(nt) += dP*dag(t.v);
            });
    //for(auto n : range(tensors))
    //    {
    //    printfln("tensors[%d] = %s\n",n,tensors.at(n));
    //    }
    auto r = stdx::accumulate(tensors,ITensor{});
    if(lambda != 0.) r = r - lambda*B;

    auto p = r;
    for(auto pass : range1(Npass))
        {
        println("  Conj grad pass ",pass);
        // Compute p*A*p
        for(auto& r : reals) r = 0.;
        ts.execute([&](int nt, TState const& t)
            {
            // The matrix A is like outer
            // product of dag(v) and v, so
            // dag(p)*A*p is |p*v|^2
            auto pv = p*t.v;
            reals.at(nt) += sqr(norm(pv));
            });
        auto pAp = stdx::accumulate(reals,0.);
        pAp += lambda*sqr(norm(p));

        auto a = sqr(norm(r))/pAp;
        B = B + a*p;
        B.scaleTo(1.);

        if(pass == Npass) break;

        // Compute new gradient and cost function
        for(auto& T : tensors) T = ITensor();
        for(auto& r : reals) r = 0.;
        ts.execute([&](int nt, TState const& t)
            {
            auto P = B*t.v;
            auto dP = deltas[t.l] - P;
            tensors.at(nt) += dP*dag(t.v);
            reals.at(nt) += sqr(norm(dP));
            });
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
       TrainStates & ts,
       Sweeps const& sweeps,
       Args args)
    {
    auto N = W.N();
    auto NT = ts.size();

    auto method = args.getString("Method");
    auto replace = args.getBool("Replace",false);
    auto pause_step = args.getBool("PauseStep",false);

    auto Nthread = ts.Nthread();
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

        //auto lc = min(c,c+dc)-1;
        //auto rc = max(c,c+dc)+1;

        ts.setBond(b);

        printfln("Sweep %d Half %d Bond %d",sw,ha,c);

        // Save old bond tensor
        auto origm = commonIndex(W.A(c),W.A(c+dc)).m();
        auto oB = W.A(c)*W.A(c+dc);

        // B is the bond tensor we will optimize
        auto B = oB;
        B.scaleTo(1.);


        //
        // Optimize bond tensor B
        //
        if(method == "conj") cgrad(B,ts,args);
        else Error(format("method type \"%s\" not recognized",method));

        //
        // Report cost after optimization
        //
        printfln("Sweep %d Half %d Bond %d",sw,ha,c);

        //auto oC = quadcost(oB,ts,cargs);
        //auto C = quadcost(B,ts,cargs);
        //printfln("Cost = %.10f -> %.10f",oC/NT,C/NT);

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
        printfln("|B-newB| = %.3E",norm(B-newB));

        auto newC = quadcost(newB,ts,{cargs,"ShowLabels",true});
        printfln("--> After SVD, Cost = %.10f",newC/NT);

        //
        // Update E's (MPS environment tensors)
        // i.e. projection of training images into current "wings"
        // of the MPS W
        //
        ts.shiftE(W,b,ha==1?Fromleft:Fromright);

        if(fileExists("WRITE_WF"))
            {
            println("File WRITE_WF found");
            system("rm -f WRITE_WF");
            println("Writing W to disk");
            writeToFile("W",W);
            }

        if(fileExists("LAMBDA"))
            {
            auto lf = std::ifstream("LAMBDA");
            Real lambda = 0.;
            lf >> lambda;
            lf.close();
            args.add("lambda",lambda);
            system("rm -f LAMBDA");
            println("new lambda = ",lambda);
            }

        if(pause_step) PAUSE;

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

    int d = 2;
    auto datadir = input.getString("datadir","/Users/mstoudenmire/software/tnml/mllib/MNIST");
    auto Ntrain = input.getInt("Ntrain",60000);
    auto Nbatch = input.getInt("Nbatch",10);
    auto Nsweep = input.getInt("Nsweep",50);
    auto cutoff = input.getReal("cutoff",1E-10);
    auto maxm = input.getInt("maxm",5000);
    auto minm = input.getInt("minm",max(10,maxm/2));
    auto ninitial = input.getInt("ninitial",100);
    auto Nthread = input.getInt("nthread",1);
    auto replace = input.getYesNo("replace",false);
    auto pause_step = input.getYesNo("pause_step",false);
    //auto feature = input.getString("feature","normal");

    //Cost function settings
    auto lambda = input.getReal("lambda",0.);

    //Gradient settings
    auto method = input.getString("method","conj");
    auto alpha = input.getReal("alpha",0.01);
    auto clip = input.getReal("clip",1.0);
    auto Npass = input.getInt("Npass",4);
    auto cconv = input.getReal("cconv",1E-10);


    auto labels = array<long,NL>{{0,1,2,3,4,5,6,7,8,9}};

    auto train = readMNIST(datadir,mllib::Train,{"NT=",Ntrain});

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
    auto phi = [](Real g, int n) -> Real
        {
        if(g < 0 || g > 255.) Error(format("Expected g=%f to be in [0,255]",g));
        auto x = g/255.;
        return pow(x/4.,n-1);
        };

    println("Converting training set to MPS");
    auto states = vector<TState>();
    auto counts = array<int,10>{};
    auto n = 1;
    for(auto& img : train)
        {
        auto l = img.label;
        states.emplace_back(n++,l,sites,img,phi);
        ++counts[l];
        }
    int totNtrain = states.size();
    printfln("Total of %d training images",totNtrain);

    auto ts = TrainStates(move(states),N,Nthread,Nbatch);

    ////
    ////Visually inspect images to see if they look ok
    ////
    //n = 1;
    //for(auto& img : train)
    //    {
    //    writeGray(img,format("img%02d_L%d.png",n++,img.label));
    //    if(n > 10) break;
    //    }

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
    else if(fileExists("W0"))
        {
        println("Found separate W0,W1,...,W9 MPS: summing");
        L = Index("L",10,Label);
        auto Lval = [&L](long n){ return L(1+n); };
        auto ipsis = vector<MPS>(labels.size());
        for(auto n : range(labels))
            {
            auto& in = ipsis.at(n);
            in = readFromFile<MPS>(format("W%d",n));
            //in.position(1);
            in.Aref(c) *= setElt(Lval(labels[n]));
            //PrintData(in.A(c));
            }
        printfln("Summing all %d label states together",ipsis.size());
        W = sum(ipsis,{"Cutoff",1E-10});
        Print(W.A(c));
        println("Done making initial W");
        writeToFile("W",W);
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
        writeToFile("W",W);
        }
    Print(overlap(W,W));
    println("Done making initial W");

    train.clear(); //to save memory

    if(!findtype(W.A(c),Label)) Error(format("Label Index not on site %d",c));

    //
    // Project training states (product states)
    // into environment of W MPS
    //
    print("Projecting training states...");
    ts.init(W);
    println("done");

    println("Calling quadcost...");
    auto C = quadcost(W.A(1)*W.A(2),ts,{"lambda",lambda});
    printfln("Before starting DMRG Cost = %.10f",C/totNtrain);
    if(pause_step) PAUSE;

    auto sweeps = Sweeps(Nsweep,minm,maxm,cutoff);

    auto args = Args{"lambda",lambda,
                     "Method",method,
                     "Npass",Npass,
                     "alpha",alpha,
                     "clip",clip,
                     "cconv",cconv,
                     "Replace",replace,
                     "PauseStep",pause_step
                    };

    mldmrg(W,ts,sweeps,args);

    println("Writing W to disk");
    writeToFile("W",W);

    return 0;
    }
