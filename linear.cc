//#include "util.h"
#include "itensor/all.h"
#include "mllib/mnist.h"

using namespace itensor;
using namespace mllib;
using std::vector;
using std::abs;
using std::min;
using std::string;

struct TImg
    {
    long l = 1000;
    int y = -1;
    Real dC = 0.;
    Vector img;

    TImg(long l_, int y_, Vector img_)
        {
        l = l_;
        y = y_;
        img = img_;
        }
    };

Real
cgrad(Vector & W,
      vector<TImg> const& train,
      Args const& args)
    {
    auto lambda = args.getReal("lambda",0.);
    auto Npass = args.getInt("Npass");
    int size = W.size();
    int NT = train.size();

    auto r = Vector(size);
    for(auto& t : train)
        {
        auto Wt = W*t.img;
        r += (t.y-Wt)*t.img;
        }
    r /= NT;
    if(lambda != 0.) r = r - lambda*W;

    auto C = 0.;
    auto p = r;
    for(auto pass : range1(Npass))
        {
        //Compute p*A*p
        Real pAp = 0.;
        for(auto& t : train)
            {
            auto pv = p*t.img;
            pAp += pv*pv;
            }
        pAp /= NT;
        pAp += lambda*(W*W);

        auto a = (r*r)/pAp;
        W = W + a*p;

        C = 0.;
        auto nr = Vector(size);
        for(auto& t : train)
            {
            auto dW = (t.y-W*t.img);
            nr += dW*t.img;
            C += sqr(dW);
            }
        nr /= NT;
        C /= NT;
        if(lambda != 0.) nr = nr - lambda*W;
        auto beta = (nr*nr)/(r*r);
        r = nr;

        C += lambda*(W*W);
        printfln("  %d C = %.10f",pass,C);

        if(fileExists("STOP"))
            {
            println("Found file STOP, exiting");
            system("rm -f STOP");
            return C;
            }

        p = r + beta*p;
        }
    return C;
    }

int
main(int argc, char* argv[])
    {
    if(argc != 2) return printfln("Usage: %s inputfile",argv[0]),0;

    auto in = InputGroup(argv[1],"input");

    auto datadir = in.getString("datadir","/Users/mstoudenmire/software/tnml/mllib/MNIST");
    auto Niter = in.getInt("Nlinear_iter",5000);
    auto Ntrain = in.getInt("Ntrain",60000);
    auto lambda = in.getReal("lambda",0.);
    auto L = in.getInt("label");

    auto d = 2;

    auto dotest = in.getYesNo("dotest",false);
    auto imgtype = dotest ? Test : Train;

    int Nimg = 0;
    print("Loading training data...");
    auto traindata = readMNIST(datadir,Train,{"NT=",Ntrain});
    auto testdata = readMNIST(datadir,Test);
    println("done");

    int N = traindata.front().data.size();

    auto phi = [](Real x, int n) -> Real
        {
        return n==1 ? 1. : x/4.;
        };

    auto size = 1+N;
    printfln("Vector size = %d",size);

    auto setup = [&](vector<TImg> & set,
                     vector<MNISTData> const & data)
        {
        for(auto& img : data)
            {
            auto l = img.label;
            auto y = (l==L) ? +1 : -1;
            auto v = Vector(size);
            int nn = 0;
            v(nn++) = 1.;
            for(auto j : range(N))
                {
                v(nn++) = phi(img[j],2);
                }
            set.emplace_back(l,y,v);
            }
        };
    print("Setting up training images...");
    auto train = vector<TImg>();
    setup(train,traindata);
    print("Setting up testing images...");
    auto test = vector<TImg>();
    setup(test,testdata);
    println("done");

    auto Vname = format("V%d",L);
    Vector V;
    if(fileExists(Vname))
        {
        println("Reading parameters from disk");
        V = readFromFile<Vector>(Vname);
        }
    else
        {
        V = Vector(size);
        randomize(V);
        V /= norm(V);
        }
    Print(norm(V));

    auto C = cgrad(V,train,{"Npass=",Niter,"lambda=",lambda});

    auto evaluate = [&](vector<TImg> const& set)
        {
        auto T = set.size();
        auto ncor = 0;
        auto Cnl = 0.;
        for(auto& t : set)
            {
            auto f = V*t.img;
            if(f*t.y > 0.) ++ncor;
            Cnl += sqr(f-t.y);
            }
        Cnl /= T;
        auto ninc = T-ncor;
        printfln("Percent correct = %.4f%%, #correct = %d/%d, #incorrect = %d/%d",
                 ncor*100./T,ncor,T,ninc,T);
        auto Cl = lambda*(V*V);
        printfln("C (= %.10f + %.10f) = %.10f",Cnl,Cl,Cnl+Cl);
        };
    println("Evaluating training set");
    evaluate(train);
    println("Evaluating testing set");
    evaluate(test);

    writeToFile(Vname,V);

    SiteSet sites;
    if(fileExists("sites"))
        {
        println("Reading previous site set from disk");
        sites = readFromFile<SiteSet>("sites");
        }
    else
        {
        sites = SiteSet(N,d);
        writeToFile("sites",sites);
        }

    //
    // Make MPS version of V
    //
    auto W = MPS(sites);
    auto M = 2;
    auto links = vector<Index>(2+N);
    for(auto j : range(2+N)) links.at(j) = Index(format("l%d",j),M,Link);

    for(auto j : range1(N))
        {
        auto l = links.at(j-1);
        auto r = links.at(j);
        auto s = sites(j);
        auto& A = W.Aref(j);
        A = ITensor(l,s,r);
        A.set(1,1,1,1.);
        A.set(2,1,2,1.);
        A.set(2,2,1,V(j));
        }
    auto l0 = links.at(0);
    auto A0 = ITensor(l0);
    A0.set(l0(1),V(0));
    A0.set(l0(2),1.);
    W.Aref(1) *= A0;
    W.Aref(N) *= setElt(links.at(N)(1));

    W.position(1);

    Print(overlap(W,W));
    Print(sqr(norm(V)));

    writeToFile(format("W%d",L),W);

    return 0;
    }

