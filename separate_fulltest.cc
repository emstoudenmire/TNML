#include "util.h"
#include "itensor/util/input.h"
#include "itensor/mps/sweeps.h"

using std::string;

template<size_t nlabel>
void
fullTest(vector<MPS> const& Ws,
         MPSArr const& set_test,
         array<long,nlabel> labels)
    {
    long NL = Ws.size();
    auto counts = array<long,10>{};
    auto costs = array<Real,10>{};
    auto ninc = array<long,10>{};
    auto tninc = 0;
    auto tncor = 0;
    auto ntest = 0;

    while(true)
        {
        bool got_state = false;
        for(auto l : labels)
            {
            auto c = counts[l];
            auto& testL = set_test.at(l);
            if(c >= (long)testL.size()) continue;
            counts[l]++;
            ++ntest;
            got_state = true;
            auto& testimg = testL.at(c);
            auto weights = array<Real,10>{};
            //printfln("Label = %d",l);
            //print("weights:");
            for(auto n : range(NL))
                {
                auto o = overlap(Ws[n],testimg);
                weights.at(n) = fabs(o);
                costs.at(n) += (n==l) ? sqr(o-1) : sqr(o);
                //print(" ",weights[n]);
                }
            //println();
            auto pl = argmax(weights);
            //printfln("Predicted Label = %d",pl);
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
        if(!got_state) break;
        }
    printfln("%d/%d correct (%.2f%%), %d/%d incorrect (%.2f%%)",
             tncor,ntest,tncor*100./ntest,tninc,ntest,tninc*100./ntest);
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

    auto tC = 0.;
    println("Cost functions:");
    for(auto l : range(10))
        {
        tC += costs[l];
        printfln("  Digit %d C = %.20f",l,costs[l]);
        }
    printfln("Total C = %.20f",tC);
    }

int
main(int argc, const char* argv[])
    {
    if(argc != 2) 
       { 
       printfln("Usage: %s inputfile",argv[0]); 
       return 0; 
       }
    auto input = InputGroup(argv[1],"input");

    int d = 2;
    auto datadir = input.getString("datadir","/Users/mstoudenmire/software/tnml/mllib/MNIST");
    auto fname = input.getString("fname","W");
    auto imglen = input.getInt("imglen",28);

    //auto labels = stdx::make_array<long>(2,5);
    //auto labels = stdx::make_array<long>(7,8);
    //auto labels = stdx::make_array<long>(7,8,9);
    auto labels = stdx::make_array<long>(0,1,2,3,4,5,6,7,8,9);
    auto NL = labels.size();

    print("Labels:"); for(auto l : labels) print(" ",l); println();

    enum Feature { Normal, Series };
    auto ftype = Normal;
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

    auto test = readMNIST(datadir,mllib::Test);

    auto N = test.front().size();
    SpinHalf sites;
    if(fileExists("sites") )
        {
        sites = readFromFile<SpinHalf>("sites");
        }
    else
        {
        Error("Couldn't find file 'sites'");
        }

    println("Converting test set to MPS");
    auto testmps = MPSArr{};
    auto counts = array<int,10>{};
    for(auto& img : test)
        {
        auto l = img.label;
        testmps.at(l).push_back(makeMPS(sites,img,phi));
        ++counts[l];
        }
    auto totNtest = stdx::accumulate(counts,0);

    printfln("Total of %d testing images",totNtest);

    //long c = 1;
    //Index L;
    //auto totnorm2 = 0.;
    auto Ws = vector<MPS>(NL);
    for(auto n : range(NL))
        {
        Ws[n] = readFromFile<MPS>(format("L%d/W%d",n,n),sites);
        //totnorm2 += sqr(norm(Ws[n]));
        }
    //for(auto n : range(NL)) Ws[n] /= sqrt(totnorm2);



    printfln("Running full test");
    fullTest(Ws,testmps,labels);


    return 0;
    }
