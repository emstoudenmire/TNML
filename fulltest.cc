#include "util.h"
#include "itensor/util/input.h"
#include "itensor/mps/sweeps.h"

using std::string;

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
    auto feature = input.getString("feature","series");

    //auto labels = stdx::make_array<long>(2,5);
    //auto labels = stdx::make_array<long>(7,8);
    //auto labels = stdx::make_array<long>(7,8,9);
    auto labels = stdx::make_array<long>(0,1,2,3,4,5,6,7,8,9);

    print("Labels:"); for(auto l : labels) print(" ",l); println();

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

    enum Feature { Normal, Series };
    auto ftype = Series;
    if(feature == "norm" || feature == "normal")
        {
        ftype = Normal;
        }
    else if(feature == "series")
        {
        ftype = Series;
        }
    else
        {
        Error(format("feature type \"%s\" not recognized",feature));
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

    println("Converting test set to MPS");
    auto testmps = MPSArr{};
    for(auto& img : test)
        {
        auto l = img.label;
        auto& testmpsL = testmps.at(l);
        testmpsL.push_back(makeMPS(sites,img,phi));
        }

    auto totNtest = test.size();

    printfln("Total of %d testing images",totNtest);

    MPS psi;
    if(fileExists(fname))
        {
        psi = readFromFile<MPS>(fname,sites);
        }
    else
        {
        Error(format("Couldn't find file '%s'",fname));
        }

    printfln("Running full test of ",fname);
    fullTest(psi,testmps,labels);


    return 0;
    }
