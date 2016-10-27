#pragma once
#include "itensor/mps/sweeps.h"
#include "paralleldo.h"


ITensor
polarU(ITensor const& T,
       std::initializer_list<Index> inds)
    {
    ITensor F{IndexSet(inds)},S,G;
    svd(T,F,S,G,{"Truncate",false});
    auto fi = commonIndex(F,S),
         gi = commonIndex(G,S);
    F *= delta(fi,gi);
    return F*G;
    }

//Struct holding info about training "states"
struct TState
    {
    long n = -1;
    int l = -1;
    ITensor v;
    vector<ITensor> E;
    };

MPS*
getTrainState(MPSArr & trainmps,
              TState const& t) 
    { 
    return &(trainmps.at(t.l).at(t.n)); 
    }

MPS const*
getTrainState(MPSArr const& trainmps,
              TState const& t) 
    { 
    return &(trainmps.at(t.l).at(t.n)); 
    }

ITensor
mult(int lc, 
     ITensor B, 
     TState const& t, 
     MPSArr const& trainmps)
    {
    auto rc = lc+3;
    auto& tmps = *getTrainState(trainmps,t);
    auto N = tmps.N();
    if(lc > 0) B *= t.E.at(lc);
    if(rc < N+1) B *= t.E.at(rc);
    B *= tmps.A(lc+1);
    B *= tmps.A(lc+2);
    return B;
    }

ITensor
mult(int lc, 
     Real f, 
     TState const& t, 
     MPSArr const& trainmps)
    {
    auto rc = lc+3;
    auto& tmps = *getTrainState(trainmps,t);
    auto N = tmps.N();
    ITensor v;
    if(lc > 0 && rc < N+1) 
        {
        v = t.E.at(lc);
        v *= t.E.at(rc);
        }
    else if(lc > 0)   { v = t.E.at(lc); }
    else if(rc < N+1) { v = t.E.at(rc); }
    v *= tmps.A(lc+1);
    v *= tmps.A(lc+2);
    return v*f;
    }


Real
quadcost(ITensor B,
         vector<TState> const& ts,
         MPSArr const& trainmps,
         int L,
         ParallelDo const& parallel_do,
         Args const& args = Args::global())
    {
    auto lambda = args.getReal("lambda",0.);
    auto precalc = args.getBool("Precalc",false);
    auto lc = precalc ? 0 : args.getInt("LC");
    if(args.getBool("Normalize",false))
        {
        B /= norm(B);
        }
    auto reals = vector<Real>(parallel_do.Nthread(),0.);
    parallel_do([&](Bound b)
        {
        for(auto i = b.begin; i < b.end; ++i)
            {
            auto& t = ts.at(i);
            auto Bt = precalc ? B*t.v : mult(lc,B,t,trainmps);
            auto P = Bt.real();
            auto dP = (t.l==L) ? (1.-P) : (-P);
            reals.at(b.n) += sqr(dP);
            }
        });
    auto C = stdx::accumulate(reals,0.);
    C += lambda*sqr(norm(B));
    return C;
    }

//
// Exact solution of linear regression problem
// Only works for rather small number of training samples
//
void
exact(ITensor & B, 
      vector<TState> & ts,
      ParallelDo const& parallel_do, 
      Args const& args)
    {
    auto lambda = args.getReal("lambda",0.);
    auto L = args.getInt("Label");
    auto pcut = args.getReal("PCut",1E-8);
    
    auto I = Index("I",ts.size());

    auto inds = stdx::reserve_vector<Index>(5);
    for(auto& i : ts.front().v.inds()) inds.push_back(i);
    inds.push_back(I);

    auto yL = ITensor(I);
    auto Phi = ITensor(inds);

    for(auto n : range(ts))
        {
        Phi += setElt(I(1+n)) * ts[n].v;
        if(ts[n].l == L) yL.set(I(1+n),1.);
        }

    auto U = ITensor(I);
    ITensor S,V;
    svd(Phi,U,S,V);

    auto pseudoInv = [pcut,lambda](Real s)
        {
        if(s > pcut) 
            {
            return s/(s*s+lambda);
            }
        return 0.;
        };
    auto Sinv = apply(S,pseudoInv);

    auto PhiInv = U*Sinv*V;
    B = yL*PhiInv;
    }

//
// Conjugate gradient
//
void
cgrad(ITensor & B,
      vector<TState> & ts,
      MPSArr const& trainmps,
      ParallelDo const& parallel_do, 
      Args const& args)
    {
    auto Ntrain = ts.size();
    auto L = args.getInt("Label");
    auto Npass = args.getInt("Npass");
    auto lambda = args.getReal("lambda",0.);
    auto cconv = args.getReal("cconv",1E-10);
    auto precalc = args.getBool("Precalc",false);
    auto lc = precalc ? 0 : args.getInt("LC");


    //Workspace for parallel ops
    auto Nthread = parallel_do.Nthread();
    auto tensors = vector<ITensor>(Nthread);
    auto reals = vector<Real>(Nthread);
    auto ints = vector<int>(Nthread);

    for(auto& T : tensors) T = ITensor{};
    parallel_do(
        [&](Bound b)
            {
            for(auto i = b.begin; i < b.end; ++i)
                {
                auto& t = ts.at(i);
                auto Bt = precalc ? B*t.v : mult(lc,B,t,trainmps);
                auto P = Bt.real();
                auto dP = (t.l==L) ? (1.-P) : (-P);
                tensors.at(b.n) += precalc ? dP*dag(t.v) : mult(lc,dP,t,trainmps);
                }
            }
        );
    auto r = stdx::accumulate(tensors,ITensor{});
    if(lambda != 0.) r = r - lambda*B;

    if(norm(r) < cconv) 
        {
        printfln("  |r| = %.1E < %.1E, not optimizing",norm(r),cconv);
        return;
        }

    auto p = r;
    for(auto pass : range1(Npass))
        {
        println("  Conj grad pass ",pass);
        //Compute p*A*p
        //
        // TODO: may be able to compute pAp
        //       implicitly from information
        //       obtainable when making next r
        //
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
                    auto pv = precalc ? p*t.v : mult(lc,p,t,trainmps);
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

        for(auto& T : tensors) T = ITensor();
        for(auto& r : reals) r = 0.;
        parallel_do(
            [&](Bound b)
                {
                for(auto i = b.begin; i < b.end; ++i)
                    {
                    auto& t = ts.at(i);
                    //
                    // TODO
                    // May be able to skip this B*t.v multiply
                    // in favor of just multiplying each (p*t.v) * dag(t.v)
                    // See Shewchuk paper Eq. (47)
                    //
                    auto Bt = precalc ? B*t.v : mult(lc,B,t,trainmps);
                    auto P = Bt.real();
                    auto dP = (t.l==L) ? (1.-P) : (-P);
                    tensors.at(b.n) += precalc ? dP*dag(t.v) : mult(lc,dP,t,trainmps);
                    reals.at(b.n) += sqr(dP);
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
        printfln("  %d C = %.10f",pass,C/Ntrain);

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

void
fast_cgrad(ITensor & B,
      vector<TState> & ts,
      MPSArr const& trainmps,
      ParallelDo const& parallel_do, 
      Args const& args)
    {
    auto Ntrain = ts.size();
    auto L = args.getInt("Label");
    auto Npass = args.getInt("Npass");
    auto lambda = args.getReal("lambda",0.);
    auto cconv = args.getReal("cconv",1E-10);
    auto precalc = args.getBool("Precalc",false);
    auto lc = precalc ? 0 : args.getInt("LC");

    //Workspace for parallel ops
    auto Nthread = parallel_do.Nthread();
    auto tensors = vector<ITensor>(Nthread);
    auto reals = vector<Real>(Nthread);
    auto ints = vector<int>(Nthread);

    for(auto& T : tensors) T = ITensor{};
    parallel_do(
        [&](Bound b)
            {
            for(auto i = b.begin; i < b.end; ++i)
                {
                auto& t = ts.at(i);
                auto Bt = precalc ? B*t.v : mult(lc,B,t,trainmps);
                auto P = Bt.real();
                auto dP = (t.l==L) ? (1.-P) : (-P);
                tensors.at(b.n) += precalc ? dP*dag(t.v) : mult(lc,dP,t,trainmps);
                }
            }
        );
    auto r = stdx::accumulate(tensors,ITensor{});
    if(lambda != 0.) r = r - lambda*B;

    if(norm(r) < cconv) 
        {
        printfln("  |r| = %.1E < %.1E, not optimizing",norm(r),cconv);
        return;
        }

    auto p = r;
    for(auto pass : range1(Npass))
        {
        print("  Conj grad pass ",pass," ");
        //Compute p*A*p
        //
        // TODO: may be able to compute pAp
        //       implicitly from information
        //       obtainable when making next r
        //
        for(auto& T : tensors) T = ITensor();
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
                    auto pv = precalc ? p*t.v : mult(lc,p,t,trainmps);
                    auto pvr = pv.real();
                    reals.at(b.n) += sqr(pvr);
                    tensors.at(b.n) += (pvr)*dag(t.v);
                    }
                }
            );
        auto pAp = stdx::accumulate(reals,0.);
        pAp += lambda*sqr(norm(p));

        auto a = sqr(norm(r))/pAp;
        B = B + a*p;
        B.scaleTo(1.);

        if(pass == Npass) 
            {
            println();
            break;
            }

        auto Ap = stdx::accumulate(tensors,ITensor{});
        auto nr = r - a*Ap;
        if(lambda != 0.) nr = nr - lambda*B;

        auto beta = sqr(norm(nr)/norm(r));
        r = nr;
        r.scaleTo(1.);

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
// Truncated pseudo inverse solver
//
void
pinv(ITensor & B, 
     vector<TState> & ts,
     ParallelDo const& parallel_do, 
     Args const& args)
    {
    auto r = args.getInt("Ntarget");
    auto L = args.getInt("Label");
    auto Npass = args.getInt("Npass");

    auto lambda = args.getReal("lambda",0.);
    auto pcut = args.getReal("PCut",1E-8);
    printfln("Using pcut = %.2E",pcut);

    auto pseudoInv = [pcut,lambda](Real s)
        {
        if(s > pcut) return s/(s*s+lambda);
        return 0.;
        };
    

    //Exact check
        //{
        //Print(ts.size());
        //auto I = Index("I",ts.size());

        //auto inds = stdx::reserve_vector<Index>(5);
        //for(auto& i : ts.front().v.inds()) inds.push_back(i);
        //inds.push_back(I);

        //auto Phi = ITensor(inds);
        //for(auto n : range(ts))
        //    {
        //    Phi += setElt(I(1+n)) * ts[n].v;
        //    }

        //Print(Phi);

        //auto U = ITensor(I);
        //ITensor S,V;
        //svd(Phi,U,S,V);
        //PrintData(S);
        //}

    auto inds = stdx::reserve_vector<Index>(5);
    for(auto& i : ts.front().v.inds()) inds.push_back(i);

    auto a = Index("a",r);
    inds.push_back(a);
    auto V = random(ITensor(move(inds)));
    V = polarU(V,{a});

    //println("Initializing V with SVD of B");
    //Print(B);
    //Print(r);
    //auto l = Index("l",1);
    //auto BB = B*setElt(l(1));
    //ITensor U(l),S,V;
    //svd(BB,U,S,V,{"Maxm",r,"Minm",r,"RightIndexName","a"});
    //PrintData(S);
    //auto a = commonIndex(V,S);
    //Print(V);

    auto E = ITensor{};
    for(auto& t : ts)
        {
        E += (V*t.v)*t.v;
        }
    auto lastVE = (V*E).real();
    printfln("Initial V*E = %.20f",lastVE);

    ITensor F{a},D,G;
    for(auto pass : range1(Npass))
        {
        println("Making E");
        auto E = ITensor{};
        for(auto& t : ts)
            {
            E += (V*t.v)*t.v;
            }

        //Polar decomp
        println("Polar U");
        //F = ITensor{a};
        //D = ITensor{};
        //G = ITensor{};
        svd(E,F,D,G,{"Truncate",false});
        auto fi = commonIndex(F,D);
        auto gi = commonIndex(G,D);
        V = (F*delta(fi,gi))*G;

        auto VE = (V*E).real();
        printfln("%d V*E = %.20f",pass,VE);

        if(fabs(VE-lastVE) < 1E-4)
            {
            break;
            }
        else
            {
            lastVE = VE;
            }
        }
    PrintData(D);
    auto Einv = F*apply(D,pseudoInv)*G;

    //Construct new B
    auto yUS = ITensor{};
    for(auto& t : ts)
        {
        if(t.l != L) continue;
        yUS += (t.v*V);
        }
    B = yUS*Einv;
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
       Args args = Args::global())
    {
    auto N = W.N();
    auto Ntrain = ts.size();
    auto Nthread = parallel_do.Nthread();
    auto reals = vector<Real>(Nthread);

    auto method = args.getString("Method");
    auto L = args.getInt("Label");
    auto Wname = args.getString("Wname");
    auto pause_steps = args.getBool("PauseSteps",false);
    auto precalc = args.getBool("Precalc",true);
    auto lambda = args.getReal("lambda",0.);

    auto cargs = Args{args,"Normalize",false,"Precalc",precalc};

    for(auto sw : range1(sweeps))
    {
    printfln("Sweep %d maxm=%d",sw,sweeps.maxm(sw));
    auto svd_args = Args{"Cutoff",sweeps.cutoff(sw),
                         "Maxm",sweeps.maxm(sw),
                         "Minm",sweeps.minm(sw),
                         "Sweep",sw};
    auto noise = sweeps.noise(sw);
    for(int b = 1, ha = 1; ha <= 2; sweepnext(b,ha,N))
        {
        auto c = (ha==1) ? b : b+1;
        auto dc = (ha==1) ? +1 : -1;

        auto lc = min(c,c+dc)-1;
        auto rc = max(c,c+dc)+1;
        args.add("LC",lc);
        cargs.add("LC",lc);

        printfln("Sweep %d Half %d Bond %d",sw,ha,c);

        //Save old core tensor
        auto origm = commonIndex(W.A(c),W.A(c+dc)).m();
        auto oB = W.A(c)*W.A(c+dc);

        Print(norm(oB));
        auto B = oB;
        B.scaleTo(1.);

        //Make effective image (4 site) tensors
        //Store in t.v of each elem t of ts
        //
        if(precalc)
            {
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
            }

        if(method == "conj") cgrad(B,ts,trainmps,parallel_do,args);
        else if(method == "fast_conj") fast_cgrad(B,ts,trainmps,parallel_do,args);
        else if(method == "exact") exact(B,ts,parallel_do,args);
        else if(method == "pinv") 
            {
            auto BB = B;
            pinv(BB,ts,parallel_do,args);
            auto pC = quadcost(BB,ts,trainmps,L,parallel_do,cargs);
            printfln("After pinv, Cost = %.20f",pC/Ntrain);

            cgrad(B,ts,trainmps,parallel_do,args);
            }
        else Error(format("method type \"%s\" not recognized",method));

        //
        // Report after optimization
        //
        printfln("Sweep %d Half %d Bond %d",sw,ha,c);

        auto oC = quadcost(oB,ts,trainmps,L,parallel_do,cargs);
        auto C = quadcost(B,ts,trainmps,L,parallel_do,cargs);
        printfln("Cost = %.10f --> %.10f",oC/Ntrain,C/Ntrain);
        if(lambda > 0.)
            {
            auto RC = lambda*sqr(norm(B));
            printfln("Reg. cost RC = %.10f (%.10f)",RC/Ntrain,RC);
            printfln("Cost - RC = %.10f (%.10f)",(C-RC)/Ntrain,C-RC);
            }

        //
        // SVD B back apart into MPS tensors
        //
        long newm = -1;
        if(noise < 1E-14)
            {
            auto U = W.A(c);
            ITensor V,S;
            auto spec = svd(B,U,S,V,svd_args);
            W.setA(c,U);
            W.setA(c+dc,S*V);
            ////Keep norm the same after SVD
            //W.Aref(c+dc) *= norm(B)/norm(W.A(c+dc));
            newm = commonIndex(W.A(c),W.A(c+dc)).m();
            printfln("SVD trunc err = %.2E",spec.truncerr());
            }
        else
            {
            auto l = uniqueIndex(W.A(c),W.A(c+dc),Link);
            auto s = findtype(W.A(c),Site);
            auto rho = B;
            if(l) rho *= prime(B,s,l);
            else  rho *= prime(B,s);
            ITensor drho;
            for(auto& t : ts)
                {
                auto& tmps = *getTrainState(trainmps,t);
                auto dr = B;
                if(ha == 1 && c > 1)
                    {
                    dr *= t.E.at(lc);
                    dr *= t.E.at(lc);
                    }
                else if(ha == 2 && c < N-1)
                    {
                    dr *= t.E.at(rc);
                    dr *= t.E.at(rc);
                    }
                if(l) dr *= prime(dr,s,l);
                else  dr *= prime(dr,s);
                drho += dr;
                drho.scaleTo(1.);
                }
            rho += noise*drho;
            ITensor UU,DD;
            auto spec = diagHermitian(rho,UU,DD,svd_args);
            W.setA(c+dc,UU*B);
            W.setA(c,UU);
            newm = commonIndex(DD,UU).m();
            printfln("Trunc err = %.2E",spec.truncerr());
            }

        printfln("Original m=%d, New m=%d",origm,newm);

        auto newB = W.A(c)*W.A(c+dc);
        Print(norm(newB));

        auto newC = quadcost(newB,ts,trainmps,L,parallel_do,cargs);
        printfln("--> After SVD, Cost = %.10f (%.10f)",newC/Ntrain,newC);

        if(newC > 1.1*C) println("> 10% larger C after SVD");

        if(pause_steps) { PAUSE }

        //
        // Update E's
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
                //Don't think we should normalize the t.E's because
                //that's a nonlinear transformation that wouldn't happen
                //to the test images as they go through the decision function
                //t.E[L].at(c) /= norm(t.E.at(c));
                E.at(c).scaleTo(1.);
                }
            });

        if(fileExists("WRITE_WF"))
            {
            println("File WRITE_WF found");
            system("rm -f WRITE_WF");
            printfln("Writing %s to disk",Wname);
            writeToFile(Wname,W);
            }

        } //loop over c,dc

    printfln("Writing %s to disk",Wname);
    writeToFile(Wname,W);

    } //loop over sweeps

    } //mldmrg
