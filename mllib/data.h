#ifndef __MLLIB_DATA_H_
#define __MLLIB_DATA_H_
#include "itensor/all.h"
#include "mllib/datatype.h"

using std::vector;
using std::string;

namespace mllib {

template<typename value_type_, int NL_>
struct Data
    {
    static const int NL = NL_;
    using value_type = value_type_;

    int n = -1;     //which data set element this is
    int label = -1; //label of which class it belongs to
    DataType type = DataType("None");
    vector<value_type> data;
    string name;

    Data() { }

    Data(int n_, int label_ = -1)
        : n(n_), label(label_)
        { }

    Data(int n_, int label_, int size_)
        : n(n_), label(label_), data(size_)
        { }

    Data(int n_, DataType type_, int label_ = -1)
        : n(n_), label(label_), type(type_)
        { }

    Data(int n_, DataType type_, int label_, int size_)
        : n(n_), label(label_), type(type_), data(size_)
        { }

    value_type static
    default_value() { return value_type{}; }

    int
    size() const { return data.size(); }

    //operator[] is 1-indexed
    value_type&
    operator[](int n) { return data[n]; }
    value_type
    operator[](int n) const { return data[n]; }

    //operator() is 1-indexed
    value_type&
    operator()(int n) { return data[n-1]; }
    value_type
    operator()(int n) const { return data[n-1]; }
    };

} //namespace mllib

#endif
