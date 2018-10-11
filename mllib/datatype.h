#ifndef __MLLIB_DATATYPE_H_
#define __MLLIB_DATATYPE_H_

#include <array>
#include <cstring>
#include <cctype>
#include <iostream>

#ifdef DEBUG
#define CHECK_IND(X) check_ind(X);
#else
#define CHECK_IND(X)
#endif

namespace mllib {

size_t inline constexpr 
DTSize() { return 7ul; }

size_t inline constexpr 
DTStoreSize() { return 1+DTSize(); }

struct DataType
    {
    using storage_type = std::array<char,DTStoreSize()>;
    private:
    storage_type name_;
    public:

    explicit
    DataType(const char* name);

    size_t static constexpr
    size() { return DTSize(); }

    const char*
    c_str() const { assert(name_[size()]=='\0'); return &(name_[0]); }

    operator const char*() const { return c_str(); }

    const char&
    operator[](size_t i) const { CHECK_IND(i) return name_[i]; }

    char&
    operator[](size_t i) { CHECK_IND(i) return name_[i]; }

    private:
    void
    check_ind(size_t j) const
        {
        if(j >= size()) throw std::runtime_error("DataType: index out of range");
        }
    };


bool inline
operator==(DataType const& t1, DataType const& t2)
    {
    for(size_t j = 0; j < DataType::size(); ++j)
        if(t1[j] != t2[j]) return false;
    return true;
    }

bool inline
operator!=(DataType const& t1, DataType const& t2)
    {
    return !operator==(t1,t2);
    }

void inline
write(std::ostream& s, DataType const& t)
    {
    for(size_t n = 0; n < DataType::size(); ++n)
        s.write((char*) &t[n],sizeof(char));
    }

void inline
read(std::istream& s, DataType& t)
    {
    for(size_t n = 0; n < DataType::size(); ++n)
        s.read((char*) &(t[n]),sizeof(char));
    }

inline DataType::
DataType(const char* name)
    {
    name_.fill('\0');
    auto len = std::min(std::strlen(name),size());
#ifdef DEBUG
    if(std::strlen(name) > size())
        {
        std::cout << "Warning: DataType name will be truncated to " << size() << " chars" << std::endl;
        }
#endif
    for(size_t j = 0; j < len; ++j)
        {
        name_[j] = name[j];
        }
    assert(name_[size()]=='\0');
    }

const auto Test   = DataType("Test");
const auto Train  = DataType("Train");

#undef CHECK_IND

} //namespace mllib
#endif
