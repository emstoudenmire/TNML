//
// Based on code by Baptiste Wicht
// Merged into a single file by E. Miles Stoudenmire
//
//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef mnist_reader_hpp
#define mnist_reader_hpp

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <cmath>

#include "itensor/all.h"
using namespace itensor;
#include "mllib/data.h"

using std::vector;
using std::array;

namespace mnist {

/*!
 * \brief Extract the MNIST header from the given buffer
 * \param buffer The current buffer
 * \param position The current reading positoin
 * \return The value of the mnist header
 */
inline uint32_t 
read_header(const std::unique_ptr<char[]>& buffer, size_t position) 
    {
    auto header = reinterpret_cast<uint32_t*>(buffer.get());
    auto value = *(header + position);
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
    }

/*!
 * \brief Read a MNIST file inside a raw buffer
 * \param path The path to the image file
 * \return The buffer of byte on success, a nullptr-unique_ptr otherwise
 */
inline std::unique_ptr<char[]> 
read_mnist_file(const std::string& path, uint32_t key) 
    {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (!file) {
        std::cout << "Error opening file" << std::endl;
        return {};
    }

    auto size = file.tellg();
    std::unique_ptr<char[]> buffer(new char[size]);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), size);
    file.close();

    auto magic = read_header(buffer, 0);

    if (magic != key) 
        {
        std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        return {};
        }

    auto count = read_header(buffer, 1);

    if(magic == 0x803) 
        {
        auto rows    = read_header(buffer, 2);
        auto columns = read_header(buffer, 3);

        if (size < count * rows * columns + 16) 
            {
            std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            return {};
            }
        } 
    else if (magic == 0x801) 
        {
        if (size < count + 8) 
            {
            std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            return {};
            }
        }

    return buffer;
    }


/*!
 * \brief Represents a complete mnist dataset
 * \tparam Container The container to use
 * \tparam Image The type of image
 * \tparam Label The type of label
 */
template <template <typename...> class Container, typename Image, typename Label>
struct MNIST_dataset 
    {
    Container<Image> training_images; ///< The training images
    Container<Image> test_images;     ///< The test images
    Container<Label> training_labels; ///< The training labels
    Container<Label> test_labels;     ///< The test labels

    /*!
     * \brief Resize the training set to new_size
     * If new_size is less than the current size, this function has no effect.
     * \param new_size The size to resize the training sets to.
     */
    void 
    resize_training(std::size_t new_size) 
        {
        if(training_images.size() > new_size) 
            {
            training_images.resize(new_size);
            training_labels.resize(new_size);
            }
        }

    /*!
     * \brief Resize the test set to new_size
     * If new_size is less than the current size, this function has no effect.
     * \param new_size The size to resize the test sets to.
     */
    void 
    resize_test(std::size_t new_size) 
        {
        if(test_images.size() > new_size) 
            {
            test_images.resize(new_size);
            test_labels.resize(new_size);
            }
        }
    };


/*!
 * \brief Read a MNIST image file inside the given container
 * \param images The container to fill with the images
 * \param path The path to the image file
 * \param limit The maximum number of elements to read (0: no limit)
 * \param func The functor to create the image object
 */
template <template <typename...> class Container = std::vector, typename Image, typename Functor>
void 
read_mnist_image_file(Container<Image>& images, 
                      const std::string& path, 
                      std::size_t limit, 
                      Functor func) 
    {
    auto buffer = read_mnist_file(path, 0x803);

    if(buffer) 
        {
        auto count   = read_header(buffer, 1);
        auto rows    = read_header(buffer, 2);
        auto columns = read_header(buffer, 3);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

        if (limit > 0 && count > limit) {
            count = limit;
        }

        images.reserve(count);

        for(size_t i = 0; i < count; ++i) 
            {
            images.push_back(func());

            for(size_t j = 0; j < rows * columns; ++j) 
                {
                auto pixel   = *image_buffer++;
                images[i][j] = static_cast<typename Image::value_type>(pixel);
                }
            }
        }
    }

/*!
 * \brief Read a MNIST label file inside the given container
 * \param labels The container to fill with the labels
 * \param path The path to the label file
 * \param limit The maximum number of elements to read (0: no limit)
 */
template <template <typename...> class Container = std::vector, typename Label = uint8_t>
void 
read_mnist_label_file(Container<Label>& labels, const std::string& path, std::size_t limit = 0) 
    {
    auto buffer = read_mnist_file(path, 0x801);

    if(buffer) 
        {
        auto count = read_header(buffer, 1);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

        if(limit > 0 && count > limit) count = limit;

        labels.resize(count);

        for(size_t i = 0; i < count; ++i) 
            {
            auto label = *label_buffer++;
            labels[i]  = static_cast<Label>(label);
            }
        }
    }


/*!
 * \brief Read all training images and return a container filled with the images.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \param func The functor to create the image objects.
 * \return Container filled with the images
 */
template <template <typename...> class Container = std::vector, typename Image, typename Functor>
Container<Image> 
read_training_images(const std::string& folder, std::size_t limit, Functor func) 
    {
    Container<Image> images;
    read_mnist_image_file<Container, Image>(images, folder + "/train-images-idx3-ubyte", limit, func);
    return images;
    }

/*!
 * \brief Read all test images and return a container filled with the images.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \param func The functor to create the image objects.
 * \return Container filled with the images
 */
template <template <typename...> class Container = std::vector, typename Image, typename Functor>
Container<Image> 
read_test_images(const std::string& folder, std::size_t limit, Functor func) 
    {
    Container<Image> images;
    read_mnist_image_file<Container, Image>(images, folder + "/t10k-images-idx3-ubyte", limit, func);
    return images;
    }

/*!
 * \brief Read all training label and return a container filled with the labels.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \return Container filled with the labels
 */
template <template <typename...> class Container = std::vector, typename Label = uint8_t>
Container<Label> 
read_training_labels(const std::string& folder, std::size_t limit) 
    {
    Container<Label> labels;
    read_mnist_label_file<Container, Label>(labels, folder + "/train-labels-idx1-ubyte", limit);
    return labels;
    }

/*!
 * \brief Read all test label and return a container filled with the labels.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \return Container filled with the labels
 */
template <template <typename...> class Container = std::vector, typename Label = uint8_t>
Container<Label> 
read_test_labels(const std::string& folder, 
                 std::size_t limit) 
    {
    Container<Label> labels;
    read_mnist_label_file<Container, Label>(labels, folder + "/t10k-labels-idx1-ubyte", limit);
    return labels;
    }


/*!
 * \brief Read dataset from some location.
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> 
read_dataset_direct(const std::string& folder, 
                    std::size_t training_limit = 0, 
                    std::size_t test_limit = 0) 
    {
    MNIST_dataset<Container, Image, Label> dataset;
    dataset.training_images = 
        read_training_images<Container, Image>(folder, 
                                               training_limit, 
                                               [] { return Image(1 * 28 * 28); });
    dataset.training_labels = 
        read_training_labels<Container, Label>(folder, training_limit);

    dataset.test_images = 
        read_test_images<Container, Image>(folder, 
                                           test_limit, 
                                           []{ return Image(1 * 28 * 28); });

    dataset.test_labels = read_test_labels<Container, Label>(folder, test_limit);

    return dataset;
    }



/*!
 * \brief Read dataset from some location.
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
MNIST_dataset<Container, Sub<Pixel>, Label> read_dataset(const std::string& folder, std::size_t training_limit = 0, std::size_t test_limit = 0) {
    return read_dataset_direct<Container, Sub<Pixel>>(folder, training_limit, test_limit);
}

//
// MNIST Utils
//


/*!
 * \brief Binarize each sub range inside the given range
 * \param values The collection of ranges to binarize
 * \param threshold The threshold for binarization
 */
template <typename Container>
void binarize_each(Container& values, double threshold = 30.0) {
    for (auto& vec : values) {
        for (auto& v : vec) {
            v = v > threshold ? 1.0 : 0.0;
        }
    }
}

/*!
 * \brief Return the mean value of the elements inside the given range
 * \param container The range to compute the average from
 * \return The average value of the range
 */
template <typename Container>
double mean(const Container& container) {
    double mean = 0.0;
    for (auto& value : container) {
        mean += value;
    }
    return mean / container.size();
}

/*!
 * \brief Return the standard deviation of the elements inside the given range
 * \param container The range to compute the standard deviation from
 * \param mean The mean of the given range
 * \return The standard deviation of the range
 */
template <typename Container>
double stddev(const Container& container, double mean) {
    double std = 0.0;
    for (auto& value : container) {
        std += (value - mean) * (value - mean);
    }
    return std::sqrt(std / container.size());
}

/*!
 * \brief Normalize each sub range inside the given range
 * \param values The collection of ranges to normalize
 */
template <typename Container>
void normalize_each(Container& values) {
    for (auto& vec : values) {
        //zero-mean
        auto m = mnist::mean(vec);
        for (auto& v : vec) {
            v -= m;
        }
        //unit variance
        auto s = mnist::stddev(vec, 0.0);
        for (auto& v : vec) {
            v /= s;
        }
    }
}

/*!
 * \brief Binarize the given MNIST dataset
 * \param dataset The dataset to binarize
 */
template <typename Dataset>
void binarize_dataset(Dataset& dataset) {
    mnist::binarize_each(dataset.training_images);
    mnist::binarize_each(dataset.test_images);
}

/*!
 * \brief Normalize the given MNIST dataset to zero-mean and unit variance
 * \param dataset The dataset to normalize
 */
template <typename Dataset>
void normalize_dataset(Dataset& dataset) {
    mnist::normalize_each(dataset.training_images);
    mnist::normalize_each(dataset.test_images);
}

} //end of namespace mnist


namespace mllib {


using MNISTData = Data<Real,10ul>;

std::vector<MNISTData> inline
readMNIST(string datadir,
          DataType type,
          Args const& args = Args::global())
    {
    using pix_t = uint8_t;
    using label_t = uint8_t;
    const auto NL = MNISTData::NL;

    auto NT = args.getInt("NT",50000);

    auto tset = vector<MNISTData>{};

    auto data = mnist::read_dataset<std::vector,std::vector,pix_t,label_t>(datadir);
    using data_container = decltype(data.training_images);
    using label_container = decltype(data.training_labels);
    data_container set;
    label_container labels;
    if(type == Train) 
        {
        set = std::move(data.training_images);
        labels = std::move(data.training_labels);
        }
    else if(type == Test) 
        {
        set = std::move(data.test_images);
        labels = std::move(data.test_labels);
        }

    auto counts = array<int,NL>{};
    auto Total = 0;
    for(auto l : labels)
        {
        if(counts[l] >= NT) continue;
        counts[l] += 1;
        Total += 1;
        }

    tset.resize(Total);
    counts = array<int,NL>{};
    int n = 0;
    for(auto i : range(set.size()))
        {
        int l = labels.at(i);
        if(counts[l] >= NT) continue;
        counts[l] += 1;
        auto& t = tset.at(n++);
        auto& img = set.at(i);
        t.n = i;
        t.label = l;
        t.type = type;
        t.data.resize(img.size());
        for(auto j : range(img)) t.data[j] = img[j]/255.;
        }

    printfln("%sing set consists of %d images:",type,tset.size());
    for(auto l : range(NL))
        {
        printfln("  %d of label %d",counts[l],l);
        }

        //Print(data.training_images.size());
        //Print(data.test_images.size());
        //Print(data.training_images[0].size());
        //Print(Real(data.training_images[0][0]));
        //Print((std::is_same<uint8_t,decltype(data.training_images[0][0])>::value));
        //for(auto j : range(10))
        //    {
        //    int r = 0;
        //    for(auto& v : data.training_images[j])
        //        {
        //        auto x = (1./255)*static_cast<Real>(v);
        //        if(x > 0.5) print("*");
        //        else if(x > 0.2) print("-");
        //        else print(" ");
        //        r += 1;
        //        if(r >= 28) 
        //            {
        //            r = 0;
        //            println();
        //            }
        //        }
        //    println();
        //    println();
        //    }

    return tset;
    }

} //namespace mllib

#endif
