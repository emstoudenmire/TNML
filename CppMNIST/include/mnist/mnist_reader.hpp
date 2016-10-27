//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>

namespace mnist {

template<template<typename...> class Container, typename Image, typename Label>
struct MNIST_dataset 
    {
    Container<Image> training_images;
    Container<Image> test_images;
    Container<Label> training_labels;
    Container<Label> test_labels;

    void 
    resize_training(std::size_t new_size)
        {
        if(training_images.size() > new_size)
            {
            training_images.resize(new_size);
            training_labels.resize(new_size);
            }
        }

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

inline uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position){
    auto header = reinterpret_cast<uint32_t*>(buffer.get());

    auto value = *(header + position);
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

template<template<typename...> class Container = std::vector, typename Image, typename Functor>
void read_mnist_image_file(Container<Image>& images, const std::string& path, std::size_t limit, Functor func){
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Error opening file" << std::endl;
    } else {
        auto size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();

        auto magic = read_header(buffer, 0);

        if(magic != 0x803){
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        } else {
            auto count = read_header(buffer, 1);
            auto rows = read_header(buffer, 2);
            auto columns = read_header(buffer, 3);

            if(size < count * rows * columns + 16){
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            } else {
                //Skip the header
                //Cast to unsigned char is necessary cause signedness of char is
                //platform-specific
                auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

                if(limit > 0 && count > limit){
                    count = limit;
                }

                images.reserve(count);

                for(size_t i = 0; i < count; ++i){
                    images.push_back(func());

                    for(size_t j = 0; j < rows * columns; ++j){
                        auto pixel = *image_buffer++;
                        images[i][j] = static_cast<typename Image::value_type>(pixel);
                    }
                }
            }
        }
    }
}

template<template<typename...> class  Container = std::vector, typename Label = uint8_t>
void 
read_mnist_label_file(Container<Label>& labels, const std::string& path, std::size_t limit = 0)
    {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Error opening file" << std::endl;
    } else {
        auto size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();

        auto magic = read_header(buffer, 0);

        if(magic != 0x801){
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        } else {
            auto count = read_header(buffer, 1);

            if(size < count + 8)
                {
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
                } 
            else 
                {
                //Skip the header
                //Cast to unsigned char is necessary cause signedness of char is
                //platform-specific
                auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

                if(limit > 0 && count > limit){
                    count = limit;
                }

                labels.resize(count);

                for(size_t i = 0; i < count; ++i)
                    {
                    auto label = *label_buffer++;
                    labels[i] = static_cast<Label>(label);
                    }
                }
            }
        }
    }

template<template<typename...> class Container = std::vector, typename Image, typename Functor>
Container<Image> 
read_training_images(std::size_t limit, Functor func, std::string dirname = "mnist")
    {
    auto fname = dirname + "/train-images-idx3-ubyte";
    Container<Image> images;
    read_mnist_image_file<Container, Image>(images, fname, limit, func);
    return images;
    }

template<template<typename...> class Container = std::vector, typename Image, typename Functor>
Container<Image> 
read_test_images(std::size_t limit, Functor func, std::string dirname = "mnist")
    {
    auto fname = dirname+"/t10k-images-idx3-ubyte";
    Container<Image> images;
    read_mnist_image_file<Container, Image>(images,fname, limit, func);
    return images;
    }

template<template<typename...> class Container = std::vector, typename Label = uint8_t>
Container<Label> 
read_training_labels(std::size_t limit, std::string dirname = "mnist")
    {
    auto fname = dirname + "/train-labels-idx1-ubyte";
    Container<Label> labels;
    read_mnist_label_file<Container, Label>(labels,fname, limit);
    return labels;
    }

template<template<typename...> class Container = std::vector, typename Label = uint8_t>
Container<Label> read_test_labels(std::size_t limit, std::string dirname = "mnist")
    {
    auto fname = dirname + "/t10k-labels-idx1-ubyte";
    Container<Label> labels;
    read_mnist_label_file<Container, Label>(labels, fname, limit);
    return labels;
    }

template<template<typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> 
read_dataset_3d(std::size_t training_limit = 0, std::size_t test_limit = 0, std::string dirname = "mnist")
    {
    MNIST_dataset<Container, Image, Label> dataset;

    dataset.training_images = read_training_images<Container, Image>(training_limit,[]{return Image(1, 28, 28);},dirname);
    dataset.training_labels = read_training_labels<Container, Label>(training_limit,dirname);

    dataset.test_images = read_test_images<Container, Image>(test_limit, []{return Image(1, 28, 28);},dirname);
    dataset.test_labels = read_test_labels<Container, Label>(test_limit,dirname);

    return dataset;
    }

template<template<typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> 
read_dataset_direct(std::size_t training_limit = 0, std::size_t test_limit = 0, std::string dirname = "mnist")
    {
    MNIST_dataset<Container, Image, Label> dataset;

    dataset.training_images = read_training_images<Container, Image>(training_limit, []{ return Image(1 * 28 * 28); },dirname);
    dataset.training_labels = read_training_labels<Container, Label>(training_limit,dirname);

    dataset.test_images = read_test_images<Container, Image>(test_limit, []{ return Image(1 * 28 * 28); },dirname);
    dataset.test_labels = read_test_labels<Container, Label>(test_limit,dirname);

    return dataset;
    }

template<template<typename...> class Container = std::vector, 
         template<typename...> class Sub = std::vector, 
         typename Pixel = uint8_t, 
         typename Label = uint8_t>
MNIST_dataset<Container,Sub<Pixel>,Label> 
read_dataset(std::size_t training_limit = 0, std::size_t test_limit = 0, std::string dirname = "mnist")
    {
    return read_dataset_direct<Container, Sub<Pixel>>(training_limit, test_limit,dirname);
    }

} //end of namespace mnist

#endif
