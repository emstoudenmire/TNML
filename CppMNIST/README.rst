mnist
=====

Simple C++ reader for MNIST dataset

Usage
-----

You have to include mnist_reader.hpp in your code:

.. code:: cpp

    #include "mnist/mnist_reader.hpp"

And then, you can use the function :code:`read_dataset()` that returns a struct with a
vector of training images, one of test images, one of training labels and one of
test labels:

.. code:: cpp

   auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

The first two template arguments defines which container will be used for the
collections. The third argument is the type that is used to store a pixel (value
between 0 and 255) and the last one is the type that is used to store a label
(value between 0 and 9). The types in the example are the types by default, you
can use any STL container for the containers and any type that is castable from
:code:`unsigned char` for the second.

Windows
-------

The mnist_reader.hpp include is known not to compile on Visual Studio and Intel
Compiler. You can use mnist_reader_less.hpp to overcome this:

.. code:: cpp

    #include "mnist/mnist_reader_less.hpp"

This is almost equivalent to mnist_reader.hpp, except that the containers are
forced to be vector.

Utilities
---------

The header mnist_utils.hpp contains two utilities that can be useful when using
MNIST in machine learning activities:

* :code:`binarize_dataset(dataset)` Binarize all the images in the data set
* :code:`normalize_dataset(dataset)` Normalize all the images in the data set to
  a zero mean and unit variance.

License
-------

The header files are distributed under the terms of the MIT License. The MNIST
files are not my property. If used in a paper, you'll need to cite the reference
paper, as indicated in the `official website
<http://yann.lecun.com/exdb/mnist/>`_.
