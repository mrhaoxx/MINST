#include <print>

#include "image.hpp"
#include "dataloader.hpp"

#include "tensor.hpp"
#include "nn.hpp"

#include <iostream>
#include <vector>

#include <mdspan>


int main()
{


    DataLoader<uint8_t, 28, 28, true> loader("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
    DataLoader<uint8_t, 28, 28, true> loader_test("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");

    std::vector<Tensor<double, 784>> images_train;
    std::vector<Tensor<double, 10>> labels_train;

    std::vector<Tensor<double, 784>> images_test;
    std::vector<uint8_t> labels_test;

    while (!loader.eof())
    {
        auto [imgs, labels] = loader.read<1000>();
        for (int i = 0; i < 1000; i++)
        {
            images_train.push_back(Image<uint8_t, 28, 28>(imgs[i]).reshape<784>().scale(1.0 / 255.0));
            labels_train.push_back(nn::function::onehot<double, 10>(Tensor<double, 1>({double(labels[i])})));
        }
    }

    while (!loader_test.eof())
    {
        auto [imgs, labels] = loader_test.read<1000>();
        for (int i = 0; i < 1000; i++)
        {
            images_test.push_back(Image<uint8_t, 28, 28>(imgs[i]).reshape<784>().scale(1.0 / 255.0));
            labels_test.push_back(labels[i]);
        }
    }

    std::print("Loaded {} images for train\n", images_train.size());
    std::print("Loaded {} images for test\n", images_test.size());

    // test:
    // {
    //     auto t1 = images_train[0];
    //     auto t2 = t1.reshape<28, 28>();
    //     std::cout << t1.reshape<28, 28>() << t2.pad<1,1>() << std::endl;

    // }
    // goto test;

    srand(42);

    // Conv2d<double, 1, 32, 3, 3, 1, 1> l1(random<double, 32, 9>());
    // ReLU<double, 32, 26, 26> l2;
    // Conv2d<double, 32, 64, 3, 3, 1, 1> l3(random<double, 64, 9 * 32>());
    // ReLU<double, 64, 24, 24> l4;
    // MaxPool2d<double, 2, 2, 2, 2> l5;
    // Dropout<double, 64, 12, 12> l6(25);
    // Flatten<double> l7;
    // Linear l8(random<double, 9216, 128>(), random<double, 128>());
    // ReLU<double, 128> l9;
    // Dropout<double, 128> l10(50);
    // Linear l11(random<double, 128, 10>(), random<double, 10>());
    // Softmax<double, 10> l12;

    // CrossEntropy<double, 10> ce;

    Linear l1(random<double, 784, 128>(), random<double, 128>());
    ReLU<double, 128> r1;
    Dropout<double,128> d1(20);
    Linear l2(random<double, 128, 10>(), random<double, 10>());
    Softmax<double, 10> s1;

    CrossEntropy<double, 10> ce;

    Sequential seq(l1, r1, d1, l2, s1);




    // Sequential seq(l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12);

    for (int step = 0; step < 100; step++)
    {
        loader.reset();

        double total_loss = 0;
        int total_train = 0;

        for (int i = 0; i < images_train.size(); i++)
        {

            auto image = images_train[i];
            auto label = labels_train[i];


            auto act = seq.forward(image);
            auto loss = ce.forward(act, label);

            auto dact = ce.backward(act, label);
            seq.backward(dact);

            l1.step(0.005);
            l2.step(0.005);

            // l1.step(5);
            // l3.step(5);
            // l8.step(0.5);
            // l11.step(0.5);

            total_loss += loss;
            total_train++;


            // std::cout << act << std::endl;
            // std::cout << label << std::endl;

            progressbar(i, images_train.size());

            // std::cout << std::endl << "Step " << step << " Loss " << loss << std::endl;
        
        }



        loader_test.reset();
        int correct = 0, total = 0;

        for (int i = 0; i < images_test.size(); i++)
        {
            auto image = images_test[i];
            auto label = labels_test[i];

            auto act = seq.forward(image);

            auto max_idx = std::distance((*act.data).begin(), std::max_element((*act.data).begin(), (*act.data).end()));

            if (max_idx == label)
            {
                correct++;
            }else{
                std::cout << "Predicted: " << max_idx << " Label: " << int(label) << std::endl;
                std::cout << Image(image.reshape<28, 28>().scale<uint8_t>(255)) << std::endl;
                std::cout << act << std::endl;
            }
            total++;
        }

        std::cout << "Step " << step << " Avg Loss " << total_loss / total_train << std::endl;

        std::cout << "Verify " << correct << "/" << total << "  " << correct / float(total) << std::endl;
    }
    return 0;
}