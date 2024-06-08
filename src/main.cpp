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

    std::vector<Tensor<double, 1, 28, 28>> images_train;
    std::vector<Tensor<double, 10>> labels_train;

    std::vector<Tensor<double, 1, 28, 28>> images_test;
    std::vector<uint8_t> labels_test;

    while (!loader.eof())
    {
        auto [imgs, labels] = loader.read<1000>();
        for (int i = 0; i < 1000; i++)
        {
            images_train.push_back(Image<uint8_t, 28, 28>(imgs[i]).reshape<1, 28, 28>().scale(1.0 / 255.0));
            labels_train.push_back(nn::function::onehot<double, 10>(Tensor<double, 1>({double(labels[i])})));
        }
    }

    while (!loader_test.eof())
    {
        auto [imgs, labels] = loader_test.read<1000>();
        for (int i = 0; i < 1000; i++)
        {
            images_test.push_back(Image<uint8_t, 28, 28>(imgs[i]).reshape<1, 28, 28>().scale(1.0 / 255.0));
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

    Conv2d<double, 1, 32, 3, 3, 1, 1> l1(random<double, 32, 9>());
    ReLU<double, 32, 26, 26> l2;
    Conv2d<double, 32, 64, 3, 3, 1, 1> l3(random<double, 64, 9 * 32>());
    ReLU<double, 64, 24, 24> l4;
    MaxPool2d<double, 2, 2, 2, 2> l5;
    Dropout<double, 64, 12, 12> l6(25);
    Flatten<double> l7;
    Linear l8(random<double, 9216, 128>(), random<double, 128>());
    ReLU<double, 128> l9;
    Dropout<double, 128> l10(50);
    Linear l11(random<double, 128, 10>(), random<double, 10>());
    Softmax<double, 10> l12;

    CrossEntropy<double, 10> ce;

    // Linear l1(random<double, 784, 128>(), random<double, 128>());
    // ReLU<double, 128> r1;
    // Dropout<double,128> d1(20);
    // Linear l2(random<double, 128, 10>(), random<double, 10>());
    // Softmax<double, 10> s1;

    // CrossEntropy<double, 10> ce;

    // Sequential seq(l1, r1, d1, l2, s1);



    Sequential seq(l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12);

    for (int step = 0; step < 100; step++)
    {
        loader.reset();

        double total_loss = 0;
        int total_train = 0;

        for (int i = 0; i < 10; i++)
        {

            auto image = images_train[i];
            auto label = labels_train[i];
         
            auto dl1 = l1.forward(image);
            auto dl2 = l2.forward(dl1);
            auto dl3 = l3.forward(dl2);
            auto dl4 = l4.forward(dl3);
            auto dl5 = l5.forward(dl4);
            auto dl6 = l6.forward(dl5);
            auto dl7 = l7.forward(dl6);
            auto dl8 = l8.forward(dl7);
            auto dl9 = l9.forward(dl8);
            auto dl10 = l10.forward(dl9);
            auto dl11 = l11.forward(dl10);
            auto act = l12.forward(dl11);

            auto loss = ce.forward(act, label);

            // std::cout << act << "   " << loss << std::endl;

            auto _dact = ce.backward(act, label);

            auto _dl11 = l11.backward(dl10, _dact);
            auto _dl10 = l10.backward(dl9, _dl11);
            auto _dl9 = l9.backward(dl8, _dl10);
            auto _dl8 = l8.backward(dl7, _dl9);
            auto _dl7 = l7.backward(dl6, _dl8);
            auto _dl6 = l6.backward(dl5, _dl7);
            auto _dl5 = l5.backward(dl4, _dl6);
            auto _dl4 = l4.backward(dl3, _dl5);
            auto _dl3 = l3.backward(dl2, _dl4);
            auto _dl2 = l2.backward(dl1, _dl3);
            auto _dl1 = l1.backward(image, _dl2);



            // auto act = seq.forward(image);
            // auto loss = ce.forward(act, label);

            // auto dact = ce.backward(act, label);
            // seq.backward(dact);

            

            l1.step(5);
            l3.step(5);
            l8.step(0.5);
            l11.step(0.5);

            total_loss += loss;
            total_train++;


            std::cout << act << std::endl;
            std::cout << label << std::endl;

            progressbar(i, images_train.size());

            std::cout << std::endl << "Step " << step << " Avg Loss " << total_loss / total_train << std::endl;
            

            // progressbar(i, images_train.size());
            // std::cout << act << "   " << loss << std::endl;
            // std::cout << img << int(b) << std::endl;
            // std::cout << l1r <<act << label;
        }



        loader_test.reset();
        int correct = 0, total = 0;

        for (int i = 0; i < images_test.size(); i++)
        {
            auto image = images_test[i];
            auto label = labels_test[i];

            auto dl1 = l1.forward(image);
            auto dl2 = l2.forward(dl1);
            auto dl3 = l3.forward(dl2);
            auto dl4 = l4.forward(dl3);
            auto dl5 = l5.forward(dl4);
            auto dl6 = l6.forward(dl5);
            auto dl7 = l7.forward(dl6);
            auto dl8 = l8.forward(dl7);
            auto dl9 = l9.forward(dl8);
            auto dl10 = l10.forward(dl9);
            auto dl11 = l11.forward(dl10);
            auto act = l12.forward(dl11);

            std::cout << dl1 << dl2 << dl3 << dl4 << dl5 << dl6 << dl7 << dl8 << dl9 << dl10 << dl11 << act << std::endl;
            
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